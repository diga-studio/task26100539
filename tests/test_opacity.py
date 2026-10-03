"""透明度与嵌套组递归合成测试。"""

import numpy as np

from flyerdash import GroupNode, ImageNode, Project, render_region


def _solid(r, g, b, a, opacity):
    return ImageNode(bytes([r, g, b, a]), 1, 1, opacity=opacity)


def _pixel(project, x=0, y=0):
    out = render_region(project, 72, (x, y, 1, 1))
    return list(out)


def test_overlapping_red_blue_with_group_opacity():
    """红在下蓝在上, 两图像节点及父组透明度均为 0.5 -> 85,0,170,96。"""
    red = _solid(255, 0, 0, 255, 0.5)
    blue = _solid(0, 0, 255, 255, 0.5)
    group = GroupNode(children=[red, blue], opacity=0.5)
    project = Project(4, 4, [group])
    assert _pixel(project) == [85, 0, 170, 96]


def test_nested_group_opacity_recursion():
    """第二层组再次施加透明度, 验证递归规则而非只验证单组。"""
    red = _solid(255, 0, 0, 255, 0.5)
    blue = _solid(0, 0, 255, 255, 0.5)
    inner = GroupNode(children=[red, blue], opacity=0.5)
    outer = GroupNode(children=[inner], opacity=0.5)
    project = Project(4, 4, [outer])

    # 实数参考: 内组预乘 (0.125, 0, 0.25, 0.375), 外组再乘 0.5
    # -> (0.0625, 0, 0.125, 0.1875)
    a = 0.1875
    r = 0.0625 / a  # 1/3
    b = 0.125 / a  # 2/3
    expected = [
        int(np.floor(r * 255 + 0.5)),
        0,
        int(np.floor(b * 255 + 0.5)),
        int(np.floor(a * 255 + 0.5)),
    ]
    got = _pixel(project)
    assert got == expected == [85, 0, 170, 48]
    # 与单组结果不同, 证明递归透明度确实生效
    assert got[3] != 96


def test_deeply_nested_groups_match_real_arithmetic():
    """8 层嵌套组, 每层透明度 0.9, 与实数计算逐通道误差 <= 1。"""
    leaf = _solid(200, 100, 50, 255, 1.0)
    node = leaf
    for _ in range(8):
        node = GroupNode(children=[node], opacity=0.9)
    project = Project(2, 2, [node])

    factor = 0.9**8
    a = factor
    expected = [
        int(np.floor((200 / 255) * 255 + 0.5)),
        int(np.floor((100 / 255) * 255 + 0.5)),
        int(np.floor((50 / 255) * 255 + 0.5)),
        int(np.floor(a * 255 + 0.5)),
    ]
    got = _pixel(project)
    for g, e in zip(got, expected):
        assert abs(g - e) <= 1, (got, expected)


def test_group_opacity_not_distributed_to_children():
    """组透明度必须整体施加: 子节点重叠时与分摊到子节点的结果不同。"""
    # 两个半透明子节点重叠; 若错误地把组透明度 0.5 分摊给子节点,
    # 重叠区 alpha 会不同 (1-(1-.25)^2=.4375 vs 整体 (1-(1-.5)^2)*.5=.375)
    c1 = _solid(255, 0, 0, 255, 0.5)
    c2 = _solid(0, 0, 255, 255, 0.5)
    group = GroupNode(children=[c1, c2], opacity=0.5)
    project = Project(2, 2, [group])
    got = _pixel(project)
    assert got[3] == 96  # 0.375*255 四舍五入; 分摊实现会得到 112
