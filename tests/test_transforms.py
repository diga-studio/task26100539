"""变换、边界与区域切分一致性测试。

用一个独立的纯 Python 逐像素参考实现核对库输出,
并改变区域切分位置验证拼接结果与完整导出一致。
"""

import math

import numpy as np

from flyerdash import (
    GroupNode,
    ImageNode,
    Project,
    render,
    render_region,
)


# ---------------- 独立参考实现 (不依赖库内部代码) ----------------

def _ref_sample_source(src, w, h, lx, ly):
    """双线性采样预乘 alpha; 越界源像素按透明黑参与。"""
    u, v = lx - 0.5, ly - 0.5
    i0, j0 = math.floor(u), math.floor(v)
    fx, fy = u - i0, v - j0
    acc = [0.0, 0.0, 0.0, 0.0]
    for dj in (0, 1):
        for di in (0, 1):
            i, j = i0 + di, j0 + dj
            wgt = ((1 - fx) if di == 0 else fx) * ((1 - fy) if dj == 0 else fy)
            if 0 <= i < w and 0 <= j < h:
                r, g, b, a = src[(j * w + i) * 4 : (j * w + i) * 4 + 4]
                a /= 255.0
                premul = [r / 255.0 * a, g / 255.0 * a, b / 255.0 * a, a]
            else:
                premul = [0.0, 0.0, 0.0, 0.0]
            for k in range(4):
                acc[k] += wgt * premul[k]
    return acc


def _ref_compose_node(node, m, px, py, dpi, dst):
    """把节点在输出像素 (px,py) 处的贡献 source-over 到 dst (预乘)。

    m 为节点父坐标到画布坐标的变换, 节点自身变换在此复合。
    """
    scale = dpi / 72.0
    cx, cy = (px + 0.5) / scale, (py + 0.5) / scale
    m = _ref_mul(m, node.transform)
    if isinstance(node, ImageNode):
        a, b, c, d, e, f = m
        det = a * d - b * c
        lx = (d * (cx - e) - c * (cy - f)) / det
        ly = (-b * (cx - e) + a * (cy - f)) / det
        s = _ref_sample_source(node.pixels, node.width, node.height, lx, ly)
        s = [v * node.opacity for v in s]
    else:  # GroupNode: 先合成子节点, 再整体施加组透明度
        sub = [0.0, 0.0, 0.0, 0.0]
        for child in node.children:
            _ref_compose_node(child, m, px, py, dpi, sub)
        s = [v * node.opacity for v in sub]
    keep = 1.0 - s[3]
    for k in range(4):
        dst[k] = s[k] + dst[k] * keep


def _ref_mul(o, i):
    oa, ob, oc, od, oe, of = o
    ia, ib, ic, id_, ie, if_ = i
    return (
        oa * ia + oc * ib, ob * ia + od * ib,
        oa * ic + oc * id_, ob * ic + od * id_,
        oa * ie + oc * if_ + oe, ob * ie + od * if_ + of,
    )


def _ref_render_pixel(project, dpi, px, py):
    dst = [0.0, 0.0, 0.0, 0.0]
    for node in project.nodes:
        _ref_compose_node(node, (1, 0, 0, 1, 0, 0), px, py, dpi, dst)
    a = dst[3]
    out = []
    for k in range(3):
        v = dst[k] / a if a > 0 else 0.0
        out.append(min(255, max(0, int(math.floor(v * 255 + 0.5)))))
    out.append(min(255, max(0, int(math.floor(a * 255 + 0.5)))))
    return out


# ---------------- 测试场景 ----------------

def _make_project():
    """旋转 + 非均匀缩放 + 负坐标 + 带非零 RGB 的全透明源像素。"""
    # 3x2 源图, 含一个全透明但 RGB 非零的像素 (预乘语义下应不可见)
    src = bytes(
        [
            255, 0, 0, 255,   0, 255, 0, 128,   0, 0, 255, 255,
            255, 255, 0, 200, 99, 55, 11, 0,    255, 0, 255, 64,
        ]
    )
    theta = math.radians(30.0)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    # 旋转 30 度 + 非均匀缩放 (2.5, 1.25) + 平移到含负局部坐标落点的位置
    sx, sy = 2.5, 1.25
    transform = (
        cos_t * sx, sin_t * sx,
        -sin_t * sy, cos_t * sy,
        6.0, 3.0,
    )
    img = ImageNode(src, 3, 2, transform=transform, opacity=0.8)
    # 负坐标: 图像平移到画布左上角之外
    img_neg = ImageNode(src, 3, 2, transform=(1, 0, 0, 1, -1.5, -0.5), opacity=1.0)
    inner = GroupNode(children=[img_neg, img], opacity=0.9)
    outer = GroupNode(
        children=[inner],
        transform=(0.5, 0.0, 0.0, 0.5, 1.0, 1.0),
        opacity=0.7,
    )
    return Project(20, 16, [outer])


def test_matches_reference_implementation():
    project = _make_project()
    dpi = 72
    full = render_region(project, dpi, (0, 0, 20, 16))
    got = np.frombuffer(full, dtype=np.uint8).reshape(16, 20, 4)
    mismatches = 0
    for py in range(16):
        for px in range(20):
            ref = _ref_render_pixel(project, dpi, px, py)
            for k in range(4):
                if abs(int(got[py, px, k]) - ref[k]) > 1:
                    mismatches += 1
    assert mismatches == 0
    # 场景确实覆盖了非平凡内容: 存在半透明与全透明像素
    assert (got[..., 3] > 0).any() and (got[..., 3] == 0).any()
    assert ((got[..., 3] > 0) & (got[..., 3] < 255)).any()


def test_region_tiling_consistency():
    """任意区域切分拼接结果必须与完整导出一致。"""
    project = _make_project()
    dpi = 150  # 非 72 dpi, 输出 42x34 (向上取整)
    out_w, out_h = 42, 34
    full = np.frombuffer(
        render_region(project, dpi, (0, 0, out_w, out_h)), dtype=np.uint8
    ).reshape(out_h, out_w, 4)

    # 多种切分: 不同原点/大小的矩形组合, 覆盖变换边缘与透明边缘
    tilings = [
        [(0, 0, out_w, out_h)],
        [(0, 0, 1, 1), (1, 0, out_w - 1, 1), (0, 1, 1, out_h - 1),
         (1, 1, out_w - 1, out_h - 1)],
        [(x, y, min(7, out_w - x), min(5, out_h - y))
         for y in range(0, out_h, 5) for x in range(0, out_w, 7)],
        [(13, 9, 11, 8), (0, 0, 13, out_h), (13, 0, out_w - 13, 9),
         (24, 0, out_w - 24, 17), (13, 17, 11, out_h - 17),
         (24, 17, out_w - 24, out_h - 17), (0, 0, 0, 0) if False else
         (0, 0, 0, 0)],
    ]
    # 最后一个 tiling 用显式覆盖重建, 去掉占位
    tilings[3] = [r for r in tilings[3] if r[2] > 0 and r[3] > 0]
    # 用前三种 tiling 中第一种全覆盖校验; 对 tiling 3 单独校验区域值
    for tiling in tilings[:3]:
        stitched = np.zeros_like(full)
        for (x, y, w, h) in tiling:
            if w == 0 or h == 0:
                continue
            part = np.frombuffer(
                render_region(project, dpi, (x, y, w, h)), dtype=np.uint8
            ).reshape(h, w, 4)
            stitched[y : y + h, x : x + w] = part
        assert (stitched == full).all()
    # tiling 3: 逐区域与完整导出对应区域比较 (区域可重叠/不全覆盖)
    for (x, y, w, h) in tilings[3]:
        part = np.frombuffer(
            render_region(project, dpi, (x, y, w, h)), dtype=np.uint8
        ).reshape(h, w, 4)
        assert (part == full[y : y + h, x : x + w]).all()


def test_render_file_matches_region(tmp_path):
    """render 写出的 PNG 与 render_region 逐区域一致。"""
    from PIL import Image

    project = _make_project()
    dpi = 72
    path = tmp_path / "out.png"
    render(project, dpi, str(path))
    with Image.open(path) as im:
        assert im.size == (20, 16)
        disk = np.array(im.convert("RGBA"))
    for (x, y, w, h) in [(0, 0, 20, 16), (3, 2, 9, 7), (15, 12, 5, 4)]:
        part = np.frombuffer(
            render_region(project, dpi, (x, y, w, h)), dtype=np.uint8
        ).reshape(h, w, 4)
        assert (part == disk[y : y + h, x : x + w]).all()


def test_transparent_source_pixel_with_nonzero_rgb_is_invisible():
    """全透明但 RGB 非零的源像素在预乘语义下不贡献颜色。"""
    src = bytes([250, 200, 150, 0])
    img = ImageNode(src, 1, 1)
    project = Project(3, 3, [img])
    got = render_region(project, 72, (0, 0, 3, 3))
    # 该像素 alpha=0 -> 预乘后全零, 双线性采样也只混入透明黑
    assert list(got) == [0] * (3 * 3 * 4)


def test_negative_coordinates_and_canvas_edge():
    """负坐标图像在画布外部分被裁掉, 边缘像素与参考一致。"""
    src = bytes([255, 0, 0, 255] * 4)  # 2x2 全红
    # 半像素偏移使像素中心落在源像素边界上, 边缘混入透明黑
    img = ImageNode(src, 2, 2, transform=(1, 0, 0, 1, -1.5, -1.5))
    project = Project(4, 4, [img])
    got = np.frombuffer(render_region(project, 72, (0, 0, 4, 4)),
                        dtype=np.uint8).reshape(4, 4, 4)
    # 图像覆盖画布点 [-1.5,0.5)x[-1.5,0.5): 像素 (0,0) 中心 (0.5,0.5)
    # 落在局部 (2.0,2.0), 只有源像素 (1,1) 以 0.25 权重在界内
    assert 0 < got[0, 0, 3] < 255
    assert got[0, 0, 0] == 255  # 未预乘化后红色不变
    assert got[3, 3, 3] == 0
    # 与独立参考实现逐像素核对
    for py in range(4):
        for px in range(4):
            ref = _ref_render_pixel(project, 72, px, py)
            for k in range(4):
                assert abs(int(got[py, px, k]) - ref[k]) <= 1
