"""FlyerDash 导出内核的数据模型与二维仿射变换工具。

坐标约定:
- 画布以点(point)为单位, 72 点 = 1 英寸, 原点左上, y 向下。
- 仿射变换为 6 元组 (a, b, c, d, e, f):
      parent_x = a * local_x + c * local_y + e
      parent_y = b * local_x + d * local_y + f
- 图像节点: 源像素 (i, j) 覆盖局部坐标 [i, i+1) x [j, j+1),
  即一个源像素对应一个局部坐标单位, 像素中心位于半整数坐标。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Union

# 仿射变换: (a, b, c, d, e, f)
Affine = tuple[float, float, float, float, float, float]

IDENTITY: Affine = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def affine_compose(outer: Affine, inner: Affine) -> Affine:
    """返回先应用 inner 再应用 outer 的复合变换 (outer @ inner)。"""
    oa, ob, oc, od, oe, of = outer
    ia, ib, ic, id_, ie, if_ = inner
    return (
        oa * ia + oc * ib,
        ob * ia + od * ib,
        oa * ic + oc * id_,
        ob * ic + od * id_,
        oa * ie + oc * if_ + oe,
        ob * ie + od * if_ + of,
    )


def affine_invert(m: Affine) -> Affine:
    """返回 m 的逆变换。输入保证可逆。"""
    a, b, c, d, e, f = m
    det = a * d - b * c
    if det == 0.0:
        raise ValueError("affine transform is not invertible")
    ia = d / det
    ib = -b / det
    ic = -c / det
    id_ = a / det
    return (ia, ib, ic, id_, -(ia * e + ic * f), -(ib * e + id_ * f))


def affine_apply(m: Affine, x: float, y: float) -> tuple[float, float]:
    """将变换作用于单个点。"""
    a, b, c, d, e, f = m
    return (a * x + c * y + e, b * x + d * y + f)


@dataclass
class ImageNode:
    """图像层。pixels 为调用方已解码的 RGBA8 字节, 行优先, 长度 width*height*4。"""

    pixels: bytes
    width: int
    height: int
    transform: Affine = IDENTITY
    opacity: float = 1.0


@dataclass
class GroupNode:
    """组。children 自底向上有序 (索引 0 在最底层)。无额外裁切区域。"""

    children: list[Node] = field(default_factory=list)
    transform: Affine = IDENTITY
    opacity: float = 1.0


Node = Union[ImageNode, GroupNode]


@dataclass
class Project:
    """项目。画布宽高以点计, 背景透明。nodes 自底向上有序。"""

    width: float
    height: float
    nodes: list[Node] = field(default_factory=list)


SUPPORTED_DPI: tuple[int, ...] = (72, 150, 300)


def output_size(project: Project, dpi: int) -> tuple[int, int]:
    """输出像素宽高: 画布宽高乘 dpi 再除以 72 后向上取整。"""
    if dpi not in SUPPORTED_DPI:
        raise ValueError(f"unsupported dpi: {dpi!r} (supported: {SUPPORTED_DPI})")
    scale = dpi / 72.0
    return (
        math.ceil(project.width * scale),
        math.ceil(project.height * scale),
    )
