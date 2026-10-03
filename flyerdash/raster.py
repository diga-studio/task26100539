"""区域光栅化核心。

所有计算在输出像素坐标系中进行: 每个输出像素独立求值,
结果不依赖请求区域的大小、位置或内部切分边界。
中间结果为 float32 预乘 alpha, 不量化为 RGBA8。
"""

from __future__ import annotations

import math

import numpy as np

from .model import (
    IDENTITY,
    Affine,
    GroupNode,
    ImageNode,
    Node,
    affine_compose,
    affine_invert,
)


def render_region_f32(
    nodes: list[Node],
    dpi: int,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> np.ndarray:
    """渲染输出像素矩形 [(x0,y0), (x0+width) x (y0+height))。

    返回形状 (height, width, 4) 的 float32 预乘 alpha 数组, 值域 [0, 1]。
    nodes 为画布层级的有序节点列表 (自底向上)。
    """
    buf = np.zeros((height, width, 4), dtype=np.float32)
    for node in nodes:
        _composite_node(node, IDENTITY, buf, x0, y0, dpi)
    return buf


# 临时数组行分块大小: 限制宽画布上的瞬时内存, 不改变逐像素结果
_CHUNK_ROWS = 16


def to_rgba8(premul: np.ndarray) -> bytes:
    """将 float32 预乘 alpha 数组转回未预乘 RGBA8 字节。

    四舍五入; alpha 为 0 时 RGB 也为 0。
    """
    out = np.empty(premul.shape, dtype=np.uint8)
    for y0 in range(0, premul.shape[0], _CHUNK_ROWS):
        chunk = premul[y0 : y0 + _CHUNK_ROWS]
        alpha = chunk[..., 3:4]
        rgb = chunk[..., 0:3]
        nonzero = alpha > 0.0
        safe_alpha = np.where(nonzero, alpha, 1.0)
        unpremul = np.clip(rgb / safe_alpha, 0.0, 1.0)
        unpremul = np.where(nonzero, unpremul, 0.0)
        rgba = np.concatenate([unpremul, np.clip(alpha, 0.0, 1.0)], axis=-1)
        out[y0 : y0 + _CHUNK_ROWS] = np.floor(rgba * 255.0 + 0.5).clip(0, 255)
    return out.tobytes()


def _node_bounds(node: Node, m: Affine) -> tuple[float, float, float, float] | None:
    """节点在画布坐标(点)下的轴对齐包围盒; 空组返回 None。

    m 为节点父坐标到画布坐标的变换; 节点自身变换在此复合。
    """
    m = affine_compose(m, node.transform)
    if isinstance(node, ImageNode):
        a, b, c, d, e, f = m
        xs = []
        ys = []
        for lx, ly in (
            (0.0, 0.0),
            (float(node.width), 0.0),
            (0.0, float(node.height)),
            (float(node.width), float(node.height)),
        ):
            xs.append(a * lx + c * ly + e)
            ys.append(b * lx + d * ly + f)
        return (min(xs), min(ys), max(xs), max(ys))
    boxes = [
        box
        for child in node.children
        if (box := _node_bounds(child, m)) is not None
    ]
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _pixel_intersection(
    bounds: tuple[float, float, float, float] | None,
    buf_x0: int,
    buf_y0: int,
    buf_w: int,
    buf_h: int,
    scale: float,
) -> tuple[int, int, int, int] | None:
    """画布点包围盒与输出像素矩形的交集, 返回相对 buf 的 (x, y, w, h)。"""
    if bounds is None:
        return None
    minx, miny, maxx, maxy = bounds
    # 输出像素 px 覆盖画布点 [px/scale, (px+1)/scale);
    # buf 覆盖画布像素 [buf_x0, buf_x0+buf_w) x [buf_y0, buf_y0+buf_h)
    ix0 = max(int(math.floor(minx * scale)), buf_x0)
    iy0 = max(int(math.floor(miny * scale)), buf_y0)
    ix1 = min(int(math.ceil(maxx * scale)), buf_x0 + buf_w)
    iy1 = min(int(math.ceil(maxy * scale)), buf_y0 + buf_h)
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    return (ix0 - buf_x0, iy0 - buf_y0, ix1 - ix0, iy1 - iy0)


def _composite_node(
    node: Node,
    m: Affine,
    buf: np.ndarray,
    buf_x0: int,
    buf_y0: int,
    dpi: int,
) -> None:
    """将节点以 source-over 合成到 buf。m 为节点父坐标到画布坐标的变换。"""
    scale = dpi / 72.0
    buf_h, buf_w = buf.shape[0], buf.shape[1]
    rect = _pixel_intersection(
        _node_bounds(node, m), buf_x0, buf_y0, buf_w, buf_h, scale
    )
    if rect is None:
        return
    m_local = affine_compose(m, node.transform)
    rx, ry, rw, rh = rect
    if isinstance(node, ImageNode):
        sampled = _sample_image(node, m_local, buf_x0 + rx, buf_y0 + ry, rw, rh, dpi)
        _source_over(buf[ry : ry + rh, rx : rx + rw], sampled)
        return

    if node.opacity >= 1.0:
        # 组不透明度为 1 时无需中间缓冲, 子节点直接合成到父缓冲
        for child in node.children:
            _composite_node(child, m_local, buf, buf_x0, buf_y0, dpi)
        return

    # 组不透明度 < 1: 先在组缓冲内完成子节点合成, 再整体施加组透明度
    group_buf = np.zeros((rh, rw, 4), dtype=np.float32)
    for child in node.children:
        _composite_node(child, m_local, group_buf, buf_x0 + rx, buf_y0 + ry, dpi)
    group_buf *= np.float32(node.opacity)
    _source_over(buf[ry : ry + rh, rx : rx + rw], group_buf)


def _sample_image(
    node: ImageNode,
    m: Affine,
    px0: int,
    py0: int,
    width: int,
    height: int,
    dpi: int,
) -> np.ndarray:
    """对图像节点在输出像素矩形内做双线性采样, 返回 float32 预乘 alpha。"""
    out = np.empty((height, width, 4), dtype=np.float32)
    for y0 in range(0, height, _CHUNK_ROWS):
        rows = min(_CHUNK_ROWS, height - y0)
        out[y0 : y0 + rows] = _sample_rows(node, m, px0, py0 + y0, width, rows, dpi)
    return out


def _sample_rows(
    node: ImageNode,
    m: Affine,
    px0: int,
    py0: int,
    width: int,
    height: int,
    dpi: int,
) -> np.ndarray:
    """_sample_image 的行分块实现。"""
    scale = dpi / 72.0
    ia, ib, ic, id_, ie, if_ = affine_invert(m)
    # 输出像素中心对应的画布点坐标
    cx = (px0 + np.arange(width, dtype=np.float64) + 0.5) / scale
    cy = (py0 + np.arange(height, dtype=np.float64) + 0.5) / scale
    # 画布点 -> 图像局部坐标
    lx = ia * cx[None, :] + ic * cy[:, None] + ie
    ly = ib * cx[None, :] + id_ * cy[:, None] + if_
    # 源像素中心位于半整数坐标: 平移到像素索引空间
    u = lx - 0.5
    v = ly - 0.5
    i0 = np.floor(u).astype(np.int64)
    j0 = np.floor(v).astype(np.int64)
    fx = (u - i0).astype(np.float32)
    fy = (v - j0).astype(np.float32)

    src = np.frombuffer(node.pixels, dtype=np.uint8).reshape(
        node.height, node.width, 4
    )
    src_f = src.astype(np.float32) * np.float32(1.0 / 255.0)
    # 预乘 alpha
    src_f[..., 0:3] *= src_f[..., 3:4]

    def gather(ii: np.ndarray, jj: np.ndarray) -> np.ndarray:
        valid = (ii >= 0) & (ii < node.width) & (jj >= 0) & (jj < node.height)
        ci = np.clip(ii, 0, node.width - 1)
        cj = np.clip(jj, 0, node.height - 1)
        out = src_f[cj, ci]
        return out * valid[..., None]

    c00 = gather(i0, j0)
    c10 = gather(i0 + 1, j0)
    c01 = gather(i0, j0 + 1)
    c11 = gather(i0 + 1, j0 + 1)
    w0 = (1.0 - fx)[..., None]
    w1 = fx[..., None]
    top = c00 * w0 + c10 * w1
    bottom = c01 * w0 + c11 * w1
    sampled = top * (1.0 - fy)[..., None] + bottom * fy[..., None]
    sampled *= np.float32(node.opacity)
    return sampled


def _source_over(dst: np.ndarray, src: np.ndarray) -> None:
    """预乘 alpha 的 source-over: dst = src + dst * (1 - src_a)。"""
    for y0 in range(0, dst.shape[0], _CHUNK_ROWS):
        d = dst[y0 : y0 + _CHUNK_ROWS]
        s = src[y0 : y0 + _CHUNK_ROWS]
        keep = 1.0 - s[..., 3:4]
        d *= keep
        d += s
