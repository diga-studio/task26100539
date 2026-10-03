"""FlyerDash PNG 导出内核对外接口。"""

from __future__ import annotations

from .model import Project, SUPPORTED_DPI, output_size
from .png import PngWriter
from .raster import render_region_f32, to_rgba8

# 完整导出时的行带高度 (输出像素行)。生产路径不分配整画布缓冲区。
_STRIP_HEIGHT = 32


def render(project: Project, dpi: int, output_path: str) -> None:
    """将项目渲染为 RGBA PNG 文件。

    dpi 仅支持 72 / 150 / 300。输出宽高为画布宽高乘 dpi 除以 72 向上取整。
    按行带流式渲染并写出, 峰值内存与画布面积无关。
    """
    _validate_dpi(dpi)
    out_w, out_h = output_size(project, dpi)
    with PngWriter(output_path, out_w, out_h) as writer:
        y = 0
        while y < out_h:
            strip_h = min(_STRIP_HEIGHT, out_h - y)
            tile = render_region_f32(project.nodes, dpi, 0, y, out_w, strip_h)
            writer.write_rows(to_rgba8(tile))
            y += strip_h


def render_region(
    project: Project, dpi: int, rect: tuple[int, int, int, int]
) -> bytes:
    """渲染指定输出像素矩形, 返回 RGBA8 字节 (行优先, 长度 w*h*4)。

    rect = (x, y, width, height), 全局输出像素坐标, 保证在画布内。
    结果与 render 完整导出后截取对应区域完全一致。
    """
    _validate_dpi(dpi)
    x, y, width, height = rect
    if width <= 0 or height <= 0:
        raise ValueError("rect width and height must be positive")
    out_w, out_h = output_size(project, dpi)
    if x < 0 or y < 0 or x + width > out_w or y + height > out_h:
        raise ValueError(f"rect {rect!r} outside output {out_w}x{out_h}")
    tile = render_region_f32(project.nodes, dpi, x, y, width, height)
    return to_rgba8(tile)


def _validate_dpi(dpi: int) -> None:
    if dpi not in SUPPORTED_DPI:
        raise ValueError(f"unsupported dpi: {dpi!r} (supported: {SUPPORTED_DPI})")
