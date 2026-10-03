"""Public API: render() and render_region()."""

from __future__ import annotations

import math

import numpy as np

from .model import Project
from .png import PngWriter
from .raster import eval_tree, unpremultiply_to_rgba8

SUPPORTED_DPIS = (72, 150, 300)

# Budget for one strip/tile buffer of float32 premultiplied RGBA (bytes).
# Nested group evaluation keeps ~2 live buffers per nesting level, so with
# the 8-level nesting limit peak tile memory stays well under 40 MiB.
_TILE_BUDGET = 2 << 20


def output_size(project: Project, dpi: int) -> tuple[int, int]:
    """Output pixel size: ceil(canvas_pt * dpi / 72) per dimension."""
    _check_dpi(dpi)
    return (
        math.ceil(project.width * dpi / 72),
        math.ceil(project.height * dpi / 72),
    )


def _check_dpi(dpi: int) -> None:
    if dpi not in SUPPORTED_DPIS:
        raise ValueError(f"unsupported dpi {dpi}; expected one of {SUPPORTED_DPIS}")


def _eval_region(
    project: Project, dpi: int, x: int, y: int, w: int, h: int
) -> np.ndarray:
    """Evaluate the tree over output pixels [x, x+w) x [y, y+h).

    Returns uint8 RGBA array of shape (h, w, 4). Every output pixel is
    evaluated independently, so results do not depend on region geometry.
    """
    scale = dpi / 72.0  # output pixels per point
    col = (np.arange(x, x + w, dtype=np.float64) + 0.5) / scale
    row = (np.arange(y, y + h, dtype=np.float64) + 0.5) / scale
    xs = np.broadcast_to(col, (h, w))
    ys = np.broadcast_to(row[:, None], (h, w))
    premul = eval_tree(project.root, xs, ys)
    return unpremultiply_to_rgba8(premul)


def render(project: Project, dpi: int, output_path: str) -> None:
    """Render the whole canvas to an RGBA PNG at `output_path`.

    Processes the canvas in horizontal strips; never allocates a
    full-canvas pixel buffer.
    """
    width, height = output_size(project, dpi)
    strip_h = max(1, min(height, _TILE_BUDGET // (width * 4 * 4)))
    with open(output_path, "wb") as fh:
        writer = PngWriter(fh, width, height)
        y = 0
        while y < height:
            h = min(strip_h, height - y)
            strip = _eval_region(project, dpi, 0, y, width, h)
            writer.write_rows(strip.tobytes())
            y += h
        writer.finish()


def render_region(project: Project, dpi: int, rect: tuple[int, int, int, int]) -> bytes:
    """Render an output-pixel rectangle (x, y, w, h) and return RGBA8 bytes.

    Result is bit-identical to the same rectangle of a full render.
    Only the requested pixels are computed.
    """
    _check_dpi(dpi)
    x, y, w, h = rect
    width, height = output_size(project, dpi)
    if w <= 0 or h <= 0 or w > 512 or h > 512:
        raise ValueError("region must be positive and at most 512x512")
    if x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError("region lies outside the canvas")
    return _eval_region(project, dpi, x, y, w, h).tobytes()
