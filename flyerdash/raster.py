"""Tile-based rasterization: evaluate the node tree on a coordinate grid.

All computation happens in float32 premultiplied-alpha space. No
intermediate is ever quantized to 8 bits, and no buffer larger than one
tile is allocated.
"""

from __future__ import annotations

import numpy as np

from .model import GroupNode, ImageNode, Node


def _raw_view(node: ImageNode) -> np.ndarray:
    """Zero-copy uint8 view of the decoded image; cached on the node."""
    cached = getattr(node, "_raw_u8", None)
    if cached is None:
        cached = np.frombuffer(node.pixels, dtype=np.uint8).reshape(
            node.height, node.width, 4
        )
        object.__setattr__(node, "_raw_u8", cached)
    return cached


def eval_tree(node: Node, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Evaluate `node` at parent-space points (xs, ys).

    xs/ys: float64 arrays of identical shape (h, w) - parent-space coords.
    Returns float32 array (h, w, 4): premultiplied RGBA in [0, 1].
    """
    inv = node.transform.inverted()
    lx, ly = inv.apply(xs, ys)
    if isinstance(node, ImageNode):
        return sample_image(node, lx, ly)
    if isinstance(node, GroupNode):
        acc = np.zeros(xs.shape + (4,), dtype=np.float32)
        for child in node.children:
            fg = eval_tree(child, lx, ly)
            # source-over: fg over acc (premultiplied)
            acc *= 1.0 - fg[..., 3:4]
            acc += fg
        if node.opacity != 1.0:
            acc *= np.float32(node.opacity)
        return acc
    raise TypeError(f"unknown node type: {type(node)!r}")


def sample_image(node: ImageNode, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinear-sample an image node at local coords (xs, ys).

    Source pixel centers sit at half-integer coords; out-of-bounds source
    pixels participate as transparent black. Result is premultiplied and
    scaled by the node opacity.
    """
    raw = _raw_view(node)
    h, w = raw.shape[:2]
    # source pixel (i, j) center is at (i + 0.5, j + 0.5)
    u = xs - 0.5
    v = ys - 0.5
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    out = np.zeros(xs.shape + (4,), dtype=np.float32)
    # Convert only the source rows/cols this tile can touch, never the
    # whole image: keeps memory independent of source image size.
    xa = max(int(x0.min()), 0)
    xb = min(int(x0.max()) + 1, w - 1)
    ya = max(int(y0.min()), 0)
    yb = min(int(y0.max()) + 1, h - 1)
    if xa > xb or ya > yb:
        if node.opacity != 1.0:
            out *= np.float32(node.opacity)
        return out
    img = raw[ya : yb + 1, xa : xb + 1].astype(np.float32)
    img *= np.float32(1.0 / 255.0)
    img[..., :3] *= img[..., 3:4]
    fx = (u - x0).astype(np.float32)
    fy = (v - y0).astype(np.float32)
    for dy in (0, 1):
        yy = y0 + dy
        wy = (1.0 - fy) if dy == 0 else fy
        y_ok = (yy >= 0) & (yy < h)
        yc = np.clip(yy, ya, yb) - ya
        for dx in (0, 1):
            xx = x0 + dx
            wx = (1.0 - fx) if dx == 0 else fx
            ok = y_ok & (xx >= 0) & (xx < w)
            xc = np.clip(xx, xa, xb) - xa
            weight = np.where(ok, wx * wy, 0.0)
            out += img[yc, xc] * weight[..., None]
    if node.opacity != 1.0:
        out *= np.float32(node.opacity)
    return out


def unpremultiply_to_rgba8(premul: np.ndarray) -> np.ndarray:
    """Convert float premultiplied RGBA (..., 4) to unpremultiplied RGBA8.

    Rounds to nearest; where alpha == 0 the RGB channels are forced to 0.
    """
    p = premul.astype(np.float64)
    alpha = p[..., 3:4]
    has_alpha = alpha > 0.0
    safe_alpha = np.where(has_alpha, alpha, 1.0)
    rgb = np.clip(p[..., :3] / safe_alpha, 0.0, 1.0)
    rgb = np.where(has_alpha, rgb, 0.0)
    out = np.concatenate([rgb, np.clip(alpha, 0.0, 1.0)], axis=-1)
    return np.rint(out * 255.0).astype(np.uint8)
