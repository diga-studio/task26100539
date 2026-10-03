"""Transforms, transparent-edge sampling, and region-tiling invariance.

Cross-checks the numpy engine against an independent, per-pixel, pure
Python reference implementation, then verifies that stitching arbitrary
region tilings reproduces the full export exactly.
"""

import math

from flyerdash import Affine, GroupNode, ImageNode, Project, render, render_region
from flyerdash.png import decode_png


# ---------------------------------------------------------------- reference

def _ref_sample_image(node, x, y):
    """Bilinear sample in premultiplied float space, pure Python."""
    px = node.pixels
    w, h = node.width, node.height
    u, v = x - 0.5, y - 0.5
    x0, y0 = math.floor(u), math.floor(v)
    fx, fy = u - x0, v - y0
    acc = [0.0, 0.0, 0.0, 0.0]
    for dy in (0, 1):
        for dx in (0, 1):
            xx, yy = x0 + dx, y0 + dy
            weight = (fx if dx else 1.0 - fx) * (fy if dy else 1.0 - fy)
            if weight == 0.0:
                continue
            if 0 <= xx < w and 0 <= yy < h:
                i = (yy * w + xx) * 4
                r, g, b, a = (px[i + k] / 255.0 for k in range(4))
                contrib = (r * a, g * a, b * a, a)
            else:
                contrib = (0.0, 0.0, 0.0, 0.0)  # transparent black
            for k in range(4):
                acc[k] += weight * contrib[k]
    return [c * node.opacity for c in acc]


def _ref_invert(t):
    det = t.a * t.d - t.b * t.c
    ia, ib = t.d / det, -t.b / det
    ic, id_ = -t.c / det, t.a / det
    ie = -(ia * t.e + ic * t.f)
    if_ = -(ib * t.e + id_ * t.f)
    return Affine(ia, ib, ic, id_, ie, if_)


def _ref_eval(node, x, y):
    """Evaluate node at parent-space point, premultiplied floats."""
    inv = _ref_invert(node.transform)
    lx = inv.a * x + inv.c * y + inv.e
    ly = inv.b * x + inv.d * y + inv.f
    if isinstance(node, ImageNode):
        return _ref_sample_image(node, lx, ly)
    acc = [0.0, 0.0, 0.0, 0.0]
    for child in node.children:
        fg = _ref_eval(child, lx, ly)
        acc = [fg[k] + acc[k] * (1.0 - fg[3]) for k in range(4)]
    return [c * node.opacity for c in acc]


def _ref_pixel(project, dpi, px_x, px_y):
    scale = dpi / 72.0
    x, y = (px_x + 0.5) / scale, (px_y + 0.5) / scale
    r, g, b, a = _ref_eval(project.root, x, y)
    if a <= 0.0:
        return (0, 0, 0, 0)
    return (
        round(min(max(r / a, 0.0), 1.0) * 255),
        round(min(max(g / a, 0.0), 1.0) * 255),
        round(min(max(b / a, 0.0), 1.0) * 255),
        round(min(max(a, 0.0), 1.0) * 255),
    )


# ---------------------------------------------------------------- scene

def _build_scene():
    # 4x4 image mixing opaque colors and fully transparent pixels that
    # carry non-zero RGB (must not bleed into the result).
    px_a = bytearray()
    cells = [
        (255, 0, 0, 255), (0, 255, 0, 128), (0, 0, 255, 64), (255, 255, 0, 0),
        (255, 0, 255, 200), (0, 0, 0, 0), (120, 40, 200, 90), (9, 99, 9, 0),
        (10, 20, 30, 255), (200, 100, 50, 33), (0, 0, 0, 255), (250, 250, 250, 1),
        (1, 2, 3, 254), (254, 253, 252, 7), (100, 100, 100, 100), (77, 0, 33, 0),
    ]
    for c in cells:
        px_a += bytes(c)
    # Rotated 30 degrees, translated so part of it sits at negative coords.
    theta = math.radians(30.0)
    rot = Affine(math.cos(theta), math.sin(theta), -math.sin(theta), math.cos(theta), -3.0, -2.0)
    img_a = ImageNode(rot, 0.9, 4, 4, bytes(px_a))

    # 3x5 image under non-uniform scale plus a slight rotation.
    px_b = bytearray()
    for j in range(5):
        for i in range(3):
            px_b += bytes([(i * 90) % 256, (j * 50) % 256, ((i + j) * 37) % 256, (i * 60 + j * 30) % 256])
    phi = math.radians(-12.0)
    sx, sy = 2.5, 0.75
    scale_rot = Affine(
        sx * math.cos(phi), sx * math.sin(phi),
        -sy * math.sin(phi), sy * math.cos(phi),
        8.0, 5.0,
    )
    img_b = ImageNode(scale_rot, 0.8, 3, 5, bytes(px_b))

    inner = GroupNode(Affine(1.0, 0.0, 0.0, 1.0, 2.0, 1.5), 0.7, [img_a, img_b])
    outer = GroupNode(Affine(0.9, 0.1, -0.1, 1.1, 1.0, -1.0), 0.85, [inner])
    return Project(40, 30, GroupNode(Affine(), 1.0, [outer]))


def test_matches_independent_reference():
    project = _build_scene()
    out_w, out_h = 40, 30  # 72 dpi
    data = render_region(project, 72, (0, 0, out_w, out_h))
    for y in range(out_h):
        for x in range(out_w):
            got = tuple(data[(y * out_w + x) * 4 : (y * out_w + x) * 4 + 4])
            want = _ref_pixel(project, 72, x, y)
            for k in range(4):
                assert abs(got[k] - want[k]) <= 1, (x, y, got, want)


def _tilings(w, h):
    yield [(0, 0, w, h)]  # single region
    yield [(x, y, min(7, w - x), min(5, h - y))
           for y in range(0, h, 5) for x in range(0, w, 7)]
    yield [(x, y, min(13, w - x), min(11, h - y))
           for y in range(0, h, 11) for x in range(0, w, 13)]
    yield [(0, y, w, 1) for y in range(h)]  # 1-pixel-high strips
    yield [(x, 0, 1, h) for x in range(w)]  # 1-pixel-wide strips


def test_region_tiling_stitch_matches_full_export(tmp_path):
    project = _build_scene()
    out_w, out_h = 40, 30
    png_path = str(tmp_path / "full.png")
    render(project, 72, png_path)
    fw, fh, full = decode_png(png_path)
    assert (fw, fh) == (out_w, out_h)

    for tiling in _tilings(out_w, out_h):
        stitched = bytearray(out_w * out_h * 4)
        for (x, y, w, h) in tiling:
            region = render_region(project, 72, (x, y, w, h))
            for row in range(h):
                dst = ((y + row) * out_w + x) * 4
                src = row * w * 4
                stitched[dst : dst + w * 4] = region[src : src + w * 4]
        assert bytes(stitched) == full


def test_transparent_pixels_with_nonzero_rgb_do_not_bleed():
    # A single fully transparent pixel with non-zero RGB, scaled up so the
    # output samples its neighbourhood: RGB must stay zero everywhere.
    img = ImageNode(Affine(4.0, 0.0, 0.0, 4.0, 0.0, 0.0), 1.0, 1, 1,
                    bytes([200, 100, 50, 0]))
    project = Project(8, 8, GroupNode(Affine(), 1.0, [img]))
    data = render_region(project, 72, (0, 0, 8, 8))
    assert set(data) == {0}
