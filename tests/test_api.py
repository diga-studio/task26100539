"""API surface: dpi handling, output sizing, region validation."""

import math

import pytest

from flyerdash import (
    Affine,
    GroupNode,
    ImageNode,
    Project,
    output_size,
    render,
    render_region,
)
from flyerdash.png import decode_png
from tests.test_transforms import _build_scene


def _empty_project(w=10, h=10):
    return Project(w, h, GroupNode(Affine(), 1.0, []))


def test_output_size_ceil_per_dpi():
    project = _empty_project(100.5, 99.2)
    assert output_size(project, 72) == (101, 100)
    assert output_size(project, 150) == (math.ceil(100.5 * 150 / 72),
                                         math.ceil(99.2 * 150 / 72))
    assert output_size(project, 300) == (math.ceil(100.5 * 300 / 72),
                                         math.ceil(99.2 * 300 / 72))


def test_unsupported_dpi_rejected():
    project = _empty_project()
    for dpi in (0, 100, 144, -72):
        with pytest.raises(ValueError):
            output_size(project, dpi)
        with pytest.raises(ValueError):
            render_region(project, dpi, (0, 0, 1, 1))


def test_region_validation():
    project = _empty_project(10, 10)  # 10x10 px at 72 dpi
    for rect in [(0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 513, 1), (0, 0, 1, 513),
                 (-1, 0, 1, 1), (0, -1, 1, 1), (9, 0, 2, 1), (0, 9, 1, 2)]:
        with pytest.raises(ValueError):
            render_region(project, 72, rect)


def test_empty_canvas_is_transparent(tmp_path):
    project = _empty_project(5, 3)
    out = str(tmp_path / "empty.png")
    render(project, 72, out)
    w, h, data = decode_png(out)
    assert (w, h) == (5, 3)
    assert set(data) == {0}


@pytest.mark.parametrize("dpi", (72, 150, 300))
def test_render_matches_region_at_each_dpi(tmp_path, dpi):
    project = _build_scene()
    out_w, out_h = output_size(project, dpi)
    out = str(tmp_path / "full.png")
    render(project, dpi, out)
    fw, fh, full = decode_png(out)
    assert (fw, fh) == (out_w, out_h)
    # stitch 100x100 tiles (plus remainders) and compare bitwise
    stitched = bytearray(out_w * out_h * 4)
    step = 100
    for y in range(0, out_h, step):
        for x in range(0, out_w, step):
            w, h = min(step, out_w - x), min(step, out_h - y)
            region = render_region(project, dpi, (x, y, w, h))
            for row in range(h):
                dst = ((y + row) * out_w + x) * 4
                src = row * w * 4
                stitched[dst : dst + w * 4] = region[src : src + w * 4]
    assert bytes(stitched) == full


def test_noninvertible_transform_rejected():
    bad = ImageNode(Affine(1, 2, 2, 4, 0, 0), 1.0, 1, 1, bytes([0, 0, 0, 255]))
    project = Project(1, 1, GroupNode(Affine(), 1.0, [bad]))
    with pytest.raises(ValueError):
        render_region(project, 72, (0, 0, 1, 1))
