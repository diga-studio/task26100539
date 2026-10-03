"""Full 12000x12000 export in a subprocess with measured peak RSS."""

import os
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from flyerdash import render_region
from tests.large_scene import DPI, OUT_PX, build_scene

RSS_LIMIT = 192 * 1024 * 1024
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def large_export(tmp_path_factory):
    out = tmp_path_factory.mktemp("large") / "out.png"
    proc = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "tests", "render_large_child.py"), str(out)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stderr
    rss = None
    for line in proc.stdout.splitlines():
        if line.startswith("RSS_BYTES="):
            rss = int(line.split("=", 1)[1])
    assert rss is not None, proc.stdout
    return str(out), rss


def test_peak_rss_within_budget(large_export):
    _, rss = large_export
    assert rss <= RSS_LIMIT, f"peak RSS {rss / 2**20:.1f} MiB exceeds 192 MiB"


def test_export_decodable_and_regions_match(large_export):
    path, _ = large_export
    with Image.open(path) as im:
        im.load()
        assert im.mode == "RGBA"
        assert im.size == (OUT_PX, OUT_PX)
        full = np.asarray(im, dtype=np.uint8)

    project = build_scene()
    probes = [
        (0, 0, 512, 512),            # corner, max size
        (OUT_PX - 512, OUT_PX - 512, 512, 512),
        (5000, 5000, 512, 512),      # overlapping transformed images
        (1234, 987, 317, 209),       # odd size/offset
        (0, 6000, 1, 512),           # 1-px-wide sliver at the edge
        (OUT_PX - 1, OUT_PX - 1, 1, 1),
    ]
    for (x, y, w, h) in probes:
        region = render_region(project, DPI, (x, y, w, h))
        got = np.frombuffer(region, dtype=np.uint8).reshape(h, w, 4)
        want = full[y : y + h, x : x + w]
        assert np.array_equal(got, want), f"region {(x, y, w, h)} mismatch"
