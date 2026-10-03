"""12000x12000 大画布导出测试: 独立进程实测峰值 RSS, 并与 render_region 核对。"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from flyerdash import render_region

sys.path.insert(0, str(Path(__file__).parent))
from large_worker import make_scene  # noqa: E402

RSS_LIMIT = 192 * 1024 * 1024  # 192 MiB


@pytest.fixture(scope="module")
def large_render(tmp_path_factory):
    out = tmp_path_factory.mktemp("large") / "out.png"
    env = dict(os.environ)
    repo_root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "large_worker.py"), str(out)],
        capture_output=True,
        text=True,
        timeout=1800,
        env=env,
        cwd=repo_root,
    )
    assert proc.returncode == 0, proc.stderr
    info = json.loads(proc.stdout.strip().splitlines()[-1])
    return out, info


def test_peak_rss_within_budget(large_render):
    _, info = large_render
    assert info["rss_bytes"] < RSS_LIMIT, (
        f"peak RSS {info['rss_bytes'] / 1024 / 1024:.1f} MiB exceeds 192 MiB"
    )


def test_png_decodable_and_matches_regions(large_render):
    from PIL import Image

    path, info = large_render
    assert (info["width"], info["height"]) == (12000, 12000)
    project = make_scene()  # 确定性场景, 与子进程一致
    with Image.open(path) as im:
        im.load()
        assert im.size == (12000, 12000) and im.mode == "RGBA"
        # 抽样区域: 覆盖四角、中心、组重叠区与纯大图像区
        rects = [
            (0, 0, 512, 512),
            (11488, 11488, 512, 512),
            (5744, 5744, 512, 512),
            (2000, 2100, 300, 300),
            (7000, 1500, 512, 512),
            (123, 11000, 257, 511),
        ]
        for (x, y, w, h) in rects:
            expected = np.frombuffer(
                render_region(project, 150, (x, y, w, h)), dtype=np.uint8
            ).reshape(h, w, 4)
            got = np.array(im.crop((x, y, x + w, y + h)))
            assert got.shape == (h, w, 4)
            assert (got == expected).all(), f"region {(x, y, w, h)} mismatch"
