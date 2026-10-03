"""在独立子进程中执行 12000x12000 完整导出并报告实测峰值 RSS。

用法: python3 tests/large_worker.py <output.png>
stdout 最后一行输出 JSON: {"rss_bytes": ..., "width": ..., "height": ...}
"""

import json
import math
import resource
import sys

import numpy as np

from flyerdash import GroupNode, ImageNode, Project, render


def make_scene() -> Project:
    rng = np.random.default_rng(20261005)
    # 跨越整个画布的大图像 (源 1024x1024 确定性渐变, 经变换放大铺满画布)
    yy, xx = np.mgrid[0:1024, 0:1024]
    big = np.empty((1024, 1024, 4), dtype=np.uint8)
    big[..., 0] = xx % 256
    big[..., 1] = yy % 256
    big[..., 2] = (xx // 64 + yy // 64) % 2 * 255
    big[..., 3] = 255
    canvas = 5760.0  # 点; dpi=150 -> 输出 12000x12000
    scale = canvas / 1024.0
    nodes = [
        ImageNode(big.tobytes(), 1024, 1024, transform=(scale, 0, 0, scale, 0, 0))
    ]
    # 重叠的半透明嵌套组, 含旋转与非均匀缩放
    for g in range(6):
        small = rng.integers(0, 256, size=(128, 128, 4), dtype=np.uint8)
        small[..., 3] = rng.integers(0, 256, size=(128, 128), dtype=np.uint8)
        theta = math.radians(15 + g * 20)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        sx, sy = 8.0 + g, 5.0 + 0.5 * g
        tx, ty = 400.0 + g * 700.0, 500.0 + g * 600.0
        img = ImageNode(
            small.tobytes(),
            128,
            128,
            transform=(
                cos_t * sx, sin_t * sx,
                -sin_t * sy, cos_t * sy,
                tx, ty,
            ),
            opacity=0.8,
        )
        inner = GroupNode(children=[img], opacity=0.6)
        outer = GroupNode(
            children=[inner],
            transform=(1.5, 0.0, 0.0, 1.5, 100.0 * g, 50.0),
            opacity=0.5,
        )
        nodes.append(outer)
    return Project(canvas, canvas, nodes)


def main() -> None:
    output_path = sys.argv[1]
    project = make_scene()
    render(project, 150, output_path)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":  # Linux 报告 KiB, macOS 报告字节
        rss *= 1024
    print(json.dumps({"rss_bytes": rss, "width": 12000, "height": 12000}))


if __name__ == "__main__":
    main()
