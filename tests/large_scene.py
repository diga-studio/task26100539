"""Deterministic 12000x12000 test scene shared by child process and test.

Canvas: 2880 x 2880 pt -> 12000 x 12000 px at 300 dpi.
Decoded source images total ~15.2 MiB (limit: 16 MiB).
"""

import math

import numpy as np

from flyerdash import Affine, GroupNode, ImageNode, Project

CANVAS_PT = 2880.0
DPI = 300
OUT_PX = 12000


def _noise_image(seed, w, h):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 4), dtype=np.uint8).tobytes()


def build_scene():
    # Big image spanning the whole canvas (1800*1800*4 = 12.96 MiB).
    big = ImageNode(
        Affine(1.7, 0.0, 0.0, 1.7, -100.0, -100.0),
        1.0,
        1800,
        1800,
        _noise_image(1, 1800, 1800),
    )

    theta = math.radians(25.0)
    mid_a = ImageNode(
        Affine(math.cos(theta) * 3.0, math.sin(theta) * 3.0,
               -math.sin(theta) * 3.0, math.cos(theta) * 3.0,
               500.0, 700.0),
        0.65,
        512,
        512,
        _noise_image(2, 512, 512),
    )
    mid_b = ImageNode(
        Affine(2.0, 0.0, 0.0, 2.0, 900.0, 900.0),
        0.5,
        512,
        512,
        _noise_image(3, 512, 512),
    )

    phi = math.radians(-40.0)
    small = ImageNode(
        Affine(math.cos(phi) * 4.0, math.sin(phi) * 4.0,
               -math.sin(phi) * 1.5, math.cos(phi) * 1.5,
               1000.0, 1000.0),
        0.9,
        256,
        256,
        _noise_image(4, 256, 256),
    )

    inner = GroupNode(Affine(1.0, 0.0, 0.0, 1.0, 30.0, -20.0), 0.8, [small])
    overlay = GroupNode(Affine(), 0.6, [mid_a, mid_b, inner])
    root = GroupNode(Affine(), 1.0, [big, overlay])
    return Project(CANVAS_PT, CANVAS_PT, root)
