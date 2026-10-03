"""Data model for FlyerDash projects: affine transforms, nodes, project."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Affine:
    """Invertible 2D affine transform mapping local coords to parent coords.

    (x, y) -> (a*x + c*y + e, b*x + d*y + f)
    """

    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    d: float = 1.0
    e: float = 0.0
    f: float = 0.0

    def inverted(self) -> "Affine":
        det = self.a * self.d - self.b * self.c
        if det == 0.0:
            raise ValueError("transform is not invertible")
        ia = self.d / det
        ib = -self.b / det
        ic = -self.c / det
        id_ = self.a / det
        ie = -(ia * self.e + ic * self.f)
        if_ = -(ib * self.e + id_ * self.f)
        return Affine(ia, ib, ic, id_, ie, if_)

    def apply(self, xs, ys):
        """Map arrays of local coords to parent coords (numpy arrays)."""
        return (
            self.a * xs + self.c * ys + self.e,
            self.b * xs + self.d * ys + self.f,
        )


@dataclass
class ImageNode:
    """A decoded RGBA8 image. One source pixel == one local coordinate unit.

    pixels: bytes-like of length width*height*4, row-major RGBA8.
    """

    transform: Affine
    opacity: float
    width: int
    height: int
    pixels: bytes


@dataclass
class GroupNode:
    """An ordered group. Children composite bottom-to-top (index 0 first)."""

    transform: Affine
    opacity: float
    children: list = field(default_factory=list)


Node = ImageNode | GroupNode


@dataclass
class Project:
    """Canvas in points (72 pt == 1 inch). Background is transparent."""

    width: float
    height: float
    root: GroupNode
