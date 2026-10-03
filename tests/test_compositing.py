"""Opacity / source-over compositing semantics, including nested groups."""

from flyerdash import Affine, GroupNode, ImageNode, Project, render_region


def _solid(r, g, b, a):
    return bytes([r, g, b, a])


def _pixel(project, x=0, y=0):
    data = render_region(project, 72, (x, y, 1, 1))
    return tuple(data[:4])


def test_red_blue_overlap_with_node_and_group_opacity():
    # Red below blue, both image nodes and their parent group at 0.5.
    red = ImageNode(Affine(), 0.5, 1, 1, _solid(255, 0, 0, 255))
    blue = ImageNode(Affine(), 0.5, 1, 1, _solid(0, 0, 255, 255))
    group = GroupNode(Affine(), 0.5, [red, blue])
    project = Project(1, 1, GroupNode(Affine(), 1.0, [group]))
    assert _pixel(project) == (85, 0, 170, 96)


def test_nested_groups_apply_opacity_recursively():
    red = ImageNode(Affine(), 0.5, 1, 1, _solid(255, 0, 0, 255))
    blue = ImageNode(Affine(), 0.5, 1, 1, _solid(0, 0, 255, 255))
    inner = GroupNode(Affine(), 0.5, [red, blue])
    outer = GroupNode(Affine(), 0.5, [inner])
    project = Project(1, 1, GroupNode(Affine(), 1.0, [outer]))
    # Inner composite premultiplied alpha 0.375 -> *0.5 -> 0.1875 -> 48.
    # Unpremultiplied RGB is unchanged by group opacity: still (85, 0, 170).
    assert _pixel(project) == (85, 0, 170, 48)


def test_group_opacity_is_not_distributed_to_children():
    # If group opacity were (incorrectly) folded into each child, two
    # overlapping children would darken differently than one whole-group
    # multiply. Two half-alpha whites composited give alpha 0.75; scaling
    # the group by 0.5 gives 0.375 -> 96, while distributing 0.5 to each
    # child would give alpha 0.4375 -> 112.
    a = ImageNode(Affine(), 0.5, 1, 1, _solid(255, 255, 255, 255))
    b = ImageNode(Affine(), 0.5, 1, 1, _solid(255, 255, 255, 255))
    group = GroupNode(Affine(), 0.5, [a, b])
    project = Project(1, 1, GroupNode(Affine(), 1.0, [group]))
    assert _pixel(project) == (255, 255, 255, 96)


def test_sibling_order_bottom_to_top():
    bottom = ImageNode(Affine(), 1.0, 1, 1, _solid(255, 0, 0, 255))
    top = ImageNode(Affine(), 1.0, 1, 1, _solid(0, 0, 255, 255))
    project = Project(1, 1, GroupNode(Affine(), 1.0, [bottom, top]))
    assert _pixel(project) == (0, 0, 255, 255)
    flipped = Project(1, 1, GroupNode(Affine(), 1.0, [top, bottom]))
    assert _pixel(flipped) == (255, 0, 0, 255)
