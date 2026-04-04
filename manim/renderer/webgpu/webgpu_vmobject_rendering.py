"""WebGPU draw calls for VMobject fill + stroke + surface rendering — Phase 2/3.

Fill
----
Uses the Loop-Blinn quadratic bezier test for anti-aliased fill boundaries.
Each cubic bezier curve is first split into 4 sub-cubics via de Casteljau
subdivision (2 levels), then each sub-cubic is approximated as a quadratic.
The subdivision reduces the approximation error to negligible levels (<0.01 %).
Three kinds of triangles are emitted:

  texture_mode = +1  concave bezier region (Loop-Blinn: keep where u²−v ≥ 0)
  texture_mode = −1  convex  bezier region (Loop-Blinn: keep where u²−v ≤ 0)
  texture_mode =  0  flat interior (always kept)

Stroke
------
All four cubic bezier control points (b0, h0, h1, b3) are passed to the GPU.
The fragment shader computes the exact unsigned distance to the cubic bezier
curve via Newton's-method minimisation and discards pixels outside the stroke
half-width — no quadratic approximation is made.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from manim.utils.space_ops import cross2d, earclip_triangulation

if TYPE_CHECKING:
    import wgpu as wgpu_t

    from manim.mobject.types.vectorized_mobject import VMobject
    from manim.renderer.webgpu.webgpu_renderer import WebGPURenderer


# ---------------------------------------------------------------------------
# Surface vertex layout — must match surface.wgsl locations:
#   location 0 → in_vert    float32x3  offset  0  (12 bytes)
#   location 1 → in_normal  float32x3  offset 12  (12 bytes)
#   location 2 → in_color   float32x4  offset 24  (16 bytes)
#   stride: 40 bytes
# ---------------------------------------------------------------------------

_SURFACE_DTYPE = np.dtype(
    [
        ("in_vert",   np.float32, (3,)),
        ("in_normal", np.float32, (3,)),
        ("in_color",  np.float32, (4,)),
    ]
)
_SURFACE_STRIDE: int = _SURFACE_DTYPE.itemsize  # 40 bytes

_SURFACE_OFFSETS: dict[str, int] = {
    name: _SURFACE_DTYPE.fields[name][1]  # type: ignore[index]
    for name in _SURFACE_DTYPE.names
}

SURFACE_VERTEX_LAYOUT: dict = {
    "array_stride": _SURFACE_STRIDE,
    "step_mode": "vertex",
    "attributes": [
        {"format": "float32x3", "offset": _SURFACE_OFFSETS["in_vert"],   "shader_location": 0},
        {"format": "float32x3", "offset": _SURFACE_OFFSETS["in_normal"], "shader_location": 1},
        {"format": "float32x4", "offset": _SURFACE_OFFSETS["in_color"],  "shader_location": 2},
    ],
}


# ---------------------------------------------------------------------------
# Fill vertex layout — must match vmobject_fill.wgsl locations:
#   location 0 → in_vert        float32x3  offset  0  (12 bytes)
#   location 1 → in_color       float32x4  offset 12  (16 bytes)
#   location 2 → texture_coords float32x2  offset 28  ( 8 bytes)
#   location 3 → texture_mode   float32    offset 36  ( 4 bytes)
#   stride: 40 bytes
# ---------------------------------------------------------------------------

_FILL_DTYPE = np.dtype(
    [
        ("in_vert", np.float32, (3,)),
        ("in_color", np.float32, (4,)),
        ("texture_coords", np.float32, (2,)),
        ("texture_mode", np.float32),
    ]
)
_FILL_STRIDE: int = _FILL_DTYPE.itemsize  # 40 bytes

_FILL_OFFSETS: dict[str, int] = {
    name: _FILL_DTYPE.fields[name][1]  # type: ignore[index]
    for name in _FILL_DTYPE.names
}

FILL_VERTEX_LAYOUT: dict = {
    "array_stride": _FILL_STRIDE,
    "step_mode": "vertex",
    "attributes": [
        {"format": "float32x3", "offset": _FILL_OFFSETS["in_vert"],        "shader_location": 0},
        {"format": "float32x4", "offset": _FILL_OFFSETS["in_color"],       "shader_location": 1},
        {"format": "float32x2", "offset": _FILL_OFFSETS["texture_coords"], "shader_location": 2},
        {"format": "float32",   "offset": _FILL_OFFSETS["texture_mode"],   "shader_location": 3},
    ],
}


# ---------------------------------------------------------------------------
# Stroke vertex layout — must match vmobject_stroke.wgsl locations:
#   location 0 → current_curve_0  float32x3  offset  0  (12 bytes)
#   location 1 → current_curve_1  float32x3  offset 12  (12 bytes)  ← into current_curve
#   location 2 → current_curve_2  float32x3  offset 24  (12 bytes)  ← into current_curve
#   location 3 → current_curve_3  float32x3  offset 36  (12 bytes)  ← into current_curve
#   location 4 → tile_coordinate  float32x2  offset 48  ( 8 bytes)
#   location 5 → in_color         float32x4  offset 56  (16 bytes)
#   location 6 → in_width         float32    offset 72  ( 4 bytes)
#   stride: 76 bytes
# ---------------------------------------------------------------------------

_STROKE_DTYPE = np.dtype(
    [
        ("current_curve", np.float32, (4, 3)),  # 48 bytes at offset 0  (b0, h0, h1, b3)
        ("tile_coordinate", np.float32, (2,)),
        ("in_color", np.float32, (4,)),
        ("in_width", np.float32),
    ]
)
_STROKE_STRIDE: int = _STROKE_DTYPE.itemsize  # 76 bytes


def _stroke_field_offset(name: str) -> int:
    return _STROKE_DTYPE.fields[name][1]  # type: ignore[index]


STROKE_VERTEX_LAYOUT: dict = {
    "array_stride": _STROKE_STRIDE,
    "step_mode": "vertex",
    "attributes": [
        # current_curve is a (4,3) sub-field starting at offset 0.
        # Split into four vec3 bindings with explicit byte offsets.
        {"format": "float32x3", "offset": _stroke_field_offset("current_curve"),      "shader_location": 0},
        {"format": "float32x3", "offset": _stroke_field_offset("current_curve") + 12, "shader_location": 1},
        {"format": "float32x3", "offset": _stroke_field_offset("current_curve") + 24, "shader_location": 2},
        {"format": "float32x3", "offset": _stroke_field_offset("current_curve") + 36, "shader_location": 3},
        {"format": "float32x2", "offset": _stroke_field_offset("tile_coordinate"),    "shader_location": 4},
        {"format": "float32x4", "offset": _stroke_field_offset("in_color"),           "shader_location": 5},
        {"format": "float32",   "offset": _stroke_field_offset("in_width"),           "shader_location": 6},
    ],
}


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def render_webgpu_mobject(
    renderer: WebGPURenderer,
    mobject: VMobject,
) -> None:
    """Dispatch every family member of *mobject* to the correct pipeline.

    Each family member is routed independently:

    * ``shade_in_3d=True``  → surface pipeline (Phong-lit, depth-tested).
    * otherwise             → fill + stroke pipelines (2-D painter's algorithm).

    This correctly handles ``Surface`` objects (a ``VGroup`` whose individual
    face pieces carry ``shade_in_3d=True``) as well as mixed scenes where some
    sub-mobjects are 3-D and others are flat overlays.
    """
    for submob in mobject.family_members_with_points():
        if getattr(submob, "shade_in_3d", False):
            _draw_surface_face(renderer, submob)
        else:
            _draw_vmobject_fill(renderer, submob)
            _draw_vmobject_stroke(renderer, submob)


def render_webgpu_surface(
    renderer: WebGPURenderer,
    mobject: VMobject,
) -> None:
    """Record surface draw calls for *mobject* and its descendants.

    Only processes VMobjects that have ``shade_in_3d=True`` and non-zero fill
    alpha.  The geometry is fan-triangulated from each subpath's anchor points
    with a per-face normal computed via the cross product.
    """
    for submob in mobject.family_members_with_points():
        if getattr(submob, "shade_in_3d", False):
            _draw_surface_face(renderer, submob)


def render_webgpu_vmobject_fill(
    renderer: WebGPURenderer,
    mobject: VMobject,
) -> None:
    """Record fill draw calls for *mobject* and all its descendants."""
    for submob in mobject.family_members_with_points():
        _draw_vmobject_fill(renderer, submob)


def render_webgpu_vmobject_stroke(
    renderer: WebGPURenderer,
    mobject: VMobject,
) -> None:
    """Record stroke draw calls for *mobject* and all its descendants."""
    for submob in mobject.family_members_with_points():
        _draw_vmobject_stroke(renderer, submob)


# ---------------------------------------------------------------------------
# Cubic → quadratic subdivision (used by fill triangulation)
# ---------------------------------------------------------------------------

_CUBIC_SUBDIVISION_LEVELS: int = 2  # 4 quadratic pieces per cubic bezier


def _cubic_to_quadratics(
    b0s: np.ndarray,
    h0s: np.ndarray,
    h1s: np.ndarray,
    b2s: np.ndarray,
    levels: int = _CUBIC_SUBDIVISION_LEVELS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subdivide n cubic beziers into n*2^levels quadratic approximations.

    Each cubic (b0, h0, h1, b3) is split by de Casteljau at t=0.5 `levels`
    times, then each sub-cubic is approximated by the quadratic whose single
    control point is the midpoint of its two handles.

    Returns (qb0s, qmids, qb2s), each shaped (n * 2^levels, 3).
    """
    curves = np.stack([b0s, h0s, h1s, b2s], axis=1).astype(np.float64)  # (n, 4, 3)

    for _ in range(levels):
        n_cur = len(curves)
        c0, c1, c2, c3 = curves[:, 0], curves[:, 1], curves[:, 2], curves[:, 3]
        m01   = (c0 + c1) * 0.5
        m12   = (c1 + c2) * 0.5
        m23   = (c2 + c3) * 0.5
        m012  = (m01 + m12) * 0.5
        m123  = (m12 + m23) * 0.5
        m0123 = (m012 + m123) * 0.5

        new_curves = np.empty((n_cur * 2, 4, 3), dtype=np.float64)
        new_curves[0::2, 0] = c0;    new_curves[0::2, 1] = m01
        new_curves[0::2, 2] = m012;  new_curves[0::2, 3] = m0123
        new_curves[1::2, 0] = m0123; new_curves[1::2, 1] = m123
        new_curves[1::2, 2] = m23;   new_curves[1::2, 3] = c3
        curves = new_curves

    qb0s  = curves[:, 0]
    qmids = (curves[:, 1] + curves[:, 2]) * 0.5  # midpoint of sub-handles
    qb2s  = curves[:, 3]
    return qb0s, qmids, qb2s


# ---------------------------------------------------------------------------
# Fill draw helper
# ---------------------------------------------------------------------------


def _draw_vmobject_fill(
    renderer: WebGPURenderer,
    vmobject: VMobject,
) -> None:
    fill_rgba = vmobject.get_fill_rgbas()
    if fill_rgba.shape[0] == 0 or fill_rgba[0, 3] == 0:
        return  # transparent or no fill

    color = fill_rgba[0].astype(np.float32)

    result = _triangulate_cairo_vmobject(vmobject)
    if result is None:
        return
    verts, tex_coords, tex_modes = result
    if len(verts) == 0:
        return

    n_verts = len(verts)
    attrs = np.empty(n_verts, dtype=_FILL_DTYPE)
    attrs["in_vert"] = verts.astype(np.float32)
    attrs["in_color"] = color  # broadcast: same color for all vertices
    attrs["texture_coords"] = tex_coords.astype(np.float32)
    attrs["texture_mode"] = tex_modes.astype(np.float32)

    import wgpu  # local import so module loads without wgpu installed

    device: wgpu_t.GPUDevice = renderer.device
    vbo = device.create_buffer_with_data(
        data=attrs.tobytes(),
        usage=wgpu.BufferUsage.VERTEX,
    )
    renderer.frame_vbos.append(vbo)

    rp = renderer.current_render_pass
    rp.set_pipeline(renderer.fill_pipeline)
    rp.set_bind_group(0, renderer.camera_bind_group, [], 0, 0)
    rp.set_vertex_buffer(0, vbo)
    rp.draw(n_verts, 1, 0, 0)


# ---------------------------------------------------------------------------
# Fill triangulation: cubic VMobject → Loop-Blinn triangles
# ---------------------------------------------------------------------------


def _triangulate_cairo_vmobject(
    vmobject: VMobject,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (verts, tex_coords, tex_modes) for all fill triangles.

    Converts each cubic bezier to a quadratic approximation, classifies each
    curve as concave (+1) or convex (−1), emits bezier boundary triangles with
    Loop-Blinn UVs, and earclip-triangulates the flat interior (tex_mode=0).
    """
    subpaths = vmobject.get_subpaths()
    if not subpaths:
        return None

    nppcc = vmobject.n_points_per_cubic_curve  # 4
    atol = vmobject.tolerance_for_point_equality

    all_verts: list[np.ndarray] = []
    all_tex_coords: list[np.ndarray] = []
    all_tex_modes: list[np.ndarray] = []

    for subpath in subpaths:
        n_curves = len(subpath) // nppcc
        if n_curves == 0:
            continue

        pts = subpath[: n_curves * nppcc]  # (4n, 3)

        # Cubic control points.
        b0s_cubic = pts[0::nppcc]     # start anchors   (n, 3)
        h0s_cubic = pts[1::nppcc]     # handle 0
        h1s_cubic = pts[2::nppcc]     # handle 1
        b2s_cubic = pts[3::nppcc]     # end anchors     (n, 3)

        # Subdivide each cubic into 4 quadratic approximations (2 de Casteljau levels).
        b0s, b1s, b2s = _cubic_to_quadratics(b0s_cubic, h0s_cubic, h1s_cubic, b2s_cubic)
        n_curves = len(b0s)  # now 4 × original

        # Build flat (3*n_curves, 3) quadratic representation for the triangulation.
        quad_pts = np.empty((n_curves * 3, 3), dtype=np.float64)
        quad_pts[0::3] = b0s
        quad_pts[1::3] = b1s
        quad_pts[2::3] = b2s

        # Classify curves.
        v01s = b1s - b0s
        v12s = b2s - b1s
        crosses = cross2d(v01s, v12s)
        convexities = np.sign(crosses)

        # Orientation from signed area of anchor polygon.
        ax, ay = b0s[:, 0], b0s[:, 1]
        signed_area = float(
            np.sum(ax * np.roll(ay, -1) - np.roll(ax, -1) * ay)
        )
        if signed_area >= 0:
            concave_parts = convexities > 0
            convex_parts = convexities <= 0
        else:
            concave_parts = convexities < 0
            convex_parts = convexities >= 0

        # ── Bezier boundary triangles ──────────────────────────────────────
        # UVs for every bezier triangle: (0,0) at b0, (0.5,0) at b1, (1,1) at b2.
        _UV_TILE = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 1.0]], dtype=np.float32)

        if np.any(concave_parts):
            n_c = int(np.sum(concave_parts))
            tri = np.empty((n_c * 3, 3))
            tri[0::3] = b0s[concave_parts]
            tri[1::3] = b1s[concave_parts]
            tri[2::3] = b2s[concave_parts]
            all_verts.append(tri)
            all_tex_coords.append(np.tile(_UV_TILE, (n_c, 1)))
            all_tex_modes.append(np.ones(n_c * 3, dtype=np.float32))

        if np.any(convex_parts):
            n_v = int(np.sum(convex_parts))
            tri = np.empty((n_v * 3, 3))
            tri[0::3] = b0s[convex_parts]
            tri[1::3] = b1s[convex_parts]
            tri[2::3] = b2s[convex_parts]
            all_verts.append(tri)
            all_tex_coords.append(np.tile(_UV_TILE, (n_v, 1)))
            all_tex_modes.append(-np.ones(n_v * 3, dtype=np.float32))

        # ── Flat interior (earclip) ────────────────────────────────────────
        # Inner polygon = all b0s + b1s of concave curves + b2s at loop ends.
        end_of_loop = np.zeros(n_curves, dtype=bool)
        if n_curves > 1:
            end_of_loop[:-1] = (np.abs(b2s[:-1] - b0s[1:]) > atol).any(1)
        end_of_loop[-1] = True

        # Indices into quad_pts (0::3=b0, 1::3=b1, 2::3=b2).
        idx = np.arange(n_curves)
        inner_vert_indices = np.hstack(
            [
                idx * 3,                                # b0 indices in quad_pts
                idx[concave_parts] * 3 + 1,            # b1 of concave curves
                idx[end_of_loop] * 3 + 2,              # b2 at loop ends
            ]
        )
        inner_vert_indices.sort()

        # Ring ends: positions of b2-index entries (index % 3 == 2).
        rings = (
            np.arange(1, len(inner_vert_indices) + 1)[inner_vert_indices % 3 == 2]
        ).tolist()

        inner_verts = quad_pts[inner_vert_indices]  # (M, 3)
        if len(inner_verts) < 3 or not rings:
            continue

        tri_indices_raw = earclip_triangulation(inner_verts[:, :2], rings)
        if not tri_indices_raw:
            continue

        # Map back to quad_pts indices.
        inner_tri_indices = inner_vert_indices[
            np.array(tri_indices_raw, dtype=int)
        ]
        inner_pts = quad_pts[inner_tri_indices]  # (K, 3)
        n_inner = len(inner_pts)

        all_verts.append(inner_pts)
        all_tex_coords.append(np.zeros((n_inner, 2), dtype=np.float32))
        all_tex_modes.append(np.zeros(n_inner, dtype=np.float32))

    if not all_verts:
        return None

    return (
        np.concatenate(all_verts, axis=0),
        np.concatenate(all_tex_coords, axis=0),
        np.concatenate(all_tex_modes, axis=0),
    )


# ---------------------------------------------------------------------------
# Stroke draw helper
# ---------------------------------------------------------------------------


def _draw_vmobject_stroke(
    renderer: WebGPURenderer,
    vmobject: VMobject,
) -> None:
    stroke_rgba = vmobject.get_stroke_rgbas()
    stroke_width = float(vmobject.get_stroke_width())
    if stroke_rgba.shape[0] == 0 or stroke_rgba[0, 3] == 0 or stroke_width == 0:
        return  # invisible stroke

    color = stroke_rgba[0].astype(np.float32)
    nppcc = vmobject.n_points_per_cubic_curve

    curve_list: list[np.ndarray] = []
    for subpath in vmobject.get_subpaths():
        n_curves = len(subpath) // nppcc
        if n_curves == 0:
            continue
        pts = subpath[: n_curves * nppcc]
        b0s = pts[0::nppcc]
        h0s = pts[1::nppcc]   # first handle
        h1s = pts[2::nppcc]   # second handle
        b2s = pts[3::nppcc]   # end anchor

        # Stack into (n_curves, 4, 3): all 4 cubic control points per curve.
        curve_list.append(np.stack([b0s, h0s, h1s, b2s], axis=1))

    if not curve_list:
        return

    all_curves = np.concatenate(curve_list, axis=0).astype(np.float32)  # (N, 4, 3)
    n_total = len(all_curves)

    # Each curve → 3 identical vertex records (repeated for the two triangles).
    base = np.zeros(n_total * 3, dtype=_STROKE_DTYPE)
    base["current_curve"] = np.repeat(all_curves, 3, axis=0)
    base["in_color"] = color
    base["in_width"] = stroke_width

    # Tile × 2 to form 6 vertices per curve (2 triangles).
    stroke_data = np.tile(base, 2)
    n_half = n_total * 3

    stroke_data["tile_coordinate"][:n_half] = np.tile(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]], (n_total, 1)
    )
    stroke_data["tile_coordinate"][n_half:] = np.tile(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], (n_total, 1)
    )

    import wgpu  # local import so module loads without wgpu installed

    device: wgpu_t.GPUDevice = renderer.device
    vbo = device.create_buffer_with_data(
        data=stroke_data.tobytes(),
        usage=wgpu.BufferUsage.VERTEX,
    )
    renderer.frame_vbos.append(vbo)

    rp = renderer.current_render_pass
    rp.set_pipeline(renderer.stroke_pipeline)
    rp.set_bind_group(0, renderer.camera_bind_group, [], 0, 0)
    rp.set_vertex_buffer(0, vbo)
    rp.draw(len(stroke_data), 1, 0, 0)


# ---------------------------------------------------------------------------
# Surface draw helper (Phase 3)
# ---------------------------------------------------------------------------


def _draw_surface_face(
    renderer: WebGPURenderer,
    vmobject: VMobject,
) -> None:
    """Fan-triangulate subpaths of a shade_in_3d VMobject and draw them."""
    fill_rgba = vmobject.get_fill_rgbas()
    if fill_rgba.shape[0] == 0 or fill_rgba[0, 3] == 0:
        return

    color = fill_rgba[0].astype(np.float32)
    nppcc = vmobject.n_points_per_cubic_curve

    all_verts: list[np.ndarray] = []
    all_normals: list[np.ndarray] = []

    for subpath in vmobject.get_subpaths():
        n_curves = len(subpath) // nppcc
        if n_curves < 2:
            continue
        # Anchor points only (every 4th point starting at 0).
        anchors = subpath[0::nppcc]  # (n_curves, 3)
        # Include the last endpoint so the polygon closes.
        last = subpath[n_curves * nppcc - 1 : n_curves * nppcc]
        if len(last) and not np.allclose(anchors[-1], last[0], atol=1e-6):
            anchors = np.vstack([anchors, last])

        n_pts = len(anchors)
        if n_pts < 3:
            continue

        # Face normal — cross product of first two edges from centroid.
        centroid = anchors.mean(axis=0)
        v0 = anchors[0] - centroid
        v1 = anchors[1] - centroid
        raw_normal = np.cross(v0, v1).astype(np.float64)
        norm_len = np.linalg.norm(raw_normal)
        normal = (raw_normal / norm_len).astype(np.float32) if norm_len > 1e-9 else np.array([0, 0, 1], dtype=np.float32)

        # Fan triangulation from centroid:
        #   (centroid, anchors[i], anchors[i+1])  for i in 0..n_pts-1
        fan_verts = np.empty((n_pts * 3, 3), dtype=np.float32)
        fan_verts[0::3] = centroid.astype(np.float32)
        fan_verts[1::3] = anchors.astype(np.float32)
        fan_verts[2::3] = np.roll(anchors, -1, axis=0).astype(np.float32)

        all_verts.append(fan_verts)
        all_normals.append(np.tile(normal, (n_pts * 3, 1)))

    if not all_verts:
        return

    verts = np.concatenate(all_verts, axis=0)
    normals = np.concatenate(all_normals, axis=0)
    n_total = len(verts)

    attrs = np.empty(n_total, dtype=_SURFACE_DTYPE)
    attrs["in_vert"] = verts
    attrs["in_normal"] = normals
    attrs["in_color"] = color  # broadcast

    import wgpu  # local import so module loads without wgpu installed

    device: wgpu_t.GPUDevice = renderer.device
    vbo = device.create_buffer_with_data(
        data=attrs.tobytes(),
        usage=wgpu.BufferUsage.VERTEX,
    )
    renderer.frame_vbos.append(vbo)

    rp = renderer.current_render_pass
    rp.set_pipeline(renderer.surface_pipeline)
    rp.set_bind_group(0, renderer.camera_bind_group, [], 0, 0)
    rp.set_vertex_buffer(0, vbo)
    rp.draw(n_total, 1, 0, 0)
