"""WebGPU draw calls for VMobject fill + stroke + surface rendering — Phase 2/3.

Fill
----
Uses the Slug algorithm (Lengyel 2017) for GPU-side analytical fill coverage.
Raw quadratic bezier control points are uploaded to a storage buffer; the
fragment shader computes exact winding-number coverage per pixel with smooth
sub-pixel anti-aliasing.  No CPU tessellation is required.

Stroke
------
All four cubic bezier control points (b0, h0, h1, b3) are passed to the GPU.
The fragment shader computes the exact unsigned distance to the cubic bezier
curve via Newton's-method minimisation and discards pixels outside the stroke
half-width — no quadratic approximation is made.

Batched rendering
-----------------
``render_webgpu_mobject`` collects geometry for *all* scene mobjects before
touching the GPU.  All fill data is concatenated into one GPU buffer, all
stroke data into another, all surface data into a third (1–3 allocations per
frame regardless of mobject count).  Draw calls reference sub-ranges of those
shared buffers via ``set_vertex_buffer(slot, buf, offset)``, preserving the
exact painter's-algorithm order.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from manim.mobject.three_d.three_dimensions import Surface
from manim.mobject.types.vectorized_mobject import VMobject

if TYPE_CHECKING:
    import wgpu as wgpu_t

    from manim.renderer.webgpu.webgpu_renderer import WebGPURenderer


@dataclass
class _OITPassData:
    """Geometry needed for the OIT accumulation render pass."""
    surface_parts: list[np.ndarray]
    surface_buf: wgpu_t.GPUBuffer
    byte_offsets: list[int]
    oit_indices: list[int]  # indices into surface_parts that are OIT


def _surface_opacity_class(part: np.ndarray) -> str:
    """Classify a surface part by its alpha distribution.

    Returns:
        "opaque" — all alpha >= 0.99  → opaque pipeline, depth write
        "oit"    — any alpha < 0.99   → Weighted Blended OIT
    """
    alphas = part["in_color"][:, 3]
    if float(alphas.min()) >= 0.99:
        return "opaque"
    return "oit"


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
# Slug fill vertex layout — must match slug_fill.wgsl locations:
#   location 0 → in_pos       float32x3  offset  0  (12 bytes)
#   location 1 → in_color     float32x4  offset 12  (16 bytes)
#   location 2 → curve_start  uint32     offset 28  ( 4 bytes)
#   location 3 → n_curves     uint32     offset 32  ( 4 bytes)
#   stride: 36 bytes
# ---------------------------------------------------------------------------

_SLUG_FILL_DTYPE = np.dtype(
    [
        ("in_pos",      np.float32, (3,)),
        ("in_color",    np.float32, (4,)),
        ("curve_start", np.uint32),
        ("n_curves",    np.uint32),
    ]
)
_SLUG_FILL_STRIDE: int = _SLUG_FILL_DTYPE.itemsize  # 36 bytes

_SLUG_FILL_OFFSETS: dict[str, int] = {
    name: _SLUG_FILL_DTYPE.fields[name][1]  # type: ignore[index]
    for name in _SLUG_FILL_DTYPE.names
}

SLUG_FILL_VERTEX_LAYOUT: dict = {
    "array_stride": _SLUG_FILL_STRIDE,
    "step_mode": "vertex",
    "attributes": [
        {"format": "float32x3", "offset": _SLUG_FILL_OFFSETS["in_pos"],      "shader_location": 0},
        {"format": "float32x4", "offset": _SLUG_FILL_OFFSETS["in_color"],    "shader_location": 1},
        {"format": "uint32",    "offset": _SLUG_FILL_OFFSETS["curve_start"], "shader_location": 2},
        {"format": "uint32",    "offset": _SLUG_FILL_OFFSETS["n_curves"],    "shader_location": 3},
    ],
}


# ---------------------------------------------------------------------------
# Geometry caches — eliminates repeated tessellation for static shapes.
#
# Both caches are WeakKeyDictionary so GC can reclaim vmobjects that have
# been removed from the scene.  Each entry maps:
#   vmobject → (points_hash: int, geometry: ndarray | tuple[ndarray, ndarray])
#
# The points_hash is recomputed cheaply (tobytes hash) every frame; a mismatch
# means the shape changed and we re-tessellate.
# ---------------------------------------------------------------------------

_slug_fill_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_stroke_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _points_hash(vmobject: VMobject) -> int:
    """Fast hash of vmobject.points — used to detect geometry changes."""
    pts = vmobject.points
    if pts.size == 0:
        return 0
    return hash(pts.tobytes())


# ---------------------------------------------------------------------------
# Public entry point — batched rendering
# ---------------------------------------------------------------------------


def render_webgpu_mobject(
    renderer: WebGPURenderer,
    mobjects: list,
) -> _OITPassData | None:
    """Batch-render all VMobjects in *mobjects* (the scene's top-level list).

    Rendering is split into typed draw commands executed in order:

    * ``slug_fill``       — 2-D Slug fill (no depth test)
    * ``surface_opaque``  — 3-D opaque surface fill (depth write + test)
    * ``stroke_2d``       — 2-D stroke (depth-tested, depth_write=False — occluded by surfaces)
    * ``stroke_3d``       — 3-D stroke for shade_in_3d objects (depth test)

    The ``surface_oit`` category is NOT executed here; the function returns
    an ``_OITPassData`` with the OIT geometry so ``update_frame`` can run it
    in a separate OIT accumulation pass.  All surfaces with any alpha < 0.99
    are routed through OIT for correct order-independent compositing.

    **shade_in_3d objects**: fill (if present) goes to slug_fill_3d; stroke
    (if present) goes to stroke_3d.  Both are emitted independently, so an
    object with both fill and stroke renders a filled surface with 3-D strokes
    on top.  Objects with no fill (axis lines, arrowheads, tick marks) emit
    only stroke_3d.

    **Issue 2 fix**: the stroke vertex shader now works in view space, so any
    3-D curve (including the world Z-axis) is rendered correctly.

    **Issue 3 fix**: stroke_3d uses depth_compare=less, so 3-D axis lines are
    occluded by surfaces in front of them.
    """
    import wgpu  # local import so module loads without wgpu installed

    view_matrix: np.ndarray = renderer.camera.view_matrix

    # ── Phase 1: tessellate ───────────────────────────────────────────────
    slug_quad_parts:  list[np.ndarray] = []
    slug_curve_parts: list[np.ndarray] = []
    stroke_parts:     list[np.ndarray] = []
    surface_parts:    list[np.ndarray] = []

    # Command types: slug_fill | slug_fill_3d
    #                surface_opaque | surface_oit
    #                stroke_2d | stroke_3d
    draw_plan: list[tuple[str, int]] = []

    for mob in mobjects:
        mob_type = type(mob).__name__
        if not isinstance(mob, VMobject):
            continue

        # ── Parametric Surface: triangle-mesh + Phong lighting ────────────
        # Also collect the stroke of each face so the mesh grid is visible,
        # matching Cairo which renders both fill and stroke for every face.
        if isinstance(mob, Surface):
            for submob in mob.family_members_with_points():
                data = _collect_surface_geometry(submob)
                if data is not None:
                    cls = _surface_opacity_class(data)
                    cmd = "surface_opaque" if cls == "opaque" else "surface_oit"
                    draw_plan.append((cmd, len(surface_parts)))
                    surface_parts.append(data)

                phash   = _points_hash(submob)
                scached = _stroke_cache.get(submob)
                if scached is not None and scached[0] == phash:
                    draw_plan.append(("stroke_surface", len(stroke_parts)))
                    stroke_parts.append(scached[1])
                else:
                    stroke_data = _collect_stroke_geometry(submob)
                    if stroke_data is not None:
                        _stroke_cache[submob] = (phash, stroke_data)
                        draw_plan.append(("stroke_surface", len(stroke_parts)))
                        stroke_parts.append(stroke_data)
            continue

        for submob in mob.family_members_with_points():
            phash   = _points_hash(submob)
            scached = _stroke_cache.get(submob)

            if getattr(submob, "shade_in_3d", False):
                fill_rgba = submob.get_fill_rgbas()
                has_fill  = fill_rgba.shape[0] > 0 and float(fill_rgba[0, 3]) > 0.01

                if has_fill:
                    # 3-D flat VMobject with fill (e.g. number-plane, polygon in 3D):
                    cached = _slug_fill_cache.get(submob)
                    if cached is not None and cached[0] == phash:
                        quad_verts, curves_flat = cached[1]
                        draw_plan.append(("slug_fill_3d", len(slug_quad_parts)))
                        slug_quad_parts.append(quad_verts.copy())
                        slug_curve_parts.append(curves_flat)
                    else:
                        slug_data = _collect_slug_fill_geometry(submob, view_matrix)
                        if slug_data is not None:
                            _slug_fill_cache[submob] = (phash, slug_data)
                            quad_verts, curves_flat = slug_data
                            draw_plan.append(("slug_fill_3d", len(slug_quad_parts)))
                            slug_quad_parts.append(quad_verts.copy())
                            slug_curve_parts.append(curves_flat)

                # 3-D stroke logic (axis lines, etc.)
                if scached is not None and scached[0] == phash:
                    draw_plan.append(("stroke_3d", len(stroke_parts)))
                    stroke_parts.append(scached[1])
                else:
                    stroke_data = _collect_stroke_geometry(submob)
                    if stroke_data is not None:
                        _stroke_cache[submob] = (phash, stroke_data)
                        draw_plan.append(("stroke_3d", len(stroke_parts)))
                        stroke_parts.append(stroke_data)

            else:
                # ── 2-D object: Slug fill + 2-D stroke ──────────────────────
                cached = _slug_fill_cache.get(submob)
                if cached is not None and cached[0] == phash:
                    quad_verts, curves_flat = cached[1]
                    draw_plan.append(("slug_fill", len(slug_quad_parts)))
                    slug_quad_parts.append(quad_verts.copy())
                    slug_curve_parts.append(curves_flat)
                else:
                    slug_data = _collect_slug_fill_geometry(submob, view_matrix)
                    if slug_data is not None:
                        _slug_fill_cache[submob] = (phash, slug_data)
                        quad_verts, curves_flat = slug_data
                        draw_plan.append(("slug_fill", len(slug_quad_parts)))
                        slug_quad_parts.append(quad_verts.copy())
                        slug_curve_parts.append(curves_flat)

                if scached is not None and scached[0] == phash:
                    draw_plan.append(("stroke_2d", len(stroke_parts)))
                    stroke_parts.append(scached[1])
                else:
                    stroke_data = _collect_stroke_geometry(submob)
                    if stroke_data is not None:
                        _stroke_cache[submob] = (phash, stroke_data)
                        draw_plan.append(("stroke_2d", len(stroke_parts)))
                        stroke_parts.append(stroke_data)

    if not draw_plan:
        return None

    # ── Phase 2: batch upload ─────────────────────────────────────────────
    device: wgpu_t.GPUDevice = renderer.device

    slug_fill_vbo = slug_fill_byte_offsets = None
    slug_bind_group = None
    stroke_buf = stroke_byte_offsets = None
    surface_buf = surface_byte_offsets = None

    if slug_quad_parts:
        curve_global_offset = 0
        for i, curves_flat in enumerate(slug_curve_parts):
            n_quads_i = len(curves_flat) // 3
            slug_quad_parts[i]["curve_start"] = curve_global_offset
            curve_global_offset += n_quads_i

        slug_fill_vbo, slug_fill_byte_offsets = _batch_upload(device, slug_quad_parts)
        renderer.frame_vbos.append(slug_fill_vbo)

        all_curves = np.concatenate(slug_curve_parts, axis=0)
        slug_curves_buf = device.create_buffer_with_data(
            data=all_curves.tobytes(),
            usage=wgpu.BufferUsage.STORAGE,
        )
        renderer.frame_vbos.append(slug_curves_buf)
        slug_bind_group = renderer._build_slug_bind_group(slug_curves_buf)

    if stroke_parts:
        stroke_buf, stroke_byte_offsets = _batch_upload(device, stroke_parts)
        renderer.frame_vbos.append(stroke_buf)

    if surface_parts:
        _smooth_surface_normals(surface_parts)
        surface_buf, surface_byte_offsets = _batch_upload(device, surface_parts)
        renderer.frame_vbos.append(surface_buf)

    # ── Phase 3: draw in the main render pass ─────────────────────────────
    # Execute in this fixed order:
    #   slug_fill → surface_opaque → stroke_2d → stroke_3d → stroke_surface
    # stroke_surface uses a depth-biased pipeline so mesh lines sitting exactly
    # on the surface never z-fight with it.
    # surface_oit entries are skipped here and returned for a separate OIT pass.
    rp = renderer.current_render_pass
    cam_bg = renderer.camera_bind_group

    def _draw_stroke(idx: int, pipeline_name: str, _cur: list) -> None:
        if pipeline_name == "stroke_2d":
            pipeline = renderer.stroke_pipeline
        elif pipeline_name == "stroke_surface":
            pipeline = renderer.stroke_3d_surface_pipeline
        else:
            pipeline = renderer.stroke_3d_pipeline
        if _cur[0] != pipeline_name:
            rp.set_pipeline(pipeline)
            rp.set_bind_group(0, cam_bg, [], 0, 0)
            _cur[0] = pipeline_name
        arr = stroke_parts[idx]
        rp.set_vertex_buffer(0, stroke_buf, stroke_byte_offsets[idx], arr.nbytes)
        rp.draw(len(arr), 1, 0, 0)

    def _draw_surface(idx: int, pipeline, pipeline_key: str, _cur: list) -> None:
        if _cur[0] != pipeline_key:
            rp.set_pipeline(pipeline)
            rp.set_bind_group(0, cam_bg, [], 0, 0)
            _cur[0] = pipeline_key
        arr = surface_parts[idx]
        rp.set_vertex_buffer(0, surface_buf, surface_byte_offsets[idx], arr.nbytes)
        rp.draw(len(arr), 1, 0, 0)

    _cur: list[str | None] = [None]  # mutable current-pipeline tracker

    # 1. Slug fills (2-D — no depth test)
    if slug_fill_vbo is not None:
        for cmd_type, idx in draw_plan:
            if cmd_type == "slug_fill":
                if _cur[0] != "slug_fill":
                    rp.set_pipeline(renderer.slug_fill_pipeline)
                    rp.set_bind_group(0, slug_bind_group, [], 0, 0)
                    _cur[0] = "slug_fill"
                arr = slug_quad_parts[idx]
                rp.set_vertex_buffer(0, slug_fill_vbo, slug_fill_byte_offsets[idx], arr.nbytes)
                rp.draw(len(arr), 1, 0, 0)

        # 1b. Slug fills (3-D — depth-tested, shade_in_3d with fill)
        for cmd_type, idx in draw_plan:
            if cmd_type == "slug_fill_3d":
                if _cur[0] != "slug_fill_3d":
                    rp.set_pipeline(renderer.slug_fill_3d_pipeline)
                    rp.set_bind_group(0, slug_bind_group, [], 0, 0)
                    _cur[0] = "slug_fill_3d"
                arr = slug_quad_parts[idx]
                rp.set_vertex_buffer(0, slug_fill_vbo, slug_fill_byte_offsets[idx], arr.nbytes)
                rp.draw(len(arr), 1, 0, 0)

    # 2. Opaque surfaces
    if surface_buf is not None:
        for cmd_type, idx in draw_plan:
            if cmd_type == "surface_opaque":
                _draw_surface(idx, renderer.surface_pipeline, "surface_opaque", _cur)

    # 5. 2-D strokes
    if stroke_buf is not None:
        for cmd_type, idx in draw_plan:
            if cmd_type == "stroke_2d":
                _draw_stroke(idx, "stroke_2d", _cur)

        # 6. 3-D strokes (depth-tested)
        for cmd_type, idx in draw_plan:
            if cmd_type == "stroke_3d":
                _draw_stroke(idx, "stroke_3d", _cur)

        # 7. Surface mesh strokes (depth-biased to prevent z-fighting)
        for cmd_type, idx in draw_plan:
            if cmd_type == "stroke_surface":
                _draw_stroke(idx, "stroke_surface", _cur)

    # ── OIT data for the caller ───────────────────────────────────────────
    oit_indices = [idx for cmd_type, idx in draw_plan if cmd_type == "surface_oit"]
    if oit_indices and surface_buf is not None:
        return _OITPassData(
            surface_parts=surface_parts,
            surface_buf=surface_buf,
            byte_offsets=surface_byte_offsets,
            oit_indices=oit_indices,
        )
    return None


# ---------------------------------------------------------------------------
# Explicit single-mobject public helpers (kept for callers outside update_frame)
# ---------------------------------------------------------------------------


def render_webgpu_surface(
    renderer: WebGPURenderer,
    mobject: VMobject,
) -> None:
    """Record surface draw calls for *mobject* and its descendants."""
    for submob in mobject.family_members_with_points():
        if getattr(submob, "shade_in_3d", False):
            _draw_surface_face(renderer, submob)


def render_webgpu_vmobject_stroke(
    renderer: WebGPURenderer,
    mobject: VMobject,
) -> None:
    """Record stroke draw calls for *mobject* and all its descendants."""
    for submob in mobject.family_members_with_points():
        _draw_vmobject_stroke(renderer, submob)


# ---------------------------------------------------------------------------
# GPU upload helpers
# ---------------------------------------------------------------------------


def _batch_upload(
    device: wgpu_t.GPUDevice,
    arrays: list[np.ndarray],
) -> tuple[wgpu_t.GPUBuffer, list[int]]:
    """Concatenate *arrays* into one bytes blob and upload as a single VERTEX buffer.

    Returns ``(gpu_buffer, byte_offsets)`` where ``byte_offsets[i]`` is the
    byte position of ``arrays[i]`` within the buffer.
    """
    import wgpu

    byte_offsets: list[int] = []
    parts: list[bytes] = []
    offset = 0
    for arr in arrays:
        byte_offsets.append(offset)
        b = arr.tobytes()
        parts.append(b)
        offset += len(b)

    buf = device.create_buffer_with_data(
        data=b"".join(parts),
        usage=wgpu.BufferUsage.VERTEX,
    )
    return buf, byte_offsets


# ---------------------------------------------------------------------------
# Geometry collectors — CPU only, no GPU calls
# ---------------------------------------------------------------------------


def _collect_stroke_geometry(vmobject: VMobject) -> np.ndarray | None:
    """Return a ``_STROKE_DTYPE`` array for *vmobject*'s stroke, or ``None``."""
    stroke_rgba  = vmobject.get_stroke_rgbas()
    stroke_width = float(vmobject.get_stroke_width())
    if stroke_rgba.shape[0] == 0 or stroke_rgba[0, 3] == 0 or stroke_width == 0:
        return None

    color  = stroke_rgba[0].astype(np.float32)
    nppcc  = vmobject.n_points_per_cubic_curve

    curve_list: list[np.ndarray] = []
    for subpath in vmobject.get_subpaths():
        n_curves = len(subpath) // nppcc
        if n_curves == 0:
            continue
        pts = subpath[: n_curves * nppcc]
        b0s = pts[0::nppcc]
        h0s = pts[1::nppcc]
        h1s = pts[2::nppcc]
        b2s = pts[3::nppcc]
        curve_list.append(np.stack([b0s, h0s, h1s, b2s], axis=1))

    if not curve_list:
        return None

    all_curves = np.concatenate(curve_list, axis=0).astype(np.float32)  # (N, 4, 3)
    n_total    = len(all_curves)

    base = np.zeros(n_total * 3, dtype=_STROKE_DTYPE)
    base["current_curve"] = np.repeat(all_curves, 3, axis=0)
    base["in_color"]      = color
    base["in_width"]      = stroke_width

    stroke_data = np.tile(base, 2)
    n_half = n_total * 3
    stroke_data["tile_coordinate"][:n_half] = np.tile(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]], (n_total, 1)
    )
    stroke_data["tile_coordinate"][n_half:] = np.tile(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], (n_total, 1)
    )
    return stroke_data


def _smooth_surface_normals(surface_parts: list[np.ndarray]) -> None:
    """Average normals at shared vertex positions to produce smooth shading.

    Modifies ``surface_parts`` in-place.  Vertices whose positions match
    (within 1e-5 world-space units) share a common averaged normal, removing
    the hard crease lines produced by per-face flat normals.
    """
    if not surface_parts:
        return

    # Stack all vertex positions and flat face normals.
    all_verts = np.concatenate([p["in_vert"]   for p in surface_parts], axis=0)  # (T, 3) f32
    all_norms = np.concatenate([p["in_normal"] for p in surface_parts], axis=0)  # (T, 3) f32

    # Quantise positions so that vertices within 1e-5 units map to the same key.
    PREC = 1e-5
    quantized = np.round(all_verts.astype(np.float64) / PREC).astype(np.int64)  # (T, 3)

    # np.unique on a 2-D array of int64 rows → unique vertex groups.
    _, inverse = np.unique(quantized, axis=0, return_inverse=True)  # inverse: (T,)

    # Accumulate face normals per unique position.
    n_unique = int(inverse.max()) + 1
    smooth = np.zeros((n_unique, 3), dtype=np.float64)
    np.add.at(smooth, inverse, all_norms.astype(np.float64))

    # Normalise.
    lengths = np.linalg.norm(smooth, axis=1, keepdims=True)
    lengths = np.where(lengths < 1e-9, 1.0, lengths)
    smooth = (smooth / lengths).astype(np.float32)

    # Write smoothed normals back into each part's structured array.
    idx = 0
    for part in surface_parts:
        n = len(part)
        part["in_normal"] = smooth[inverse[idx : idx + n]]
        idx += n


def _collect_surface_geometry(vmobject: VMobject) -> np.ndarray | None:
    """Return a ``_SURFACE_DTYPE`` array for a shade_in_3d VMobject, or ``None``."""
    fill_rgba = vmobject.get_fill_rgbas()
    if fill_rgba.shape[0] == 0 or fill_rgba[0, 3] == 0:
        return None

    color = fill_rgba[0].astype(np.float32)
    nppcc = vmobject.n_points_per_cubic_curve

    all_verts:   list[np.ndarray] = []
    all_normals: list[np.ndarray] = []

    for subpath in vmobject.get_subpaths():
        n_curves = len(subpath) // nppcc
        if n_curves < 2:
            continue
        anchors = subpath[0::nppcc]
        last    = subpath[n_curves * nppcc - 1 : n_curves * nppcc]
        if len(last) and not np.allclose(anchors[-1], last[0], atol=1e-6):
            anchors = np.vstack([anchors, last])

        n_pts = len(anchors)
        if n_pts < 3:
            continue

        centroid   = anchors.mean(axis=0)
        v0         = anchors[0] - centroid
        v1         = anchors[1] - centroid
        # WebGPU evaluates front_face="ccw" in framebuffer space (Y points DOWN).
        # Clip space has Y pointing UP, so Y is negated going clip → framebuffer,
        # which reverses the apparent winding.  A triangle that is CW in clip/
        # world Y-up space becomes CCW in framebuffer space = FRONT FACE.
        #
        # Manim's anchor ordering is CCW in world Y-up space when viewed from
        # outside the surface (confirmed analytically for standard sphere/torus
        # parameterisations).  Therefore:
        #   (centroid, curr, next) → CW in world Y-up → CCW in framebuffer → FRONT FACE ✓
        #   (centroid, next, curr) → CCW in world Y-up → CW in framebuffer → BACK FACE (culled) ✗
        #
        # Normal: v1 × v0 with CCW-from-outside anchors gives the outward-pointing
        # normal (v0 × v1 would be inward).
        raw_normal = np.cross(v1, v0).astype(np.float64)
        norm_len   = np.linalg.norm(raw_normal)
        normal     = (
            (raw_normal / norm_len).astype(np.float32)
            if norm_len > 1e-9
            else np.array([0.0, 0.0, 1.0], dtype=np.float32)
        )

        fan_verts = np.empty((n_pts * 3, 3), dtype=np.float32)
        fan_verts[0::3] = centroid.astype(np.float32)
        fan_verts[1::3] = anchors.astype(np.float32)                        # curr
        fan_verts[2::3] = np.roll(anchors, -1, axis=0).astype(np.float32)  # next

        all_verts.append(fan_verts)
        all_normals.append(np.tile(normal, (n_pts * 3, 1)))

    if not all_verts:
        return None

    verts   = np.concatenate(all_verts,   axis=0)
    normals = np.concatenate(all_normals, axis=0)
    n_total = len(verts)

    attrs = np.empty(n_total, dtype=_SURFACE_DTYPE)
    attrs["in_vert"]   = verts
    attrs["in_normal"] = normals
    attrs["in_color"]  = color
    return attrs


# ---------------------------------------------------------------------------
# Slug fill geometry collector
# ---------------------------------------------------------------------------


def _collect_slug_fill_geometry(
    vmobject: VMobject,
    view_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(quad_verts, curves_flat)`` for the Slug fill pipeline, or ``None``.

    *quad_verts* is a ``_SLUG_FILL_DTYPE`` array of 6 vertices (world-space
    3-D positions) forming a bounding quad for this shape in view space.
    ``curve_start`` is set to 0 and must be patched to the global offset by
    the caller before upload.

    *curves_flat* is a ``float32`` array of shape ``(n_quads * 3, 3)``
    containing the world-space XYZ coordinates of every quadratic bezier
    control point — three consecutive entries (p1, p2, p3) per curve.

    Parameters
    ----------
    view_matrix
        4×4 float32 view matrix (world → camera space).  When ``None`` an
        identity matrix is used (2-D orthographic default).  The bounding quad
        is computed in view space so it covers the projected shape regardless
        of the object's 3-D orientation.
    """
    fill_rgba = vmobject.get_fill_rgbas()
    if fill_rgba.shape[0] == 0 or fill_rgba[0, 3] == 0:
        return None

    color    = fill_rgba[0].astype(np.float32)
    subpaths = vmobject.get_subpaths()
    if not subpaths:
        return None

    nppcc = vmobject.n_points_per_cubic_curve

    per_subpath: list[np.ndarray] = []  # each: (n, 3, 3) world-space XYZ

    for subpath in subpaths:
        n_curves = len(subpath) // nppcc
        if n_curves == 0:
            continue
        pts = subpath[: n_curves * nppcc]

        b0s = pts[0::nppcc]
        h0s = pts[1::nppcc]
        h1s = pts[2::nppcc]
        b2s = pts[3::nppcc]

        # Subdivide cubics into quadratics (2 de Casteljau levels → 4 per cubic).
        qb0s, qmids, qb2s = _cubic_to_quadratics(b0s, h0s, h1s, b2s)

        # Stack to (n_quads, 3, 3): [p1, p2, p3] in world-space XYZ.
        curves_xyz = np.stack([qb0s, qmids, qb2s], axis=1)  # (n, 3, 3)

        # Implicit close: if the subpath is open (last anchor ≠ first anchor),
        # add a degenerate quadratic representing the straight closing line from
        # the last anchor back to the first anchor.  This matches Cairo's
        # implicit-close behaviour during partial animations (e.g. Create):
        # without this segment the winding-number integral is wrong for open
        # paths and the fill spills outside the intended region.
        first_anchor = b0s[0].astype(np.float32)
        last_anchor  = b2s[-1].astype(np.float32)
        if not np.allclose(first_anchor, last_anchor, atol=1e-6):
            mid_pt  = ((first_anchor + last_anchor) * 0.5).astype(np.float32)
            closing = np.array(
                [[last_anchor, mid_pt, first_anchor]], dtype=np.float32
            )  # (1, 3, 3)
            curves_xyz = np.concatenate([curves_xyz, closing], axis=0)

        per_subpath.append(curves_xyz.astype(np.float32))

    if not per_subpath:
        return None

    curves_stacked = np.concatenate(per_subpath, axis=0)  # (N_total, 3, 3)
    n_quads        = len(curves_stacked)
    curves_flat    = curves_stacked.reshape(-1, 3)         # (N_total * 3, 3)

    # ── Bounding quad in view space ───────────────────────────────────────
    # Transform all control points to view space, compute XY extent there,
    # then map the 4 bounding corners back to world space for storage.
    # This guarantees correct screen coverage for tilted 3-D objects.
    if view_matrix is None:
        view_matrix = np.eye(4, dtype=np.float32)
    vm = view_matrix.astype(np.float32)
    R, t = vm[:3, :3], vm[:3, 3]

    pts_v = (R @ curves_flat.T).T + t          # (N*3, 3) view space

    bbox_min = pts_v[:, :2].min(axis=0) - 0.05
    bbox_max = pts_v[:, :2].max(axis=0) + 0.05
    x0, y0   = float(bbox_min[0]), float(bbox_min[1])
    x1, y1   = float(bbox_max[0]), float(bbox_max[1])
    avg_z_v  = float(pts_v[:, 2].mean())

    # Four corners in view space (same average Z).
    corners_v = np.array(
        [[x0, y0, avg_z_v], [x1, y0, avg_z_v],
         [x0, y1, avg_z_v], [x1, y1, avg_z_v]],
        dtype=np.float32,
    )
    # Invert the view matrix (rigid-body: R^T, -R^T t).
    R_inv = R.T
    t_inv = -(R_inv @ t)
    corners_w = (R_inv @ corners_v.T).T + t_inv   # (4, 3) world space

    # 6 vertices: two CCW triangles.
    quad_pos = corners_w[[0, 1, 2, 1, 3, 2]]      # (6, 3)

    quad_verts = np.empty(6, dtype=_SLUG_FILL_DTYPE)
    quad_verts["in_pos"]      = quad_pos
    quad_verts["in_color"]    = color
    quad_verts["curve_start"] = 0       # patched to global offset by caller
    quad_verts["n_curves"]    = n_quads

    return quad_verts, curves_flat


# ---------------------------------------------------------------------------
# Single-mobject draw helpers (used by the explicit public helpers above)
# ---------------------------------------------------------------------------


def _draw_vmobject_stroke(renderer: WebGPURenderer, vmobject: VMobject) -> None:
    data = _collect_stroke_geometry(vmobject)
    if data is None:
        return

    import wgpu

    device: wgpu_t.GPUDevice = renderer.device
    vbo = device.create_buffer_with_data(
        data=data.tobytes(), usage=wgpu.BufferUsage.VERTEX
    )
    renderer.frame_vbos.append(vbo)

    rp = renderer.current_render_pass
    rp.set_pipeline(renderer.stroke_pipeline)
    rp.set_bind_group(0, renderer.camera_bind_group, [], 0, 0)
    rp.set_vertex_buffer(0, vbo)
    rp.draw(len(data), 1, 0, 0)


def _draw_surface_face(renderer: WebGPURenderer, vmobject: VMobject) -> None:
    data = _collect_surface_geometry(vmobject)
    if data is None:
        return

    import wgpu

    device: wgpu_t.GPUDevice = renderer.device
    vbo = device.create_buffer_with_data(
        data=data.tobytes(), usage=wgpu.BufferUsage.VERTEX
    )
    renderer.frame_vbos.append(vbo)

    rp = renderer.current_render_pass
    rp.set_pipeline(renderer.surface_pipeline)
    rp.set_bind_group(0, renderer.camera_bind_group, [], 0, 0)
    rp.set_vertex_buffer(0, vbo)
    rp.draw(len(data), 1, 0, 0)


# ---------------------------------------------------------------------------
# Cubic → quadratic subdivision (used by Slug fill geometry collector)
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
