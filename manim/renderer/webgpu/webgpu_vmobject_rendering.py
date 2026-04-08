"""WebGPU draw calls for VMobject fill + stroke + surface rendering.

Fill + Stroke (combined pipeline)
----------------------------------
``cubic_to_quads.wgsl`` — GPU compute shader converts raw cubic Bezier
control points to quadratic approximations (4 per cubic, two-level de
Casteljau subdivision).

``vmobject_fill_stroke.wgsl`` — combined render shader: one bounding quad
per object, one fragment loop accumulates both Slug winding-number fill
coverage (in NDC space) and SDF stroke distance (in pixel space).  Porter-
Duff "over" compositing produces the final colour.

Closing segments
~~~~~~~~~~~~~~~~
Every open subpath gets a linear closing cubic (degree-elevated line from
the last anchor back to the first) appended to the fill cubic list.  This
makes the winding-number integral correct for partial paths (e.g. during
``Create`` animations).  The closing cubic is NOT added to the stroke cubic
list — strokes should follow the visible part of the curve only.

Surfaces
--------
Parametric surfaces use a combined triangle-mesh pipeline
(``surface_combined.wgsl`` / ``surface_oit.wgsl``) with Phong lighting and
barycentric wireframe in a single draw call.  The centroid vertex of each
triangle fan carries bary=(1,0,0); the outer edge (anchor_i ↔ anchor_{i+1})
has bary.x=0 — this is the visible mesh-grid edge.  Transparent surfaces go
through the OIT accumulation + composition passes.

Batching
--------
``collect_frame_data`` tessellates *all* scene mobjects on the CPU, uploads
one cubics buffer (fill then stroke, all objects) and one vertex buffer (one
bounding quad per object), then returns a ``_FrameData`` ready for the GPU.
``draw_frame_data`` records draw calls into the active render pass.
"""

from __future__ import annotations

import struct
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from manim.mobject.three_d.three_dimensions import Surface
from manim.mobject.types.vectorized_mobject import VMobject

if TYPE_CHECKING:
    import wgpu as wgpu_t

    from manim.renderer.webgpu.webgpu_renderer import WebGPURenderer


# ---------------------------------------------------------------------------
# Combined surface vertex layout — must match surface_combined.wgsl /
# surface_oit.wgsl locations.
#
#   location 0 — in_vert          float32x3  offset  0  (12 B)
#   location 1 — in_normal        float32x3  offset 12  (12 B)
#   location 2 — in_fill_color    float32x4  offset 24  (16 B)
#   location 3 — in_stroke_color  float32x4  offset 40  (16 B)
#   location 4 — in_bary          float32x3  offset 56  (12 B)
#   location 5 — stroke_half_px   float32    offset 68  ( 4 B)
#   stride: 72 bytes
# ---------------------------------------------------------------------------

_SURFACE_COMBINED_DTYPE = np.dtype(
    [
        ("in_vert",         np.float32, (3,)),
        ("in_normal",       np.float32, (3,)),
        ("in_fill_color",   np.float32, (4,)),
        ("in_stroke_color", np.float32, (4,)),
        ("in_bary",         np.float32, (3,)),
        ("stroke_half_px",  np.float32),
    ]
)
_SURFACE_COMBINED_STRIDE: int = _SURFACE_COMBINED_DTYPE.itemsize  # 72 bytes

_SURFACE_COMBINED_OFFSETS: dict[str, int] = {
    name: _SURFACE_COMBINED_DTYPE.fields[name][1]  # type: ignore[index]
    for name in _SURFACE_COMBINED_DTYPE.names
}

SURFACE_COMBINED_VERTEX_LAYOUT: dict = {
    "array_stride": _SURFACE_COMBINED_STRIDE,
    "step_mode": "vertex",
    "attributes": [
        {"format": "float32x3", "offset": _SURFACE_COMBINED_OFFSETS["in_vert"],         "shader_location": 0},
        {"format": "float32x3", "offset": _SURFACE_COMBINED_OFFSETS["in_normal"],       "shader_location": 1},
        {"format": "float32x4", "offset": _SURFACE_COMBINED_OFFSETS["in_fill_color"],   "shader_location": 2},
        {"format": "float32x4", "offset": _SURFACE_COMBINED_OFFSETS["in_stroke_color"], "shader_location": 3},
        {"format": "float32x3", "offset": _SURFACE_COMBINED_OFFSETS["in_bary"],         "shader_location": 4},
        {"format": "float32",   "offset": _SURFACE_COMBINED_OFFSETS["stroke_half_px"],  "shader_location": 5},
    ],
}


# ---------------------------------------------------------------------------
# Combined fill+stroke vertex layout — must match vmobject_fill_stroke.wgsl.
#
#   location 0 — in_pos             float32x3  offset  0   (12 B)
#   location 1 — in_fill_color      float32x4  offset 12   (16 B)
#   location 2 — in_stroke_color    float32x4  offset 28   (16 B)
#   location 3 — stroke_half_ndc    float32    offset 44   ( 4 B)
#   location 4 — fill_curve_start   uint32     offset 48   ( 4 B)
#   location 5 — n_fill_curves      uint32     offset 52   ( 4 B)
#   location 6 — stroke_curve_start uint32     offset 56   ( 4 B)
#   location 7 — n_stroke_curves    uint32     offset 60   ( 4 B)
#   stride: 64 bytes
# ---------------------------------------------------------------------------

_FILL_STROKE_DTYPE = np.dtype(
    [
        ("in_pos",             np.float32, (3,)),
        ("in_fill_color",      np.float32, (4,)),
        ("in_stroke_color",    np.float32, (4,)),
        ("stroke_half_ndc",    np.float32),
        ("fill_curve_start",   np.uint32),
        ("n_fill_curves",      np.uint32),
        ("stroke_curve_start", np.uint32),
        ("n_stroke_curves",    np.uint32),
    ]
)
_FILL_STROKE_STRIDE: int = _FILL_STROKE_DTYPE.itemsize  # 64 bytes

_FILL_STROKE_OFFSETS: dict[str, int] = {
    name: _FILL_STROKE_DTYPE.fields[name][1]  # type: ignore[index]
    for name in _FILL_STROKE_DTYPE.names
}

FILL_STROKE_VERTEX_LAYOUT: dict = {
    "array_stride": _FILL_STROKE_STRIDE,
    "step_mode": "vertex",
    "attributes": [
        {"format": "float32x3", "offset": _FILL_STROKE_OFFSETS["in_pos"],             "shader_location": 0},
        {"format": "float32x4", "offset": _FILL_STROKE_OFFSETS["in_fill_color"],      "shader_location": 1},
        {"format": "float32x4", "offset": _FILL_STROKE_OFFSETS["in_stroke_color"],    "shader_location": 2},
        {"format": "float32",   "offset": _FILL_STROKE_OFFSETS["stroke_half_ndc"],    "shader_location": 3},
        {"format": "uint32",    "offset": _FILL_STROKE_OFFSETS["fill_curve_start"],   "shader_location": 4},
        {"format": "uint32",    "offset": _FILL_STROKE_OFFSETS["n_fill_curves"],      "shader_location": 5},
        {"format": "uint32",    "offset": _FILL_STROKE_OFFSETS["stroke_curve_start"], "shader_location": 6},
        {"format": "uint32",    "offset": _FILL_STROKE_OFFSETS["n_stroke_curves"],    "shader_location": 7},
    ],
}


# ---------------------------------------------------------------------------
# Per-frame data container
# ---------------------------------------------------------------------------


@dataclass
class _FrameData:
    """All GPU-ready data for one group of mobjects (one camera bind group).

    Produced by ``collect_frame_data``; consumed by ``draw_frame_data`` and
    the caller's OIT / fixed-frame passes.
    """

    # VMobject fill+stroke via combined pipeline
    fs_parts: list[np.ndarray]       # _FILL_STROKE_DTYPE arrays, one per draw call
    fs_buf: wgpu_t.GPUBuffer | None  # concatenated vertex buffer
    fs_byte_offsets: list[int]       # byte offset of each part in fs_buf

    # GPU compute: cubic → quadratic conversion
    cubics_buf: wgpu_t.GPUBuffer | None    # input (12 floats/cubic), all objects
    quads_out_buf: wgpu_t.GPUBuffer | None # output (36 floats/cubic = 4 quads × 9)
    n_cubics_total: int
    compute_bg: wgpu_t.GPUBindGroup | None  # compute pass bind group
    render_bg: wgpu_t.GPUBindGroup | None   # fragment bind group (camera + quads)

    # Parametric surfaces (combined fill + barycentric wireframe pipeline)
    surface_parts: list[np.ndarray]
    surface_buf: wgpu_t.GPUBuffer | None
    surface_byte_offsets: list[int]

    # Ordered draw commands:
    #   "fill_stroke_2d"  — 2-D VMobject (no depth write)
    #   "fill_stroke_3d"  — shade_in_3d VMobject (depth write + test)
    #   "surface_opaque"  — opaque parametric surface
    #   "surface_oit"     — transparent parametric surface (OIT pass, caller handles)
    draw_plan: list[tuple[str, int]]

    # Indices into surface_parts that need OIT (handled by the caller).
    oit_indices: list[int]


# ---------------------------------------------------------------------------
# Geometry caches
# ---------------------------------------------------------------------------

# fill_stroke_cache: vmobject → (points_hash, (fill_cubics, stroke_cubics))
#   fill_cubics  : (N, 4, 3) float32 — includes closing segments for winding
#   stroke_cubics: (M, 4, 3) float32 — no closing segments (visible curve only)
# Geometry only; colors/widths are fetched fresh every frame.
_fill_stroke_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _points_hash(vmobject: VMobject) -> int:
    pts = vmobject.points
    if pts.size == 0:
        return 0
    return hash(pts.tobytes())


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def collect_frame_data(
    renderer: WebGPURenderer,
    mobjects: list,
    camera_uniform_buf: wgpu_t.GPUBuffer,
) -> _FrameData | None:
    """Tessellate *mobjects*, upload to GPU, return a ``_FrameData``.

    Does NOT record any GPU commands — only uploads buffers and creates bind
    groups.  The caller must run the compute pass (via ``_FrameData.compute_bg``)
    before the render pass.

    *camera_uniform_buf* is the 176-byte uniform buffer for this camera group.
    It is stored in the render bind group so the fragment shader can project
    world-space curve data into the correct NDC space.
    """
    import wgpu

    view_matrix: np.ndarray = renderer.camera.view_matrix
    proj_matrix: np.ndarray = renderer.camera.projection_matrix

    # Per-draw-call data collected across all mobjects.
    fs_parts: list[np.ndarray] = []
    # Cubics: fill first (all objects), then stroke (all objects).
    all_fill_cubics:   list[np.ndarray] = []  # (Ni, 4, 3) per draw call
    all_stroke_cubics: list[np.ndarray] = []  # (Mi, 4, 3) per draw call
    n_fill_cubics_per:   list[int] = []       # Ni per draw call
    n_stroke_cubics_per: list[int] = []       # Mi per draw call

    surface_parts: list[np.ndarray] = []
    draw_plan: list[tuple[str, int]] = []

    for mob in mobjects:
        if not isinstance(mob, VMobject):
            continue

        # ── Parametric Surface ────────────────────────────────────────────
        if isinstance(mob, Surface):
            for submob in mob.family_members_with_points():
                data = _collect_surface_geometry(
                    submob, view_matrix, proj_matrix
                )
                if data is not None:
                    cls = _surface_opacity_class(data)
                    cmd = "surface_opaque" if cls == "opaque" else "surface_oit"
                    draw_plan.append((cmd, len(surface_parts)))
                    surface_parts.append(data)
            continue

        # ── Regular VMobject (2-D or shade_in_3d) ────────────────────────
        for submob in mob.family_members_with_points():
            phash  = _points_hash(submob)
            cached = _fill_stroke_cache.get(submob)
            if cached is None or cached[0] != phash:
                result = _collect_cubics(submob)
                if result is not None:
                    _fill_stroke_cache[submob] = (phash, result)
                else:
                    _fill_stroke_cache.pop(submob, None)
                    continue
            cached = _fill_stroke_cache.get(submob)
            if cached is None:
                continue
            fill_cubics, stroke_cubics = cached[1]

            # Fetch current colors every frame (they change during animations).
            fill_rgba   = submob.get_fill_rgbas()
            stroke_rgba = submob.get_stroke_rgbas()
            fill_color   = (fill_rgba[0].astype(np.float32)
                            if fill_rgba.shape[0] > 0
                            else np.zeros(4, dtype=np.float32))
            stroke_color = (stroke_rgba[0].astype(np.float32)
                            if stroke_rgba.shape[0] > 0
                            else np.zeros(4, dtype=np.float32))
            stroke_width = (float(submob.get_stroke_width())
                            if stroke_rgba.shape[0] > 0
                            else 0.0)

            # Skip entirely invisible objects (both fill and stroke transparent).
            if fill_color[3] < 0.001 and (stroke_color[3] < 0.001 or stroke_width < 0.001):
                continue

            # Build bounding quad with placeholder curve indices.
            quad_verts = _build_fill_stroke_quad(
                fill_cubics=fill_cubics,
                stroke_cubics=stroke_cubics,
                fill_color=fill_color,
                stroke_color=stroke_color,
                stroke_width=stroke_width,
                fill_curve_start=0,    # assigned below after all objects are collected
                stroke_curve_start=0,  # assigned below
                view_matrix=view_matrix,
                proj_matrix=proj_matrix,
            )
            if len(quad_verts) == 0:
                continue

            is_3d = getattr(submob, "shade_in_3d", False)
            draw_plan.append(("fill_stroke_3d" if is_3d else "fill_stroke_2d",
                              len(fs_parts)))
            fs_parts.append(quad_verts)
            all_fill_cubics.append(fill_cubics)
            all_stroke_cubics.append(stroke_cubics)
            n_fill_cubics_per.append(len(fill_cubics))
            n_stroke_cubics_per.append(len(stroke_cubics))

    if not draw_plan:
        return None

    device: wgpu_t.GPUDevice = renderer.device

    # ── Assign global curve start indices ────────────────────────────────
    # Cubics buffer layout: [fill_cubics_obj0, fill_cubics_obj1, ...,
    #                        stroke_cubics_obj0, stroke_cubics_obj1, ...]
    # Quads output layout:  [fill_quads_obj0, fill_quads_obj1, ...,
    #                        stroke_quads_obj0, stroke_quads_obj1, ...]
    total_fill_cubics   = sum(n_fill_cubics_per)
    total_stroke_cubics = sum(n_stroke_cubics_per)
    n_cubics_total      = total_fill_cubics + total_stroke_cubics

    fill_global   = 0               # running fill cubic index
    stroke_global = total_fill_cubics  # stroke cubics follow all fill cubics

    for i, part in enumerate(fs_parts):
        part["fill_curve_start"]   = fill_global * 4
        part["n_fill_curves"]      = n_fill_cubics_per[i] * 4
        part["stroke_curve_start"] = stroke_global * 4
        part["n_stroke_curves"]    = n_stroke_cubics_per[i] * 4
        fill_global   += n_fill_cubics_per[i]
        stroke_global += n_stroke_cubics_per[i]

    # ── Upload vertex data ───────────────────────────────────────────────
    fs_buf, fs_byte_offsets = None, []
    if fs_parts:
        fs_buf, fs_byte_offsets = _batch_upload(device, fs_parts)
        renderer.frame_vbos.append(fs_buf)

    # ── Upload cubics and create compute/render bind groups ──────────────
    cubics_buf = quads_out_buf = compute_bg = render_bg = None

    if n_cubics_total > 0:
        # Build flat float32 array: [all fill cubics..., all stroke cubics...]
        fill_arrays   = [c for c in all_fill_cubics   if len(c) > 0]
        stroke_arrays = [c for c in all_stroke_cubics if len(c) > 0]
        all_arrays    = fill_arrays + stroke_arrays
        all_cubics    = np.concatenate(all_arrays, axis=0)  # (N, 4, 3)
        cubics_flat   = all_cubics.astype(np.float32).ravel()  # N*12 floats

        cubics_buf = device.create_buffer_with_data(
            data=cubics_flat.tobytes(),
            usage=wgpu.BufferUsage.STORAGE,
        )
        renderer.frame_vbos.append(cubics_buf)

        quads_size    = n_cubics_total * 36 * 4  # 4 quads × 9 floats × 4 bytes
        quads_out_buf = device.create_buffer(
            size=max(quads_size, 16),  # WebGPU minimum binding size
            usage=wgpu.BufferUsage.STORAGE,
        )
        renderer.frame_vbos.append(quads_out_buf)

        # Params uniform (n_cubics, padded to 16 bytes for WebGPU alignment).
        params_bytes = struct.pack("<4I", n_cubics_total, 0, 0, 0)
        params_buf   = device.create_buffer_with_data(
            data=params_bytes,
            usage=wgpu.BufferUsage.UNIFORM,
        )
        renderer.frame_vbos.append(params_buf)

        compute_bg = device.create_bind_group(
            layout=renderer._compute_bgl,
            entries=[
                {"binding": 0, "resource": {"buffer": cubics_buf,    "offset": 0, "size": cubics_buf.size}},
                {"binding": 1, "resource": {"buffer": quads_out_buf, "offset": 0, "size": quads_out_buf.size}},
                {"binding": 2, "resource": {"buffer": params_buf,    "offset": 0, "size": 16}},
            ],
        )

        render_bg = device.create_bind_group(
            layout=renderer._fill_stroke_bgl,
            entries=[
                {"binding": 0, "resource": {"buffer": camera_uniform_buf, "offset": 0, "size": 176}},
                {"binding": 1, "resource": {"buffer": quads_out_buf,      "offset": 0, "size": quads_out_buf.size}},
            ],
        )

    # ── Upload surface data ──────────────────────────────────────────────
    surface_buf, surface_byte_offsets = None, []
    if surface_parts:
        _smooth_surface_normals(surface_parts)
        surface_buf, surface_byte_offsets = _batch_upload(device, surface_parts)
        renderer.frame_vbos.append(surface_buf)

    oit_indices = [idx for cmd, idx in draw_plan if cmd == "surface_oit"]

    return _FrameData(
        fs_parts=fs_parts,
        fs_buf=fs_buf,
        fs_byte_offsets=fs_byte_offsets,
        cubics_buf=cubics_buf,
        quads_out_buf=quads_out_buf,
        n_cubics_total=n_cubics_total,
        compute_bg=compute_bg,
        render_bg=render_bg,
        surface_parts=surface_parts,
        surface_buf=surface_buf,
        surface_byte_offsets=surface_byte_offsets,
        draw_plan=draw_plan,
        oit_indices=oit_indices,
    )


def draw_frame_data(
    renderer: WebGPURenderer,
    fd: _FrameData,
    cam_bg: wgpu_t.GPUBindGroup,
) -> None:
    """Record draw commands for *fd* into ``renderer.current_render_pass``.

    Draw order
    ----------
    1. 2-D fill+stroke objects — interleaved in ``draw_plan`` order (painter's
       algorithm; no depth write so objects paint over each other correctly).
    2. 3-D fill+stroke objects — depth write + test (shade_in_3d).
    3. Opaque parametric surfaces — depth write (includes barycentric wireframe).

    OIT surfaces are NOT drawn here; the caller reads ``fd.oit_indices`` and
    handles them in a separate accumulation pass.
    """
    rp = renderer.current_render_pass

    cur_pipeline: list[str | None]      = [None]
    cur_bg:       list[object | None]   = [None]

    def _activate(name: str, bg: wgpu_t.GPUBindGroup) -> None:
        if cur_pipeline[0] != name:
            if name == "fill_stroke_2d":
                rp.set_pipeline(renderer.fill_stroke_pipeline)
            elif name == "fill_stroke_3d":
                rp.set_pipeline(renderer.fill_stroke_3d_pipeline)
            elif name == "surface_opaque":
                rp.set_pipeline(renderer.surface_pipeline)
            cur_pipeline[0] = name
        if cur_bg[0] is not bg:
            rp.set_bind_group(0, bg, [], 0, 0)
            cur_bg[0] = bg

    # 1. 2-D fill+stroke: interleaved in draw_plan order (painter's algorithm).
    if fd.fs_buf is not None and fd.render_bg is not None:
        for cmd, idx in fd.draw_plan:
            if cmd != "fill_stroke_2d":
                continue
            _activate("fill_stroke_2d", fd.render_bg)
            arr = fd.fs_parts[idx]
            rp.set_vertex_buffer(0, fd.fs_buf, fd.fs_byte_offsets[idx], arr.nbytes)
            rp.draw(len(arr), 1, 0, 0)

    # 2. 3-D fill+stroke: depth-tested and depth-written.
    if fd.fs_buf is not None and fd.render_bg is not None:
        for cmd, idx in fd.draw_plan:
            if cmd != "fill_stroke_3d":
                continue
            _activate("fill_stroke_3d", fd.render_bg)
            arr = fd.fs_parts[idx]
            rp.set_vertex_buffer(0, fd.fs_buf, fd.fs_byte_offsets[idx], arr.nbytes)
            rp.draw(len(arr), 1, 0, 0)

    # 3. Opaque parametric surfaces (combined fill + barycentric wireframe).
    if fd.surface_buf is not None:
        for cmd, idx in fd.draw_plan:
            if cmd != "surface_opaque":
                continue
            _activate("surface_opaque", cam_bg)
            arr = fd.surface_parts[idx]
            rp.set_vertex_buffer(0, fd.surface_buf, fd.surface_byte_offsets[idx], arr.nbytes)
            rp.draw(len(arr), 1, 0, 0)


# ---------------------------------------------------------------------------
# GPU upload helpers
# ---------------------------------------------------------------------------


def _batch_upload(
    device: wgpu_t.GPUDevice,
    arrays: list[np.ndarray],
) -> tuple[wgpu_t.GPUBuffer, list[int]]:
    """Concatenate *arrays* into one bytes blob and upload as a VERTEX buffer."""
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
# VMobject cubic collector — geometry cache
# ---------------------------------------------------------------------------


def _collect_cubics(
    vmobject: VMobject,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(fill_cubics, stroke_cubics)`` for the GPU compute shader.

    *fill_cubics*  — ``(N, 4, 3)`` float32.  All subpath cubics **plus** one
        linear closing cubic per open subpath (required for correct winding-
        number coverage during partial animations such as ``Create``).

    *stroke_cubics* — ``(M, 4, 3)`` float32.  Only the actual subpath cubics,
        no closing segment — the stroke should follow the visible curve only.

    Colors are NOT stored here; they are fetched fresh every frame in
    ``collect_frame_data`` so that opacity animations work correctly.

    Returns ``None`` if the vmobject has no usable bezier curves.
    """
    nppcc = vmobject.n_points_per_cubic_curve

    fill_cubics_list:   list[np.ndarray] = []
    stroke_cubics_list: list[np.ndarray] = []

    for subpath in vmobject.get_subpaths():
        n_curves = len(subpath) // nppcc
        if n_curves == 0:
            continue
        pts = subpath[: n_curves * nppcc]
        b0s = pts[0::nppcc].astype(np.float32)
        h0s = pts[1::nppcc].astype(np.float32)
        h1s = pts[2::nppcc].astype(np.float32)
        b3s = pts[3::nppcc].astype(np.float32)

        cubics = np.stack([b0s, h0s, h1s, b3s], axis=1)  # (n, 4, 3)
        stroke_cubics_list.append(cubics)
        fill_cubics_list.append(cubics)

        # Closing segment: linear cubic from the last anchor back to the first.
        # Degree-elevation from a line (last→first) to a cubic:
        #   b0 = last, b1 = last + (first-last)/3,
        #   b2 = last + 2*(first-last)/3, b3 = first.
        first = b0s[0]
        last  = b3s[-1]
        if not np.allclose(first, last, atol=1e-6):
            diff = first - last
            closing = np.array(
                [[last, last + diff * (1.0 / 3.0), last + diff * (2.0 / 3.0), first]],
                dtype=np.float32,
            )
            fill_cubics_list.append(closing)

    if not fill_cubics_list and not stroke_cubics_list:
        return None

    fill_cubics = (np.concatenate(fill_cubics_list,   axis=0)
                   if fill_cubics_list
                   else np.empty((0, 4, 3), dtype=np.float32))
    stroke_cubics = (np.concatenate(stroke_cubics_list, axis=0)
                     if stroke_cubics_list
                     else np.empty((0, 4, 3), dtype=np.float32))
    return fill_cubics, stroke_cubics


# ---------------------------------------------------------------------------
# Bounding-quad builder
# ---------------------------------------------------------------------------


def _build_fill_stroke_quad(
    fill_cubics: np.ndarray,
    stroke_cubics: np.ndarray,
    fill_color: np.ndarray,
    stroke_color: np.ndarray,
    stroke_width: float,
    fill_curve_start: int,
    stroke_curve_start: int,
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
) -> np.ndarray:
    """Build a ``_FILL_STROKE_DTYPE`` bounding quad (6 vertices) for one object.

    The bounding box is computed in NDC space (clip.xy / clip.w) from the
    anchor points of both fill and stroke cubics, then mapped back to world
    space at the average view-space Z.  This is correct for both orthographic
    (w = 1) and perspective projections.

    *stroke_half_ndc* is the stroke half-width in NDC units, computed from
    the current projection matrix and average clip-w so that stroke width is
    consistent across perspective depths.
    """
    # Gather all anchor points (b0 and b3 of every cubic).
    anchor_lists: list[np.ndarray] = []
    if len(fill_cubics) > 0:
        anchor_lists.append(fill_cubics[:, 0])
        anchor_lists.append(fill_cubics[:, 3])
    if len(stroke_cubics) > 0:
        anchor_lists.append(stroke_cubics[:, 0])
        anchor_lists.append(stroke_cubics[:, 3])

    if not anchor_lists:
        return np.empty(0, dtype=_FILL_STROKE_DTYPE)

    anchors = np.concatenate(anchor_lists, axis=0).astype(np.float32)  # (N, 3)

    vm = view_matrix.astype(np.float32)
    pm = proj_matrix.astype(np.float32)
    R, t = vm[:3, :3], vm[:3, 3]

    pts_v   = (R @ anchors.T).T + t    # (N, 3) view space
    avg_z_v = float(pts_v[:, 2].mean())

    # Perspective divide → NDC.
    ones  = np.ones((len(pts_v), 1), dtype=np.float32)
    clips = (pm @ np.hstack([pts_v, ones]).T).T          # (N, 4)
    w     = clips[:, 3:4]
    w_s   = np.where(np.abs(w) > 1e-8, w, np.sign(w + 1e-38) * 1e-8)
    ndcs  = clips[:, :2] / w_s                           # (N, 2) NDC

    PAD     = 0.05
    ndc_min = ndcs.min(axis=0) - PAD
    ndc_max = ndcs.max(axis=0) + PAD

    # Stroke half-width in NDC.
    # v_thickness = 0.004 * stroke_width  (view-space, matching vmobject_stroke.wgsl)
    # stroke_half_ndc = v_thickness * pm[0,0] / avg_clip_w
    #   where avg_clip_w = pm[3,2]*avg_z + pm[3,3]
    avg_clip_w = float(pm[3, 2] * avg_z_v + pm[3, 3])
    avg_clip_w = avg_clip_w if abs(avg_clip_w) > 1e-8 else 1.0
    stroke_half_ndc = 0.0
    if stroke_width > 0.0 and float(stroke_color[3]) > 0.001:
        stroke_half_ndc = float(0.004 * stroke_width * abs(pm[0, 0]) / abs(avg_clip_w))
        # Add stroke padding so the bounding quad covers the stroke edges.
        ndc_min -= stroke_half_ndc * 2.0
        ndc_max += stroke_half_ndc * 2.0

    # Invert NDC bounding corners to view space.
    inv_px = 1.0 / (pm[0, 0] if abs(pm[0, 0]) > 1e-8 else 1.0)
    inv_py = 1.0 / (pm[1, 1] if abs(pm[1, 1]) > 1e-8 else 1.0)
    x0_v = (float(ndc_min[0]) * avg_clip_w - float(pm[0, 3])) * inv_px
    x1_v = (float(ndc_max[0]) * avg_clip_w - float(pm[0, 3])) * inv_px
    y0_v = (float(ndc_min[1]) * avg_clip_w - float(pm[1, 3])) * inv_py
    y1_v = (float(ndc_max[1]) * avg_clip_w - float(pm[1, 3])) * inv_py

    corners_v = np.array(
        [[x0_v, y0_v, avg_z_v], [x1_v, y0_v, avg_z_v],
         [x0_v, y1_v, avg_z_v], [x1_v, y1_v, avg_z_v]],
        dtype=np.float32,
    )
    R_inv    = R.T
    t_inv    = -(R_inv @ t)
    corners_w = (R_inv @ corners_v.T).T + t_inv  # (4, 3) world space
    quad_pos  = corners_w[[0, 1, 2, 1, 3, 2]]    # (6, 3) two CCW triangles

    n_fill_quads   = len(fill_cubics)   * 4  # 4 quadratics per cubic
    n_stroke_quads = len(stroke_cubics) * 4

    verts = np.empty(6, dtype=_FILL_STROKE_DTYPE)
    verts["in_pos"]             = quad_pos
    verts["in_fill_color"]      = fill_color
    verts["in_stroke_color"]    = stroke_color
    verts["stroke_half_ndc"]    = stroke_half_ndc
    verts["fill_curve_start"]   = fill_curve_start
    verts["n_fill_curves"]      = n_fill_quads
    verts["stroke_curve_start"] = stroke_curve_start
    verts["n_stroke_curves"]    = n_stroke_quads
    return verts


# ---------------------------------------------------------------------------
# Surface geometry collectors (unchanged from original)
# ---------------------------------------------------------------------------


def _surface_opacity_class(part: np.ndarray) -> str:
    alphas = part["in_fill_color"][:, 3]
    return "opaque" if float(alphas.min()) >= 0.99 else "oit"


def _collect_surface_geometry(
    vmobject: VMobject,
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
) -> np.ndarray | None:
    """Return a ``_SURFACE_COMBINED_DTYPE`` array for a shade_in_3d VMobject.

    Barycentric coordinates are assigned per triangle in the centroid fan:
      centroid     → bary = (1, 0, 0)   (bary.x = 0 on outer edge)
      anchor_i     → bary = (0, 1, 0)
      anchor_{i+1} → bary = (0, 0, 1)

    ``stroke_half_px`` is computed from the stroke width, projection matrix
    and average clip-w of the surface anchors so that wireframe line width
    is consistent across perspective depths.
    """
    from manim import config

    fill_rgba = vmobject.get_fill_rgbas()
    if fill_rgba.shape[0] == 0 or fill_rgba[0, 3] == 0:
        return None

    fill_color   = fill_rgba[0].astype(np.float32)
    stroke_rgba  = vmobject.get_stroke_rgbas()
    stroke_color = (stroke_rgba[0].astype(np.float32)
                    if stroke_rgba.shape[0] > 0
                    else np.zeros(4, dtype=np.float32))
    stroke_width = float(vmobject.get_stroke_width()) if stroke_rgba.shape[0] > 0 else 0.0

    nppcc = vmobject.n_points_per_cubic_curve

    all_verts:   list[np.ndarray] = []
    all_normals: list[np.ndarray] = []
    all_bary:    list[np.ndarray] = []

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

        centroid = anchors.mean(axis=0)
        v0 = anchors[0] - centroid
        v1 = anchors[1] - centroid
        raw_normal = np.cross(v1, v0).astype(np.float64)
        norm_len   = np.linalg.norm(raw_normal)
        normal     = (
            (raw_normal / norm_len).astype(np.float32)
            if norm_len > 1e-9
            else np.array([0.0, 0.0, 1.0], dtype=np.float32)
        )

        # Triangle fan: (centroid, anchor_i, anchor_{i+1})
        fan_verts = np.empty((n_pts * 3, 3), dtype=np.float32)
        fan_verts[0::3] = centroid.astype(np.float32)
        fan_verts[1::3] = anchors.astype(np.float32)
        fan_verts[2::3] = np.roll(anchors, -1, axis=0).astype(np.float32)

        # Barycentric coords: centroid=(1,0,0), anchor_i=(0,1,0), next=(0,0,1)
        bary_block = np.zeros((n_pts * 3, 3), dtype=np.float32)
        bary_block[0::3] = [1.0, 0.0, 0.0]
        bary_block[1::3] = [0.0, 1.0, 0.0]
        bary_block[2::3] = [0.0, 0.0, 1.0]

        all_verts.append(fan_verts)
        all_normals.append(np.tile(normal, (n_pts * 3, 1)))
        all_bary.append(bary_block)

    if not all_verts:
        return None

    verts   = np.concatenate(all_verts,   axis=0)
    normals = np.concatenate(all_normals, axis=0)
    bary    = np.concatenate(all_bary,    axis=0)
    n_total = len(verts)

    # Compute stroke_half_px: half the wireframe line width in screen pixels.
    # Formula matches _build_fill_stroke_quad: 0.004 * width * |pm[0,0]| / |avg_clip_w|
    # then multiplied by pixel_width/2 to convert NDC to pixels.
    stroke_half_px = 0.0
    if stroke_width > 0.0 and float(stroke_color[3]) > 0.001:
        pm = proj_matrix.astype(np.float32)
        vm = view_matrix.astype(np.float32)
        R, t = vm[:3, :3], vm[:3, 3]
        pts_v = (R @ verts.T).T + t         # (N, 3) view space
        avg_z_v = float(pts_v[:, 2].mean())
        avg_clip_w = float(pm[3, 2] * avg_z_v + pm[3, 3])
        avg_clip_w = avg_clip_w if abs(avg_clip_w) > 1e-8 else 1.0
        stroke_half_ndc = float(0.004 * stroke_width * abs(pm[0, 0]) / abs(avg_clip_w))
        stroke_half_px  = stroke_half_ndc * config.pixel_width * 0.5

    attrs = np.empty(n_total, dtype=_SURFACE_COMBINED_DTYPE)
    attrs["in_vert"]         = verts
    attrs["in_normal"]       = normals
    attrs["in_fill_color"]   = fill_color
    attrs["in_stroke_color"] = stroke_color
    attrs["in_bary"]         = bary
    attrs["stroke_half_px"]  = stroke_half_px
    return attrs


def _smooth_surface_normals(surface_parts: list[np.ndarray]) -> None:
    """Average normals at shared vertex positions (modifies in-place)."""
    if not surface_parts:
        return

    all_verts = np.concatenate([p["in_vert"]    for p in surface_parts], axis=0)
    all_norms = np.concatenate([p["in_normal"]  for p in surface_parts], axis=0)

    PREC = 1e-5
    quantized = np.round(all_verts.astype(np.float64) / PREC).astype(np.int64)
    _, inverse = np.unique(quantized, axis=0, return_inverse=True)

    n_unique = int(inverse.max()) + 1
    smooth   = np.zeros((n_unique, 3), dtype=np.float64)
    np.add.at(smooth, inverse, all_norms.astype(np.float64))

    lengths = np.linalg.norm(smooth, axis=1, keepdims=True)
    lengths = np.where(lengths < 1e-9, 1.0, lengths)
    smooth  = (smooth / lengths).astype(np.float32)

    idx = 0
    for part in surface_parts:
        n = len(part)
        part["in_normal"] = smooth[inverse[idx : idx + n]]
        idx += n


