"""WebGPU renderer for Manim — Phase 1.

Phase 1 scope
-------------
* Headless rendering (no preview window).
* VMobject fill only.
* ``config.save_last_frame = True`` → saves a PNG.
* ``config.write_to_movie = True`` → writes video frames.

Design
------
Reads geometry directly from Cairo ``VMobject``.  No dependency on OpenGL
classes (``OpenGLCamera``, ``OpenGLVMobject``, ``moderngl``).

Camera
------
A simple orthographic projection matrix maps Manim's frame coordinate system
(centre at origin, width = config.frame_width, height = config.frame_height)
to WebGPU NDC (x, y ∈ [-1, 1], z ∈ [0, 1]).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from manim import config, logger
from manim.constants import OUT, PI, RIGHT
from manim.mobject.types.vectorized_mobject import VMobject
from manim.scene.scene_file_writer import SceneFileWriter
from manim.utils.color import color_to_rgba
from manim.utils.exceptions import EndSceneEarlyException
from manim.utils.simple_functions import clip
from manim.utils.space_ops import (
    quaternion_from_angle_axis,
    quaternion_mult,
    rotation_matrix_transpose_from_quaternion,
)

from .webgpu_vmobject_rendering import (
    SLUG_FILL_VERTEX_LAYOUT,
    STROKE_VERTEX_LAYOUT,
    SURFACE_VERTEX_LAYOUT,
    render_webgpu_mobject,
)

if TYPE_CHECKING:
    import wgpu as wgpu_t

    from manim.scene.scene import Scene

try:
    import wgpu
except ImportError as exc:
    msg = (
        "wgpu-py is required for the WebGPU renderer. "
        "Install it with:  pip install wgpu"
    )
    raise ImportError(msg) from exc


# ---------------------------------------------------------------------------
# Camera — feature-parity with OpenGLCamera for 2-D + 3-D scenes.
# ---------------------------------------------------------------------------


class WebGPUCamera:
    """Camera for the WebGPU renderer.

    Matches the attribute / method surface of ``OpenGLCamera`` so that
    scene code that inspects ``renderer.camera`` works without changes.

    Projection
    ----------
    * 2-D scenes (default): orthographic, z mapped to the WebGPU [0, 1] NDC
      range.
    * 3-D scenes (Phase 3): perspective projection driven by ``focal_distance``
      and the Euler-angle view matrix.

    Parameters
    ----------
    frame_shape
        (width, height) of the rendered frame.  Defaults to
        ``(config.frame_width, config.frame_height)``.
    center_point
        World-space origin of the camera frame.  Defaults to the origin.
    euler_angles
        (theta, phi, gamma) camera orientation angles in radians.
        Defaults to (0, 0, 0) — looking straight down the −Z axis.
    focal_distance
        Perspective focal distance expressed as a multiple of ``frame_height``.
        Only used when ``orthographic=False``.
    light_source_position
        World-space position of the key light.  Defaults to (−10, 10, 10).
    orthographic
        Use orthographic (True) or perspective (False) projection.
        Default is True (matching Manim's default 2-D look).
    minimum_polar_angle / maximum_polar_angle
        Clamp range for the phi Euler angle during interactive orbit.
    """

    near: float = -100.0
    far: float = 100.0
    use_z_index: bool = True

    def __init__(
        self,
        frame_shape: tuple[float, float] | None = None,
        center_point: np.ndarray | None = None,
        euler_angles: np.ndarray | None = None,
        focal_distance: float = 2.0,
        light_source_position: np.ndarray | None = None,
        orthographic: bool = True,
        minimum_polar_angle: float = -PI / 2,
        maximum_polar_angle: float = PI / 2,
    ) -> None:
        self.use_z_index = True
        self.frame_rate: int = config.get("frame_rate", 60)
        self.orthographic = orthographic
        self.minimum_polar_angle = minimum_polar_angle
        self.maximum_polar_angle = maximum_polar_angle
        self.focal_distance = focal_distance

        self.frame_shape: tuple[float, float] = (
            frame_shape
            if frame_shape is not None
            else (float(config["frame_width"]), float(config["frame_height"]))
        )
        self.center_point: np.ndarray = (
            np.asarray(center_point, dtype=float)
            if center_point is not None
            else np.zeros(3)
        )
        self.light_source_position: np.ndarray = np.asarray(
            light_source_position if light_source_position is not None else [-10, 10, 10],
            dtype=float,
        )
        self.euler_angles: np.ndarray = np.asarray(
            euler_angles if euler_angles is not None else [0.0, 0.0, 0.0],
            dtype=float,
        )
        self.refresh_rotation_matrix()

    # ------------------------------------------------------------------
    # Frame geometry helpers (mirrors OpenGLCamera)
    # ------------------------------------------------------------------

    def get_width(self) -> float:
        """Width of the camera frame in scene units."""
        return self.frame_shape[0]

    def get_height(self) -> float:
        """Height of the camera frame in scene units."""
        return self.frame_shape[1]

    def get_shape(self) -> tuple[float, float]:
        """(width, height) of the camera frame in scene units."""
        return self.frame_shape

    def get_center(self) -> np.ndarray:
        """World-space centre of the camera frame."""
        return self.center_point.copy()

    def get_focal_distance(self) -> float:
        """Perspective focal distance in scene units."""
        return self.focal_distance * self.get_height()

    # ------------------------------------------------------------------
    # Camera reset
    # ------------------------------------------------------------------

    def to_default_state(self) -> WebGPUCamera:
        """Reset frame size, position, and orientation to config defaults."""
        self.frame_shape = (
            float(config["frame_width"]),
            float(config["frame_height"]),
        )
        self.center_point = np.zeros(3)
        self.euler_angles = np.zeros(3)
        self.refresh_rotation_matrix()
        return self

    # ------------------------------------------------------------------
    # Rotation — matches OpenGLCamera.set/increment_* interface
    # ------------------------------------------------------------------

    def refresh_rotation_matrix(self) -> None:
        """Recompute ``inverse_rotation_matrix`` from current Euler angles."""
        theta, phi, gamma = self.euler_angles
        quat = quaternion_mult(
            quaternion_from_angle_axis(theta, OUT, axis_normalized=True),
            quaternion_from_angle_axis(phi, RIGHT, axis_normalized=True),
            quaternion_from_angle_axis(gamma, OUT, axis_normalized=True),
        )
        self.inverse_rotation_matrix: np.ndarray = np.array(
            rotation_matrix_transpose_from_quaternion(np.asarray(quat, dtype=float)),
            dtype=float,
        )

    def set_euler_angles(
        self,
        theta: float | None = None,
        phi: float | None = None,
        gamma: float | None = None,
    ) -> WebGPUCamera:
        if theta is not None:
            self.euler_angles[0] = theta
        if phi is not None:
            self.euler_angles[1] = phi
        if gamma is not None:
            self.euler_angles[2] = gamma
        self.refresh_rotation_matrix()
        return self

    def set_theta(self, theta: float) -> WebGPUCamera:
        return self.set_euler_angles(theta=theta)

    def set_phi(self, phi: float) -> WebGPUCamera:
        return self.set_euler_angles(phi=phi)

    def set_gamma(self, gamma: float) -> WebGPUCamera:
        return self.set_euler_angles(gamma=gamma)

    def increment_theta(self, dtheta: float) -> WebGPUCamera:
        self.euler_angles[0] += dtheta
        self.refresh_rotation_matrix()
        return self

    def increment_phi(self, dphi: float) -> WebGPUCamera:
        self.euler_angles[1] = clip(
            self.euler_angles[1] + dphi,
            self.minimum_polar_angle,
            self.maximum_polar_angle,
        )
        self.refresh_rotation_matrix()
        return self

    def increment_gamma(self, dgamma: float) -> WebGPUCamera:
        self.euler_angles[2] += dgamma
        self.refresh_rotation_matrix()
        return self

    # ------------------------------------------------------------------
    # View matrix (world → camera space)
    # ------------------------------------------------------------------

    @property
    def view_matrix(self) -> np.ndarray:
        """4×4 float32 view matrix: rotates and translates world space into
        camera space.

        For default 2-D scenes (no rotation, center at origin) this is the
        identity matrix, so 2-D rendering is unaffected.
        """
        R = np.asarray(self.inverse_rotation_matrix, dtype=np.float32)  # 3×3
        c = self.center_point.astype(np.float32)
        view = np.eye(4, dtype=np.float32)
        view[:3, :3] = R
        view[:3, 3] = -(R @ c)
        return view

    # ------------------------------------------------------------------
    # Projection matrix (used by the shader uniform upload)
    # ------------------------------------------------------------------

    @property
    def projection_matrix(self) -> np.ndarray:
        """4×4 float32 projection matrix in WebGPU NDC convention (z ∈ [0, 1]).

        Orthographic when ``self.orthographic`` is True (default).
        Perspective otherwise — focal distance drives the field of view.
        """
        fw, fh = self.frame_shape
        near, far = self.near, self.far

        if self.orthographic:
            # Orthographic: map frame to NDC with z ∈ [0, 1].
            return np.array(
                [
                    [2.0 / fw, 0.0,      0.0,                 0.0],
                    [0.0,      2.0 / fh, 0.0,                 0.0],
                    [0.0,      0.0,      1.0 / (far - near),  -near / (far - near)],
                    [0.0,      0.0,      0.0,                  1.0],
                ],
                dtype=np.float32,
            )
        else:
            # Perspective: symmetric frustum, z ∈ [0, 1] (WebGPU NDC).
            fd = self.get_focal_distance()
            return np.array(
                [
                    [2.0 * fd / fw, 0.0,            0.0,                              0.0],
                    [0.0,           2.0 * fd / fh,  0.0,                              0.0],
                    [0.0,           0.0,            far / (far - near),              -far * near / (far - near)],
                    [0.0,           0.0,            1.0,                              0.0],
                ],
                dtype=np.float32,
            )


# ---------------------------------------------------------------------------
# Main renderer class
# ---------------------------------------------------------------------------


class WebGPURenderer:
    """Headless WebGPU renderer (Phase 1: fill rendering to PNG / video)."""

    def __init__(
        self,
        file_writer_class: type[SceneFileWriter] = SceneFileWriter,
        skip_animations: bool = False,
    ) -> None:
        self._file_writer_class = file_writer_class
        self._original_skipping_status = skip_animations
        self.skip_animations = skip_animations

        self.animation_start_time: float = 0.0
        self.animation_elapsed_time: float = 0.0
        self.time: float = 0.0
        self.num_plays: int = 0
        self.animations_hashes: list[str | None] = []

        self.camera: WebGPUCamera = WebGPUCamera()
        self.window: None = None
        self.static_image: Any = None
        self.file_writer: SceneFileWriter | None = None  # set by init_scene()

        self.background_color = config["background_color"]

        # Filled by init_scene():
        self._device: wgpu_t.GPUDevice | None = None
        self._render_texture: wgpu_t.GPUTexture | None = None
        self._render_texture_view: wgpu_t.GPUTextureView | None = None
        self._depth_texture: wgpu_t.GPUTexture | None = None
        self._depth_texture_view: wgpu_t.GPUTextureView | None = None
        self._proj_bgl: wgpu_t.GPUBindGroupLayout | None = None
        self._slug_bgl: wgpu_t.GPUBindGroupLayout | None = None
        self._slug_fill_pipeline: wgpu_t.GPURenderPipeline | None = None
        self._stroke_pipeline: wgpu_t.GPURenderPipeline | None = None
        self._surface_pipeline: wgpu_t.GPURenderPipeline | None = None

        # Per-frame state (set during update_frame, cleared after submit).
        self.current_render_pass: wgpu_t.GPURenderPassEncoder | None = None
        self.camera_bind_group: wgpu_t.GPUBindGroup | None = None
        self._camera_uniform_buf: wgpu_t.GPUBuffer | None = None
        self.frame_vbos: list[wgpu_t.GPUBuffer] = []

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_scene(self, scene: Scene) -> None:
        """Create the wgpu device, offscreen texture, and file writer."""
        self.scene = scene
        self.partial_movie_files: list[str | None] = []
        self.file_writer: SceneFileWriter = self._file_writer_class(
            self,
            scene.__class__.__name__,
        )

        self.background_color = config["background_color"]

        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self._device = adapter.request_device_sync(
            required_features=[],
            required_limits={},
        )
        logger.debug("WebGPU adapter: %s", adapter.info)

        width = config.pixel_width
        height = config.pixel_height
        self._render_texture = self._device.create_texture(
            size=(width, height, 1),
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )
        self._render_texture_view = self._render_texture.create_view()

        self._depth_texture = self._device.create_texture(
            size=(width, height, 1),
            format=wgpu.TextureFormat.depth24plus,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
        )
        self._depth_texture_view = self._depth_texture.create_view()

        self._proj_bgl = self._create_camera_bgl()
        self._stroke_pipeline = self._create_stroke_pipeline(self._proj_bgl)
        self._surface_pipeline = self._create_surface_pipeline(self._proj_bgl)
        self._slug_bgl, self._slug_fill_pipeline = self._create_slug_fill_pipeline()

    # ------------------------------------------------------------------
    # Pipeline creation
    # ------------------------------------------------------------------

    def _create_camera_bgl(self) -> wgpu_t.GPUBindGroupLayout:
        """Create the bind group layout shared by stroke, surface, and Slug pipelines.

        Layout: binding 0 — one uniform buffer carrying projection (64 B) +
        view (64 B) + light_pos+pad (16 B) = 144 bytes total.
        """
        assert self._device is not None
        return self._device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": "uniform"},
                }
            ]
        )

    def _create_stroke_pipeline(
        self, proj_bgl: wgpu_t.GPUBindGroupLayout
    ) -> wgpu_t.GPURenderPipeline:
        assert self._device is not None
        shader_path = Path(__file__).parent / "shaders" / "vmobject_stroke.wgsl"
        shader_module = self._device.create_shader_module(
            code=shader_path.read_text(encoding="utf-8")
        )
        _blend = {
            "color": {
                "src_factor": "src-alpha",
                "dst_factor": "one-minus-src-alpha",
                "operation": "add",
            },
            "alpha": {
                "src_factor": "one",
                "dst_factor": "one",
                "operation": "add",
            },
        }
        return self._device.create_render_pipeline(
            layout=self._device.create_pipeline_layout(
                bind_group_layouts=[proj_bgl]
            ),
            vertex={
                "module": shader_module,
                "entry_point": "vs_main",
                "buffers": [STROKE_VERTEX_LAYOUT],
            },
            fragment={
                "module": shader_module,
                "entry_point": "fs_main",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm, "blend": _blend}],
            },
            primitive={"topology": "triangle-list", "cull_mode": "none"},
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": False,
                "depth_compare": "always",
                "stencil_front": {"compare": "always", "fail_op": "keep", "depth_fail_op": "keep", "pass_op": "keep"},
                "stencil_back":  {"compare": "always", "fail_op": "keep", "depth_fail_op": "keep", "pass_op": "keep"},
                "stencil_read_mask": 0,
                "stencil_write_mask": 0,
            },
            multisample={
                "count": 1,
                "mask": 0xFFFF_FFFF,
                "alpha_to_coverage_enabled": False,
            },
        )

    def _create_surface_pipeline(
        self, proj_bgl: wgpu_t.GPUBindGroupLayout
    ) -> wgpu_t.GPURenderPipeline:
        assert self._device is not None
        shader_path = Path(__file__).parent / "shaders" / "surface.wgsl"
        shader_module = self._device.create_shader_module(
            code=shader_path.read_text(encoding="utf-8")
        )
        _blend = {
            "color": {
                "src_factor": "src-alpha",
                "dst_factor": "one-minus-src-alpha",
                "operation": "add",
            },
            "alpha": {
                "src_factor": "one",
                "dst_factor": "one",
                "operation": "add",
            },
        }
        return self._device.create_render_pipeline(
            layout=self._device.create_pipeline_layout(
                bind_group_layouts=[proj_bgl]
            ),
            vertex={
                "module": shader_module,
                "entry_point": "vs_main",
                "buffers": [SURFACE_VERTEX_LAYOUT],
            },
            fragment={
                "module": shader_module,
                "entry_point": "fs_main",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm, "blend": _blend}],
            },
            primitive={"topology": "triangle-list", "cull_mode": "none"},
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True,
                "depth_compare": "less",
                "stencil_front": {"compare": "always", "fail_op": "keep", "depth_fail_op": "keep", "pass_op": "keep"},
                "stencil_back":  {"compare": "always", "fail_op": "keep", "depth_fail_op": "keep", "pass_op": "keep"},
                "stencil_read_mask": 0,
                "stencil_write_mask": 0,
            },
            multisample={
                "count": 1,
                "mask": 0xFFFF_FFFF,
                "alpha_to_coverage_enabled": False,
            },
        )

    def _create_slug_fill_pipeline(
        self,
    ) -> tuple[wgpu_t.GPUBindGroupLayout, wgpu_t.GPURenderPipeline]:
        """Create the Slug fill pipeline with a storage-buffer bind group layout."""
        assert self._device is not None
        shader_path = Path(__file__).parent / "shaders" / "slug_fill.wgsl"
        shader_module = self._device.create_shader_module(
            code=shader_path.read_text(encoding="utf-8")
        )

        # Group 0: binding 0 = camera uniform, binding 1 = curves storage (read-only).
        slug_bgl = self._device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": "uniform"},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": "read-only-storage", "has_dynamic_offset": False},
                },
            ]
        )

        _blend = {
            "color": {
                "src_factor": "src-alpha",
                "dst_factor": "one-minus-src-alpha",
                "operation": "add",
            },
            "alpha": {
                "src_factor": "one",
                "dst_factor": "one",
                "operation": "add",
            },
        }

        pipeline = self._device.create_render_pipeline(
            layout=self._device.create_pipeline_layout(bind_group_layouts=[slug_bgl]),
            vertex={
                "module": shader_module,
                "entry_point": "vs_main",
                "buffers": [SLUG_FILL_VERTEX_LAYOUT],
            },
            fragment={
                "module": shader_module,
                "entry_point": "fs_main",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm, "blend": _blend}],
            },
            primitive={"topology": "triangle-list", "cull_mode": "none"},
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": False,
                "depth_compare": "always",
                "stencil_front": {"compare": "always", "fail_op": "keep", "depth_fail_op": "keep", "pass_op": "keep"},
                "stencil_back":  {"compare": "always", "fail_op": "keep", "depth_fail_op": "keep", "pass_op": "keep"},
                "stencil_read_mask": 0,
                "stencil_write_mask": 0,
            },
            multisample={"count": 1, "mask": 0xFFFF_FFFF, "alpha_to_coverage_enabled": False},
        )
        return slug_bgl, pipeline

    # ------------------------------------------------------------------
    # Camera bind group (rebuilt each frame when projection changes)
    # ------------------------------------------------------------------

    def _build_camera_uniform_buf(self) -> wgpu_t.GPUBuffer:
        """Pack the 144-byte camera uniform and upload it; return the buffer."""
        assert self._device is not None
        proj_bytes  = self.camera.projection_matrix.T.flatten().tobytes()
        view_bytes  = self.camera.view_matrix.T.flatten().tobytes()
        light       = np.zeros(4, dtype=np.float32)
        light[:3]   = self.camera.light_source_position.astype(np.float32)
        light_bytes = light.tobytes()

        buf = self._device.create_buffer_with_data(
            data=proj_bytes + view_bytes + light_bytes,
            usage=wgpu.BufferUsage.UNIFORM,
        )
        self.frame_vbos.append(buf)
        return buf

    def _build_camera_bind_group(self) -> wgpu_t.GPUBindGroup:
        assert self._device is not None
        assert self._proj_bgl is not None

        self._camera_uniform_buf = self._build_camera_uniform_buf()

        return self._device.create_bind_group(
            layout=self._proj_bgl,
            entries=[
                {"binding": 0, "resource": {"buffer": self._camera_uniform_buf, "offset": 0, "size": 144}}
            ],
        )

    def _build_slug_bind_group(
        self, curves_buf: wgpu_t.GPUBuffer
    ) -> wgpu_t.GPUBindGroup:
        """Build the Slug fill bind group: camera uniform + curves storage buffer."""
        assert self._device is not None
        assert self._slug_bgl is not None
        assert self._camera_uniform_buf is not None, (
            "_build_camera_bind_group() must be called before _build_slug_bind_group()"
        )
        return self._device.create_bind_group(
            layout=self._slug_bgl,
            entries=[
                {"binding": 0, "resource": {"buffer": self._camera_uniform_buf, "offset": 0, "size": 144}},
                {"binding": 1, "resource": {"buffer": curves_buf, "offset": 0, "size": curves_buf.size}},
            ],
        )

    # ------------------------------------------------------------------
    # Pipeline / device accessors (used by webgpu_vmobject_rendering)
    # ------------------------------------------------------------------

    @property
    def device(self) -> wgpu_t.GPUDevice:
        assert self._device is not None, "init_scene() has not been called"
        return self._device

    @property
    def stroke_pipeline(self) -> wgpu_t.GPURenderPipeline:
        assert self._stroke_pipeline is not None, "init_scene() has not been called"
        return self._stroke_pipeline

    @property
    def surface_pipeline(self) -> wgpu_t.GPURenderPipeline:
        assert self._surface_pipeline is not None, "init_scene() has not been called"
        return self._surface_pipeline

    @property
    def slug_fill_pipeline(self) -> wgpu_t.GPURenderPipeline:
        assert self._slug_fill_pipeline is not None, "init_scene() has not been called"
        return self._slug_fill_pipeline

    # ------------------------------------------------------------------
    # Frame rendering
    # ------------------------------------------------------------------

    def update_frame(self, scene: Scene) -> None:
        """Render one frame into the offscreen texture."""
        assert self._device is not None
        assert self._render_texture_view is not None
        assert self._depth_texture_view is not None

        bg = self._background_color  # (r, g, b, a) floats in [0, 1]

        encoder = self._device.create_command_encoder()
        render_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self._render_texture_view,
                    "load_op": "clear",
                    "store_op": "store",
                    "clear_value": tuple(float(c) for c in bg),
                }
            ],
            depth_stencil_attachment={
                "view": self._depth_texture_view,
                "depth_clear_value": 1.0,
                "depth_load_op": "clear",
                "depth_store_op": "discard",
            },
        )

        self.current_render_pass = render_pass
        self.frame_vbos = []

        # One camera bind group per frame (projection may change).
        self.camera_bind_group = self._build_camera_bind_group()

        # Batch render: collect all geometry first, then 1–3 GPU uploads total.
        render_webgpu_mobject(self, scene.mobjects)

        render_pass.end()
        self._device.queue.submit([encoder.finish()])

        self.current_render_pass = None
        self.camera_bind_group = None
        self.frame_vbos = []

        self.animation_elapsed_time = time.time() - self.animation_start_time

    # ------------------------------------------------------------------
    # Frame readback
    # ------------------------------------------------------------------

    def _get_raw_frame_data(self) -> bytes:
        """Copy the render texture to CPU memory and return tightly-packed RGBA bytes."""
        assert self._device is not None
        assert self._render_texture is not None

        width = config.pixel_width
        height = config.pixel_height
        bpr = width * 4  # bytes per row (unpadded)

        # WebGPU requires bytes_per_row to be a multiple of 256.
        aligned_bpr = (bpr + 255) & ~255

        readback_buf = self._device.create_buffer(
            size=aligned_bpr * height,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )

        encoder = self._device.create_command_encoder()
        encoder.copy_texture_to_buffer(
            {"texture": self._render_texture, "mip_level": 0, "origin": (0, 0, 0)},
            {
                "buffer": readback_buf,
                "offset": 0,
                "bytes_per_row": aligned_bpr,
                "rows_per_image": height,
            },
            (width, height, 1),
        )
        self._device.queue.submit([encoder.finish()])

        readback_buf.map_sync(wgpu.MapMode.READ)
        raw = bytes(readback_buf.read_mapped())
        readback_buf.unmap()

        if aligned_bpr != bpr:
            raw = b"".join(
                raw[i * aligned_bpr : i * aligned_bpr + bpr] for i in range(height)
            )
        return raw

    def get_image(self) -> Image.Image:
        """Return the current frame as a PIL Image (RGBA)."""
        raw = self._get_raw_frame_data()
        return Image.frombytes(
            "RGBA", (config.pixel_width, config.pixel_height), raw
        )

    def get_frame(self) -> np.ndarray:
        """Return the current frame as a (height, width, 4) uint8 NumPy array."""
        raw = self._get_raw_frame_data()
        return np.frombuffer(raw, dtype=np.uint8).reshape(
            (config.pixel_height, config.pixel_width, 4)
        )

    # ------------------------------------------------------------------
    # Scene lifecycle
    # ------------------------------------------------------------------

    def render(self, scene: Scene, frame_offset: float, moving_mobjects: list) -> None:
        self.update_frame(scene)
        if not self.skip_animations:
            self.file_writer.write_frame(self)

    def play(self, scene: Scene, *animations: Any, **kwargs: Any) -> None:
        self.animation_start_time = time.time()
        self.skip_animations = self._original_skipping_status
        self._update_skipping_status()

        self.animations_hashes.append(None)
        self.file_writer.add_partial_movie_file(None)

        self.file_writer.begin_animation(not self.skip_animations)
        scene.compile_animation_data(*animations, **kwargs)
        scene.begin_animations()

        if scene.is_current_animation_frozen_frame():
            self.update_frame(scene)
            if not self.skip_animations:
                self.file_writer.write_frame(
                    self, num_frames=int(config.frame_rate * scene.duration)
                )
            self.animation_elapsed_time = scene.duration
        else:
            scene.play_internal()

        self.file_writer.end_animation(not self.skip_animations)
        self.time += scene.duration
        self.num_plays += 1

    def scene_finished(self, scene: Scene) -> None:
        if self.num_plays > 0:
            self.file_writer.finish()
        elif self.num_plays == 0 and config.write_to_movie:
            config.save_last_frame = True
            config.write_to_movie = False

        if self._should_save_last_frame():
            config.save_last_frame = True
            self.update_frame(scene)
            self.file_writer.save_image(self.get_image())

    def save_static_frame_data(self, scene: Scene, static_mobjects: Any) -> None:
        pass  # not implemented in Phase 1

    def clear_screen(self) -> None:
        pass  # headless — no window

    # ------------------------------------------------------------------
    # Skipping helpers
    # ------------------------------------------------------------------

    def _update_skipping_status(self) -> None:
        if self.file_writer.sections[-1].skip_animations:
            self.skip_animations = True
        if config["save_last_frame"]:
            self.skip_animations = True
        if (
            config.from_animation_number > 0
            and self.num_plays < config.from_animation_number
        ):
            self.skip_animations = True
        if (
            config.upto_animation_number >= 0
            and self.num_plays > config.upto_animation_number
        ):
            self.skip_animations = True
            raise EndSceneEarlyException()

    def _should_save_last_frame(self) -> bool:
        if config["save_last_frame"]:
            return True
        if self.scene.interactive_mode:
            return False
        return self.num_plays == 0

    # ------------------------------------------------------------------
    # Background colour
    # ------------------------------------------------------------------

    @property
    def background_color(self):
        return self._background_color

    @background_color.setter
    def background_color(self, value) -> None:
        self._background_color = color_to_rgba(value, 1.0)

    def get_pixel_shape(self) -> tuple[int, int]:
        return (config.pixel_width, config.pixel_height)
