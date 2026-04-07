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
from manim.constants import IN, OUT, PI, RIGHT, DOWN, LEFT
from manim.mobject.mobject import Mobject
from manim.mobject.types.vectorized_mobject import VMobject
from manim.scene.scene_file_writer import SceneFileWriter
from manim.utils.color import color_to_rgba
from manim.utils.exceptions import EndSceneEarlyException
from manim.utils.hashing import get_hash_from_play_call
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
    from .webgpu_renderer_window import WebGPUWindow

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


class WebGPUCamera(Mobject):
    """Camera for the WebGPU renderer.

    Inherits from ``Mobject`` so it can carry updaters and be added to the
    scene, matching the pattern used by ``OpenGLCamera(OpenGLMobject)``.

    Matches the attribute / method surface of ``OpenGLCamera`` so that
    scene code that inspects ``renderer.camera`` works without changes.

    Projection
    ----------
    * 2-D scenes: orthographic, z mapped to the WebGPU [0, 1] NDC
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
        orthographic: bool = False,
        minimum_polar_angle: float = -PI / 2,
        maximum_polar_angle: float = PI / 2,
    ) -> None:
        super().__init__()
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
            else np.array([0.0, 0.0, 11.0], dtype=float)
        )
        self.euler_angles: np.ndarray = np.asarray(
            euler_angles if euler_angles is not None else [0.0, 0.0, 0.0],
            dtype=float,
        )
        self.refresh_rotation_matrix()

        # Fixed-mobject registries — populated by ThreeDScene helpers.
        # fixed_in_frame: objects rendered with identity rotation + ortho
        #   projection as a 2-D overlay on top of the 3-D scene (e.g. title text).
        # fixed_orientation: objects rendered with identity rotation + current
        #   projection so they don't tilt as the camera orbits (e.g. 3-D labels).
        self._fixed_in_frame_mobjects: set[Mobject] = set()
        self._fixed_orientation_mobjects: set[Mobject] = set()

        # ThreeDScene.get_moving_mobjects() checks _frame_center and
        # get_value_trackers() to detect camera-driven animation.
        # These are defined on ThreeDCamera (Cairo) but not on Mobject,
        # so we provide equivalent stubs here.
        self._frame_center: Mobject = Mobject()

    def get_value_trackers(self) -> list:
        """Required by ThreeDScene.get_moving_mobjects."""
        return []

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
        """Refresh the camera's inverse rotation matrix based on its Euler angles.
        Matches Cairo's orientation.
        """
        theta, phi, gamma = self.euler_angles
        quat = quaternion_mult(
            quaternion_from_angle_axis(theta, IN, axis_normalized=True),
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

    _PERSPECTIVE_FAR: float = 50.0  # must match projection_matrix

    def set_focal_distance(self, focal_distance: float) -> WebGPUCamera:
        """Set the perspective focal distance (= near plane distance).

        Larger values zoom in (more telephoto); smaller values zoom out.
        Only has an effect when ``orthographic=False``.
        Matches ``OpenGLCamera.focal_distance`` convention.

        ``focal_distance`` must be positive and strictly less than the far
        plane (50.0).  Values outside that range are clamped.
        """
        max_near = self._PERSPECTIVE_FAR * (1.0 - 1e-4)
        clamped = float(np.clip(focal_distance, 1e-4, max_near))
        if clamped != focal_distance:
            logger.warning(
                "WebGPUCamera.set_focal_distance: value %.4g clamped to %.4g "
                "(must be in (0, far=%.4g))",
                focal_distance, clamped, self._PERSPECTIVE_FAR,
            )
        self.focal_distance = clamped
        return self

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

        Uses T(-c) @ R_inv, which rotates the world around the origin (matches
        OpenGLCamera behavior where the camera orbits the focal point).
        """
        R = np.asarray(self.inverse_rotation_matrix, dtype=np.float32)  # 3×3
        c = self.center_point.astype(np.float32)
        view = np.eye(4, dtype=np.float32)
        view[:3, :3] = R
        # Translation in camera space: T(-c) followed by rotation R is equivalent
        # to rotating the origin then translating, or translating the origin then rotating.
        # To stay centered on origin: rotate first, then translate by -distance.
        # V = translation(0, 0, -11) @ R_inv
        view[:3, 3] = [0.0, 0.0, -c[2]]
        return view

    @property
    def fixed_view_matrix(self) -> np.ndarray:
        """View matrix with camera rotation stripped — z-translation only.

        Used for fixed-orientation and fixed-in-frame mobjects so they don't
        tilt or spin when the camera orbits.  The z-translation is preserved so
        depth ordering within the fixed layer is consistent with the main scene.
        """
        view = np.eye(4, dtype=np.float32)
        view[2, 3] = -float(self.center_point[2])
        return view

    @property
    def ortho_projection_matrix(self) -> np.ndarray:
        """Forced orthographic projection matrix, regardless of self.orthographic.

        Fixed-in-frame overlays always use orthographic so that screen-space
        coordinates map directly to Manim scene units (matching 2-D scenes).
        """
        # Orthographic: map frame to NDC with z ∈ [0, 1].
        # Note: Manim's +Z is out of the screen (towards the viewer).
        # So +Z should map to 0 (near) and -Z should map to 1 (far).
        # Z_clip = -1/(far-near) * Z + far/(far-near)
        fw, fh = self.frame_shape
        near, far = self.near, self.far
        return np.array(
            [
                [2.0 / fw, 0.0,      0.0,                  0.0],
                [0.0,      2.0 / fh, 0.0,                  0.0],
                [0.0,      0.0,     -1.0 / (far - near),   far / (far - near)],
                [0.0,      0.0,      0.0,                  1.0],
            ],
            dtype=np.float32,
        )
        
    # ------------------------------------------------------------------
    # Projection matrix (used by the shader uniform upload)
    # ------------------------------------------------------------------

    @property
    def projection_matrix(self) -> np.ndarray:
        """4×4 float32 projection matrix in WebGPU NDC convention (z ∈ [0, 1]).

        Perspective when ``self.orthographic`` is False (default).
        Perspective otherwise — focal distance drives the field of view.
        """
        fw, fh = self.frame_shape
        near, far = self.near, self.far

        if self.orthographic:
            return self.ortho_projection_matrix
        else:
            # Perspective mapping for WebGPU: W_clip = -z_view, z_clip ∈ [0, 1].
            # near = focal_distance (matches OpenGLCamera's implicit convention where
            # the default focal_distance=2.0 equals OpenGL's hardcoded near=2).
            # Changing focal_distance zooms the scene: larger → more telephoto.
            # FOV is set by w=fw/6, h=fh/6 (same as opengl.perspective_projection_matrix).
            f = self._PERSPECTIVE_FAR
            n = float(np.clip(self.focal_distance, 1e-4, f * (1.0 - 1e-4)))
            w, h = fw / 6.0, fh / 6.0
            return np.array(
                [
                    [2.0 * n / w, 0.0,         0.0,            0.0],
                    [0.0,         2.0 * n / h, 0.0,            0.0],
                    [0.0,         0.0,         f / (n - f),    n * f / (n - f)],
                    [0.0,         0.0,        -1.0,            0.0],
                ],
                dtype=np.float32,
            )


    # ------------------------------------------------------------------
    # Fixed-mobject registry (used by ThreeDScene)
    # ------------------------------------------------------------------

    def add_fixed_in_frame_mobjects(self, *mobjects: Mobject) -> None:
        """Register mobjects to be rendered as 2-D screen-space overlays.

        These objects are drawn after the 3-D scene with a fresh depth buffer,
        identity camera rotation, and an orthographic projection so they always
        appear on top at their 2-D screen-space coordinates.
        """
        self._fixed_in_frame_mobjects.update(mobjects)

    def remove_fixed_in_frame_mobjects(self, *mobjects: Mobject) -> None:
        """Unregister mobjects previously added with add_fixed_in_frame_mobjects."""
        self._fixed_in_frame_mobjects.difference_update(mobjects)

    def add_fixed_orientation_mobjects(self, *mobjects: Mobject) -> None:
        """Register mobjects whose orientation is frozen relative to the camera.

        These objects still move in 3-D space (their world coordinates are used
        normally) but the camera rotation is not applied — they remain upright as
        the camera orbits.  Useful for 3-D labels that should always face forward.
        """
        self._fixed_orientation_mobjects.update(mobjects)

    def remove_fixed_orientation_mobjects(self, *mobjects: Mobject) -> None:
        """Unregister mobjects previously added with add_fixed_orientation_mobjects."""
        self._fixed_orientation_mobjects.difference_update(mobjects)


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
        self.window: WebGPUWindow | None = None
        self.pressed_keys: set[int] = set()
        self.static_image: Any = None
        self.file_writer: SceneFileWriter | None = None  # set by init_scene()

        # SpecialThreeDScene reads renderer.camera_config["pixel_width"] to decide
        # whether to apply low-quality overrides.  Mirrors the pattern used by
        # OpenGLRenderer so that SpecialThreeDScene works unchanged with WebGPU.
        self.camera_config: dict = {
            "pixel_width": config.pixel_width,
            "pixel_height": config.pixel_height,
        }

        self.background_color = config["background_color"]

        # Scene-wide lighting — read by _build_camera_uniform_buf() each frame.
        # light_color / ambient_color are RGB floats in [0, 1].
        self.light_source_position: np.ndarray = np.array([-10.0, 10.0, 5.0])
        self.light_color: np.ndarray           = np.array([1.0, 1.0, 1.0])
        self.light_intensity: float            = 100.0
        self.ambient_color: np.ndarray         = np.array([1.0, 1.0, 1.0])
        self.ambient_intensity: float          = 0.4

        # Filled by init_scene():
        self._device: wgpu_t.GPUDevice | None = None
        self._render_texture: wgpu_t.GPUTexture | None = None
        self._render_texture_view: wgpu_t.GPUTextureView | None = None
        self._depth_texture: wgpu_t.GPUTexture | None = None
        self._depth_texture_view: wgpu_t.GPUTextureView | None = None
        self._proj_bgl: wgpu_t.GPUBindGroupLayout | None = None
        self._slug_bgl: wgpu_t.GPUBindGroupLayout | None = None
        # Slug fill: 2-D (no depth) and 3-D (depth-tested, shade_in_3d with fill).
        self._slug_fill_pipeline: wgpu_t.GPURenderPipeline | None = None
        self._slug_fill_3d_pipeline: wgpu_t.GPURenderPipeline | None = None
        # Stroke pipelines: 2-D, 3-D, and 3-D with depth bias (surface mesh lines).
        self._stroke_pipeline: wgpu_t.GPURenderPipeline | None = None
        self._stroke_3d_pipeline: wgpu_t.GPURenderPipeline | None = None
        self._stroke_3d_surface_pipeline: wgpu_t.GPURenderPipeline | None = None
        # Surface pipelines: opaque (depth write + backface cull) and OIT.
        self._surface_pipeline: wgpu_t.GPURenderPipeline | None = None  # opaque
        self._surface_oit_pipeline: wgpu_t.GPURenderPipeline | None = None
        # OIT accumulation textures (rgba16float each).
        self._oit_accum_texture: wgpu_t.GPUTexture | None = None
        self._oit_accum_view: wgpu_t.GPUTextureView | None = None
        self._oit_reveal_texture: wgpu_t.GPUTexture | None = None
        self._oit_reveal_view: wgpu_t.GPUTextureView | None = None
        # OIT composition pipeline + bind group.
        self._oit_compose_pipeline: wgpu_t.GPURenderPipeline | None = None
        self._oit_compose_bgl: wgpu_t.GPUBindGroupLayout | None = None
        self._oit_compose_bind_group: wgpu_t.GPUBindGroup | None = None

        # Compact readback compute pipeline (GPU row-depadding + B↔R fix).
        self._readback_compute_pipeline: wgpu_t.GPUComputePipeline | None = None
        self._readback_compute_bgl: wgpu_t.GPUBindGroupLayout | None = None
        self._readback_compute_bind_group: wgpu_t.GPUBindGroup | None = None
        # Storage buffer the compute shader writes into (STORAGE | COPY_SRC).
        self._readback_storage_buf: wgpu_t.GPUBuffer | None = None
        # Mappable buffer we copy into before CPU readback (COPY_DST | MAP_READ).
        self._readback_map_buf: wgpu_t.GPUBuffer | None = None

        # Per-frame state (set during update_frame, cleared after submit).
        self.current_render_pass: wgpu_t.GPURenderPassEncoder | None = None
        self.camera_bind_group: wgpu_t.GPUBindGroup | None = None
        self._camera_uniform_buf: wgpu_t.GPUBuffer | None = None
        # Fixed-mobject bind groups (rebuilt each frame with stripped-rotation view).
        #   fixed_camera_bind_group: identity rotation + current projection (fixed-orientation)
        #   fixed_frame_bind_group:  identity rotation + orthographic projection (fixed-in-frame)
        self.fixed_camera_bind_group: wgpu_t.GPUBindGroup | None = None
        self._fixed_orient_uniform_buf: wgpu_t.GPUBuffer | None = None
        self.fixed_frame_bind_group: wgpu_t.GPUBindGroup | None = None
        self._fixed_frame_uniform_buf: wgpu_t.GPUBuffer | None = None
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
        # bgra8unorm matches the window surface format on all major platforms
        # (Metal/Vulkan/DX12), enabling copy_texture_to_texture without a
        # blit shader.  Readback in _get_raw_frame_data() swaps B↔R to
        # produce the RGBA output expected by PIL / numpy callers.
        self._render_texture = self._device.create_texture(
            size=(width, height, 1),
            format=wgpu.TextureFormat.bgra8unorm,
            usage=(
                wgpu.TextureUsage.RENDER_ATTACHMENT
                | wgpu.TextureUsage.COPY_SRC
                | wgpu.TextureUsage.TEXTURE_BINDING  # read by compact-readback compute shader
            ),
        )
        self._render_texture_view = self._render_texture.create_view()

        self._depth_texture = self._device.create_texture(
            size=(width, height, 1),
            format=wgpu.TextureFormat.depth24plus,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
        )
        self._depth_texture_view = self._depth_texture.create_view()

        self._proj_bgl = self._create_camera_bgl()
        self._stroke_pipeline    = self._create_stroke_pipeline(self._proj_bgl, depth_test=True)
        self._stroke_3d_pipeline = self._create_stroke_pipeline(self._proj_bgl, depth_test=True)
        # Surface mesh lines sit exactly on the surface triangles.  A negative
        # depth bias pulls each fragment slightly toward the camera so the mesh
        # always wins the depth test without visually offsetting the lines.
        # depth_bias=-10000 gives ~6e-4 constant offset in [0,1] depth space
        # (depth24plus unit ≈ 6e-8), which is large enough to reliably beat
        # floating-point depth jitter on flat/low-slope surface regions where
        # depth_bias_slope_scale alone contributes nearly zero.
        self._stroke_3d_surface_pipeline = self._create_stroke_pipeline(
            self._proj_bgl,
            depth_test=True,
            depth_bias=-10000,
            depth_bias_slope_scale=-1.0,
            depth_bias_clamp=0.00001,
        )
        self._surface_pipeline            = self._create_surface_pipeline(self._proj_bgl, cull_mode="none",  depth_write=True)
        self._slug_bgl, self._slug_fill_pipeline    = self._create_slug_fill_pipeline(depth_test=True)
        _,              self._slug_fill_3d_pipeline = self._create_slug_fill_pipeline(depth_test=True)
        self._create_oit_resources(width, height)
        self._create_readback_pipeline(width, height)

        if self.should_create_window():
            from .webgpu_renderer_window import WebGPUWindow
            self.window = WebGPUWindow(self)

    # ------------------------------------------------------------------
    # Pipeline creation
    # ------------------------------------------------------------------

    def _create_camera_bgl(self) -> wgpu_t.GPUBindGroupLayout:
        """Create the bind group layout shared by stroke, surface, and Slug pipelines.

        Layout: binding 0 — one uniform buffer (176 bytes total):
          offset   0 — projection        mat4x4<f32>  64 B
          offset  64 — view              mat4x4<f32>  64 B
          offset 128 — light_pos         vec3<f32>    12 B
          offset 140 — light_intensity   f32           4 B
          offset 144 — light_color       vec3<f32>    12 B
          offset 156 — ambient_intensity f32           4 B
          offset 160 — ambient_color     vec3<f32>    12 B
          offset 172 — _pad              f32           4 B
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
        self,
        proj_bgl: wgpu_t.GPUBindGroupLayout,
        depth_test: bool = False,
        depth_bias: int = 0,
        depth_bias_slope_scale: float = 0.0,
        depth_bias_clamp: float = 0.0,
    ) -> wgpu_t.GPURenderPipeline:
        """Create a stroke pipeline.

        depth_test controls depth *writing* only — both 2-D and 3-D strokes
        always depth-test (depth_compare="less") so they are correctly occluded
        by opaque geometry.

        depth_test=False  — 2-D strokes: depth-read-only.  Occluded by any
                            opaque surface in front of them, but do not themselves
                            occlude later geometry.
        depth_test=True   — 3-D strokes (shade_in_3d): depth-write + depth-test
                            so they occlude geometry drawn behind them.
        depth_bias / depth_bias_slope_scale / depth_bias_clamp
                          — WebGPU depth bias applied to every fragment.  Use
                            negative values to push geometry toward the camera,
                            which prevents z-fighting when strokes lie exactly
                            on a surface (e.g. Surface mesh lines).
        """
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
                "targets": [{"format": wgpu.TextureFormat.bgra8unorm, "blend": _blend}],
            },
            primitive={"topology": "triangle-list", "cull_mode": "none"},
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": depth_test,
                "depth_compare": "less",  # always depth-test; write only for 3-D strokes
                "depth_bias": depth_bias,
                "depth_bias_slope_scale": depth_bias_slope_scale,
                "depth_bias_clamp": depth_bias_clamp,
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
        self,
        proj_bgl: wgpu_t.GPUBindGroupLayout,
        cull_mode: str = "none",
        depth_write: bool = True,
    ) -> wgpu_t.GPURenderPipeline:
        """Create a surface (mesh) pipeline.

        cull_mode   — WebGPU cull mode passed directly to the pipeline.
                      Use "back" for opaque surfaces (back faces are never
                      visible and culling them halves fragment work).
                      Use "none" for OIT transparent surfaces (both faces
                      must contribute so the interior is visible through
                      the front face).
        depth_write — True for opaque surfaces so they occlude later geometry.
                      False for OIT surfaces so transparent layers do not block
                      each other (they still depth-test against opaque geometry).
        """
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
                "targets": [{"format": wgpu.TextureFormat.bgra8unorm, "blend": _blend}],
            },
            primitive={"topology": "triangle-list", "cull_mode": cull_mode},
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": depth_write,
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

    def _create_oit_resources(self, width: int, height: int) -> None:
        """Create OIT accumulation textures, pipelines, and bind groups."""
        assert self._device is not None
        assert self._proj_bgl is not None

        # ── Accumulation textures ──────────────────────────────────────────
        oit_usage = wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING
        self._oit_accum_texture = self._device.create_texture(
            size=(width, height, 1),
            format=wgpu.TextureFormat.rgba16float,
            usage=oit_usage,
        )
        self._oit_accum_view = self._oit_accum_texture.create_view()

        self._oit_reveal_texture = self._device.create_texture(
            size=(width, height, 1),
            format=wgpu.TextureFormat.rgba16float,
            usage=oit_usage,
        )
        self._oit_reveal_view = self._oit_reveal_texture.create_view()

        # ── OIT accumulation pipeline ──────────────────────────────────────
        oit_shader_path = Path(__file__).parent / "shaders" / "surface_oit.wgsl"
        oit_shader = self._device.create_shader_module(
            code=oit_shader_path.read_text(encoding="utf-8")
        )
        _accum_blend = {
            "color": {"src_factor": "one", "dst_factor": "one", "operation": "add"},
            "alpha": {"src_factor": "one", "dst_factor": "one", "operation": "add"},
        }
        _reveal_blend = {
            "color": {"src_factor": "zero", "dst_factor": "one-minus-src-alpha", "operation": "add"},
            "alpha": {"src_factor": "zero", "dst_factor": "one",                 "operation": "add"},
        }
        self._surface_oit_pipeline = self._device.create_render_pipeline(
            layout=self._device.create_pipeline_layout(
                bind_group_layouts=[self._proj_bgl]
            ),
            vertex={
                "module": oit_shader,
                "entry_point": "vs_main",
                "buffers": [SURFACE_VERTEX_LAYOUT],
            },
            fragment={
                "module": oit_shader,
                "entry_point": "fs_main",
                "targets": [
                    {"format": wgpu.TextureFormat.rgba16float, "blend": _accum_blend},
                    {"format": wgpu.TextureFormat.rgba16float, "blend": _reveal_blend},
                ],
            },
            primitive={"topology": "triangle-list", "cull_mode": "none"},
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": False,
                "depth_compare": "less",
                "stencil_front": {"compare": "always", "fail_op": "keep", "depth_fail_op": "keep", "pass_op": "keep"},
                "stencil_back":  {"compare": "always", "fail_op": "keep", "depth_fail_op": "keep", "pass_op": "keep"},
                "stencil_read_mask": 0,
                "stencil_write_mask": 0,
            },
            multisample={"count": 1, "mask": 0xFFFF_FFFF, "alpha_to_coverage_enabled": False},
        )

        # ── OIT composition pipeline ───────────────────────────────────────
        compose_path = Path(__file__).parent / "shaders" / "oit_compose.wgsl"
        compose_shader = self._device.create_shader_module(
            code=compose_path.read_text(encoding="utf-8")
        )
        self._oit_compose_bgl = self._device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": "unfilterable-float",
                        "view_dimension": "2d",
                        "multisampled": False,
                    },
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": "unfilterable-float",
                        "view_dimension": "2d",
                        "multisampled": False,
                    },
                },
            ]
        )
        _compose_blend = {
            "color": {"src_factor": "src-alpha", "dst_factor": "one-minus-src-alpha", "operation": "add"},
            "alpha": {"src_factor": "one",       "dst_factor": "one",                 "operation": "add"},
        }
        self._oit_compose_pipeline = self._device.create_render_pipeline(
            layout=self._device.create_pipeline_layout(
                bind_group_layouts=[self._oit_compose_bgl]
            ),
            vertex={"module": compose_shader, "entry_point": "vs_main", "buffers": []},
            fragment={
                "module": compose_shader,
                "entry_point": "fs_main",
                "targets": [{"format": wgpu.TextureFormat.bgra8unorm, "blend": _compose_blend}],
            },
            primitive={"topology": "triangle-list", "cull_mode": "none"},
            multisample={"count": 1, "mask": 0xFFFF_FFFF, "alpha_to_coverage_enabled": False},
        )
        self._oit_compose_bind_group = self._device.create_bind_group(
            layout=self._oit_compose_bgl,
            entries=[
                {"binding": 0, "resource": self._oit_accum_view},
                {"binding": 1, "resource": self._oit_reveal_view},
            ],
        )

    def _create_slug_fill_pipeline(
        self,
        depth_test: bool = False,
    ) -> tuple[wgpu_t.GPUBindGroupLayout, wgpu_t.GPURenderPipeline]:
        """Create a Slug fill pipeline.

        depth_test controls depth *writing* only — both 2-D and 3-D fills
        always depth-test (depth_compare="less") so they are occluded by any
        opaque surface rendered before them.

        depth_test=False — 2-D fills: depth-read-only (default).
        depth_test=True  — 3-D fills (shade_in_3d): depth-write + depth-test
                           so they occlude geometry drawn behind them.
        """
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
                "targets": [{"format": wgpu.TextureFormat.bgra8unorm, "blend": _blend}],
            },
            primitive={"topology": "triangle-list", "cull_mode": "none"},
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": depth_test,
                "depth_compare": "less",  # always depth-test; write only for 3-D fills
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

    def _pack_camera_uniforms(
        self,
        proj: np.ndarray,
        view: np.ndarray,
    ) -> wgpu_t.GPUBuffer:
        """Pack a 176-byte camera+lighting uniform buffer from explicit proj/view matrices.

        Layout (matches Uniforms struct in surface.wgsl / surface_oit.wgsl):
          offset   0 — projection        mat4x4<f32>  64 B
          offset  64 — view              mat4x4<f32>  64 B
          offset 128 — light_pos         vec3<f32>    12 B
          offset 140 — light_intensity   f32           4 B
          offset 144 — light_color       vec3<f32>    12 B
          offset 156 — ambient_intensity f32           4 B
          offset 160 — ambient_color     vec3<f32>    12 B
          offset 172 — _pad              f32           4 B

        Called by _build_camera_bind_group for each of the three per-frame
        variants: normal, fixed-orientation, and fixed-in-frame.
        """
        assert self._device is not None

        proj_bytes = proj.T.flatten().astype(np.float32).tobytes()
        view_bytes = view.T.flatten().astype(np.float32).tobytes()

        # block A: light_pos (xyz) + light_intensity (w)
        block_a = np.zeros(4, dtype=np.float32)
        block_a[:3] = np.asarray(self.light_source_position, dtype=np.float32)
        block_a[3]  = np.float32(self.light_intensity)

        # block B: light_color (xyz) + ambient_intensity (w)
        block_b = np.zeros(4, dtype=np.float32)
        block_b[:3] = np.asarray(self.light_color, dtype=np.float32)
        block_b[3]  = np.float32(self.ambient_intensity)

        # block C: ambient_color (xyz) + _pad (w)
        block_c = np.zeros(4, dtype=np.float32)
        block_c[:3] = np.asarray(self.ambient_color, dtype=np.float32)

        buf = self._device.create_buffer_with_data(
            data=(proj_bytes + view_bytes
                  + block_a.tobytes() + block_b.tobytes() + block_c.tobytes()),
            usage=wgpu.BufferUsage.UNIFORM,
        )
        self.frame_vbos.append(buf)
        return buf

    def _build_camera_uniform_buf(self) -> wgpu_t.GPUBuffer:
        """Pack the 176-byte camera+lighting uniform with the current view/projection."""
        return self._pack_camera_uniforms(
            self.camera.projection_matrix,
            self.camera.view_matrix,
        )

    def _build_camera_bind_group(self) -> wgpu_t.GPUBindGroup:
        """Build all three per-frame camera bind groups.

        normal (camera_bind_group)
            Full camera rotation + current projection.  Used for all regular
            mobjects.

        fixed_camera_bind_group
            Rotation-stripped view + current projection.  Used for
            fixed-orientation mobjects: they don't tilt with the camera but
            are still depth-sorted with the rest of the scene.

        fixed_frame_bind_group
            Rotation-stripped view + forced orthographic projection.  Used for
            fixed-in-frame mobjects: 2-D overlays rendered after the 3-D scene
            with a fresh depth buffer so they always appear on top.
        """
        assert self._device is not None
        assert self._proj_bgl is not None

        def _make_bg(buf: wgpu_t.GPUBuffer) -> wgpu_t.GPUBindGroup:
            return self._device.create_bind_group(
                layout=self._proj_bgl,
                entries=[{"binding": 0, "resource": {"buffer": buf, "offset": 0, "size": 176}}],
            )

        # Normal bind group
        self._camera_uniform_buf = self._build_camera_uniform_buf()
        normal_bg = _make_bg(self._camera_uniform_buf)

        fixed_view = self.camera.fixed_view_matrix

        # Fixed-orientation: rotation-stripped view, same projection as scene
        self._fixed_orient_uniform_buf = self._pack_camera_uniforms(
            self.camera.projection_matrix, fixed_view
        )
        self.fixed_camera_bind_group = _make_bg(self._fixed_orient_uniform_buf)

        # Fixed-in-frame: rotation-stripped view, always orthographic
        self._fixed_frame_uniform_buf = self._pack_camera_uniforms(
            self.camera.ortho_projection_matrix, fixed_view
        )
        self.fixed_frame_bind_group = _make_bg(self._fixed_frame_uniform_buf)

        return normal_bg

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
                {"binding": 0, "resource": {"buffer": self._camera_uniform_buf, "offset": 0, "size": 176}},
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
    def stroke_3d_pipeline(self) -> wgpu_t.GPURenderPipeline:
        assert self._stroke_3d_pipeline is not None, "init_scene() has not been called"
        return self._stroke_3d_pipeline

    @property
    def stroke_3d_surface_pipeline(self) -> wgpu_t.GPURenderPipeline:
        """3-D stroke pipeline with depth bias — for surface mesh lines."""
        assert self._stroke_3d_surface_pipeline is not None, "init_scene() has not been called"
        return self._stroke_3d_surface_pipeline

    @property
    def surface_pipeline(self) -> wgpu_t.GPURenderPipeline:
        """Opaque surface pipeline (cull_back, depth_write=True)."""
        assert self._surface_pipeline is not None, "init_scene() has not been called"
        return self._surface_pipeline

    @property
    def surface_oit_pipeline(self) -> wgpu_t.GPURenderPipeline:
        assert self._surface_oit_pipeline is not None, "init_scene() has not been called"
        return self._surface_oit_pipeline

    @property
    def slug_fill_pipeline(self) -> wgpu_t.GPURenderPipeline:
        assert self._slug_fill_pipeline is not None, "init_scene() has not been called"
        return self._slug_fill_pipeline

    @property
    def slug_fill_3d_pipeline(self) -> wgpu_t.GPURenderPipeline:
        assert self._slug_fill_3d_pipeline is not None, "init_scene() has not been called"
        return self._slug_fill_3d_pipeline

    # ------------------------------------------------------------------
    # Frame rendering
    # ------------------------------------------------------------------

    def update_frame(self, scene: Scene) -> None:
        """Render one frame into the offscreen texture.

        Pass structure
        --------------
        1. **Main pass** — clears the frame; draws normal slug fills, opaque
           surfaces, strokes.  Fixed-orientation mobjects are drawn at the end
           of this pass with a rotation-stripped camera bind group so they share
           the same depth buffer as the rest of the scene.
        2. **OIT accumulation pass** — if any normal surface has alpha < 0.99,
           renders those fragments into two OIT accumulation textures (rgba16float)
           with Weighted Blended blending.
        3. **OIT composition pass** — full-screen triangle composites OIT result
           onto the main texture.
        4. **Fixed-in-frame overlay pass** — only if fixed-in-frame mobjects exist.
           Loads existing colour, clears depth, and renders overlays with a
           rotation-stripped orthographic camera so they always appear on top.
        """
        assert self._device is not None
        assert self._render_texture_view is not None
        assert self._depth_texture_view is not None

        bg = self._background_color

        # Build all three camera bind groups for this frame.
        # camera_bind_group          — normal rotated view  (set on renderer for vmobject_rendering)
        # fixed_camera_bind_group    — rotation-stripped, current projection (fixed-orientation)
        # fixed_frame_bind_group     — rotation-stripped, ortho projection   (fixed-in-frame)
        self.camera_bind_group = self._build_camera_bind_group()
        self.frame_vbos = []

        # ── Partition mobjects ────────────────────────────────────────────
        cam = self.camera
        fixed_in_frame = cam._fixed_in_frame_mobjects
        fixed_orient   = cam._fixed_orientation_mobjects

        normal_mobs       = [m for m in scene.mobjects if m not in fixed_in_frame and m not in fixed_orient]
        fixed_orient_mobs = [m for m in scene.mobjects if m in fixed_orient]
        fixed_frame_mobs  = [m for m in scene.mobjects if m in fixed_in_frame]

        encoder = self._device.create_command_encoder()

        # ── Pass 1: main ──────────────────────────────────────────────────
        # Renders normal mobjects and fixed-orientation mobjects (same depth
        # buffer; fixed-orient uses the rotation-stripped bind group).
        main_pass = encoder.begin_render_pass(
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
                "depth_store_op": "store",
            },
        )
        self.current_render_pass = main_pass
        oit_data = render_webgpu_mobject(self, normal_mobs)

        # Fixed-orientation: same pass, swap to rotation-stripped bind group.
        # The normal bind group and uniform buffer are saved and restored so
        # the OIT accumulation pass (below) still uses the correct camera.
        if fixed_orient_mobs:
            _saved_bg  = self.camera_bind_group
            _saved_buf = self._camera_uniform_buf
            self.camera_bind_group  = self.fixed_camera_bind_group
            self._camera_uniform_buf = self._fixed_orient_uniform_buf
            render_webgpu_mobject(self, fixed_orient_mobs)
            self.camera_bind_group  = _saved_bg
            self._camera_uniform_buf = _saved_buf

        main_pass.end()

        # ── Pass 2: OIT accumulation ──────────────────────────────────────
        if oit_data is not None:
            oit_pass = encoder.begin_render_pass(
                color_attachments=[
                    {
                        "view": self._oit_accum_view,
                        "load_op": "clear",
                        "store_op": "store",
                        "clear_value": (0.0, 0.0, 0.0, 0.0),
                    },
                    {
                        "view": self._oit_reveal_view,
                        "load_op": "clear",
                        "store_op": "store",
                        "clear_value": (1.0, 1.0, 1.0, 1.0),
                    },
                ],
                depth_stencil_attachment={
                    "view": self._depth_texture_view,
                    "depth_load_op": "load",    # read depth from main pass
                    "depth_store_op": "discard",
                },
            )
            oit_pass.set_pipeline(self.surface_oit_pipeline)
            oit_pass.set_bind_group(0, self.camera_bind_group, [], 0, 0)
            for idx in oit_data.oit_indices:
                arr = oit_data.surface_parts[idx]
                oit_pass.set_vertex_buffer(
                    0, oit_data.surface_buf,
                    oit_data.byte_offsets[idx], arr.nbytes,
                )
                oit_pass.draw(len(arr), 1, 0, 0)
            oit_pass.end()

            # ── Pass 3: OIT composition ───────────────────────────────────
            compose_pass = encoder.begin_render_pass(
                color_attachments=[
                    {
                        "view": self._render_texture_view,
                        "load_op": "load",
                        "store_op": "store",
                    }
                ],
            )
            compose_pass.set_pipeline(self._oit_compose_pipeline)
            compose_pass.set_bind_group(0, self._oit_compose_bind_group, [], 0, 0)
            compose_pass.draw(3, 1, 0, 0)
            compose_pass.end()

        # ── Fixed-in-frame overlay pass ───────────────────────────────────
        # Rendered last, after OIT composition, so overlays always appear on
        # top of the 3-D scene.  The depth buffer is cleared to 1.0 (far) and
        # discarded afterward — fixed-in-frame objects only depth-test against
        # each other, not against the main scene.
        if fixed_frame_mobs:
            fixed_pass = encoder.begin_render_pass(
                color_attachments=[
                    {
                        "view": self._render_texture_view,
                        "load_op": "load",   # preserve the composited 3-D scene
                        "store_op": "store",
                    }
                ],
                depth_stencil_attachment={
                    "view": self._depth_texture_view,
                    "depth_clear_value": 1.0,
                    "depth_load_op": "clear",   # fresh depth — overlays on top
                    "depth_store_op": "discard",
                },
            )
            self.current_render_pass = fixed_pass
            self.camera_bind_group  = self.fixed_frame_bind_group
            self._camera_uniform_buf = self._fixed_frame_uniform_buf
            render_webgpu_mobject(self, fixed_frame_mobs)
            fixed_pass.end()

        self._device.queue.submit([encoder.finish()])

        self.current_render_pass = None
        self.camera_bind_group = None
        self.frame_vbos = []

        self.animation_elapsed_time = time.time() - self.animation_start_time

    # ------------------------------------------------------------------
    # Frame readback
    # ------------------------------------------------------------------

    def _create_readback_pipeline(self, width: int, height: int) -> None:
        """Create the compact-readback compute pipeline and its persistent buffers.

        The compute shader (readback_compact.wgsl) reads every pixel from the
        bgra8unorm render texture and writes tightly-packed RGBA u32 values into
        a storage buffer — eliminating two CPU operations that previously ran on
        every frame:

          * Row-padding strip  — copy_texture_to_buffer requires bytes_per_row
            to be a multiple of 256; the shader writes directly to tight index
            ``y * width + x``, so the output is already compact.

          * B↔R channel swap  — textureLoad() returns components as (r, g, b, a)
            regardless of the bgra physical layout, so the output is already in
            RGBA byte order.
        """
        assert self._device is not None
        assert self._render_texture_view is not None

        shader_path = Path(__file__).parent / "shaders" / "readback_compact.wgsl"
        shader = self._device.create_shader_module(
            code=shader_path.read_text(encoding="utf-8")
        )

        self._readback_compute_bgl = self._device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "texture": {
                        "sample_type": "float",
                        "view_dimension": "2d",
                        "multisampled": False,
                    },
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "storage"},
                },
            ]
        )

        self._readback_compute_pipeline = self._device.create_compute_pipeline(
            layout=self._device.create_pipeline_layout(
                bind_group_layouts=[self._readback_compute_bgl]
            ),
            compute={"module": shader, "entry_point": "main"},
        )

        packed_size = width * height * 4
        self._readback_storage_buf = self._device.create_buffer(
            size=packed_size,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        self._readback_map_buf = self._device.create_buffer(
            size=packed_size,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )
        self._readback_compute_bind_group = self._device.create_bind_group(
            layout=self._readback_compute_bgl,
            entries=[
                {"binding": 0, "resource": self._render_texture_view},
                {
                    "binding": 1,
                    "resource": {
                        "buffer": self._readback_storage_buf,
                        "offset": 0,
                        "size": packed_size,
                    },
                },
            ],
        )

    def _get_raw_frame_data(self) -> bytes:
        """Readback the current frame as tightly-packed RGBA bytes.

        A compute shader (readback_compact.wgsl) handles both row-depadding and
        the bgra→rgba channel fix on the GPU.  The CPU no longer needs to loop
        over rows or touch a numpy channel-swap.
        """
        assert self._device is not None
        assert self._readback_compute_pipeline is not None
        assert self._readback_compute_bind_group is not None
        assert self._readback_storage_buf is not None
        assert self._readback_map_buf is not None

        width  = config.pixel_width
        height = config.pixel_height
        packed_size = width * height * 4

        encoder = self._device.create_command_encoder()

        # Compact pass: row-depad + bgra→rgba in one GPU dispatch.
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(self._readback_compute_pipeline)
        compute_pass.set_bind_group(0, self._readback_compute_bind_group)
        compute_pass.dispatch_workgroups(
            (width  + 15) // 16,
            (height + 15) // 16,
        )
        compute_pass.end()

        # Copy packed storage buffer → mappable buffer.
        encoder.copy_buffer_to_buffer(
            self._readback_storage_buf, 0,
            self._readback_map_buf,     0,
            packed_size,
        )

        self._device.queue.submit([encoder.finish()])

        self._readback_map_buf.map_sync(wgpu.MapMode.READ)
        raw = bytes(self._readback_map_buf.read_mapped())
        self._readback_map_buf.unmap()
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
    # Window helpers
    # ------------------------------------------------------------------

    def should_create_window(self) -> bool:
        """Mirror of ``OpenGLRenderer.should_create_window``.

        A preview window is opened when ``--preview`` is active and the
        renderer is not writing a movie or saving a still frame.
        """
        if config["force_window"]:
            logger.warning(
                "'--force_window' is enabled; this is intended for debugging "
                "and may impact performance when combined with file output.",
            )
            return True
        return (
            config["preview"]
            and not config["save_last_frame"]
            and not config["format"]
            and not config["write_to_movie"]
            and not config["dry_run"]
        )

    def pixel_coords_to_space_coords(
        self,
        px: float,
        py: float,
        relative: bool = False,
        top_left: bool = False,
    ) -> np.ndarray:
        """Convert pixel coordinates to Manim scene-space coordinates.

        Parameters
        ----------
        px, py:
            Pixel position.  For ``relative=False``, these are absolute
            pixel coordinates within the render texture.
        relative:
            When True, treat *px*/*py* as a delta and return the
            corresponding scene-space delta (normalised to ``[-1, 1]``
            then scaled).
        top_left:
            When True (the default for ``rendercanvas``), the origin is
            at the top-left corner; y increases downward.
        """
        pixel_width  = config.pixel_width
        pixel_height = config.pixel_height
        frame_height = config.frame_height
        frame_center = self.camera.get_center()

        if relative:
            return 2.0 * np.array([px / pixel_width, py / pixel_height, 0.0])

        scale = frame_height / pixel_height
        y_direction = -1 if top_left else 1
        return (
            frame_center
            + scale
            * np.array(
                [(px - pixel_width / 2), y_direction * (py - pixel_height / 2), 0.0]
            )
        )

    # ------------------------------------------------------------------
    # Scene rendering
    # ------------------------------------------------------------------

    def render(self, scene: Scene, frame_offset: float, moving_mobjects: list) -> None:
        self.update_frame(scene)
        if self.skip_animations:
            return
        self.file_writer.write_frame(self)
        if self.window is not None:
            self.window.present()
            while self.animation_elapsed_time < frame_offset:
                if self.window.is_closing:
                    break
                self.update_frame(scene)
                self.window.present()

    def play(self, scene: Scene, *animations: Any, **kwargs: Any) -> None:
        self.animation_start_time = time.time()
        self.skip_animations = self._original_skipping_status
        self.update_skipping_status()

        # Compile first so we can compute a real hash (same order as CairoRenderer).
        scene.compile_animation_data(*animations, **kwargs)

        if self.skip_animations:
            hash_current_animation = None
            self.time += scene.duration
        elif config["disable_caching"]:
            hash_current_animation = f"uncached_{self.num_plays:05}"
        else:
            assert scene.animations is not None
            hash_current_animation = get_hash_from_play_call(
                scene, self.camera, scene.animations, scene.mobjects
            )
            if self.file_writer.is_already_cached(hash_current_animation):
                logger.info(
                    "Animation %d: using cached data (hash: %s)",
                    self.num_plays,
                    hash_current_animation,
                )
                self.skip_animations = True
                self.time += scene.duration

        self.animations_hashes.append(hash_current_animation)
        self.file_writer.add_partial_movie_file(hash_current_animation)

        self.file_writer.begin_animation(not self.skip_animations)
        scene.begin_animations()

        if scene.is_current_animation_frozen_frame():
            self.update_frame(scene)
            if not self.skip_animations:
                self.file_writer.write_frame(
                    self, num_frames=int(config.frame_rate * scene.duration)
                )
            if self.window is not None:
                self.window.present()
                while time.time() - self.animation_start_time < scene.duration:
                    if self.window.is_closing:
                        break
                    self.window.present()
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
        if self.window is not None:
            self.window.present()

    # ------------------------------------------------------------------
    # Skipping helpers
    # ------------------------------------------------------------------

    def update_skipping_status(self) -> None:
        """Check and update ``skip_animations`` for the current animation.

        Mirrors ``CairoRenderer.update_skipping_status`` and
        ``OpenGLRenderer.update_skipping_status`` so the WebGPU renderer
        honours the same configuration knobs:

        * ``file_writer.sections[-1].skip_animations`` — section-level skip
          (e.g. the section was marked skip via ``scene.next_section``).
        * ``config.save_last_frame`` — only the final frame matters; all
          intermediate animation frames can be skipped.
        * ``config.from_animation_number`` — skip animations before the given
          index (useful for scrubbing to a specific animation).
        * ``config.upto_animation_number`` — stop rendering after the given
          index and raise ``EndSceneEarlyException``.
        """
        # there is always at least one section → no out-of-bounds here
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
