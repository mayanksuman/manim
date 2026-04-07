"""Preview window for the WebGPU renderer.

Uses rendercanvas (bundled with wgpu-py) to open a native OS window.
The offscreen render texture (``bgra8unorm``) is copied directly to the
window surface via ``copy_texture_to_texture`` — no blit shader needed
because both textures share the same format.

Event mapping
-------------
rendercanvas uses the Web standard key-name strings (``"ArrowLeft"``,
``"q"``, ...).  Manim scene callbacks expect pyglet-compatible integer
key codes.  A small mapping table is included below.  Single printable
characters are mapped with ``ord()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from manim import __version__, config

if TYPE_CHECKING:
    from .webgpu_renderer import WebGPURenderer

try:
    import wgpu
    from rendercanvas.auto import RenderCanvas
except ImportError as exc:
    raise ImportError(
        "wgpu-py (with rendercanvas) is required for the preview window. "
        "Install it with:  pip install wgpu"
    ) from exc


# ---------------------------------------------------------------------------
# Key / modifier mapping
# ---------------------------------------------------------------------------

# rendercanvas key strings → pyglet-compatible integer codes.
# Printable single chars use ord() directly (see _key_to_int below).
_SPECIAL_KEY_MAP: dict[str, int] = {
    "ArrowLeft":   65361,
    "ArrowRight":  65363,
    "ArrowUp":     65362,
    "ArrowDown":   65364,
    "Escape":      65307,
    "Enter":       65293,
    "Backspace":   65288,
    "Tab":         65289,
    "Delete":      65535,
    "Home":        65360,
    "End":         65367,
    "PageUp":      65365,
    "PageDown":    65366,
    "Insert":      65379,
    "F1":  65470, "F2":  65471, "F3":  65472, "F4":  65473,
    "F5":  65474, "F6":  65475, "F7":  65476, "F8":  65477,
    "F9":  65478, "F10": 65479, "F11": 65480, "F12": 65481,
    "Shift":      65505,  # SHIFT_VALUE in manim/constants.py
    "Control":    65507,
    "Alt":        65513,
    "Meta":       65511,
    "CapsLock":   65509,
    "NumLock":    65407,
    "ScrollLock": 65300,
}

# rendercanvas modifier strings → pyglet modifier bitmask bits
_MODIFIER_BITS: dict[str, int] = {
    "Shift":   1,
    "Control": 4,
    "Alt":     8,
    "Meta":    16,
}


def _key_to_int(key: str) -> int:
    """Convert a rendercanvas key string to a pyglet-compatible integer."""
    if key in _SPECIAL_KEY_MAP:
        return _SPECIAL_KEY_MAP[key]
    if len(key) == 1:
        return ord(key)
    # Unknown multi-char key — use a stable hash to avoid collisions with
    # printable chars.
    return (hash(key) & 0x7FFF_FFFF) | 0x8000_0000


def _modifiers_to_int(modifiers: tuple | list) -> int:
    """Convert a rendercanvas modifiers collection to a pyglet modifier int."""
    result = 0
    for m in modifiers:
        result |= _MODIFIER_BITS.get(m, 0)
    return result


# ---------------------------------------------------------------------------
# Window size helper
# ---------------------------------------------------------------------------

def _compute_window_size() -> tuple[int, int]:
    """Return ``(width, height)`` for the preview window in logical pixels.

    Defaults to ``(pixel_width, pixel_height)`` so that the surface texture
    produced by the canvas always matches the offscreen render texture,
    making ``copy_texture_to_texture`` safe without any size bookkeeping.
    """
    return config.pixel_width, config.pixel_height


# ---------------------------------------------------------------------------
# Window class
# ---------------------------------------------------------------------------

class WebGPUWindow:
    """Preview window wrapping a ``rendercanvas.RenderCanvas``.

    Each call to :meth:`present` polls OS events and blits the offscreen
    render texture to the window surface via ``copy_texture_to_texture``.

    Interface expected by ``scene.py`` and ``WebGPURenderer``:

    * ``is_closing`` — True once the user closes the window
    * ``destroy()``  — tear down the underlying canvas
    """

    def __init__(self, renderer: WebGPURenderer) -> None:
        self._renderer = renderer

        win_w, win_h = _compute_window_size()
        self._canvas = RenderCanvas(
            size=(win_w, win_h),
            title=f"Manim Community {__version__}",
            # "manual" means we call force_draw() ourselves; the scheduler
            # never schedules draws on its own.
            update_mode="manual",
        )

        # Configure the wgpu context.
        # bgra8unorm matches the render texture format → copy_texture_to_texture
        # works without any format conversion.
        # COPY_DST is needed on the surface texture so we can copy into it.
        self._context = self._canvas.get_wgpu_context()
        self._context.configure(
            device=renderer._device,
            format=wgpu.TextureFormat.bgra8unorm,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_DST,
        )

        # Register the draw callback (executed inside the rendercanvas lifecycle
        # on every force_draw() call).
        self._canvas.request_draw(self._draw_frame)

        # Register event handlers.
        self._canvas.add_event_handler(self._on_key_down,     "key_down")
        self._canvas.add_event_handler(self._on_key_up,       "key_up")
        self._canvas.add_event_handler(self._on_pointer_move, "pointer_move")
        self._canvas.add_event_handler(self._on_pointer_down, "pointer_down")
        self._canvas.add_event_handler(self._on_wheel,        "wheel")

    # ------------------------------------------------------------------
    # Public interface (consumed by WebGPURenderer and scene.py)
    # ------------------------------------------------------------------

    @property
    def is_closing(self) -> bool:
        """True once the OS window has been closed."""
        return self._canvas.get_closed()

    def destroy(self) -> None:
        """Close the OS window."""
        self._canvas.close()

    def present(self) -> None:
        """Poll OS events and blit the current render texture to the window.

        Call this after every :meth:`~WebGPURenderer.update_frame` that
        should be visible in the preview window.
        """
        # Process pending OS events (keyboard, mouse, resize, close …).
        self._canvas._process_events()
        # Trigger _draw_frame → copy_texture_to_texture → present to screen.
        self._canvas.force_draw()

    # ------------------------------------------------------------------
    # Draw callback (runs inside the rendercanvas present lifecycle)
    # ------------------------------------------------------------------

    def _draw_frame(self) -> None:
        """Copy the offscreen render texture to the window surface texture."""
        renderer = self._renderer
        if renderer._render_texture is None or renderer._device is None:
            return

        surface_tex = self._context.get_current_texture()
        w = config.pixel_width
        h = config.pixel_height

        encoder = renderer._device.create_command_encoder()
        encoder.copy_texture_to_texture(
            {"texture": renderer._render_texture, "mip_level": 0, "origin": (0, 0, 0)},
            {"texture": surface_tex,              "mip_level": 0, "origin": (0, 0, 0)},
            (w, h, 1),
        )
        renderer._device.queue.submit([encoder.finish()])

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_key_down(self, event: dict) -> None:
        key = event.get("key", "")
        symbol = _key_to_int(key)
        self._renderer.pressed_keys.add(symbol)
        # scene.on_key_press asserts OpenGLCamera/Renderer — skip for WebGPU.

    def _on_key_up(self, event: dict) -> None:
        key = event.get("key", "")
        symbol = _key_to_int(key)
        self._renderer.pressed_keys.discard(symbol)

    def _on_pointer_move(self, event: dict) -> None:
        point = self._renderer.pixel_coords_to_space_coords(
            event["x"], event["y"], top_left=True
        )
        d_point = self._renderer.pixel_coords_to_space_coords(
            event.get("dx", 0), event.get("dy", 0), relative=True
        )
        # scene.on_mouse_motion asserts OpenGLCamera — skip for WebGPU.
        _ = point, d_point  # suppress unused-var warnings

    def _on_pointer_down(self, event: dict) -> None:
        point = self._renderer.pixel_coords_to_space_coords(
            event["x"], event["y"], top_left=True
        )
        _ = point

    def _on_wheel(self, event: dict) -> None:
        point = self._renderer.pixel_coords_to_space_coords(
            event["x"], event["y"], top_left=True
        )
        _ = point
