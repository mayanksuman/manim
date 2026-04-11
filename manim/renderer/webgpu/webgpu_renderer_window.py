"""Preview window for the WebGPU renderer.

Uses ``rendercanvas`` to open a native OS window.  The offscreen render
texture (``bgra8unorm``) is copied directly to the window surface via
``copy_texture_to_texture`` — no format conversion needed.

Interactive camera controls
---------------------------

========================  ================================================
Input                     Action
========================  ================================================
Left-drag                 **Orbit** — horizontal drag rotates theta (yaw);
                          vertical drag tilts phi (pitch), clamped to ±90°.
Right-drag / Middle-drag  **Pan** — translates the view laterally in camera
                          space; one screen-width drag = one frame width.
Scroll wheel              **Zoom** — perspective: ``focal_distance`` scales
                          exponentially (~12 % per notch); orthographic:
                          ``frame_shape`` scales by the same factor.
Key ``r``                 **Reset** — restores default orbit, zoom, and
                          clears the accumulated pan offset.
Key ``q``                 **Quit** — closes the preview window.
========================  ================================================

Pan state (``_pan_x``, ``_pan_y``) is stored on :class:`WebGPUWindow` and
injected into the camera just before each render; it is not part of the
scripted camera model.

Event mapping
-------------
rendercanvas delivers Web-standard key strings (``"ArrowLeft"``, ``"q"``
…).  A mapping table converts these to pyglet-compatible integer codes so
that ``scene.on_key_press`` callbacks work unchanged.
"""

from __future__ import annotations

import math
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
# Interactive camera sensitivity constants
# ---------------------------------------------------------------------------

# Wheel: zoom sensitivity.
# rendercanvas delivers wheel deltas in CSS pixels (≈100–120 per notch on
# most platforms).  A factor of 0.001 gives ≈10 % zoom per notch
# (exp(120 * 0.001) ≈ 1.13).
_ZOOM_SCROLL_FACTOR: float = 0.001

# Minimum/maximum focal distance (perspective only).
_ZOOM_MIN_FD: float = 0.5
_ZOOM_MAX_FD: float = 100.0

# Maximum pointer movement (in window pixels) between press and release that
# is still classified as a click rather than a drag.
_CLICK_THRESHOLD_PX: float = 4.0


# ---------------------------------------------------------------------------
# Window class
# ---------------------------------------------------------------------------

class WebGPUWindow:
    """Preview window wrapping a ``rendercanvas.RenderCanvas``.

    Each call to :meth:`present` polls OS events and blits the offscreen
    render texture to the window surface via ``copy_texture_to_texture``.

    Interface expected by ``scene.py`` and ``WebGPURenderer``:

    * ``is_closing`` — True once the OS window has been closed.
    * ``destroy()``  — tear down the underlying canvas.

    Interactive camera controls
    ---------------------------
    Mouse events are translated into camera mutations and an immediate
    re-render so the view updates in real time.  Pan state (``_pan_x``,
    ``_pan_y``) is owned by this class and pushed to the camera just before
    each render via :meth:`_sync_pan_to_camera`; orbit and zoom are applied
    directly to :class:`~.WebGPUCamera` fields.

    ========================  ================================================
    Input                     Action
    ========================  ================================================
    Left-drag                 **Orbit** — horizontal drag: ``increment_theta``;
                              vertical drag: ``increment_phi`` (clamped).
    Right-drag / Middle-drag  **Pan** — accumulates ``_pan_x`` / ``_pan_y``
                              proportional to ``frame_shape / pixel_size``.
    Scroll wheel              **Zoom** — perspective: ``focal_distance *= exp(dy
                              * 0.001)``; orthographic: ``frame_shape *= same``.
    Key ``r``                 **Reset** — clears ``_pan_x``, ``_pan_y`` and
                              calls ``camera.to_default_state()``.
    Key ``q``                 **Quit** — closes the preview window.
    ========================  ================================================

    Customising interaction
    -----------------------
    Subclass :class:`WebGPUWindow` and override any of the four interaction
    hooks to change behaviour without touching internal event dispatch:

    ========================  ================================================
    Method                    When called
    ========================  ================================================
    :meth:`on_mouse_drag`     Pointer moves while a button is held.
    :meth:`on_scroll`         Wheel (scroll) event.
    :meth:`on_key_press`      Key pressed down.
    :meth:`on_key_release`    Key released.
    :meth:`on_mouse_left_click`   Left button pressed and released in place.
    :meth:`on_mouse_right_click`  Right button pressed and released in place.
    ========================  ================================================

    Each hook can call the building-block helpers :meth:`orbit`, :meth:`pan`,
    and :meth:`zoom` and finish with :meth:`_render_from_window` to trigger a
    re-render.  Pass the subclass to the renderer via
    ``WebGPURenderer(window_class=MyWindow)``.

    Example — swap orbit and pan, double zoom speed::

        class MyWindow(WebGPUWindow):
            def on_mouse_drag(self, x, y, dx, dy, button):
                if button == 3:        # right-drag → orbit
                    self.orbit(dx, dy)
                elif button == 1:      # left-drag → pan
                    self.pan(dx, dy)
                else:
                    return
                self._render_from_window()

            def on_scroll(self, x, y, dy):
                self.zoom(dy * 2)      # 2× sensitivity
                self._render_from_window()

        renderer = WebGPURenderer(window_class=MyWindow)
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

        # ── Drag / click state ────────────────────────────────────────────
        # _drag_button: 1 = left (orbit), 2 = middle (pan), 3 = right (pan)
        # None when no button is held.
        self._drag_button: int | None = None
        self._last_px: float = 0.0
        self._last_py: float = 0.0
        # _press_x/y: pointer position at the moment the button was pressed,
        # used to distinguish a click (≤ _CLICK_THRESHOLD_PX movement) from a drag.
        self._press_x: float = 0.0
        self._press_y: float = 0.0

        # ── Pan state ─────────────────────────────────────────────────────
        # Camera-space lateral offset in scene units, accumulated from
        # right/middle-drag events.  Stored here (not on the camera) because
        # pan is interactive view navigation, not part of the scripted camera
        # model.  Synced to camera._cam_pan_x/y just before every render.
        self._pan_x: float = 0.0
        self._pan_y: float = 0.0

        # Register event handlers.
        self._canvas.add_event_handler(self._on_key_down,     "key_down")
        self._canvas.add_event_handler(self._on_key_up,       "key_up")
        self._canvas.add_event_handler(self._on_pointer_move, "pointer_move")
        self._canvas.add_event_handler(self._on_pointer_down, "pointer_down")
        self._canvas.add_event_handler(self._on_pointer_up,   "pointer_up")
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
    # Interactive camera helpers
    # ------------------------------------------------------------------

    def _sync_pan_to_camera(self) -> None:
        """Write the window's pan offset into the camera before rendering.

        ``WebGPUCamera.view_matrix`` reads ``_cam_pan_x`` / ``_cam_pan_y``
        via ``getattr`` so they don't need to be initialised in ``__init__``.
        This call makes the camera pick up the current window pan without
        the camera model needing to know about interactive navigation.
        """
        cam = self._renderer.camera
        cam._cam_pan_x = self._pan_x
        cam._cam_pan_y = self._pan_y

    def _render_from_window(self) -> None:
        """Sync pan, re-render the current scene, and present to the window.

        Called after every camera mutation so the preview updates immediately —
        even when no animation is running and the main loop is not calling
        ``update_frame`` in a tight loop.
        """
        renderer = self._renderer
        if renderer._device is None:
            return
        scene = getattr(renderer, "scene", None)
        if scene is None:
            return
        self._sync_pan_to_camera()
        renderer.update_frame(scene)
        self._canvas.force_draw()

    # ------------------------------------------------------------------
    # Overridable interaction building blocks
    # ------------------------------------------------------------------

    def orbit(self, dx: float, dy: float) -> None:
        """Rotate the camera by *dx* horizontal and *dy* vertical pixel deltas.

        Override to change orbit behaviour or sensitivity.

        Horizontal drag rotates around the vertical axis (theta).
        Vertical drag tilts up/down (phi, clamped to ±90°).

        A full horizontal swipe (pixel_width pixels) = one full revolution.
        A full vertical swipe (pixel_height pixels) = 180° tilt.

        Sign convention (rendercanvas y-axis points downward):
          * Drag right (dx > 0) → scene rotates to the right → theta increases.
          * Drag down  (dy > 0) → scene tilts down            → phi decreases.
        """
        cam = self._renderer.camera
        pw = max(config.pixel_width,  1)
        ph = max(config.pixel_height, 1)
        dtheta =  dx * (2.0 * math.pi / pw)
        dphi   = -dy * (math.pi       / ph)
        cam.increment_theta(dtheta)
        cam.increment_phi(dphi)

    def pan(self, dx: float, dy: float) -> None:
        """Translate the camera laterally by *dx* / *dy* pixel deltas.

        Override to change pan behaviour or sensitivity.

        Updates the window-owned pan state.  The new values are pushed to the
        camera by ``_render_from_window`` → ``_sync_pan_to_camera``.

        One full horizontal swipe (pixel_width pixels) shifts the scene by
        exactly one ``frame_width`` scene unit.

        Signs (rendercanvas y-axis downward; Manim y-axis upward):
          * Drag right (dx > 0) → scene moves right → _pan_x increases.
          * Drag down  (dy > 0) → scene moves down  → _pan_y decreases.
        """
        fw, fh = self._renderer.camera.frame_shape
        pw = max(config.pixel_width,  1)
        ph = max(config.pixel_height, 1)
        self._pan_x += dx * (fw / pw)
        self._pan_y -= dy * (fh / ph)

    def zoom(self, scroll_dy: float) -> None:
        """Zoom in/out by *scroll_dy* CSS-pixel scroll units.

        Override to change zoom behaviour, sensitivity, or limits.

        Positive *scroll_dy* (scroll down) zooms out; negative zooms in.

        Perspective camera: adjusts ``focal_distance``.
        Orthographic camera: scales ``frame_shape`` proportionally.

        The zoom is exponential so that successive zoom steps are perceptually
        uniform: each notch (≈ 120 CSS pixels) changes the scale by ~12 %.
        """
        cam = self._renderer.camera
        factor = math.exp(scroll_dy * _ZOOM_SCROLL_FACTOR)

        if cam.orthographic:
            fw, fh = cam.frame_shape
            new_fw = max(fw * factor, 0.01)
            new_fh = max(fh * factor, 0.01)
            cam.frame_shape = (new_fw, new_fh)
        else:
            new_fd = float(np.clip(
                cam.focal_distance * factor,
                _ZOOM_MIN_FD,
                _ZOOM_MAX_FD,
            ))
            cam.set_focal_distance(new_fd)

    # ------------------------------------------------------------------
    # Overridable interaction hooks
    # ------------------------------------------------------------------

    def on_mouse_drag(self, x: float, y: float, dx: float, dy: float, button: int) -> None:
        """Called on every pointer-move event while a mouse button is held.

        Override to customise drag behaviour.  The default implementation
        maps button 1 (left) to :meth:`orbit` and buttons 2/3
        (middle/right) to :meth:`pan`.

        Parameters
        ----------
        x, y:
            Current pointer position in window pixels (y-axis downward).
        dx, dy:
            Delta from the previous pointer position in window pixels.
        button:
            Web-standard button code: 1 = left, 2 = middle, 3 = right.
        """
        if button == 1:
            self.orbit(dx, dy)
        elif button in (2, 3):
            self.pan(dx, dy)
        else:
            return
        self._render_from_window()

    def on_scroll(self, x: float, y: float, dy: float) -> None:
        """Called on every wheel (scroll) event.

        Override to customise scroll behaviour.  The default implementation
        calls :meth:`zoom`.

        Parameters
        ----------
        x, y:
            Pointer position at the time of the scroll, in window pixels.
        dy:
            Vertical scroll delta in CSS pixels (positive = scroll down =
            zoom out).
        """
        self.zoom(dy)
        self._render_from_window()

    def on_key_press(self, key: str, modifiers: int) -> None:
        """Called when a key is pressed.

        Override to add or replace key bindings.  Call ``super().on_key_press(key,
        modifiers)`` to keep the default ``r`` → reset and ``q`` → quit bindings.

        Parameters
        ----------
        key:
            Web-standard key string (e.g. ``"r"``, ``"ArrowLeft"``,
            ``"Escape"``).
        modifiers:
            Pyglet-compatible modifier bitmask (Shift=1, Control=4, Alt=8).
        """
        if key == "r":
            self._pan_x = 0.0
            self._pan_y = 0.0
            self._renderer.camera.to_default_state()
            self._render_from_window()
        elif key == "q":
            self._canvas.close()

    def on_key_release(self, key: str, modifiers: int) -> None:
        """Called when a key is released.

        Override to react to key-release events.  The default implementation
        does nothing (key tracking is handled internally).

        Parameters
        ----------
        key:
            Web-standard key string.
        modifiers:
            Pyglet-compatible modifier bitmask.
        """

    def on_mouse_left_click(self, x: float, y: float) -> None:
        """Called when the left mouse button is clicked (pressed and released
        without dragging more than ``_CLICK_THRESHOLD_PX`` pixels).

        Override to add left-click behaviour.  The default implementation
        does nothing.

        Parameters
        ----------
        x, y:
            Pointer position at release, in window pixels (y-axis downward).
        """

    def on_mouse_right_click(self, x: float, y: float) -> None:
        """Called when the right mouse button is clicked (pressed and released
        without dragging more than ``_CLICK_THRESHOLD_PX`` pixels).

        Override to add right-click behaviour.  The default implementation
        does nothing.

        Parameters
        ----------
        x, y:
            Pointer position at release, in window pixels (y-axis downward).
        """

    # ------------------------------------------------------------------
    # Raw event handlers — dispatch to the overridable hooks above.
    # Subclasses should override the hooks, not these methods.
    # ------------------------------------------------------------------

    def _on_key_down(self, event: dict) -> None:
        key = event.get("key", "")
        modifiers = _modifiers_to_int(event.get("modifiers", []))
        self._renderer.pressed_keys.add(_key_to_int(key))
        self.on_key_press(key, modifiers)

    def _on_key_up(self, event: dict) -> None:
        key = event.get("key", "")
        modifiers = _modifiers_to_int(event.get("modifiers", []))
        self._renderer.pressed_keys.discard(_key_to_int(key))
        self.on_key_release(key, modifiers)

    def _on_pointer_down(self, event: dict) -> None:
        # button: 1 = left, 2 = middle, 3 = right (Web standard)
        self._drag_button = event.get("button", 1)
        x = float(event.get("x", 0))
        y = float(event.get("y", 0))
        self._last_px = x
        self._last_py = y
        self._press_x = x
        self._press_y = y

    def _on_pointer_up(self, event: dict) -> None:
        button = self._drag_button
        x = float(event.get("x", 0))
        y = float(event.get("y", 0))
        self._drag_button = None
        # Fire a click hook only when the pointer barely moved (not a drag).
        if math.hypot(x - self._press_x, y - self._press_y) < _CLICK_THRESHOLD_PX:
            if button == 1:
                self.on_mouse_left_click(x, y)
            elif button == 3:
                self.on_mouse_right_click(x, y)

    def _on_pointer_move(self, event: dict) -> None:
        if self._drag_button is None:
            return
        px = float(event.get("x", 0))
        py = float(event.get("y", 0))
        dx = px - self._last_px
        dy = py - self._last_py
        self._last_px = px
        self._last_py = py
        if dx == 0.0 and dy == 0.0:
            return
        self.on_mouse_drag(px, py, dx, dy, self._drag_button)

    def _on_wheel(self, event: dict) -> None:
        dy = float(event.get("dy", 0))
        if dy == 0.0:
            return
        self.on_scroll(float(event.get("x", 0)), float(event.get("y", 0)), dy)
