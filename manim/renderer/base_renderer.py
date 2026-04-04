"""Structural interface shared by all Manim renderers.

Using ``typing.Protocol`` (rather than an ABC) means the existing
``CairoRenderer`` and ``OpenGLRenderer`` satisfy the interface automatically
without any inheritance changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, runtime_checkable

import numpy as np
from typing import Protocol

if TYPE_CHECKING:
    from PIL import Image

    from manim.scene.scene import Scene


@runtime_checkable
class RendererProtocol(Protocol):
    """Protocol that every Manim renderer must satisfy.

    ``scene.py`` accesses these attributes and methods on the renderer object.
    Declaring them here makes the contract explicit and enables static type
    checking without breaking existing renderer classes.
    """

    camera: Any
    skip_animations: bool
    num_plays: int
    time: float
    file_writer: Any
    window: Any
    animation_start_time: float
    static_image: Any

    def init_scene(self, scene: Scene) -> None: ...

    def play(self, scene: Scene, *args: Any, **kwargs: Any) -> None: ...

    def render(
        self, scene: Scene, frame_offset: float, moving_mobjects: list
    ) -> None: ...

    def update_frame(self, scene: Scene) -> None: ...

    def scene_finished(self, scene: Scene) -> None: ...

    def clear_screen(self) -> None: ...

    def get_image(self) -> Image.Image: ...

    def get_frame(self) -> np.ndarray: ...
