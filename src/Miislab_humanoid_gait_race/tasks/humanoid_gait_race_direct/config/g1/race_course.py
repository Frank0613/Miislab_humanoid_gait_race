# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Layout of the race course shared by the terrain, the decorations and the environment.

The course runs along +x. Lanes are side by side along y and centered on y = 0; the start line is at x = 0 and the
finish line at x = :attr:`RaceCourseCfg.track_length`. Between them, every lane crosses the same obstacles, made
like the training terrain: random rough ground, stairs, boxes and slopes.
"""

from __future__ import annotations

from isaaclab.utils import configclass

OBSTACLE_KINDS = ("rough", "stairs_up", "stairs_down", "boxes", "slope")


@configclass
class RaceCourseCfg:
    """Geometry of the race course."""

    num_lanes: int | None = None
    """Number of lanes. The race environment sets it to the number of environments, one robot per lane."""

    lane_width: float = 3.0
    """Distance between two fence rows [m]. With narrower lanes, a robot drifting off center sees the fences in its
    height scan (1 m wide), which the policy never met in training."""

    track_length: float = 100.0
    """Distance from the start line to the finish line [m]."""

    run_up: float = 3.0
    """Track length behind the start line [m]."""

    run_out: float = 4.0
    """Track length past the finish line [m]."""

    # obstacles
    obstacles: tuple[str, ...] = ("rough", "stairs_up", "boxes", "stairs_down", "slope")
    """Obstacles from start to finish, any of :data:`OBSTACLE_KINDS`, spread evenly between the flat ends."""

    difficulty: float = 0.4
    """Obstacle difficulty from 0 (easiest) to 1 (hardest), like the training terrain levels."""

    seed: int = 0
    """Seed of the random obstacles; every lane gets the same ones."""

    flat_start: float = 2.0
    """Flat ground between the start line and the first obstacle [m]."""

    flat_finish: float = 2.0
    """Flat ground between the last obstacle and the finish line [m]."""

    obstacle_margin: float = 0.4
    """Flat strip along each fence, where the obstacles stop [m]."""

    # obstacle parameters, ranges as in the training terrain (difficulty interpolates them)
    rough_length: float = 3.0
    rough_fill: bool = False
    """Rough ground from the end of the flat start to the start of the flat finish, instead of ``rough_length``.
    Only for a course whose sole obstacle is ``rough``."""
    rough_noise_range: tuple[float, float] = (0.02, 0.10)
    rough_noise_step: float = 0.02
    stair_height_range: tuple[float, float] = (0.05, 0.23)
    stair_width: float = 0.3
    stair_count: int = 4
    stair_platform: float = 1.2
    box_height_range: tuple[float, float] = (0.05, 0.2)
    box_size: float = 0.45
    box_count: int = 6
    slope_range: tuple[float, float] = (0.0, 0.4)
    slope_length: float = 1.5
    slope_platform: float = 1.0

    # height field, as in the training terrain
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    slope_threshold: float = 0.75

    def width(self) -> float:
        """Width of all lanes together [m]."""
        return self.num_lanes * self.lane_width

    def lane_centers_y(self) -> list[float]:
        """Lateral position of the center of every lane [m], lane 0 first."""
        return [(i - (self.num_lanes - 1) / 2) * self.lane_width for i in range(self.num_lanes)]

    def obstacle_length(self, kind: str) -> float:
        """Length of an obstacle along the track [m]."""
        if kind == "rough":
            return self.track_length - self.flat_start - self.flat_finish if self.rough_fill else self.rough_length
        if kind in ("stairs_up", "stairs_down"):
            return 2 * self.stair_count * self.stair_width + self.stair_platform
        if kind == "boxes":
            return self.box_count * self.box_size
        if kind == "slope":
            return 2 * self.slope_length + self.slope_platform
        raise ValueError(f"Unknown obstacle '{kind}', expected one of {OBSTACLE_KINDS}.")

    def obstacle_layout(self) -> list[tuple[str, float, float]]:
        """``(kind, start x, length)`` of every obstacle, with equal flat gaps between them."""
        lengths = [self.obstacle_length(kind) for kind in self.obstacles]
        free = self.track_length - self.flat_start - self.flat_finish - sum(lengths)
        if free < 0.0:
            raise ValueError(
                f"The obstacles need {sum(lengths) + self.flat_start + self.flat_finish:.1f} m of track:"
                f" increase track_length (now {self.track_length:.1f} m) or remove obstacles."
            )
        gap = free / (len(lengths) - 1) if len(lengths) > 1 else free / 2
        x = self.flat_start if len(lengths) > 1 else self.flat_start + gap
        layout = []
        for kind, length in zip(self.obstacles, lengths, strict=True):
            layout.append((kind, x, length))
            x += length + gap
        return layout
