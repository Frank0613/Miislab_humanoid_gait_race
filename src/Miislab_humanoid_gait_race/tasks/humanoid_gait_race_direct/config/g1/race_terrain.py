# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain of the race course: flat ground with the same obstacles in every lane, inside a flat border.

Plugs into :class:`~isaaclab.terrains.TerrainImporterCfg` as its ``terrain_generator``, in place of Isaac Lab's
:class:`~isaaclab.terrains.TerrainGenerator`.
"""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh
from isaaclab.terrains.trimesh.utils import make_border
from isaaclab.utils import configclass

from .race_course import RaceCourseCfg

# colors of the terrain mesh [0-255]
TRACK_COLOR = (165, 72, 52)
OBSTACLE_COLOR = (150, 150, 155)
GRASS_COLOR = (70, 125, 60)


class RaceTerrainGenerator:
    """Height field of the race course, one lane wide and repeated for every lane."""

    def __init__(self, cfg: RaceTerrainGeneratorCfg, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        course = cfg.course
        if course.num_lanes is None:
            raise ValueError("RaceCourseCfg.num_lanes must be set before the terrain is generated.")
        hs, vs = course.horizontal_scale, course.vertical_scale
        rng = np.random.default_rng(course.seed)

        # one lane: rows along x from the run-up to the run-out, columns along y from one fence to the next
        x_begin, x_end = -course.run_up, course.track_length + course.run_out
        x = x_begin + np.arange(round((x_end - x_begin) / hs) + 1) * hs
        y = np.arange(round(course.lane_width / hs)) * hs
        inside = (y >= course.obstacle_margin) & (y <= course.lane_width - course.obstacle_margin)
        lane = np.zeros((len(x), len(y)))
        for kind, start, length in course.obstacle_layout():
            rows = (x >= start) & (x < start + length)
            lane[np.ix_(rows, inside)] = self._obstacle(kind, x[rows] - start, y[inside], course, rng)

        # all lanes, closed by the fence line of the last lane
        height_field = np.concatenate([np.tile(lane, (1, course.num_lanes)), np.zeros((len(x), 1))], axis=1)
        vertices, triangles = convert_height_field_to_mesh(
            np.rint(height_field / vs).astype(np.int16), hs, vs, course.slope_threshold
        )
        track = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
        track.apply_translation((x_begin, -course.width() / 2, 0.0))
        colors = np.where(np.abs(track.vertices[:, 2:3]) > 1e-3, OBSTACLE_COLOR, TRACK_COLOR)
        track.visual.vertex_colors = np.concatenate([colors, np.full((len(colors), 1), 255)], axis=1).astype(np.uint8)

        # flat border around the track, its top at z = 0; its boxes stay closed so the terrain mesh has a volume
        inner_size = (x_end - x_begin, course.width())
        border_size = (inner_size[0] + 2 * cfg.border_width, inner_size[1] + 2 * cfg.border_width)
        center = ((x_begin + x_end) / 2, 0.0, -cfg.border_height / 2)
        border = trimesh.util.concatenate(make_border(border_size, inner_size, cfg.border_height, center))
        border.visual.vertex_colors = np.tile((*GRASS_COLOR, 255), (len(border.vertices), 1)).astype(np.uint8)

        self.terrain_mesh = trimesh.util.concatenate([track, border])
        self.terrain_origins = np.zeros((1, 1, 3))
        self.flat_patches = {}

    @staticmethod
    def _obstacle(kind: str, x: np.ndarray, y: np.ndarray, course: RaceCourseCfg, rng) -> np.ndarray:
        """Heights [m] of an obstacle at local positions ``x`` (from its start) and ``y`` (across the lane)."""
        d = course.difficulty

        def lerp(value_range: tuple[float, float]) -> float:
            return value_range[0] + d * (value_range[1] - value_range[0])

        if kind == "rough":
            # random heights on the height-field grid, as the training "random_rough" terrain
            levels = np.arange(course.rough_noise_range[0], course.rough_noise_range[1] + 1e-6, course.rough_noise_step)
            return rng.choice(levels, size=(len(x), len(y)))
        if kind in ("stairs_up", "stairs_down"):
            # up (or down) stair_count steps, a platform, and back to the ground
            step = lerp(course.stair_height_range)
            up = np.minimum(np.floor(x / course.stair_width) + 1, course.stair_count)
            down = np.minimum(np.floor((x[-1] - x) / course.stair_width) + 1, course.stair_count)
            heights = np.minimum(up, down) * step
            return np.repeat((heights if kind == "stairs_up" else -heights)[:, None], len(y), axis=1)
        if kind == "boxes":
            # square cells of random heights, as the training "boxes" terrain
            height = lerp(course.box_height_range)
            cells = rng.uniform(-height, height, size=(course.box_count, int(np.ceil(y[-1] / course.box_size)) + 1))
            ix = np.minimum((x / course.box_size).astype(int), course.box_count - 1)
            iy = ((y - y[0]) / course.box_size).astype(int)
            return cells[np.ix_(ix, iy)]
        if kind == "slope":
            # a ramp up, a platform and a ramp down
            slope = lerp(course.slope_range)
            heights = slope * np.minimum(np.minimum(x, x[-1] - x), course.slope_length)
            return np.repeat(heights[:, None], len(y), axis=1)
        raise ValueError(f"Unknown obstacle '{kind}'.")


@configclass
class RaceTerrainGeneratorCfg:
    """Configuration of :class:`RaceTerrainGenerator`."""

    class_type: type = RaceTerrainGenerator

    course: RaceCourseCfg = RaceCourseCfg()
    """Course geometry. The race environment replaces it with its own :attr:`course`."""

    border_width: float = 30.0
    """Width of the flat ground around the track [m]."""

    border_height: float = 1.0
    """Depth of the border boxes below the ground [m]."""

    sub_terrains: dict = {}
    """Empty: the terrain importer reads it to decide on height-field collision, which this terrain does not use."""
