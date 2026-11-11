# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Decorations of the race course: guard rails between the lanes, grass, a start arch and a finish gantry.

Only the guard rails collide; the rest is visual-only. The arch and the gantry keep :attr:`RaceTrackCfg.clearance`
free above the track, and the race environment starts its height-scan rays below that, so they never see them.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import clone, get_current_stage
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path

from .race_course import RaceCourseCfg
from .race_labels import PIXEL_FONT

if TYPE_CHECKING:
    from pxr import Usd

# SimReady yellow guard rail, 2.79 m long (x), 0.25 m wide (y) and 1.01 m tall (z), origin at its center on the
# ground. ``payloads/base.usda`` holds its mesh, material and collider without the asset's physics variants.
FENCE_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/SimReady/Industrial/Warehouse/Barriers/Rail_Guard_A09/payloads/base.usda"

WHITE = (0.95, 0.95, 0.95)
BLACK = (0.05, 0.05, 0.05)
RED = (0.85, 0.08, 0.08)
GRAY = (0.35, 0.35, 0.38)
GRASS = (0.25, 0.5, 0.2)


class _Builder:
    """Creates visual-only shapes under one root, sharing one material per color."""

    def __init__(self, stage: Usd.Stage, root: str):
        self.stage = stage
        self.root = root
        self.materials: dict[tuple[float, float, float], str] = {}
        self.count = 0

    def _material(self, color: tuple[float, float, float]) -> str:
        if color not in self.materials:
            path = f"{self.root}/Looks/color_{len(self.materials)}"
            material_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.6)
            material_cfg.func(path, material_cfg)
            self.materials[color] = path
        return self.materials[color]

    def shape(self, group: str, prim_type: str, color, translation, orientation=None, scale=None, attributes=None):
        from pxr import Gf, UsdGeom  # noqa: PLC0415

        path = f"{self.root}/{group}/{prim_type.lower()}_{self.count}"
        self.count += 1
        prim = sim_utils.create_prim(
            path,
            prim_type,
            translation=translation,
            orientation=orientation,
            scale=scale,
            attributes=attributes,
            stage=self.stage,
        )
        UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*color)])
        sim_utils.bind_visual_material(path, self._material(color), stage=self.stage)

    def box(self, group: str, color, center, size):
        self.shape(group, "Cube", color, center, scale=size, attributes={"size": 1.0})

    def capsule_between(self, group: str, color, start, end, radius: float):
        """Capsule along the segment ``start``-``end``, which lies in a plane x = const."""
        dy, dz = end[1] - start[1], end[2] - start[2]
        length = math.hypot(dy, dz)
        # rotate the capsule axis (z) about x onto the segment
        angle = math.atan2(-dy, dz)
        orientation = (math.sin(angle / 2), 0.0, 0.0, math.cos(angle / 2))
        center = tuple((a + b) / 2 for a, b in zip(start, end, strict=True))
        attributes = {"radius": radius, "height": length, "axis": "Z"}
        self.shape(group, "Capsule", color, center, orientation=orientation, attributes=attributes)

    def text(self, group: str, color, text: str, x: float, center_y: float, top_z: float, pixel: float):
        """Pixel letters on the plane x = const, readable when looking along +x."""
        columns = 6 * len(text) - 1
        for i, letter in enumerate(text):
            for row, line in enumerate(PIXEL_FONT[letter]):
                for col, on in enumerate(line):
                    if on == "1":
                        c = 6 * i + col
                        # looking along +x, the reader's right is -y
                        y = center_y + (columns / 2 - c - 0.5) * pixel
                        z = top_z - (row + 0.5) * pixel
                        self.box(group, color, (x, y, z), (pixel * 0.5, pixel * 1.01, pixel * 1.01))

    def checkers(self, group: str, colors, center, size, cell: float, axis: str):
        """Checkerboard on a thin box, in the ground plane (``axis="z"``) or the banner plane (``axis="x"``)."""
        u_size, v_size = (size[0], size[1]) if axis == "z" else (size[1], size[2])
        nu, nv = max(1, round(u_size / cell)), max(1, round(v_size / cell))
        du, dv = u_size / nu, v_size / nv
        for i in range(nu):
            for j in range(nv):
                u = -u_size / 2 + (i + 0.5) * du
                v = -v_size / 2 + (j + 0.5) * dv
                color = colors[(i + j) % 2]
                if axis == "z":
                    self.box(group, color, (center[0] + u, center[1] + v, center[2]), (du, dv, size[2]))
                else:
                    self.box(group, color, (center[0], center[1] + u, center[2] + v), (size[0], du, dv))


def _spawn_fences(stage: Usd.Stage, root: str, cfg: RaceTrackCfg):
    """Guard rails, each as a hidden static collider and a visual-only copy.

    Newton hides the colliders of a body that has visual-only shapes, and the static world carries the arch and
    the gantry, so the rails are drawn from separate visual copies.
    """
    from pxr import Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

    course = cfg.course
    fence_usd_path = retrieve_file_path(cfg.fence_usd_path)
    x_begin, x_end = -course.run_up, course.track_length + course.run_out
    num_segments = max(1, math.ceil((x_end - x_begin) / cfg.fence_length))
    for row in range(course.num_lanes + 1):
        y = row * course.lane_width - course.width() / 2
        for seg in range(num_segments):
            x = x_begin + (seg + 0.5) * cfg.fence_length
            for kind in ("collider", "visual"):
                fence = stage.DefinePrim(f"{root}/Fences/{kind}/row_{row}_fence_{seg}", "Xform")
                fence.GetReferences().AddReference(fence_usd_path)
                sim_utils.standardize_xform_ops(fence, translation=(x, y, 0.0))
                if kind == "collider":
                    # static collider: no rigid body, so it is attached to the world, and not drawn
                    UsdGeom.Imageable(fence).CreatePurposeAttr(UsdGeom.Tokens.guide)
                    for prim in Usd.PrimRange(fence):
                        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                else:
                    # visual-only: open the instances so their physics schemas can be removed
                    for prim in [prim for prim in Usd.PrimRange(fence) if prim.IsInstance()]:
                        prim.SetInstanceable(False)
                    for prim in Usd.PrimRange(fence):
                        for api in (UsdPhysics.RigidBodyAPI, UsdPhysics.CollisionAPI, UsdPhysics.MassAPI):
                            if prim.HasAPI(api):
                                prim.RemoveAPI(api)


def _spawn_grass(builder: _Builder, cfg: RaceTrackCfg):
    """Green strips on the ground around the track."""
    course = cfg.course
    x_begin, x_end = -course.run_up, course.track_length + course.run_out
    # 1 cm thick so it does not flicker against the ground in the distance
    half_width, margin, z = course.width() / 2, cfg.grass_width, 0.005
    length = x_end - x_begin + 2 * margin
    for side in (-1, 1):
        center = ((x_begin + x_end) / 2, side * (half_width + margin / 2), z)
        builder.box("Grass", GRASS, center, (length, margin, z * 2))
        center = (x_begin - margin / 2 if side < 0 else x_end + margin / 2, 0.0, z)
        builder.box("Grass", GRASS, center, (margin, 2 * half_width, z * 2))


def _spawn_start(builder: _Builder, cfg: RaceTrackCfg):
    """White start line under an inflatable-style arch, striped red and white, with a START banner."""
    course = cfg.course
    line_size = (cfg.line_width, course.width(), cfg.line_thickness)
    builder.box("StartLine", WHITE, (0.0, 0.0, cfg.line_thickness / 2), line_size)

    # arch path: two straight legs and a half ellipse over the top, in the plane x = 0
    half_span = course.width() / 2 + cfg.arch_margin
    leg_height = cfg.clearance + cfg.arch_radius
    rise = min(max(0.35 * half_span, 1.5), 3.5)
    points = [(0.0, -half_span, 0.0), (0.0, -half_span, leg_height)]
    num_arc = max(8, round(math.pi * (half_span + rise) / 2 / cfg.arch_segment))
    for i in range(1, num_arc):
        angle = math.pi * i / num_arc
        points.append((0.0, -half_span * math.cos(angle), leg_height + rise * math.sin(angle)))
    points += [(0.0, half_span, leg_height), (0.0, half_span, 0.0)]
    # split the legs into segments of about the same length as the arc
    path = []
    for start, end in zip(points[:-1], points[1:], strict=True):
        n = max(1, round(math.dist(start, end) / cfg.arch_segment))
        path += [tuple(s + (e - s) * k / n for s, e in zip(start, end, strict=True)) for k in range(n)]
    path.append(points[-1])
    for i, (start, end) in enumerate(zip(path[:-1], path[1:], strict=True)):
        builder.capsule_between("Arch", (RED, WHITE)[i % 2], start, end, cfg.arch_radius)
    for side in (-1, 1):
        builder.box("Arch", GRAY, (0.0, side * half_span, 0.2), (1.0, 1.0, 0.4))

    # banner hanging from the arch on two straps, START facing the robots behind the start line
    top = leg_height + 0.6 * rise
    columns = 6 * len("START") + 1
    pixel = min(cfg.banner_pixel, 2 * (0.8 * half_span - cfg.arch_radius - 0.1) / columns, (top - cfg.clearance) / 9)
    width, height = columns * pixel, 9 * pixel
    builder.box("Banner", WHITE, (0.0, 0.0, top - height / 2), (0.06, width, height))
    builder.text("Banner", RED, "START", -0.03 - pixel / 4, 0.0, top - pixel, pixel)
    for side in (-1, 1):
        y = side * (width / 2 - 0.2)
        arch_z = leg_height + rise * math.sqrt(max(0.0, 1.0 - (y / half_span) ** 2))
        builder.box("Banner", GRAY, (0.0, y, (top + arch_z) / 2), (0.04, 0.06, arch_z - top))


def _spawn_finish(builder: _Builder, cfg: RaceTrackCfg):
    """Checkered finish line under a gantry with a checkered FINISH banner and two checkered flags."""
    course = cfg.course
    x = course.track_length
    line_size = (0.5, course.width(), cfg.line_thickness)
    builder.checkers("FinishLine", (BLACK, WHITE), (x, 0.0, cfg.line_thickness / 2), line_size, 0.25, "z")

    half_span = course.width() / 2 + cfg.arch_margin
    pixel = min(cfg.banner_pixel, 1.6 * half_span / (6 * len("FINISH") - 1))
    banner_height = 11 * pixel
    beam_z = cfg.clearance + banner_height + 0.25
    post_height = beam_z + 0.25
    for side in (-1, 1):
        builder.box("Gantry", GRAY, (x, side * half_span, post_height / 2), (0.35, 0.35, post_height))
        builder.box("Gantry", GRAY, (x, side * half_span, 0.2), (1.0, 1.0, 0.4))
        # flag pole on top of the post, flag waving toward the inside of the track
        pole_top = post_height + 1.5
        pole_size = (0.05, 0.05, pole_top - post_height)
        builder.box("Flags", GRAY, (x, side * half_span, (post_height + pole_top) / 2), pole_size)
        builder.checkers(
            "Flags", (BLACK, WHITE), (x, side * (half_span - 0.45), pole_top - 0.3), (0.02, 0.8, 0.6), 0.2, "x"
        )
    builder.box("Gantry", GRAY, (x, 0.0, beam_z), (0.4, 2 * half_span, 0.5))

    # banner: checkered stripes above and below FINISH
    width = (6 * len("FINISH") + 3) * pixel
    banner_z = beam_z - 0.25 - banner_height / 2
    builder.box("Banner", WHITE, (x, 0.0, banner_z), (0.06, width, banner_height))
    for z in (banner_z + 4.75 * pixel, banner_z - 4.75 * pixel):
        builder.checkers("Banner", (BLACK, WHITE), (x - 0.04, 0.0, z), (0.03, width, 1.5 * pixel), 0.75 * pixel, "x")
    builder.text("Banner", BLACK, "FINISH", x - 0.03 - pixel / 4, 0.0, banner_z + 3.5 * pixel, pixel)


@clone
def spawn_race_track(
    prim_path: str,
    cfg: RaceTrackCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn the guard rails, the grass, the start arch and the finish gantry under ``prim_path``."""
    if cfg.course.num_lanes is None:
        raise ValueError("RaceCourseCfg.num_lanes must be set before the scene is created.")
    stage = get_current_stage()
    root = sim_utils.create_prim(prim_path, "Xform", translation=translation, orientation=orientation, stage=stage)
    _spawn_fences(stage, prim_path, cfg)
    builder = _Builder(stage, prim_path)
    _spawn_grass(builder, cfg)
    _spawn_start(builder, cfg)
    _spawn_finish(builder, cfg)
    return root


@configclass
class RaceTrackCfg(sim_utils.SpawnerCfg):
    """Decorations of the race course."""

    func: Callable = spawn_race_track

    course: RaceCourseCfg = RaceCourseCfg()
    """Course geometry. The race environment replaces it with its own :attr:`course`."""

    fence_usd_path: str = FENCE_USD_PATH
    """USD file of one guard-rail segment, centered on its origin along x."""

    fence_length: float = 2.786
    """Length of one guard-rail segment along x [m]."""

    grass_width: float = 30.0
    """Width of the grass around the track [m]."""

    clearance: float = 2.6
    """Free height above the ground under the arch and the gantry [m]."""

    arch_margin: float = 1.0
    """Distance from the outer fences to the arch legs and the gantry posts [m]."""

    arch_radius: float = 0.35
    """Radius of the arch tube [m]."""

    arch_segment: float = 0.7
    """Length of one colored segment of the arch tube [m]."""

    banner_pixel: float = 0.12
    """Size of one letter pixel on the banners [m]."""

    line_width: float = 0.15
    """Width of the start line along x [m]."""

    line_thickness: float = 0.004
    """Height of the start and finish lines above the ground [m]."""
