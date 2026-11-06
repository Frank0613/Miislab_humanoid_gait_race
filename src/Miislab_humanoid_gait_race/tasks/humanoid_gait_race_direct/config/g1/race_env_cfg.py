# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from .env_cfg import HumanoidGaitRaceEnvCfg, HumanoidGaitRaceSceneCfg
from .race_course import RaceCourseCfg
from .race_terrain import RaceTerrainGeneratorCfg
from .race_track import RaceTrackCfg


@configclass
class HumanoidGaitRaceTrackSceneCfg(HumanoidGaitRaceSceneCfg):
    """Race course with obstacles, guard rails between the lanes, a start arch and a finish gantry."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=RaceTerrainGeneratorCfg(),
        use_terrain_origins=False,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        # no visual material: the terrain mesh carries its own colors (track, obstacles, grass)
        visual_material=None,
        debug_vis=False,
    )
    race_track = AssetBaseCfg(prim_path="/World/RaceTrack", spawn=RaceTrackCfg())


@configclass
class HumanoidGaitRaceTrackEnvCfg(HumanoidGaitRaceEnvCfg):
    """Race of G1 walking policies: every robot runs down its own lane, the first across the finish line wins."""

    # race time limit: the race ends when every robot has finished or after this time [s]
    episode_length_s = 120.0

    scene: HumanoidGaitRaceTrackSceneCfg = HumanoidGaitRaceTrackSceneCfg(
        num_envs=2, env_spacing=2.5, replicate_physics=True
    )

    # course geometry and obstacles, shared with the terrain and the decorations
    course: RaceCourseCfg = RaceCourseCfg()

    # commands: run forward along the lane (+x) at this speed, stand still after the finish
    race_lin_vel_x = 1.0  # [m/s]
    # heading: face straight down the track (+x), with a firmer yaw-rate command than in training
    heading_control_stiffness = 1.0
    # a robot whose heading is off by more than this gets no forward command and turns in place [rad]
    turn_in_place_angle = math.pi / 2
    is_play = True

    # start: robots stand this far behind the start line, facing down the track [m]
    start_offset = 0.5
    reset_pose_range = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}

    # rules: a robot whose torso touches anything restarts this far before the obstacle where it fell [m]
    restart_offset = 1.0
    # height-scan rays this close to a fence line hit a rail, which is 0.25 m thick, and see the ground instead [m]
    rail_scan_band = 0.2
    # a torso contact within this distance of a fence line [m], with the body tilted less than this [rad], is a robot
    # walking into a rail, not a fall: it races on without a restart
    rail_contact_distance = 0.5
    rail_contact_max_tilt = math.pi / 4
    terrain_curriculum = False

    def __post_init__(self) -> None:
        """Look down the track from behind the start arch, and keep the height scan below the decorations."""
        super().__post_init__()
        # the scan value is ``torso height - hit height - 0.5`` whatever the ray start, so it matches training;
        # starting the rays 1 m above the torso keeps the arch and the gantry out of the scan
        self.scene.height_scanner.offset.pos = (0.0, 0.0, 1.0)
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(-7.0, -9.0, 6.0), lookat=(6.0, 0.0, 0.0))
        # the collision buffers of training are sized for thousands of robots; a race needs a small part of them,
        # and leaves the GPU memory to other jobs
        for physics in (self.sim.physics.newton_mjwarp, self.sim.physics.default):
            physics.collision_cfg.max_triangle_pairs = 100_000

    def play_mode(self):
        """Keep the race setup; only apply the generic play defaults."""
        DirectRLEnvCfg.play_mode(self)


@configclass
class HumanoidGaitRaceTrackMidtermEnvCfg(HumanoidGaitRaceTrackEnvCfg):
    """Midterm race: the same course as the final race, without obstacles, a straight sprint on flat ground."""

    course: RaceCourseCfg = RaceCourseCfg(obstacles=())
