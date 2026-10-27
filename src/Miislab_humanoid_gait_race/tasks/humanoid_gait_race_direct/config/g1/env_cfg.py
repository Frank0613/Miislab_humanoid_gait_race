# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class HumanoidGaitRaceSceneCfg(InteractiveSceneCfg):
    """Define robot assets, ground, lighting, and other scene objects."""

    robot: ArticulationCfg = MISSING


@configclass
class HumanoidGaitRaceEnvCfg(DirectRLEnvCfg):
    """Configure the task before constructing the environment."""

    decimation: int = MISSING
    episode_length_s: float = MISSING
    action_space: int = MISSING
    observation_space: int = MISSING
    state_space: int = MISSING

    sim: SimulationCfg = SimulationCfg()
    scene: HumanoidGaitRaceSceneCfg = HumanoidGaitRaceSceneCfg(num_envs=1, env_spacing=4.0)

    def __post_init__(self) -> None:
        """Set simulation timing, physics, spaces, and task parameters."""
        raise NotImplementedError("Configure simulation, scene, and task parameters.")
