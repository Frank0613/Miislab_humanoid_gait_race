# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Miislab-Humanoid-Gait-Race-G1-Direct-v1",
    entry_point=f"{__name__}.env:HumanoidGaitRaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:HumanoidGaitRaceEnvCfg",
        "default_agent": "skrl",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# races of trained policies on a fenced track, see scripts/race.py
# final: obstacles along the track, as the training terrain
gym.register(
    id="Miislab-Humanoid-Gait-Rl-G1-Race-Final",
    entry_point=f"{__name__}.race_env:HumanoidGaitRaceTrackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.race_env_cfg:HumanoidGaitRaceTrackEnvCfg",
        "default_agent": "skrl",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# midterm: the same track without obstacles, a straight sprint
gym.register(
    id="Miislab-Humanoid-Gait-Rl-G1-Race-Midterm",
    entry_point=f"{__name__}.race_env:HumanoidGaitRaceTrackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.race_env_cfg:HumanoidGaitRaceTrackMidtermEnvCfg",
        "default_agent": "skrl",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
