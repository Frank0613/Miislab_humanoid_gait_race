# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs import DirectRLEnv

if TYPE_CHECKING:
    from .env_cfg import HumanoidGaitRaceEnvCfg


class HumanoidGaitRaceEnv(DirectRLEnv):
    """Implement the task's environment lifecycle."""

    cfg: HumanoidGaitRaceEnvCfg

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        raise NotImplementedError("Process policy actions.")

    def _apply_action(self) -> None:
        raise NotImplementedError("Apply actions to the simulation.")

    def _get_observations(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError("Compute policy observations.")

    def _get_rewards(self) -> torch.Tensor:
        raise NotImplementedError("Compute task rewards.")

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Compute termination and timeout conditions.")

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        raise NotImplementedError("Reset selected environments.")
