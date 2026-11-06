# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import wrap_to_pi

from .env import HumanoidGaitRaceEnv

if TYPE_CHECKING:
    from .race_env_cfg import HumanoidGaitRaceTrackEnvCfg


class HumanoidGaitRaceTrackEnv(HumanoidGaitRaceEnv):
    """One G1 per lane racing down a straight course with obstacles; the first robot across the finish line wins.

    Rules:

    * every robot gets the same forward-velocity command and the same target heading, straight down the track (+x)
      toward the finish line; a robot that drifts sideways is not steered back. A robot that turned away always
      turns back the way it came, even past 180 degrees, instead of being told to complete the turn, and turns in
      place while it faces away from the track,
    * the height scan does not see the guard rails: the policy never met walls in training, and turns away from
      them, so the rails only stop a robot that walks into them,
    * a robot whose torso touches anything (a fall) restarts 1 m before the obstacle where it fell, behind the start
      line if it fell before the first obstacle; its race clock keeps running. Walking into a rail while upright is
      not a fall: the robot stays where it is and races on,
    * a robot that crosses the finish line gets a zero command and stands still,
    * the race ends when every robot has finished or after ``episode_length_s``. Finishers are ranked by time, the
      others by the farthest point they reached. :attr:`race_results` holds the ranking of every race.

    :attr:`lane_names` labels the lanes in the printed messages, e.g. with the checkpoint driving each robot.
    """

    cfg: HumanoidGaitRaceTrackEnvCfg

    def __init__(self, cfg: HumanoidGaitRaceTrackEnvCfg, render_mode: str | None = None, **kwargs):
        # one lane per robot: the terrain and the decorations are built with the scene, so the course they share
        # is sized before the scene exists
        cfg.course.num_lanes = cfg.scene.num_envs
        cfg.scene.terrain.terrain_generator.course = cfg.course
        cfg.scene.race_track.spawn.course = cfg.course
        super().__init__(cfg, render_mode, **kwargs)
        self.track_length = self.cfg.course.track_length

        # start positions: lane centers, behind the start line; a fall moves the robot's origin to its restart point
        origins = torch.zeros(self.num_envs, 3, device=self.device)
        origins[:, 0] = -self.cfg.start_offset
        origins[:, 1] = torch.tensor(self.cfg.course.lane_centers_y(), device=self.device)
        self._terrain.env_origins = origins
        self._restart_points = [start - self.cfg.restart_offset for _, start, _ in self.cfg.course.obstacle_layout()]
        course = self.cfg.course
        self._fence_y = torch.tensor(
            [row * course.lane_width - course.width() / 2 for row in range(course.num_lanes + 1)], device=self.device
        )

        # race state
        self.lane_names = [f"lane {i}" for i in range(self.num_envs)]
        self.race_count = 0
        self.race_results: list[list[dict]] = []
        self._race_step = 0
        self._race_over = False
        self._finish_time = torch.full((self.num_envs,), math.nan, device=self.device)
        self._falls = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._max_distance = torch.zeros(self.num_envs, device=self.device)
        # heading unwrapped over full turns [rad]
        self._heading = torch.zeros(self.num_envs, device=self.device)

    @property
    def race_time(self) -> float:
        """Time since the start of the current race [s]."""
        return self._race_step * self.step_dt

    def _resample_commands(self, env_ids: torch.Tensor):
        self._commands[env_ids, 1] = 0.0
        self._command_time_left[env_ids] = self.cfg.resampling_command_time
        self._update_yaw_rate_command()

    def _update_yaw_rate_command(self):
        # face the finish line: straight down the track, wherever the robot is in its lane
        self._target_heading.zero_()
        # the heading error is not wrapped: a robot turned 200 degrees to the left is told to turn 200 degrees back
        # to the right, not 160 more to the left, which would spin it round to face the same obstacle again
        self._heading += wrap_to_pi(self._robot.data.heading_w.torch - self._heading)
        heading_error = self._target_heading - self._heading
        self._commands[:, 2] = torch.clamp(heading_error * self.cfg.heading_control_stiffness, min=-1.0, max=1.0)
        # forward command: none after the finish, nor while facing away from the track, so a robot turning back
        # does not walk down the track the wrong way
        running = torch.isnan(self._finish_time) & (heading_error.abs() < self.cfg.turn_in_place_angle)
        self._commands[:, 0] = torch.where(running, self.cfg.race_lin_vel_x, 0.0)

    def _height_scan_hit_heights(self) -> torch.Tensor:
        # rays on a rail see the ground under it, which is flat: the obstacles stop short of the fences
        hits = self._height_scanner.data.ray_hits_w.torch
        rail_distance = torch.abs(hits[..., 1:2] - self._fence_y).min(dim=-1)[0]
        return torch.where(rail_distance < self.cfg.rail_scan_band, 0.0, hits[..., 2])

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        died, _ = super()._get_dones()
        self._race_step += 1
        x = self._robot.data.root_pos_w.torch[:, 0]
        self._max_distance = torch.maximum(self._max_distance, x.clamp(max=self.track_length))
        finished = ~torch.isnan(self._finish_time)
        # a robot that falls after the finish keeps its result and is left where it is
        died &= ~finished
        # an upright robot next to a fence touched a rail, it did not fall: no restart
        y = self._robot.data.root_pos_w.torch[:, 1]
        near_rail = torch.abs(y.unsqueeze(1) - self._fence_y).min(dim=1)[0] < self.cfg.rail_contact_distance
        upright = self._robot.data.projected_gravity_b.torch[:, 2] < -math.cos(self.cfg.rail_contact_max_tilt)
        died &= ~(near_rail & upright)
        self._falls += died
        for i in died.nonzero().flatten().tolist():
            fall_x = x[i].item()
            restart_x = max((p for p in self._restart_points if p < fall_x), default=-self.cfg.start_offset)
            self._terrain.env_origins[i, 0] = restart_x
            print(f"[RACE] {self.lane_names[i]} fell at {fall_x:.1f} m, restarts at {restart_x:.1f} m")

        crossed = (x >= self.track_length) & ~finished & ~died
        if torch.any(crossed):
            # interpolate the crossing time within the step from the distance past the line and the speed
            speed = self._robot.data.root_lin_vel_w.torch[:, 0].clamp(min=0.1)
            self._finish_time[crossed] = self.race_time - ((x - self.track_length) / speed)[crossed]
            self._commands[crossed, 0] = 0.0
            place = int(finished.sum().item())
            crossed_ids = crossed.nonzero().flatten().tolist()
            for i in sorted(crossed_ids, key=lambda i: self._finish_time[i].item()):
                place += 1
                finish_time = self._finish_time[i].item()
                print(f"[RACE] {self.lane_names[i]} crossed the finish line in {finish_time:.2f} s (place {place})")

        all_finished = bool(torch.all(~torch.isnan(self._finish_time)))
        self._race_over = all_finished or self._race_step >= self.max_episode_length
        if self._race_over:
            self._report_results()
        time_out = torch.full_like(died, self._race_over)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        # a new race starts after the previous one ends; single robots are reset after a fall
        if self._race_over or self._race_step == 0:
            self._race_step = 0
            self._race_over = False
            self._finish_time.fill_(math.nan)
            self._falls.zero_()
            self._max_distance.fill_(-self.cfg.start_offset)
            self._terrain.env_origins[:, 0] = -self.cfg.start_offset
            env_ids = None
        # robots restart facing down the track
        if env_ids is None:
            self._heading.zero_()
        else:
            self._heading[env_ids] = 0.0
        super()._reset_idx(env_ids)

    def _report_results(self):
        """Rank the robots of the race that just ended into :attr:`race_results`."""
        self.race_count += 1
        finish_time = self._finish_time.tolist()
        distance = self._max_distance.tolist()
        falls = self._falls.tolist()
        results = [
            {
                "lane": i,
                "name": self.lane_names[i],
                "finish_time": None if math.isnan(finish_time[i]) else finish_time[i],
                "distance": distance[i],
                "falls": falls[i],
            }
            for i in range(self.num_envs)
        ]
        # finishers by time, then the others by the farthest point they reached
        results.sort(key=lambda r: (r["finish_time"] is None, r["finish_time"] or 0.0, -r["distance"]))
        self.race_results.append(results)
        winner = results[0]
        if winner["finish_time"] is not None:
            print(f"[RACE] Race {self.race_count} over, winner: {winner['name']} ({winner['finish_time']:.2f} s)")
        else:
            print(f"[RACE] Race {self.race_count} over, nobody finished within {self.max_episode_length_s:.0f} s")
