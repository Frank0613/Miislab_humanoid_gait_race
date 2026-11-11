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
from .race_labels import RaceLabels

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
    * a robot whose torso touches anything (a fall) is out of the race: it is not reset, gets a zero command and
      stays where it fell. Walking into a rail while upright is not a fall: the robot races on from where it is,
    * a robot that crosses the finish line gets a zero command and stands still,
    * the race ends when every robot has finished or fallen, or after ``episode_length_s``. Finishers are ranked by
      time, the others by the farthest point they reached. :attr:`race_results` holds the ranking of every race.

    :attr:`lane_names` labels the lanes in the printed messages, e.g. with the checkpoint driving each robot, and
    floats above each robot in the visualizers.
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

        # start positions: lane centers, behind the start line
        origins = torch.zeros(self.num_envs, 3, device=self.device)
        origins[:, 0] = -self.cfg.start_offset
        origins[:, 1] = torch.tensor(self.cfg.course.lane_centers_y(), device=self.device)
        self._terrain.env_origins = origins
        course = self.cfg.course
        self._fence_y = torch.tensor(
            [row * course.lane_width - course.width() / 2 for row in range(course.num_lanes + 1)], device=self.device
        )

        # race state
        # the labels show once the lanes are named
        self._labels: RaceLabels | None = None
        self._lane_names = [f"lane {i}" for i in range(self.num_envs)]
        self.race_count = 0
        self.race_results: list[list[dict]] = []
        self._race_step = 0
        self._race_over = False
        self._finish_time = torch.full((self.num_envs,), math.nan, device=self.device)
        # where each robot fell, NaN while it is still racing [m]
        self._fall_x = torch.full((self.num_envs,), math.nan, device=self.device)
        self._max_distance = torch.zeros(self.num_envs, device=self.device)
        # heading unwrapped over full turns [rad]
        self._heading = torch.zeros(self.num_envs, device=self.device)
        # chase camera: (look-at x, distance) [m], None until the first step
        self._camera: tuple[float, float] | None = None

    @property
    def lane_names(self) -> list[str]:
        """Name of the robot in every lane, lane 0 first."""
        return self._lane_names

    @lane_names.setter
    def lane_names(self, names: list[str]):
        self._lane_names = list(names)
        if self.cfg.show_labels:
            if self._labels is not None:
                self._labels.hide()
            self._labels = self._create_labels(self._lane_names)

    def _create_labels(self, names: list[str]) -> RaceLabels:
        # the letters face the camera: the follow camera looks along its azimuth, the fixed one from the default view
        viewer = self.cfg.sim.default_visualizer_cfg
        if self.cfg.camera_follow:
            yaw = math.radians(self.cfg.camera_azimuth_deg)
        else:
            yaw = math.atan2(viewer.lookat[1] - viewer.eye[1], viewer.lookat[0] - viewer.eye[0])
        return RaceLabels(
            names,
            max_width=self.cfg.label_max_width_ratio * self.cfg.course.lane_width,
            pixel=self.cfg.label_pixel,
            yaw=yaw,
            color=self.cfg.label_color,
            height=self.cfg.label_height,
        )

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
        # forward command: none after the finish or a fall, nor while facing away from the track, so a robot turning
        # back does not walk down the track the wrong way
        running = (
            torch.isnan(self._finish_time)
            & torch.isnan(self._fall_x)
            & (heading_error.abs() < self.cfg.turn_in_place_angle)
        )
        self._commands[:, 0] = torch.where(running, self.cfg.race_lin_vel_x, 0.0)

    def _height_scan_hit_heights(self) -> torch.Tensor:
        # rays on a rail see the ground under it, which is flat: the obstacles stop short of the fences
        hits = self._height_scanner.data.ray_hits_w.torch
        rail_distance = torch.abs(hits[..., 1:2] - self._fence_y).min(dim=-1)[0]
        return torch.where(rail_distance < self.cfg.rail_scan_band, 0.0, hits[..., 2])

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        died, _ = super()._get_dones()
        self._race_step += 1
        root_pos = self._robot.data.root_pos_w.torch
        # a policy that outputs NaN (e.g. a checkpoint whose training diverged) blows up its robot's simulation
        broken = ~torch.isfinite(root_pos).all(dim=1)
        if self._labels is not None:
            # the label of a broken robot is moved out of sight
            self._labels.update(
                torch.where(broken.unsqueeze(1), torch.tensor([0.0, 0.0, -100.0], device=self.device), root_pos)
            )
        x = root_pos[:, 0]
        self._max_distance = torch.fmax(self._max_distance, x.clamp(max=self.track_length))
        finished = ~torch.isnan(self._finish_time)
        fallen = ~torch.isnan(self._fall_x)
        # a robot that falls after the finish keeps its result; one that already fell stays out
        died &= ~finished & ~fallen
        # an upright robot next to a fence touched a rail, it did not fall
        y = self._robot.data.root_pos_w.torch[:, 1]
        near_rail = torch.abs(y.unsqueeze(1) - self._fence_y).min(dim=1)[0] < self.cfg.rail_contact_distance
        upright = self._robot.data.projected_gravity_b.torch[:, 2] < -math.cos(self.cfg.rail_contact_max_tilt)
        died &= ~(near_rail & upright)
        # a fallen robot is not reset: it is out of the race and lies where it fell
        self._fall_x[died] = x[died]
        self._commands[died, 0] = 0.0
        for i in died.nonzero().flatten().tolist():
            print(f"[RACE] {self.lane_names[i]} fell at {x[i].item():.1f} m, out of the race")
        fallen |= died
        # a broken robot is out of the race too, at the farthest point it reached
        broken &= ~finished & ~fallen
        self._fall_x[broken] = self._max_distance[broken]
        for i in broken.nonzero().flatten().tolist():
            print(f"[RACE] {self.lane_names[i]} produced NaN (broken checkpoint?), out of the race")
        fallen |= broken

        crossed = (x >= self.track_length) & ~finished & ~fallen
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

        if self.cfg.camera_follow:
            self._follow_pack(x, ~torch.isnan(self._finish_time) | fallen)

        all_done = bool(torch.all(~torch.isnan(self._finish_time) | fallen))
        self._race_over = all_done or self._race_step >= self.max_episode_length
        if self._race_over:
            self._report_results()
        time_out = torch.full_like(died, self._race_over)
        # nobody is reset before the race is over
        return torch.zeros_like(died), time_out

    def _follow_pack(self, x: torch.Tensor, out: torch.Tensor):
        """Follow the robots still racing, far enough away to keep all of them, and the track, in view."""
        if not self.sim.visualizers:
            return
        racing = [v for v, o in zip(x.tolist(), out.tolist(), strict=True) if not o and math.isfinite(v)]
        if racing:
            x_min, x_max = min(racing), max(racing)
            target = ((x_min + x_max) / 2, self._camera_distance(x_min, x_max))
            if self._camera is None or self._race_step <= 1:
                self._camera = target
            else:
                # smooth the motion, so the camera glides instead of shaking with the robots' steps
                alpha = 1.0 - math.exp(-self.step_dt / self.cfg.camera_smoothing_time)
                self._camera = tuple(c + alpha * (t - c) for c, t in zip(self._camera, target, strict=True))
        if self._camera is None:
            return
        center_x, distance = self._camera
        forward = self._camera_forward()
        eye = (center_x - distance * forward[0], -distance * forward[1], -distance * forward[2])
        self.sim.set_camera_view(eye, (center_x, 0.0, 0.0))

    def _camera_forward(self) -> tuple[float, float, float]:
        """Viewing direction of the follow camera, from its azimuth and pitch."""
        azimuth = math.radians(self.cfg.camera_azimuth_deg)
        pitch = math.radians(self.cfg.camera_pitch_deg)
        return (math.cos(pitch) * math.cos(azimuth), math.cos(pitch) * math.sin(azimuth), -math.sin(pitch))

    def _camera_distance(self, x_min: float, x_max: float) -> float:
        """Smallest distance from the look-at point that frames the track between ``x_min`` and ``x_max``."""
        cfg = self.cfg
        azimuth = math.radians(cfg.camera_azimuth_deg)
        pitch = math.radians(cfg.camera_pitch_deg)
        tan_h = math.tan(math.radians(cfg.camera_fov_h_deg) / 2)
        tan_v = tan_h / cfg.camera_aspect
        center_x = (x_min + x_max) / 2
        half_width = self.cfg.course.width() / 2
        margin = cfg.camera_margin
        # corners of the box to frame: the track width, from the last to the first robot, ground to above the labels
        corners = [
            (px - center_x, py, pz)
            for px in (x_min - margin, x_max + margin)
            for py in (-half_width, half_width)
            for pz in (0.0, cfg.label_height + 1.0)
        ]
        forward = self._camera_forward()
        right = (math.sin(azimuth), -math.cos(azimuth), 0.0)
        up = (math.sin(pitch) * math.cos(azimuth), math.sin(pitch) * math.sin(azimuth), math.cos(pitch))
        distance = cfg.camera_min_distance
        while distance < 500.0:
            eye = tuple(-distance * f for f in forward)
            fits = True
            for corner in corners:
                d = [c - e for c, e in zip(corner, eye, strict=True)]
                depth = sum(a * b for a, b in zip(d, forward, strict=True))
                horizontal = sum(a * b for a, b in zip(d, right, strict=True))
                vertical = sum(a * b for a, b in zip(d, up, strict=True))
                if depth <= 0.1 or abs(horizontal) > depth * tan_h or abs(vertical) > depth * tan_v:
                    fits = False
                    break
            if fits:
                return distance
            distance *= 1.05
        return distance

    def _reset_idx(self, env_ids: torch.Tensor | None):
        # robots are reset together, when a new race starts
        if self._race_over or self._race_step == 0:
            self._race_step = 0
            self._race_over = False
            self._finish_time.fill_(math.nan)
            self._fall_x.fill_(math.nan)
            self._max_distance.fill_(-self.cfg.start_offset)
            self._terrain.env_origins[:, 0] = -self.cfg.start_offset
            env_ids = None
        # robots start facing down the track
        self._heading.zero_()
        super()._reset_idx(env_ids)

    def _report_results(self):
        """Rank the robots of the race that just ended into :attr:`race_results`."""
        self.race_count += 1
        finish_time = self._finish_time.tolist()
        distance = self._max_distance.tolist()
        fall_x = self._fall_x.tolist()
        results = [
            {
                "lane": i,
                "name": self.lane_names[i],
                "finish_time": None if math.isnan(finish_time[i]) else finish_time[i],
                "distance": distance[i],
                "fell_at": None if math.isnan(fall_x[i]) else fall_x[i],
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
