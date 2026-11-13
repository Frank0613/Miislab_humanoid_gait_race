# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Race skrl checkpoints against each other on the race course, round after round, in one simulator window.

The races come from a schedule file (``schedule.json`` by default): a list of rounds, each a list of groups, each
group the checkpoints that race together. Rounds and groups are numbered in order (``Round 1 - Group 1``, ...), and a
group with no checkpoint is filled in later, while the window is open, e.g. with the winners of the round before. A
checkpoint is the name of a file in ``models/`` (``a`` or ``a.pt`` is ``models/a.pt``), or a path with a folder in
it:

.. code-block:: json

    {
        "task": "final",
        "lanes": 3,
        "rounds": [
            [
                ["a", "b", "c"],
                ["d", "e"]
            ],
            [
                []
            ]
        ]
    }

Every checkpoint drives one G1 in its own lane, named after its checkpoint file. Each group loads its robots at
the start and waits: in the Kit window, press **S** to start the race, **N** to go to the next group once it is over
(or to stop a race early), **R** to race the current group again (its new result replaces the old one), and **F** to
switch between the follow camera and your own camera. The result of every group is written to
``results/<date-time>/round_<n>/group_<m>.txt``; the ranking of all races is printed when the window is closed.
Without a Kit window (``--viz newton``, no visualizer) or with ``--auto``, the races run one after the other.

Checkpoints on the command line race instead of the schedule, ``--rounds`` times. ``--task`` picks the race:
``final`` (obstacles, the default), ``midterm`` (the same track without obstacles), ``midterm-plus`` (rough ground
all the way, no stairs) or any task id.

.. code-block:: bash

    uv run python scripts/race.py --viz kit
    uv run python scripts/race.py --schedule finals.json --viz kit
    uv run python scripts/race.py a b --task midterm --viz kit
    uv run python scripts/race.py models/a.pt models/b.pt --viz newton --difficulty 0.7 --rounds 3 --real-time
"""

import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import tempfile

RACE_TASKS = {
    "final": "Miislab-Humanoid-Gait-Rl-G1-Race-Final",
    "midterm": "Miislab-Humanoid-Gait-Rl-G1-Race-Midterm",
    "midterm-plus": "Miislab-Humanoid-Gait-Rl-G1-Race-Midterm-Plus",
}
RESULTS_FLAG = "--results_file"
DEFAULT_SCHEDULE = "schedule.json"


def build_parser() -> argparse.ArgumentParser:
    # imported here so that the ranking process stays lightweight
    from isaaclab.app import add_launcher_args

    from isaaclab_rl.entrypoints.backends import cli_args_skrl

    parser = argparse.ArgumentParser(description="Race skrl checkpoints against each other, one robot per lane.")
    parser.add_argument(
        "agents", nargs="*", metavar="CHECKPOINT", help="Checkpoints to race, one lane each, instead of the schedule."
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default=DEFAULT_SCHEDULE,
        help="JSON file with the checkpoints of every round, used when no checkpoint is given.",
    )
    parser.add_argument(
        "--lanes",
        type=int,
        default=0,
        help="Number of lanes, at least the largest round of the schedule; more leaves room for bigger rounds "
        "written into the schedule while the window is open.",
    )
    parser.add_argument("--results_dir", type=str, default="results", help="Folder of the round results.")
    parser.add_argument(
        "--models_dir", type=str, default="models", help="Folder of the checkpoints named without a folder."
    )
    parser.add_argument(
        "--auto", action="store_true", default=False, help="Start every round at once, without waiting for S and N."
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Race task: final (obstacles), midterm (no obstacles), midterm-plus (rough ground) or a task id. "
        "Default: the schedule's task, else final.",
    )
    parser.add_argument("--agent", type=str, default=None, help="Agent configuration entry point.")
    parser.add_argument("--rounds", type=int, default=1, help="Races of the checkpoints given on the command line.")
    parser.add_argument("--track_length", type=float, default=None, help="Start-to-finish distance [m].")
    parser.add_argument("--lane_width", type=float, default=None, help="Distance between fence rows [m].")
    parser.add_argument("--difficulty", type=float, default=None, help="Obstacle difficulty from 0 to 1.")
    parser.add_argument(
        "--obstacles",
        type=str,
        nargs="*",
        default=None,
        help="Obstacles from start to finish: rough, stairs_up, stairs_down, boxes, slope (none for a flat track).",
    )
    parser.add_argument("--course_seed", type=int, default=None, help="Seed of the random obstacles.")
    parser.add_argument("--time_limit", type=float, default=None, help="Race time limit [s].")
    parser.add_argument("--speed", type=float, default=None, help="Forward-velocity command of every robot [m/s].")
    parser.add_argument(
        "--fixed_camera",
        action="store_true",
        default=False,
        help="Start with your own camera instead of the follow camera (F switches).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
    parser.add_argument(
        "--real-time",
        action="store_true",
        default=False,
        help="Run in real-time, if possible (always, while waiting for S and N in the Kit window).",
    )
    parser.add_argument(RESULTS_FLAG, type=str, default=None, help=argparse.SUPPRESS)
    cli_args_skrl.add_skrl_args(parser)
    add_launcher_args(parser)
    return parser


def checkpoint_path(entry: str, base: str, models_dir: str) -> str:
    """Path of a checkpoint: a bare name is a file in ``models_dir`` (``.pt`` added if missing), else a path.

    Relative paths and ``models_dir`` are relative to ``base``.
    """
    if not isinstance(entry, str) or not entry:
        raise ValueError(f"a checkpoint is a file name, not {entry!r}")
    if not os.path.isabs(entry) and not os.path.dirname(entry):
        entry = os.path.join(models_dir, entry if entry.endswith(".pt") else f"{entry}.pt")
    return entry if os.path.isabs(entry) else os.path.relpath(os.path.join(base, entry))


def parse_agents(entries: list[str], base: str, models_dir: str) -> list[tuple[str, str]]:
    """``(name, checkpoint)`` of every checkpoint, named after its file name, with unique names."""
    agents = []
    # an empty name is a slot not filled in yet
    for entry in (entry for entry in entries if entry != ""):
        path = checkpoint_path(entry, base, models_dir)
        if not os.path.isfile(path):
            raise ValueError(f"checkpoint not found: {path}")
        agents.append((os.path.splitext(os.path.basename(path))[0], path))
    names = [name for name, _ in agents]
    return [
        (f"{name} #{names[:i].count(name) + 1}" if names.count(name) > 1 else name, path)
        for i, (name, path) in enumerate(agents)
    ]


def load_rounds(args_cli) -> tuple[list[dict], dict]:
    """The rounds to race, ``{"name", "agents": [(name, checkpoint)]}``, and the schedule's other settings."""
    if args_cli.agents:
        agents = parse_agents(args_cli.agents, os.getcwd(), args_cli.models_dir)
        return [
            {"name": f"Race {i + 1}", "file": f"race_{i + 1:02d}.txt", "agents": agents} for i in range(args_cli.rounds)
        ], {}
    if not os.path.isfile(args_cli.schedule):
        raise ValueError(
            f"no checkpoint given and no schedule file {args_cli.schedule}; write one like this:\n"
            '{"task": "final", "rounds": [[["a", "b"]]]}'
        )
    with open(args_cli.schedule) as f:
        try:
            schedule = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"{args_cli.schedule} is not valid JSON: {e}") from e
    base = os.path.dirname(os.path.abspath(args_cli.schedule))
    # a list of rounds, each a list of groups, each group a list of checkpoints; a round written as one list of
    # checkpoints is a round of one group
    schedule_rounds = schedule.get("rounds")
    if not isinstance(schedule_rounds, list):
        raise ValueError(f'{args_cli.schedule} needs "rounds": a list of rounds, each a list of groups')
    rounds = []
    for r, groups in enumerate(schedule_rounds, start=1):
        if not isinstance(groups, list):
            raise ValueError(f"round {r} of {args_cli.schedule} must be a list of groups, not {groups!r}")
        if groups and all(isinstance(group, str) for group in groups):
            groups = [groups]
        for g, group in enumerate(groups, start=1):
            # a group without checkpoints is not filled in yet, e.g. before the winners of the round before are known
            if not isinstance(group, list):
                raise ValueError(f"round {r} of {args_cli.schedule}: a group is a list of checkpoints, not {group!r}")
            rounds.append(
                {
                    "name": f"Round {r} - Group {g}",
                    "file": os.path.join(f"round_{r}", f"group_{g:02d}.txt"),
                    "agents": parse_agents(group, base, args_cli.models_dir),
                }
            )
    if not rounds:
        raise ValueError(f"{args_cli.schedule} has no group to race")
    if not rounds[0]["agents"]:
        raise ValueError(f"the first group of {args_cli.schedule} has no checkpoint")
    lanes = schedule.get("lanes", 0)
    if not isinstance(lanes, int) or lanes < 0:
        raise ValueError(f'"lanes" of {args_cli.schedule} must be a whole number, not {lanes!r}')
    return rounds, schedule


class RaceKeys:
    """Key presses in the Kit window; unavailable without Kit."""

    def __init__(self, keys: tuple[str, ...]):
        self.available = False
        self._pressed: set[str] = set()
        try:
            import carb.input
            import omni.appwindow

            app_window = omni.appwindow.get_default_app_window()
            if app_window is None:
                return
            self._carb = carb.input
            self._keys = keys
            self._input = carb.input.acquire_input_interface()
            self._keyboard = app_window.get_keyboard()
            self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_event)
            self.available = True
        except Exception:
            return

    def _on_event(self, event, *args, **kwargs) -> bool:
        if event.type == self._carb.KeyboardEventType.KEY_PRESS and event.input.name in self._keys:
            self._pressed.add(event.input.name)
        return True

    def take(self) -> set[str]:
        """The keys pressed since the last call; a key not used right away is dropped, not kept for later."""
        pressed, self._pressed = self._pressed, set()
        return pressed


class RaceHud:
    """A small Kit window with the round, the race clock, the camera mode, the key to press and, once a round is over,
    its ranking; nothing without Kit."""

    def __init__(self):
        self._labels = None
        self._ranking = None
        try:
            import omni.ui as ui

            self._ui = ui
            self._window = ui.Window("Race", width=520, height=360)
            with self._window.frame, ui.VStack(spacing=6):
                self._labels = {
                    "title": ui.Label("", height=30, style={"font_size": 24, "color": 0xFF3333FF}),
                    "clock": ui.Label("", height=40, style={"font_size": 34}),
                    "status": ui.Label("", height=24, style={"font_size": 18}),
                    "camera": ui.Label("", height=24, style={"font_size": 18, "color": 0xFFAAAAAA}),
                }
                self._table = ui.VStack(spacing=2)
        except Exception:
            self._labels = None

    def show(self, ranking: dict | None = None, **texts: str):
        if self._labels is None:
            return
        for key, text in texts.items():
            self._labels[key].text = text
        if ranking is not self._ranking:
            self._ranking = ranking
            self._show_ranking(ranking)

    def _show_ranking(self, ranking: dict | None):
        """One row per robot: place, name, lane, result; empty while the round is not over."""
        ui = self._ui
        self._table.clear()
        if ranking is None:
            return
        with self._table:
            for r in ranking["results"]:
                place, result = ranking_cells(r)
                color = 0xFF33CCFF if r["finish_time"] is not None and r["place"] == 1 else 0xFFFFFFFF
                with ui.HStack(height=24):
                    for text, width in ((place, 50), (r["name"], 200), (f"lane {r['lane']}", 70), (result, 0)):
                        style = {"font_size": 18, "color": color}
                        if width:
                            ui.Label(text, width=width, style=style)
                        else:
                            ui.Label(text, style=style)


def run_race(argv: list[str]):
    """Race the rounds in the simulator, writing every round's result and the ``--results_file`` after it."""
    import contextlib
    import copy
    import time

    import gymnasium as gym
    import torch

    from isaaclab.app import launch_simulation
    from isaaclab.app.sim_launcher import fuse_kit_args

    from isaaclab_rl.entrypoints.common import apply_env_overrides, close_env, set_hydra_args
    from isaaclab_rl.skrl import (
        SkrlVecEnvWrapper,
        check_skrl_version,
        import_skrl_runner,
        resolve_skrl_agent_cfg_entry_point,
    )

    from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

    import Miislab_humanoid_gait_race.tasks  # noqa: F401

    args_cli, hydra_args = setup_preset_cli(build_parser(), fuse_kit_args(argv))
    set_hydra_args(hydra_args)
    try:
        rounds, schedule = load_rounds(args_cli)
    except ValueError as e:
        raise SystemExit(f"race.py: {e}") from e
    task_name = args_cli.task or schedule.get("task") or "final"
    args_cli.task = RACE_TASKS.get(task_name, task_name)

    check_skrl_version()
    agent_cfg_entry_point = resolve_skrl_agent_cfg_entry_point(args_cli.agent, args_cli.algorithm)
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, agent_cfg_entry_point, play_mode=True)

    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
        runner_cls = import_skrl_runner(args_cli.ml_framework)
        apply_env_overrides(args_cli, env_cfg)
        # one lane per robot of the largest round; smaller rounds leave the last lanes empty
        # lanes: as many as asked for (--lanes, else the schedule's "lanes"), and at least the largest round
        lanes = args_cli.lanes or schedule.get("lanes", 0)
        env_cfg.scene.num_envs = max(lanes, *(len(r["agents"]) for r in rounds))
        course = env_cfg.course
        for option in ("track_length", "lane_width", "difficulty"):
            if getattr(args_cli, option) is not None:
                setattr(course, option, getattr(args_cli, option))
        if args_cli.obstacles is not None:
            course.obstacles = tuple(args_cli.obstacles)
        if args_cli.course_seed is not None:
            course.seed = args_cli.course_seed
        course.obstacle_layout()  # fail early if the obstacles do not fit on the track
        if args_cli.time_limit is not None:
            env_cfg.episode_length_s = args_cli.time_limit
        if args_cli.speed is not None:
            env_cfg.race_lin_vel_x = args_cli.speed
        # the robots stay where they are after a race, until the next round is loaded
        env_cfg.auto_restart = False
        if args_cli.seed is not None:
            agent_cfg["seed"] = args_cli.seed
        env_cfg.seed = agent_cfg["seed"]

        env = gym.make(args_cli.task, cfg=env_cfg)
        cleanup.callback(lambda: close_env(env))
        race = env.unwrapped
        race.camera_follow = not args_cli.fixed_camera
        env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

        # one skrl agent per checkpoint, all built from the same agent configuration, loaded once
        agent_cfg["trainer"]["close_environment_at_exit"] = False
        agent_cfg["agent"]["experiment"]["write_interval"] = 0
        agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
        policies = {}

        def load_policy(path: str):
            if path not in policies:
                runner = runner_cls(env, copy.deepcopy(agent_cfg))
                print(f"[INFO] Loading model checkpoint from: {os.path.abspath(path)}")
                runner.agent.load(os.path.abspath(path))
                runner.agent.enable_training_mode(False, apply_to_models=True)
                weights = [p for model in runner.agent.models.values() for p in model.parameters()]
                if not all(torch.isfinite(p).all() for p in weights):
                    print(f"[WARNING] {path} has NaN or infinite weights: its robot will be out of the race at once")
                policies[path] = runner.agent
            return policies[path]

        results_dir = os.path.join(args_cli.results_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        summary = {"task": task_name, "track_length": course.track_length, "obstacles": list(course.obstacles)}
        summary.update(difficulty=course.difficulty, results_dir=results_dir, rounds=[])
        limit = race.max_episode_length_s
        length = course.track_length
        print(f"[RACE] {len(rounds)} race(s), {race.num_envs} lanes, {length:.0f} m, time limit {limit:.0f} s")
        print(f"[RACE] obstacles: {', '.join(course.obstacles) or 'none'} (difficulty {course.difficulty:.2f})")

        def save_round(round_cfg: dict, stopped: bool):
            results = [{"place": place, **r} for place, r in enumerate(race.race_results[-1], start=1)]
            rnd = {"name": round_cfg["name"], "stopped_early": stopped, "results": results}
            path = os.path.join(results_dir, round_cfg["file"])
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write("\n".join(format_round(rnd)) + "\n")
            print(f"[RACE] results of {round_cfg['name']} written to {path}")
            # a round raced again replaces its earlier result
            done = [i for i, r in enumerate(summary["rounds"]) if r["name"] == rnd["name"]]
            if done:
                summary["rounds"][done[0]] = rnd
            else:
                summary["rounds"].append(rnd)
            with open(args_cli.results_file, "w") as f:
                json.dump(summary, f)

        def schedule_mtime() -> float | None:
            if args_cli.agents or not os.path.isfile(args_cli.schedule):
                return None
            return os.path.getmtime(args_cli.schedule)

        schedule_seen = schedule_mtime()

        def reload_rounds(current: list[dict]) -> list[dict]:
            """The rounds of the schedule file as it is now; the current ones if it did not change or is broken."""
            nonlocal schedule_seen
            mtime = schedule_mtime()
            if mtime is None or mtime == schedule_seen:
                return current
            schedule_seen = mtime
            try:
                new_rounds, new_schedule = load_rounds(args_cli)
            except ValueError as e:
                print(f"[RACE] {args_cli.schedule} not reloaded, keeping the races as they were: {e}")
                return current
            print(f"[RACE] {args_cli.schedule} reloaded: {len(new_rounds)} race(s)")
            new_task = new_schedule.get("task")
            new_lanes = new_schedule.get("lanes", 0)
            if new_lanes != schedule.get("lanes", 0) and not args_cli.lanes:
                print(f"[RACE] the lanes changed to {new_lanes}: that takes a restart, the track keeps {race.num_envs}")
            if new_task is not None and RACE_TASKS.get(new_task, new_task) != args_cli.task:
                print(f"[RACE] the task changed to {new_task}: that takes a restart, the track stays {task_name}")
            return new_rounds

        def load_round(index: int):
            """Put the robots of round ``index`` at the start; None if they do not fit on the track."""
            round_cfg = rounds[index]
            if not round_cfg["agents"]:
                print(f"[RACE] {round_cfg['name']} has no checkpoint yet: fill it in the schedule, then press N")
                return None
            if len(round_cfg["agents"]) > race.num_envs:
                print(
                    f"[RACE] {round_cfg['name']} has {len(round_cfg['agents'])} robots, the track only "
                    f"{race.num_envs} lanes: fix the schedule (or restart with --lanes)"
                )
                return None
            lane_policies = [load_policy(path) for _, path in round_cfg["agents"]]
            race.set_lanes([name for name, _ in round_cfg["agents"]])
            print(f"[RACE] {round_cfg['name']} ({index + 1}/{len(rounds)}):")
            for lane, (name, path) in enumerate(round_cfg["agents"]):
                print(f"[RACE]   lane {lane}: {name} ({path})")
            # the skrl wrapper resets the env only once, unless told otherwise
            env._reset_once = True
            obs, _ = env.reset()
            race.refresh_view()
            return round_cfg, lane_policies, obs, env.state()

        keys = RaceKeys(("S", "N", "R", "F"))
        interactive = keys.available and not args_cli.auto
        hud = RaceHud()
        if interactive:
            print("[RACE] keys in the Isaac Sim window: S start, N next race (or stop), R race again, F camera")

        index = 0
        loaded = load_round(index)
        round_cfg, lane_policies, obs, states = loaded
        state = "waiting"
        dt = race.step_dt
        sim = race.sim
        last_hud = last_check = 0.0
        with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
            while sim.is_running():
                start_time = time.time()
                # keys pressed since the last step, used now or never: an S pressed during a race must not start
                # the next one, an N pressed at the start must not stop the race once it runs
                pressed = keys.take() if interactive else set()
                if "F" in pressed:
                    race.camera_follow = not race.camera_follow
                    print(f"[RACE] camera: {'follow' if race.camera_follow else 'free, yours to move'}")
                # the schedule may be edited while the window is open: read it again when it changes
                if state != "racing" and time.time() - last_check > 0.5:
                    last_check = time.time()
                    rounds = reload_rounds(rounds)
                    if state == "waiting" and index < len(rounds) and rounds[index] != round_cfg:
                        # the round waiting at the start was changed: load it again
                        loaded = load_round(index)
                        if loaded is not None:
                            round_cfg, lane_policies, obs, states = loaded

                if "R" in pressed and state != "waiting":
                    # race the current round again, as the schedule has it now; its new result replaces the old one
                    if state == "racing":
                        race.end_race()
                    rounds = reload_rounds(rounds)
                    loaded = load_round(index) if index < len(rounds) else None
                    if loaded is not None:
                        round_cfg, lane_policies, obs, states = loaded
                        state = "waiting"
                        print(f"[RACE] {round_cfg['name']} again: press S to start")
                    elif state == "racing":
                        state = "over"

                if state == "waiting":
                    if not interactive or "S" in pressed:
                        print(f"[RACE] {round_cfg['name']}: go!")
                        state = "racing"
                    else:
                        race.refresh_view()
                        sim.render()
                elif state == "racing":
                    stopped = "N" in pressed
                    if stopped:
                        race.end_race()
                    else:
                        actions = torch.zeros(env.num_envs, *env.action_space.shape, device=env.device)
                        for lane, policy in enumerate(lane_policies):
                            sampled, outputs = policy.act(obs, states, timestep=0, timesteps=0)
                            actions[lane] = outputs.get("mean_actions", sampled)[lane]
                        obs, _, _, _, _ = env.step(actions)
                        states = env.state()
                    if race.race_over:
                        save_round(round_cfg, stopped)
                        state = "over"
                        if index + 1 >= len(rounds):
                            print("[RACE] that was the last race of the schedule; add groups and press N, or close")
                elif state == "over":
                    if not interactive or "N" in pressed:
                        rounds = reload_rounds(rounds)
                        if index + 1 < len(rounds):
                            loaded = load_round(index + 1)
                            if loaded is not None:
                                index += 1
                                round_cfg, lane_policies, obs, states = loaded
                                state = "waiting"
                            elif not interactive:
                                break
                        elif not sim.visualizers or not interactive:
                            break
                        else:
                            print("[RACE] the schedule has no race after this one; add a group and press N again")
                    if state == "over":
                        race.refresh_view()
                        sim.render()

                if time.time() - last_hud > 0.2:
                    last_hud = time.time()
                    if state == "waiting":
                        status = f"{race.num_racers} robots at the start - press S to start"
                    elif state == "racing":
                        status = "racing - N stops the race now, R starts it over"
                    elif index + 1 < len(rounds):
                        status = f"race over, results saved - N: {rounds[index + 1]['name']}, R: race again"
                    else:
                        status = "last race over - R races it again; add groups, or close the window"
                    camera = "follow" if race.camera_follow else "free (yours to move)"
                    ranking = summary["rounds"][-1] if state == "over" and summary["rounds"] else None
                    hud.show(
                        ranking=ranking,
                        title=f"{round_cfg['name']}  ({index + 1}/{len(rounds)})",
                        clock=f"{race.race_time:5.1f} / {race.max_episode_length_s:.0f} s",
                        status=status,
                        camera=f"camera: {camera} - F switches",
                    )

                sleep_time = dt - (time.time() - start_time)
                # a watched race runs at real speed, so its clock matches the wall clock
                if (args_cli.real_time or interactive or state != "racing") and sleep_time > 0:
                    time.sleep(sleep_time)


MEDALS = {1: "🥇", 2: "🥈", 3: "🥉"}


def ranking_cells(r: dict) -> tuple[str, str]:
    """Place (``1.``, or ``DNF``) and result of one robot."""
    if r["finish_time"] is not None:
        return f"{r['place']}.", f"{r['finish_time']:.2f} s"
    if r.get("broken_at") is not None:
        return "DNF", f"out (NaN) at {r['broken_at']:.1f} m"
    if r["fell_at"] is not None:
        return "DNF", f"fell at {r['fell_at']:.1f} m"
    return "DNF", f"{r['distance']:.1f} m reached"


def format_round(rnd: dict) -> list[str]:
    """Lines of the ranking of one round."""
    width = max(len(r["name"]) for r in rnd["results"]) + 2
    lines = [rnd["name"] + ("  (stopped early)" if rnd["stopped_early"] else "")]
    for r in rnd["results"]:
        place, result = ranking_cells(r)
        rank = f"{MEDALS.get(r['place'], '  ')} {place:>3}" if place != "DNF" else "   DNF"
        result = f"{result:>9}" if r["finish_time"] is not None else f"{result:>15}"
        lines.append(f"  {rank}  {r['name']:<{width}} lane {r['lane']}  {result}")
    return lines


def print_ranking(summary: dict):
    """Print the ranking of every round."""
    print()
    print("=" * 60)
    obstacles = ", ".join(summary["obstacles"]) or "none"
    print(f"🏁  RACE RESULTS  🏁   {summary['track_length']:.0f} m, obstacles: {obstacles}")
    print("=" * 60)
    for rnd in summary["rounds"]:
        print("\n".join(format_round(rnd)))
        print()
    print(f"Round results: {summary['results_dir']}/")


def main():
    if RESULTS_FLAG in sys.argv:
        run_race(sys.argv[1:])
        return

    # run the races in a child process: closing the simulator ends its process, so the ranking is printed here,
    # once the window is closed
    with tempfile.TemporaryDirectory() as tmp_dir:
        results_file = os.path.join(tmp_dir, "results.json")
        child = subprocess.Popen([sys.executable, os.path.abspath(__file__), *sys.argv[1:], RESULTS_FLAG, results_file])
        # Ctrl+C reaches the child too, which ends the race and closes the window
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        return_code = child.wait()
        if os.path.isfile(results_file):
            with open(results_file) as f:
                print_ranking(json.load(f))
        elif return_code == 0:
            print("[RACE] No race was completed.")
    sys.exit(return_code)


if __name__ == "__main__":
    main()
