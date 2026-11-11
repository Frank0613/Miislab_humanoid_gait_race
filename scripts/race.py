# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Race skrl checkpoints against each other on the race course.

Every checkpoint drives one G1 in its own lane; the first robot across the finish line wins. Each robot is named
after its checkpoint file. When the races are over the simulation window closes and the final ranking is printed.
``--task`` picks the race: ``final`` (obstacles, the default), ``midterm`` (the same track without obstacles),
``midterm-plus`` (rough ground all the way, no stairs) or any task id.

.. code-block:: bash

    uv run python scripts/race.py models/a.pt models/b.pt models/c.pt --viz kit
    uv run python scripts/race.py models/a.pt models/b.pt --task midterm --viz kit
    uv run python scripts/race.py models/a.pt models/b.pt --task midterm-plus --viz kit
    uv run python scripts/race.py models/a.pt models/b.pt --viz newton --difficulty 0.7 --rounds 3 --real-time
"""

import argparse
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


def build_parser() -> argparse.ArgumentParser:
    # imported here so that the ranking process stays lightweight
    from isaaclab.app import add_launcher_args

    from isaaclab_rl.entrypoints.backends import cli_args_skrl

    parser = argparse.ArgumentParser(description="Race skrl checkpoints against each other, one robot per lane.")
    parser.add_argument("agents", nargs="+", metavar="CHECKPOINT", help="Checkpoints to race, one lane each.")
    parser.add_argument(
        "--task",
        type=str,
        default="final",
        help="Race task: final (obstacles), midterm (no obstacles), midterm-plus (rough ground) or a task id.",
    )
    parser.add_argument("--agent", type=str, default=None, help="Agent configuration entry point.")
    parser.add_argument("--rounds", type=int, default=1, help="Number of races to run before exiting.")
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
        help="Keep the camera at the start instead of chasing the robots from behind.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
    parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
    parser.add_argument(RESULTS_FLAG, type=str, default=None, help=argparse.SUPPRESS)
    cli_args_skrl.add_skrl_args(parser)
    add_launcher_args(parser)
    return parser


def parse_agents(paths: list[str]) -> list[tuple[str, str]]:
    """``(name, checkpoint)`` of every checkpoint, named after its file name, with unique names."""
    agents = []
    for path in paths:
        if not os.path.isfile(path):
            raise SystemExit(f"race.py: checkpoint not found: {path}")
        agents.append((os.path.splitext(os.path.basename(path))[0], path))
    names = [name for name, _ in agents]
    return [
        (f"{name} #{names[:i].count(name) + 1}" if names.count(name) > 1 else name, path)
        for i, (name, path) in enumerate(agents)
    ]


def run_race(argv: list[str]):
    """Run the races in the simulator, writing the results to the ``--results_file`` after every race."""
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
    args_cli.task = RACE_TASKS.get(args_cli.task, args_cli.task)
    agents = parse_agents(args_cli.agents)

    check_skrl_version()
    agent_cfg_entry_point = resolve_skrl_agent_cfg_entry_point(args_cli.agent, args_cli.algorithm)
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, agent_cfg_entry_point, play_mode=True)

    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
        runner_cls = import_skrl_runner(args_cli.ml_framework)
        apply_env_overrides(args_cli, env_cfg)
        # one lane per checkpoint
        env_cfg.scene.num_envs = len(agents)
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
        if args_cli.fixed_camera:
            env_cfg.camera_follow = False
        if args_cli.seed is not None:
            agent_cfg["seed"] = args_cli.seed
        env_cfg.seed = agent_cfg["seed"]

        env = gym.make(args_cli.task, cfg=env_cfg)
        cleanup.callback(lambda: close_env(env))
        race = env.unwrapped
        race.lane_names = [name for name, _ in agents]
        env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

        # one skrl agent per checkpoint, all built from the same agent configuration
        agent_cfg["trainer"]["close_environment_at_exit"] = False
        agent_cfg["agent"]["experiment"]["write_interval"] = 0
        agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
        policies = []
        for _, path in agents:
            runner = runner_cls(env, copy.deepcopy(agent_cfg))
            print(f"[INFO] Loading model checkpoint from: {os.path.abspath(path)}")
            runner.agent.load(os.path.abspath(path))
            runner.agent.enable_training_mode(False, apply_to_models=True)
            weights = [p for model in runner.agent.models.values() for p in model.parameters()]
            if not all(torch.isfinite(p).all() for p in weights):
                print(f"[WARNING] {path} has NaN or infinite weights: its robot will be out of the race at once")
            policies.append(runner.agent)

        summary = {
            "agents": [{"lane": i, "name": name, "checkpoint": path} for i, (name, path) in enumerate(agents)],
            "track_length": course.track_length,
            "obstacles": list(course.obstacles),
            "difficulty": course.difficulty,
            "races": [],
        }
        print(f"[RACE] {len(agents)} robots, {course.track_length:.0f} m, time limit {race.max_episode_length_s:.0f} s")
        print(f"[RACE] obstacles: {', '.join(course.obstacles) or 'none'} (difficulty {course.difficulty:.2f})")
        for i, (name, path) in enumerate(agents):
            print(f"[RACE]   lane {i}: {name} ({path})")

        obs, _ = env.reset()
        states = env.state()
        dt = race.step_dt
        with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
            while race.race_count < args_cli.rounds:
                start_time = time.time()
                actions = torch.zeros(env.num_envs, *env.action_space.shape, device=env.device)
                for lane, policy in enumerate(policies):
                    sampled, outputs = policy.act(obs, states, timestep=0, timesteps=0)
                    actions[lane] = outputs.get("mean_actions", sampled)[lane]
                obs, _, _, _, _ = env.step(actions)
                states = env.state()
                if len(race.race_results) > len(summary["races"]):
                    summary["races"] = list(race.race_results)
                    with open(args_cli.results_file, "w") as f:
                        json.dump(summary, f)
                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)


def print_ranking(summary: dict):
    """Print the ranking of every race and, after several races, the overall standings."""
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    races = summary["races"]
    width = max(len(agent["name"]) for agent in summary["agents"]) + 2
    print()
    print("=" * 60)
    obstacles = ", ".join(summary["obstacles"]) or "none"
    print(f"🏁  RACE RESULTS  🏁   {summary['track_length']:.0f} m, obstacles: {obstacles}")
    print("=" * 60)
    for number, results in enumerate(races, start=1):
        if len(races) > 1:
            print(f"Race {number}")
        for place, r in enumerate(results, start=1):
            if r["finish_time"] is not None:
                rank = f"{medals.get(place, '  ')} {place:>2}."
                result = f"{r['finish_time']:7.2f} s"
            elif r["fell_at"] is not None:
                rank, result = "   DNF", f"fell at {r['fell_at']:5.1f} m"
            else:
                rank, result = "   DNF", f"{r['distance']:5.1f} m reached"
            print(f"  {rank}  {r['name']:<{width}} lane {r['lane']}  {result}")
        winner = results[0]
        print(f"  Winner: {winner['name']}" if winner["finish_time"] is not None else "  Nobody finished")
        print()

    if len(races) > 1:
        standings = []
        for agent in summary["agents"]:
            places = [next(p for p, r in enumerate(res, 1) if r["lane"] == agent["lane"]) for res in races]
            wins = sum(1 for res in races if res[0]["lane"] == agent["lane"] and res[0]["finish_time"] is not None)
            standings.append((agent["name"], wins, sum(places) / len(places)))
        standings.sort(key=lambda s: (-s[1], s[2]))
        print(f"Overall after {len(races)} races")
        for place, (name, wins, mean_place) in enumerate(standings, start=1):
            rank = f"{medals.get(place, '  ')} {place:>2}."
            print(f"  {rank}  {name:<{width}} wins: {wins}   average place: {mean_place:.1f}")
        print()


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
