# Miislab_humanoid_gait_race

An installable downstream Isaac Lab task package with a standard uv `src` layout.

The registered tasks are:
- `Miislab-Humanoid-Gait-Race-G1-Direct-v1`: velocity-tracking locomotion on rough terrain, used for training
- `Miislab-Humanoid-Gait-Rl-G1-Race-Final`: race of trained policies on an obstacle course, see [Race](#race)
- `Miislab-Humanoid-Gait-Rl-G1-Race-Midterm`: the same race on a flat track without obstacles, a straight sprint
- `Miislab-Humanoid-Gait-Rl-G1-Race-Midterm-Plus`: the same track with rough ground (bumps up to 15 cm) from start to finish, no stairs

## Installation

Tested on Ubuntu with an NVIDIA RTX 4080 (driver 580), uv 0.13 and Python 3.12. The project expects an Isaac Lab
v3.0.0 checkout two directories up, at `../../isaac/IsaacLab-3.0.0` (see `[tool.uv.sources]` in `pyproject.toml`), so
the commands below use `~/isaac` and `~/codes`. Run them in order on a new machine:

```bash
# 1. uv (skip if `uv --version` already works), then open a new shell or source the env file it prints
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Isaac Lab v3.0.0, next to this project
mkdir -p ~/isaac ~/codes
git clone --branch v3.0.0 --depth 1 https://github.com/isaac-sim/IsaacLab.git ~/isaac/IsaacLab-3.0.0

# 3. this project
git clone <this-repo-url> ~/codes/Miislab_humanoid_gait_race
cd ~/codes/Miislab_humanoid_gait_race

# 4. the environment: Python 3.12, Isaac Lab, Newton, Isaac Sim and skrl, as pinned in uv.lock
uv sync --locked

# 5. check: the three tasks are registered
uv run pytest
uv run isaaclab list_envs

# 6. the checkpoints to race go in models/ (git-ignored), then race
mkdir -p models
cp /path/to/*.pt models/
uv run python scripts/race.py models/a.pt models/b.pt --task midterm --viz kit
```

The first `uv sync` downloads several GB. The first Isaac Sim start asks you to accept the NVIDIA Omniverse EULA; set
`OMNI_KIT_ACCEPT_EULA=YES` to accept it without the prompt. The first race also downloads the guard rail and other
SimReady assets and caches them. If Isaac Lab lives somewhere else, change the paths in `[tool.uv.sources]` and run
`uv lock` before `uv sync`.

## Race

`scripts/race.py` puts one G1 per checkpoint in its own lane of a straight 100 m course. Yellow SimReady guard rails
(`Rail_Guard_A09`) separate the lanes; the robots start under a red and white arch with a START banner, which widens
with the number of lanes, and finish on a checkered line under a FINISH gantry. Between start and finish, every lane
crosses the same obstacles, made like the training terrain: random rough ground, stairs up, boxes, stairs down and a
slope. Each robot is named after its checkpoint file, and the name floats above it in the visualizer. When the races are over the simulation window closes and the
ranking is printed.

There are three races on the same track: the final (`--task final`, the default) with the obstacles, the midterm
(`--task midterm`) without them, a straight sprint on flat ground, and the midterm-plus (`--task midterm-plus`) with
rough ground over the whole track, bumpier than the final's (2-15 cm instead of 2-10 cm), and no stairs.

```bash
uv run python scripts/race.py models/a.pt models/b.pt models/c.pt --viz kit
uv run python scripts/race.py models/a.pt models/b.pt --task midterm --viz kit
uv run python scripts/race.py models/a.pt models/b.pt --task midterm-plus --viz kit
uv run python scripts/race.py models/a.pt models/b.pt --viz newton --difficulty 0.7 --rounds 3 --real-time
uv run python scripts/race.py models/a.pt models/b.pt --obstacles        # flat track
uv run python scripts/race.py models/a.pt models/b.pt --obstacles stairs_up stairs_up slope
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--task` | `final` | `final` (obstacles), `midterm` (flat track), `midterm-plus` (rough all the way) or a task id |
| `--difficulty` | 0.4 | obstacle difficulty from 0 to 1, as the training terrain levels (stairs 5-23 cm, boxes 5-20 cm, slopes 0-0.4) |
| `--obstacles` | `rough stairs_up boxes stairs_down slope` (final), none (midterm), `rough` over the whole track (midterm-plus) | obstacles from start to finish; none for a flat track |
| `--course_seed` | 0 | seed of the random rough ground and boxes |
| `--track_length` | 100 | start-to-finish distance [m] |
| `--lane_width` | 3.0 | distance between guard rails [m] |
| `--time_limit` | 300 | race time limit [s] |
| `--speed` | 1.0 | forward-velocity command of every robot [m/s] |
| `--rounds` | 1 | number of races before the window closes |
| `--fixed_camera` | off | keep the camera at the start; by default it follows the robots from the side, looking down, and keeps every robot still racing in view |

Rules:

- Every robot gets the same forward command and the same target heading, straight down the track toward the finish
  line; a robot that drifts sideways is not steered back to its lane center. A robot that turned away is always
  steered back the way it came, so it does not spin on in full circles, and turns in place while it faces more than
  90 degrees away from the track.
- The height scan does not see the guard rails (the policy never met walls in training): they only stop a robot
  that walks into them. Walking into a rail while upright is not a fall: the robot races on from where it is.
- A robot whose torso touches anything has fallen: it is not reset, it stays where it fell and is out of the race
  (DNF, ranked by the farthest point it reached).
- A robot that crosses the finish line stops. The race ends when every robot has finished or fallen, or the time
  limit is reached. Finishers are ranked by time, the others by the farthest point they reached.

All checkpoints must come from the agent configuration of this task (same network shape). Put the checkpoints before
`--obstacles`, which takes a list.

## Run the tasks

Replace the placeholders below with a task listed above and a selected RL library.

```bash
# List this project's environments and their available presets
uv run isaaclab list_envs --show_presets

# Exercise an environment without a trained policy
uv run isaaclab zero_agent --task <TASK_NAME> --num_envs 16
uv run isaaclab random_agent --task <TASK_NAME> --num_envs 16

# Train and play
uv run isaaclab train --rl_library <RL_LIBRARY> --task <TASK_NAME>
uv run isaaclab play --rl_library <RL_LIBRARY> --task <TASK_NAME> --checkpoint latest

# Distributed training
uv run isaaclab train_multigpu --rl_library <RL_LIBRARY> --task <TASK_NAME> --num_gpus 2

# Benchmark startup, runtime, training, or play
uv run isaaclab benchmark runtime --task <TASK_NAME> --num_envs 16 --num_steps 1000
uv run isaaclab benchmark training --rl_library <RL_LIBRARY> --task <TASK_NAME> --max_iterations 10
```

Use `physics=<PRESET>` to select one of the presets shown by `isaaclab list_envs`.

## Project structure

Task families live in `src/Miislab_humanoid_gait_race/tasks`. Within a manager-based family, task-wide MDP terms live in `mdp`, while
robot-specific scenes, registrations, and agent configurations live in `config/g1`. Add another robot by
creating a sibling of `config/g1`; add another task by creating a sibling task-family directory.

## Project assets

Keep project-owned USD files and related data under `src/Miislab_humanoid_gait_race/assets/data`. The asset module exposes a
stable path that works from an editable checkout and an installed wheel:

```python
from Miislab_humanoid_gait_race.assets import MIISLAB_HUMANOID_GAIT_RACE_ASSETS_DIR

robot_usd_path = MIISLAB_HUMANOID_GAIT_RACE_ASSETS_DIR / "robots" / "my_robot.usd"
```

Keep asset configuration in the `Miislab_humanoid_gait_race.assets` package and pass `str(robot_usd_path)` to configuration fields that
expect a string. USD files are configured for Git LFS by the project's `.gitattributes`; install Git LFS before adding
large binary assets.

## Development

Run the registration test and code-quality checks through the project environment:

```bash
uv run pytest
uv run pre-commit run --all-files
```

The test helpers under `source/isaaclab_tasks/test` in the Isaac Lab repository are not part of the installed
`isaaclab_tasks` package. Keep test fixtures in this project and use public Isaac Lab APIs. If you copy
`env_test_utils.py`, it becomes vendored code whose upstream changes you must track.

To configure VS Code or Cursor, run the `setup_python_env` task or invoke its command directly:

```bash
uv run isaaclab --editor
```

The setup command selects the active interpreter and writes a git-ignored `pyrightconfig.json`. The resulting
configuration inherits the project's checked-in Pyright settings and adds the Isaac Sim extensions, project `src` root,
and any Isaac Lab packages discovered in the active Python environment. This supports both Pylance in VS Code and
basedpyright in Cursor.

In VS Code, use Pylance and select the interpreter that ran the setup command. In Cursor, install the
[basedpyright extension](https://marketplace.visualstudio.com/items?itemName=detachhead.basedpyright) instead of
Pylance, select the same interpreter, and reload the window. Both language servers read `pyrightconfig.json`.

When using the `isaacsim` extra, include it while generating the editor configuration so the command can discover the
Isaac Sim installation:

```bash
uv run --extra isaacsim isaaclab --editor
```

For an Isaac Sim binaries installation that is not available in the project environment, provide its path explicitly:

```bash
# Linux
uv run isaaclab --editor --isaac_path <isaac-sim-path>

# Windows
uv run isaaclab --editor --isaac_path <isaac-sim-path>
```

## Troubleshooting

If Pylance or basedpyright cannot resolve modules, confirm that the selected interpreter matches the one used to run the
setup command, then reload the editor window. To add a missing extension or reduce indexing memory, edit the `extraPaths`
array in the root `pyrightconfig.json`; remove simulator extension directories that the project does not use.
