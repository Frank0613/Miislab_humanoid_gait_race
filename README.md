# Miislab_humanoid_gait_race

An installable downstream Isaac Lab task package with a standard uv `src` layout.

The registered tasks are:
- `Miislab-Humanoid-Gait-Race-G1-Direct-v1`: velocity-tracking locomotion on rough terrain, used for training
- `Miislab-Humanoid-Gait-Rl-G1-Race-Final`: race of trained policies on an obstacle course, see [Race](#race)
- `Miislab-Humanoid-Gait-Rl-G1-Race-Midterm`: the same race on a flat track without obstacles, a straight sprint
- `Miislab-Humanoid-Gait-Rl-G1-Race-Midterm-Plus`: the same track with rough ground (bumps up to 20 cm) from start to finish, no stairs

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
# write schedule.json (see Race), then
uv run python scripts/race.py --viz kit
```

- First `uv sync` downloads several GB
- `OMNI_KIT_ACCEPT_EULA=YES` skips the EULA prompt
- Isaac Lab elsewhere: edit `[tool.uv.sources]`, then `uv lock`

## Race

100 m straight track, one G1 per checkpoint, one lane each. Name label above each robot = checkpoint file name.

| Task (`--task`) | Track |
| --- | --- |
| `final` (default) | rough, stairs up, boxes, stairs down, slope |
| `midterm` | flat |
| `midterm-plus` | rough all the way (bumps 2-20 cm), no stairs |

### Schedule (`schedule.json`)

```json
{
  "task": "midterm",
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
```

- `rounds`: list of rounds → each round a list of groups → each group the checkpoints racing together
- Names map to `models/<name>.pt` (`a` or `a.pt`); a name with a folder is a path
- Numbered automatically: `Round 1 - Group 1`, `Round 1 - Group 2`, ...
- `[]` or `""` = not filled in yet; winners are written in by hand
- `lanes`: number of lanes; a bigger group wins. Fixed at startup
- Editable while running: saved changes load on the next `N` / `R`. `task` and `lanes` need a restart

### Run

```bash
uv run python scripts/race.py --viz kit
uv run python scripts/race.py --schedule schedule-plus.json --viz kit
```

| Key | Action |
| --- | --- |
| `S` | start |
| `N` | next group (during a race: stop and score now) |
| `R` | race the current group again; new result overwrites the old |
| `F` | follow camera ↔ free camera |

- 60 s per race, then scored: finishers by time, others by distance
- Race window: group, clock, next key, camera mode, ranking
- Results: `results/<date>-<time>/round_<n>/group_<m>.txt`

```text
Round 1 - Group 1
  🥇  1.  pon             lane 0   112.97 s
     DNF  jasper          lane 2   77.6 m reached
```

### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--schedule` | `schedule.json` | schedule file |
| `--lanes` | schedule's `lanes` | number of lanes |
| `--models_dir` | `models` | folder for bare names |
| `--results_dir` | `results` | results folder |
| `--auto` | off | no keys, races run back to back |
| `--task` | schedule's `task`, else `final` | `final` / `midterm` / `midterm-plus` |
| `--time_limit` | 60 | seconds per race |
| `--speed` | 1.0 | forward command [m/s] |
| `--difficulty` | 0.4 | obstacle difficulty 0-1 |
| `--obstacles` | per task | obstacle list; empty = flat |
| `--course_seed` | 0 | seed of rough ground and boxes |
| `--track_length` | 100 | [m] |
| `--lane_width` | 3.0 | [m] |
| `--fixed_camera` | off | start with the free camera |

### Rules

- Same command for every robot: 1.0 m/s forward, heading straight at the finish
- Start: lane center, 0.5 m behind the line, facing the finish
- Facing > 90° away: no forward command, turns back in place
- Falling does not end the race; the robot may get up (`out_on_fall = True` to disqualify)
- Simulation blow-up (NaN) = out
- Guard rails are invisible to the height scan; they only block
- All checkpoints must share this task's network shape

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
