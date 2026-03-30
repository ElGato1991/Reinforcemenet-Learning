# Reinforcement Learning – Pac-Man Q-Learning

A tabular Q-Learning agent that learns to play a classic Pac-Man maze.
The project ships with a **pre-trained Q-table** (`q_table.json`) so you can
watch the agent play immediately, and a **headless trainer** so you can keep
improving it yourself.

A second, standalone demo (`QLearning.py`) shows the same Q-Learning algorithm
applied to a simpler 2-D world with green/red pills.

---

## Requirements

```
pip install PySide6 matplotlib
```

Python 3.10+ is recommended.

---

## Quick start

### Watch the pre-trained agent (GUI)

```bash
python PacManQLearning.py --view
```

Opens the graphical viewer and loads `q_table.json` automatically.

### Train from scratch (headless, no window)

```bash
python PacManQLearning.py --train 50000
```

Runs 50,000 training episodes without a GUI and saves the updated Q-table to
`q_table.json`.

### Train and then open the viewer

```bash
python PacManQLearning.py --train 50000 --view
```

---

## All arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--train <N>` | int | `0` | Number of headless training episodes to run. Set to `0` (or omit) to skip training. |
| `--view` | flag | off | Open the GUI viewer after training (or immediately if `--train` is `0`). |
| `--qtable <path>` | str | `q_table.json` | Path to the Q-table JSON file to load/save. |
| `--workers <N>` | int | `1` | Number of parallel worker processes used during training. |
| `--chunk-episodes <N>` | int | `2000` | Number of episodes each worker processes per chunk. |

### Examples

```bash
# Watch the agent with a custom Q-table
python PacManQLearning.py --view --qtable my_qtable.json

# Train with 4 parallel workers and save to a custom file
python PacManQLearning.py --train 100000 --workers 4 --qtable my_qtable.json

# Train in large chunks (fewer synchronisation points)
python PacManQLearning.py --train 200000 --workers 8 --chunk-episodes 5000

# Train silently, then inspect the result
python PacManQLearning.py --train 50000
python PacManQLearning.py --view
```

---

## Project structure

| File | Description |
|---|---|
| `PacManQLearning.py` | Main entry point – parses CLI arguments, starts training and/or the GUI. |
| `pacman_qlearning.py` | Q-Learning agent, environment, and training loop. |
| `pacman_gui.py` | PySide6 GUI: maze renderer, stats plots, Q-table viewer. |
| `pacman_globals.py` | All tuneable constants (rewards, learning-rate, maze layout, …). |
| `pacman_headless_train.py` | Thin wrapper that delegates to `PacManQLearning.py` via subprocess. |
| `QLearning.py` | Self-contained Q-Learning demo in a 2-D pill world (no Pac-Man). |
| `q_table.json` | Pre-trained Q-table (ready to use with `--view`). |

---

## Tuning hyperparameters

Open `pacman_globals.py` to adjust learning behaviour without touching the
algorithm code:

```python
ALPHA          = 0.2      # learning rate
GAMMA          = 0.95     # discount factor
EPS_START      = 1.0      # initial exploration rate
EPS_MIN        = 0.05     # minimum exploration rate
EPS_DECAY      = 0.995    # per-episode epsilon decay

REWARD_PELLET       =  5.0    # reward for eating a pellet
REWARD_CAUGHT       = -250.0  # penalty for being caught by a ghost
REWARD_ALL_COLLECTED =  250.0 # bonus for clearing the board
```

---

## Simple Q-Learning demo

`QLearning.py` is a standalone demo that requires no Pac-Man assets:

```bash
python QLearning.py
```

A circle "robot" navigates a 2-D torus world and learns to collect green pills
while avoiding red pills using the same tabular Q-Learning algorithm.
