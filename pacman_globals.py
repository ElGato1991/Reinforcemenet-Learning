from dataclasses import dataclass


# -----------------------------
# Basic game configuration
# -----------------------------

CELL_SIZE = 40
MAX_STEPS_PER_EPISODE = 800
STEPS_PER_TICK = 1
RUN_TIMER_INTERVAL_MS = 10
SPACE_STEP_INTERVAL_MS = 180

# Rewards
REWARD_STEP = -0.1
REWARD_INVALID = -0.2
REWARD_PELLET = 5.0
REWARD_CAUGHT = -250.0
REWARD_ALL_COLLECTED = 250.0
REWARD_TIMEOUT = -80.0

# Training defaults
ALPHA = 0.2
GAMMA = 0.95
EPS_START = 1.0
EPS_MIN = 0.05
EPS_DECAY = 0.995

# Improved state representation + shaping
GHOST_DXY_CLIP = 6
DIST_BINS = 6
DIST_BIN_CAP = 18
USE_REWARD_SHAPING = True
SHAPING_PELLET_DELTA = 0.25
SHAPING_GHOST_DELTA = 0.20

# Exploration schedule (periodic epsilon boosts)
EPS_BOOST_EVERY_EPISODES = 300
EPS_BOOST_VALUE = 0.20

# Domain randomization for training/evaluation
USE_DOMAIN_RANDOMIZATION = True

# Evaluation settings
EVAL_EVERY_EPISODES = 5000
EVAL_EPISODES = 50

# Advanced learning improvements
USE_DOUBLE_Q = True
N_STEP_RETURNS = 3
CURRICULUM_STAGE1_FRAC = 0.35
CURRICULUM_STAGE2_FRAC = 0.75

# Plot/Q-table viewer knobs
MA_WINDOW = 500
MA_HISTORY_MAX = 5000
QTABLE_VIEW_REFRESH_MS = 500
QTABLE_VIEW_MAX_ROWS = 2000

QTABLE_PATH = "q_table.json"


# -----------------------------
# Maze like classic Pacman
# # = wall, . = pellet corridor
# P = pacman start, G = ghost start
# -----------------------------

MAZE_LAYOUT = [
    "#####################",
    "#.....#...#...#.....#",
    "#.###.#.#.#.#.#.###.#",
    "#.#.....#...#.....#.#",
    "#.#.###.#####.###.#.#",
    "#...#...........#...#",
    "###.#.###.#.###.#.###",
    "#...#.#...P...#.#...#",
    "#.###.#.#####.#.###.#",
    "#.....#...#...#.....#",
    "#.###.###.#.###.###.#",
    "#..G....#...#....G..#",
    "#.###.#.#####.#.###.#",
    "#.....#...G...#.....#",
    "#####################",
]


# 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
ACTIONS = {
    0: (0, -1),
    1: (1, 0),
    2: (0, 1),
    3: (-1, 0),
}

ACTION_NAMES = {
    0: "UP",
    1: "RIGHT",
    2: "DOWN",
    3: "LEFT",
}

OPPOSITE_ACTION = {
    0: 2,
    1: 3,
    2: 0,
    3: 1,
}


@dataclass
class StepInfo:
    pellet: bool = False
    caught: bool = False
    invalid: bool = False
