import argparse
import json
import multiprocessing as mp
import os
import random
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from math import ceil

from PySide6.QtCore import QAbstractTableModel, QEvent, QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    _HAS_MPL = True
except Exception:
    FigureCanvas = None
    Figure = None
    FuncFormatter = None
    MaxNLocator = None
    _HAS_MPL = False


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


class TabularQLearningAgent:
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.q_table = defaultdict(lambda: [0.0] * self.n_actions)
        self.q_table_a = defaultdict(lambda: [0.0] * self.n_actions)
        self.q_table_b = defaultdict(lambda: [0.0] * self.n_actions)

        self.alpha = float(ALPHA)
        self.gamma = float(GAMMA)
        self.epsilon = float(EPS_START)
        self.epsilon_min = float(EPS_MIN)
        self.epsilon_decay = float(EPS_DECAY)
        self.use_epsilon_greedy = True
        self.episodes_seen = 0

    def _combined_q_values(self, state: tuple[int, ...]) -> list[float]:
        qa = self.q_table_a[state]
        qb = self.q_table_b[state]
        return [(qa[i] + qb[i]) * 0.5 for i in range(self.n_actions)]

    def _sync_combined_state(self, state: tuple[int, ...]):
        self.q_table[state] = self._combined_q_values(state)

    def _sync_all_combined_states(self):
        all_states = set(self.q_table_a.keys()) | set(self.q_table_b.keys()) | set(self.q_table.keys())
        for state in all_states:
            self._sync_combined_state(state)

    def set_hyperparams(
        self,
        *,
        alpha: float | None = None,
        gamma: float | None = None,
        eps_start: float | None = None,
        eps_min: float | None = None,
        eps_decay: float | None = None,
        set_current_eps_to_start: bool = False,
    ):
        if alpha is not None:
            self.alpha = float(alpha)
        if gamma is not None:
            self.gamma = float(gamma)
        if eps_min is not None:
            self.epsilon_min = float(eps_min)
        if eps_decay is not None:
            self.epsilon_decay = float(eps_decay)
        if eps_start is not None and set_current_eps_to_start:
            self.epsilon = float(eps_start)

        self.epsilon = max(self.epsilon_min, self.epsilon)

    def choose_greedy_action(self, state: tuple[int, ...]) -> int:
        q_values = self._combined_q_values(state) if USE_DOUBLE_Q else self.q_table[state]
        max_q = max(q_values)
        best = [a for a, q in enumerate(q_values) if q == max_q]
        return random.choice(best)

    def choose_action(self, state: tuple[int, ...]) -> int:
        if self.use_epsilon_greedy and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return self.choose_greedy_action(state)

    def update(self, state, action, reward, next_state, done):
        old_q = self.q_table[state][action]
        next_max = max(self.q_table[next_state])
        target = reward if done else reward + self.gamma * next_max
        self.q_table[state][action] = old_q + self.alpha * (target - old_q)

        # Keep Double-Q tables loosely in sync for GUI learning mode.
        self.q_table_a[state][action] = self.q_table[state][action]
        self.q_table_b[state][action] = self.q_table[state][action]
        self._sync_combined_state(state)
        self._sync_combined_state(next_state)

    def update_double_q_n_step(
        self,
        state: tuple[int, ...],
        action: int,
        return_n: float,
        next_state: tuple[int, ...],
        terminal: bool,
        n_steps: int,
    ):
        if not USE_DOUBLE_Q:
            target = return_n if terminal else return_n + (self.gamma ** n_steps) * max(self.q_table[next_state])
            old_q = self.q_table[state][action]
            self.q_table[state][action] = old_q + self.alpha * (target - old_q)
            return

        update_a = random.random() < 0.5
        q_upd = self.q_table_a if update_a else self.q_table_b
        q_eval = self.q_table_b if update_a else self.q_table_a

        if terminal:
            target = return_n
        else:
            q_upd_next = q_upd[next_state]
            best_action = max(range(self.n_actions), key=lambda a: q_upd_next[a])
            target = return_n + (self.gamma ** n_steps) * q_eval[next_state][best_action]

        old_q = q_upd[state][action]
        q_upd[state][action] = old_q + self.alpha * (target - old_q)

        self._sync_combined_state(state)
        self._sync_combined_state(next_state)

    def decay_epsilon(self):
        if self.use_epsilon_greedy:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def on_episode_end(self):
        self.episodes_seen += 1
        self.decay_epsilon()
        if self.use_epsilon_greedy and self.episodes_seen % EPS_BOOST_EVERY_EPISODES == 0:
            self.epsilon = max(self.epsilon, EPS_BOOST_VALUE)

    def reset_q_table(self):
        self.q_table.clear()
        self.q_table_a.clear()
        self.q_table_b.clear()

    def save(self, path: str):
        self._sync_all_combined_states()
        serializable = {
            "format": "pacman_qtable",
            "version": 1,
            "hyperparams": {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "use_epsilon_greedy": self.use_epsilon_greedy,
                "episodes_seen": self.episodes_seen,
            },
            "q_table": [
                {"state": list(state), "q_values": q_values}
                for state, q_values in self.q_table.items()
            ],
            "q_table_a": [
                {"state": list(state), "q_values": q_values}
                for state, q_values in self.q_table_a.items()
            ],
            "q_table_b": [
                {"state": list(state), "q_values": q_values}
                for state, q_values in self.q_table_b.items()
            ],
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(serializable, handle, indent=2)

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        hp = data.get("hyperparams", {})
        self.alpha = float(hp.get("alpha", self.alpha))
        self.gamma = float(hp.get("gamma", self.gamma))
        self.epsilon = float(hp.get("epsilon", self.epsilon))
        self.epsilon_min = float(hp.get("epsilon_min", self.epsilon_min))
        self.epsilon_decay = float(hp.get("epsilon_decay", self.epsilon_decay))
        self.use_epsilon_greedy = bool(hp.get("use_epsilon_greedy", self.use_epsilon_greedy))
        self.episodes_seen = int(hp.get("episodes_seen", self.episodes_seen))

        self.q_table = defaultdict(lambda: [0.0] * self.n_actions)
        self.q_table_a = defaultdict(lambda: [0.0] * self.n_actions)
        self.q_table_b = defaultdict(lambda: [0.0] * self.n_actions)

        def _normalize_state(raw_state) -> tuple[int, ...] | None:
            state = tuple(raw_state)
            if len(state) == 4:
                return (int(state[0]), int(state[1]), 0, 0, int(state[2]), int(state[3]))
            if len(state) == 6:
                return tuple(int(v) for v in state)
            return None

        q_table_a_rows = data.get("q_table_a")
        q_table_b_rows = data.get("q_table_b")

        if isinstance(q_table_a_rows, list) and isinstance(q_table_b_rows, list):
            for row in q_table_a_rows:
                q_values = list(row.get("q_values", []))
                if len(q_values) != self.n_actions:
                    continue
                state = _normalize_state(row.get("state", []))
                if state is None:
                    continue
                self.q_table_a[state] = [float(v) for v in q_values]

            for row in q_table_b_rows:
                q_values = list(row.get("q_values", []))
                if len(q_values) != self.n_actions:
                    continue
                state = _normalize_state(row.get("state", []))
                if state is None:
                    continue
                self.q_table_b[state] = [float(v) for v in q_values]
        else:
            # Backward-compatible loading from single-table files.
            for row in data.get("q_table", []):
                q_values = list(row.get("q_values", []))
                if len(q_values) != self.n_actions:
                    continue
                state = _normalize_state(row.get("state", []))
                if state is None:
                    continue
                vals = [float(v) for v in q_values]
                self.q_table_a[state] = vals[:]
                self.q_table_b[state] = vals[:]

        self._sync_all_combined_states()

        # Keep compatibility with older files where only q_table was present.
        for row in data.get("q_table", []):
            state = tuple(row.get("state", []))
            q_values = list(row.get("q_values", []))
            if len(q_values) != self.n_actions:
                continue

            norm_state = _normalize_state(state)
            if norm_state is None:
                continue

            if norm_state not in self.q_table:
                self.q_table[norm_state] = [float(v) for v in q_values]


class PacmanWorld:
    def __init__(self):
        self.layout = [list(row) for row in MAZE_LAYOUT]
        self.height = len(self.layout)
        self.width = len(self.layout[0])
        self.max_steps = MAX_STEPS_PER_EPISODE

        self.pacman_start = self._find_single("P")
        self.ghost_starts = self._find_all("G")
        if len(self.ghost_starts) != 3:
            raise ValueError("Maze must define exactly three ghost starts 'G'.")

        self.pellet_template = set()
        self.walkable_cells: list[tuple[int, int]] = []
        for y in range(self.height):
            for x in range(self.width):
                if self.layout[y][x] != "#":
                    self.walkable_cells.append((x, y))
                if self.layout[y][x] == ".":
                    self.pellet_template.add((x, y))

        self.agent = TabularQLearningAgent(n_actions=len(ACTIONS))

        self.pacman = self.pacman_start
        self.ghosts = list(self.ghost_starts)
        self.ghost_last_actions: list[int | None] = [None] * len(self.ghosts)
        self.pellets = set()
        self.steps = 0
        self.last_action = 1
        self.last_reward = 0.0
        self.last_info = StepInfo()
        self.score = 0.0
        self.done = False
        self.collected_pellets = 0
        self.caught_count = 0
        self.ghost_chase_prob = 0.75

        self._reward_block_sum = 0.0
        self._reward_block_count = 0
        self.reward_ma = 0.0
        self.reward_ma_history: list[float] = []
        self.reward_ma_dirty = False

        self._undo_snapshot: dict | None = None

        self.reset(clear_learning_curve=True)

    def _copy_step_info(self, info: StepInfo) -> StepInfo:
        return StepInfo(pellet=bool(info.pellet), caught=bool(info.caught), invalid=bool(info.invalid))

    def _append_ma_history(self, value: float):
        self.reward_ma_history.append(value)
        if len(self.reward_ma_history) > MA_HISTORY_MAX:
            old = self.reward_ma_history
            compressed: list[float] = []
            for i in range(0, len(old) - 1, 2):
                compressed.append((old[i] + old[i + 1]) * 0.5)
            if len(old) % 2 == 1:
                compressed.append(old[-1])
            self.reward_ma_history = compressed
        self.reward_ma_dirty = True

    def _find_single(self, marker: str) -> tuple[int, int]:
        for y in range(self.height):
            for x in range(self.width):
                if self.layout[y][x] == marker:
                    return x, y
        raise ValueError(f"Marker '{marker}' not found in maze")

    def _find_all(self, marker: str) -> list[tuple[int, int]]:
        coords = []
        for y in range(self.height):
            for x in range(self.width):
                if self.layout[y][x] == marker:
                    coords.append((x, y))
        return coords

    def clear_undo(self):
        self._undo_snapshot = None

    def reset(self, clear_learning_curve: bool = True, randomize_positions: bool = False):
        if randomize_positions and USE_DOMAIN_RANDOMIZATION:
            self.pacman = random.choice(self.walkable_cells)
            ghost_candidates = [c for c in self.walkable_cells if c != self.pacman]
            if len(ghost_candidates) >= len(self.ghost_starts):
                self.ghosts = random.sample(ghost_candidates, len(self.ghost_starts))
            else:
                self.ghosts = list(self.ghost_starts)
        else:
            self.pacman = self.pacman_start
            self.ghosts = list(self.ghost_starts)

        self.ghost_last_actions = [None] * len(self.ghosts)
        self.pellets = set(self.pellet_template)
        self.pellets.discard(self.pacman)
        self.steps = 0
        self.last_action = 1
        self.last_reward = 0.0
        self.last_info = StepInfo()
        self.score = 0.0
        self.done = False
        self.collected_pellets = 0
        self.caught_count = 0

        if clear_learning_curve:
            self._reward_block_sum = 0.0
            self._reward_block_count = 0
            self.reward_ma = 0.0
            self.reward_ma_history = []
            self.reward_ma_dirty = True

        self.clear_undo()

    def _is_walkable(self, pos: tuple[int, int]) -> bool:
        x, y = pos
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return self.layout[y][x] != "#"

    def _move(self, pos: tuple[int, int], action: int) -> tuple[tuple[int, int], bool]:
        dx, dy = ACTIONS[action]
        candidate = (pos[0] + dx, pos[1] + dy)
        if self._is_walkable(candidate):
            return candidate, True
        return pos, False

    def _ghost_policy(self, ghost_pos: tuple[int, int], ghost_index: int) -> int:
        last_action = self.ghost_last_actions[ghost_index]

        # Get all walkable actions (no backtracking allowed)
        walkable_actions = []
        for action in ACTIONS:
            # Never go backwards (follow the corridor)
            if last_action is not None and action == OPPOSITE_ACTION[last_action]:
                continue
            nxt, moved = self._move(ghost_pos, action)
            if moved:
                walkable_actions.append(action)
        
        # Should always have at least one valid action if no dead ends exist
        if not walkable_actions:
            # Fallback (should not happen): pick any valid move
            for action in ACTIONS:
                nxt, moved = self._move(ghost_pos, action)
                if moved:
                    return action
            # Last resort (should really not happen): pick any action
            return random.choice(list(ACTIONS.keys()))

        # Chase phase: pick best action to get closer to pacman
        if random.random() < self.ghost_chase_prob:
            best_dist = float("inf")
            best_actions = []
            for action in walkable_actions:
                nxt, moved = self._move(ghost_pos, action)
                dist = abs(nxt[0] - self.pacman[0]) + abs(nxt[1] - self.pacman[1])
                if dist < best_dist:
                    best_dist = dist
                    best_actions = [action]
                elif dist == best_dist:
                    best_actions.append(action)
            if best_actions:
                return random.choice(best_actions)

        # Random direction from walkable actions only (no backtracking)
        return random.choice(walkable_actions)

    def _nearest_pellet_direction(self) -> int:
        if not self.pellets:
            return 4
        px, py = self.pacman
        nearest = min(self.pellets, key=lambda p: abs(p[0] - px) + abs(p[1] - py))
        dx = nearest[0] - px
        dy = nearest[1] - py
        if abs(dx) > abs(dy):
            return 1 if dx > 0 else 3
        return 2 if dy > 0 else 0

    def _wall_flags(self) -> int:
        flags = 0
        px, py = self.pacman
        checks = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        for i, (dx, dy) in enumerate(checks):
            if not self._is_walkable((px + dx, py + dy)):
                flags |= (1 << i)
        return flags

    def _nearest_ghost(self) -> tuple[int, int]:
        px, py = self.pacman
        return min(self.ghosts, key=lambda g: abs(g[0] - px) + abs(g[1] - py))

    def _nearest_ghost_distance(self) -> int:
        px, py = self.pacman
        return min(abs(gx - px) + abs(gy - py) for (gx, gy) in self.ghosts)

    def _nearest_pellet_distance(self) -> int:
        if not self.pellets:
            return 0
        px, py = self.pacman
        return min(abs(px2 - px) + abs(py2 - py) for (px2, py2) in self.pellets)

    def _discretize_dist(self, value: int) -> int:
        clipped = max(0, min(DIST_BIN_CAP, int(value)))
        idx = int(clipped / max(1, DIST_BIN_CAP) * DIST_BINS)
        return max(0, min(DIST_BINS - 1, idx))

    def get_state(self) -> tuple[int, int, int, int, int, int]:
        px, py = self.pacman
        gx, gy = self._nearest_ghost()
        dx_ghost = max(-GHOST_DXY_CLIP, min(GHOST_DXY_CLIP, gx - px))
        dy_ghost = max(-GHOST_DXY_CLIP, min(GHOST_DXY_CLIP, gy - py))

        d_ghost_bin = self._discretize_dist(self._nearest_ghost_distance())
        d_pellet_bin = self._discretize_dist(self._nearest_pellet_distance())
        pellet_dir = self._nearest_pellet_direction()
        wall_bits = self._wall_flags()
        return dx_ghost, dy_ghost, d_ghost_bin, d_pellet_bin, pellet_dir, wall_bits

    def choose_auto_action(self, mode: str) -> int:
        if mode == "Random":
            return random.choice(list(ACTIONS.keys()))

        return self.agent.choose_greedy_action(self.get_state())

    def _capture_snapshot(self, *, update_q: bool, state: tuple[int, ...], action: int):
        snap = {
            "pacman": self.pacman,
            "ghosts": list(self.ghosts),
            "ghost_last_actions": list(self.ghost_last_actions),
            "pellets": set(self.pellets),
            "steps": self.steps,
            "last_action": self.last_action,
            "last_reward": self.last_reward,
            "last_info": self._copy_step_info(self.last_info),
            "score": self.score,
            "done": self.done,
            "collected_pellets": self.collected_pellets,
            "caught_count": self.caught_count,
            "reward_block_sum": self._reward_block_sum,
            "reward_block_count": self._reward_block_count,
            "reward_ma": self.reward_ma,
            "reward_ma_history": list(self.reward_ma_history),
            "reward_ma_dirty": self.reward_ma_dirty,
            "agent_eps": self.agent.epsilon,
            "update_q": update_q,
            "state": state,
            "action": action,
            "state_existed_before": state in self.agent.q_table,
            "q_sa_before": self.agent.q_table[state][action],
        }
        self._undo_snapshot = snap

    def _step_action(self, action: int, update_q: bool) -> tuple[float, bool, StepInfo]:
        if self.done:
            return 0.0, True, self.last_info

        state = self.get_state()
        d_pellet_before = self._nearest_pellet_distance()
        d_ghost_before = self._nearest_ghost_distance()
        self._capture_snapshot(update_q=update_q, state=state, action=action)

        self.steps += 1
        self.last_action = action
        reward = REWARD_STEP
        info = StepInfo()

        self.pacman, moved = self._move(self.pacman, action)
        if not moved:
            reward += REWARD_INVALID
            info.invalid = True

        if self.pacman in self.pellets:
            self.pellets.remove(self.pacman)
            reward += REWARD_PELLET
            info.pellet = True
            self.collected_pellets += 1

        new_ghost_positions = []
        for idx, ghost in enumerate(self.ghosts):
            ghost_action = self._ghost_policy(ghost, idx)
            moved_ghost, _ = self._move(ghost, ghost_action)
            new_ghost_positions.append(moved_ghost)
            self.ghost_last_actions[idx] = ghost_action
        self.ghosts = new_ghost_positions

        done = False
        if self.pacman in self.ghosts:
            reward += REWARD_CAUGHT
            info.caught = True
            self.caught_count += 1
            done = True

        if not self.pellets:
            reward += REWARD_ALL_COLLECTED
            done = True

        if self.steps >= self.max_steps:
            if (not done) and (not info.caught):
                reward += REWARD_TIMEOUT
            done = True

        next_state = self.get_state()
        train_reward = reward

        if USE_REWARD_SHAPING:
            d_pellet_after = self._nearest_pellet_distance()
            d_ghost_after = self._nearest_ghost_distance()
            pellet_term = SHAPING_PELLET_DELTA * float(d_pellet_before - d_pellet_after)
            ghost_term = SHAPING_GHOST_DELTA * float(d_ghost_after - d_ghost_before)
            train_reward += (pellet_term + ghost_term)

        if update_q:
            self._undo_snapshot["state_next"] = next_state
            self._undo_snapshot["state_next_existed_before"] = next_state in self.agent.q_table
            self.agent.update(state, action, train_reward, next_state, done)
            if done:
                self.agent.on_episode_end()

        self.done = done
        self.last_reward = reward
        self.last_info = info
        self.score += reward

        self._reward_block_sum += reward
        self._reward_block_count += 1
        if self._reward_block_count >= MA_WINDOW:
            self.reward_ma = self._reward_block_sum / self._reward_block_count
            self._append_ma_history(self.reward_ma)
            self._reward_block_sum = 0.0
            self._reward_block_count = 0

        return reward, done, info

    def step_learning(self) -> tuple[float, bool, StepInfo]:
        action = self.agent.choose_action(self.get_state())
        return self._step_action(action, update_q=True)

    def step_manual(self, action: int) -> tuple[float, bool, StepInfo]:
        return self._step_action(action, update_q=False)

    def undo_last_step(self) -> bool:
        snap = self._undo_snapshot
        if snap is None:
            return False

        self.pacman = snap["pacman"]
        self.ghosts = list(snap["ghosts"])
        self.ghost_last_actions = list(snap["ghost_last_actions"])
        self.pellets = set(snap["pellets"])
        self.steps = snap["steps"]
        self.last_action = snap["last_action"]
        self.last_reward = snap["last_reward"]
        self.last_info = self._copy_step_info(snap["last_info"])
        self.score = snap["score"]
        self.done = snap["done"]
        self.collected_pellets = snap["collected_pellets"]
        self.caught_count = snap["caught_count"]

        self._reward_block_sum = snap["reward_block_sum"]
        self._reward_block_count = snap["reward_block_count"]
        self.reward_ma = snap["reward_ma"]
        self.reward_ma_history = list(snap["reward_ma_history"])
        self.reward_ma_dirty = snap["reward_ma_dirty"]

        self.agent.epsilon = snap["agent_eps"]

        if snap["update_q"]:
            s = snap["state"]
            a = snap["action"]
            s_next = snap.get("state_next")
            s_existed_before = bool(snap["state_existed_before"])
            s_next_existed_before = bool(snap.get("state_next_existed_before", True))

            self.agent.q_table[s][a] = snap["q_sa_before"]

            if (not s_existed_before) and (s in self.agent.q_table):
                if not (s == s_next and s_next_existed_before):
                    del self.agent.q_table[s]

            if (
                s_next is not None
                and (not s_next_existed_before)
                and (s_next in self.agent.q_table)
                and (s_next != s)
            ):
                del self.agent.q_table[s_next]

        self._undo_snapshot = None
        return True


class WorldView(QWidget):
    GHOST_COLORS = [QColor("#ff4f64"), QColor("#61e7ff"), QColor("#ffb347")]

    def __init__(self, world: PacmanWorld):
        super().__init__()
        self.world = world
        self.setFixedSize(world.width * CELL_SIZE, world.height * CELL_SIZE)
        self.setFocusPolicy(Qt.StrongFocus)

    def _draw_walls(self, p: QPainter):
        for y in range(self.world.height):
            for x in range(self.world.width):
                if self.world.layout[y][x] != "#":
                    continue
                x0 = x * CELL_SIZE
                y0 = y * CELL_SIZE
                p.fillRect(x0, y0, CELL_SIZE, CELL_SIZE, QColor("#1f53d5"))
                p.setPen(QPen(QColor("#2f7bff"), 1))
                p.drawRect(x0, y0, CELL_SIZE, CELL_SIZE)

    def _draw_pellets(self, p: QPainter):
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#f7f4a5"))
        r = CELL_SIZE * 0.10
        for px, py in self.world.pellets:
            cx = px * CELL_SIZE + CELL_SIZE / 2
            cy = py * CELL_SIZE + CELL_SIZE / 2
            p.drawEllipse(int(cx - r), int(cy - r), int(2 * r), int(2 * r))

    def _draw_pacman(self, p: QPainter):
        px, py = self.world.pacman
        x0 = px * CELL_SIZE + 5
        y0 = py * CELL_SIZE + 5
        x1 = x0 + CELL_SIZE - 10
        y1 = y0 + CELL_SIZE - 10

        mouth_angles = {0: 120, 1: 30, 2: 300, 3: 210}
        start = mouth_angles.get(self.world.last_action, 30)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#ffd84d"))
        p.drawPie(x0, y0, x1 - x0, y1 - y0, int(start * 16), int(300 * 16))

    def _draw_ghost(self, p: QPainter, gx: int, gy: int, color: QColor):
        x0 = gx * CELL_SIZE + 5
        y0 = gy * CELL_SIZE + 5
        x1 = x0 + CELL_SIZE - 10
        y1 = y0 + CELL_SIZE - 10

        p.setPen(Qt.NoPen)
        p.setBrush(color)

        p.drawPie(x0, y0, x1 - x0, y1 - y0, 0, int(180 * 16))

        body_top = int((y0 + y1) / 2)
        p.drawRect(x0, body_top, x1 - x0, (y1 - 6) - body_top)

        step = (x1 - x0) / 4.0
        points = [
            (x0, y1 - 6),
            (x0 + step, y1),
            (x0 + 2 * step, y1 - 6),
            (x0 + 3 * step, y1),
            (x1, y1 - 6),
        ]
        p.drawPolygon(QPolygonF([QPointF(x, y) for (x, y) in points]))

        p.setBrush(QColor("white"))
        eye_r = 3
        lx = x0 + (x1 - x0) * 0.35
        rx = x0 + (x1 - x0) * 0.65
        ey = y0 + (y1 - y0) * 0.38
        p.drawEllipse(int(lx - eye_r), int(ey - eye_r), 2 * eye_r, 2 * eye_r)
        p.drawEllipse(int(rx - eye_r), int(ey - eye_r), 2 * eye_r, 2 * eye_r)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        p.fillRect(0, 0, self.width(), self.height(), QColor("#070714"))
        self._draw_walls(p)
        self._draw_pellets(p)
        self._draw_pacman(p)

        for index, (gx, gy) in enumerate(self.world.ghosts):
            color = self.GHOST_COLORS[index % len(self.GHOST_COLORS)]
            self._draw_ghost(p, gx, gy, color)

        p.end()


class RewardPlot(QWidget):
    def __init__(self, world: PacmanWorld):
        super().__init__()
        self.world = world
        self.setFixedSize(380, 120)
        self._last_signature: tuple[int, float] | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not _HAS_MPL:
            self._fallback = QLabel("Matplotlib not installed.\nRun: pip install matplotlib")
            self._fallback.setAlignment(Qt.AlignCenter)
            self._fallback.setFrameStyle(QFrame.Panel | QFrame.Sunken)
            layout.addWidget(self._fallback)
            self.canvas = None
            self.fig = None
            self.ax = None
            self._line = None
            return

        dpi = 100
        self.fig = Figure(figsize=(self.width() / dpi, self.height() / dpi), dpi=dpi)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.ax = self.fig.add_subplot(111)
        self._line, = self.ax.plot([], [], linewidth=1.8)
        self.ax.grid(True, alpha=0.25)

        for spine in ("top", "right"):
            self.ax.spines[spine].set_visible(False)

        self.ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))

        def _fmt_steps(x, _pos):
            n = int(x)
            if n >= 1_000_000:
                return f"{n / 1_000_000:.1f}M"
            if n >= 1_000:
                return f"{n / 1_000:.0f}k"
            return str(n)

        self.ax.xaxis.set_major_formatter(FuncFormatter(_fmt_steps))
        self.ax.tick_params(axis="both", labelsize=8)

        self.fig.subplots_adjust(left=0.18, right=0.98, top=0.96, bottom=0.28)
        self.refresh(force=True)

    def refresh(self, force: bool = False):
        if not _HAS_MPL or self.canvas is None:
            return

        history = self.world.reward_ma_history
        if not history:
            self._line.set_data([], [])
            self.ax.set_xlim(0, MA_WINDOW)
            self.canvas.draw_idle()
            return

        signature = (len(history), float(history[-1]))
        if (not force) and self._last_signature == signature:
            return
        self._last_signature = signature

        n = len(history)
        xs = [i * MA_WINDOW for i in range(n)]
        ys = history
        self._line.set_data(xs, ys)

        self.ax.set_xlim(0, max(MA_WINDOW, int((n - 1) * MA_WINDOW)))
        ymin = min(ys)
        ymax = max(ys)
        if abs(ymax - ymin) < 1e-9:
            ymax = ymin + 1.0
        pad = 0.08 * (ymax - ymin)
        self.ax.set_ylim(ymin - pad, ymax + pad)

        self.canvas.draw_idle()


class QTableModel(QAbstractTableModel):
    _STATE_COLUMNS = ["dx_g", "dy_g", "d_g_bin", "d_p_bin", "pellet_dir", "walls"]

    def __init__(self):
        super().__init__()
        self._rows: list[tuple[tuple[int, int, int, int], list[float]]] = []

    def set_rows(self, rows: list[tuple[tuple[int, int, int, int], list[float]]]):
        self.beginResetModel()
        self._rows = rows
        self.endResetModel()

    def rowCount(self, _parent=None) -> int:
        return len(self._rows)

    def columnCount(self, _parent=None) -> int:
        return 11

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            cols = self._STATE_COLUMNS + ["Q(U)", "Q(R)", "Q(D)", "Q(L)", "best"]
            return cols[section] if 0 <= section < len(cols) else None
        return str(section)

    def data(self, index, role: int = Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None

        state, q = self._rows[index.row()]
        col = index.column()

        if 0 <= col <= 5:
            return str(state[col])

        if 6 <= col <= 9:
            return f"{q[col - 6]:.3f}"

        if col == 10:
            best = max(range(4), key=lambda a: q[a])
            return ACTION_NAMES[best]

        return None


class QTableDialog(QDialog):
    def __init__(self, world: PacmanWorld, parent=None):
        super().__init__(parent)
        self.world = world
        self.setWindowTitle("Q-table")
        self.setMinimumSize(720, 480)

        self.lbl_info = QLabel()
        self.lbl_info.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.model = QTableModel()
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.setSelectionMode(QTableView.SingleSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        layout = QVBoxLayout()
        layout.addWidget(self.lbl_info)
        layout.addWidget(self.table, 1)
        self.setLayout(layout)

        self._timer = QTimer(self)
        self._timer.setInterval(QTABLE_VIEW_REFRESH_MS)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()

        self.refresh()

    def closeEvent(self, event):
        self._timer.stop()
        return super().closeEvent(event)

    def refresh(self):
        items = list(self.world.agent.q_table.items())
        items.sort(key=lambda kv: max(kv[1]), reverse=True)

        total = len(items)
        shown = min(total, QTABLE_VIEW_MAX_ROWS)
        rows = items[:shown]

        self.lbl_info.setText(
            f"states learned: {total}    showing: {shown}    refresh: {QTABLE_VIEW_REFRESH_MS}ms"
        )
        self.model.set_rows(rows)


class MainWindow(QMainWindow):
    def __init__(self, qtable_path: str = QTABLE_PATH):
        super().__init__()
        self.setWindowTitle("Pacman Q-Learning (PySide6)")

        self.qtable_path = qtable_path
        self.world = PacmanWorld()
        self.view = WorldView(self.world)
        self.reward_plot = RewardPlot(self.world)
        self.q_table_dialog: QTableDialog | None = None

        if os.path.exists(self.qtable_path):
            try:
                self.world.agent.load(self.qtable_path)
            except Exception:
                pass

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_tick)
        self.timer.setInterval(RUN_TIMER_INTERVAL_MS)

        self.space_timer = QTimer(self)
        self.space_timer.timeout.connect(self.single_step)
        self.space_timer.setInterval(SPACE_STEP_INTERVAL_MS)

        self.btn_run = QPushButton("Run")
        self.btn_step = QPushButton("Step")
        self.btn_undo = QPushButton("Undo")
        self.btn_reset = QPushButton("Reset")
        self.btn_save_q = QPushButton("Save Q-table")
        self.btn_load_q = QPushButton("Load Q-table")
        self.btn_view_q = QPushButton("View Q-table")
        self.btn_reset_q = QPushButton("Reset Q-table")

        self.combo_auto = QComboBox()
        self.combo_auto.addItems(["Q-Learning", "Q-Table", "Random"])

        self.chk_auto_restart = QCheckBox("Auto-Restart")
        self.chk_auto_restart.setChecked(True)

        self.chk_eps_greedy = QCheckBox("epsilon-greedy")
        self.chk_eps_greedy.setChecked(True)
        self.chk_eps_greedy.toggled.connect(self.on_toggle_eps_greedy)

        self.spin_alpha = QDoubleSpinBox()
        self.spin_alpha.setDecimals(4)
        self.spin_alpha.setSingleStep(0.01)
        self.spin_alpha.setRange(0.0, 1.0)
        self.spin_alpha.setValue(ALPHA)
        self.spin_alpha.valueChanged.connect(self.on_hyperparams_changed)

        self.spin_gamma = QDoubleSpinBox()
        self.spin_gamma.setDecimals(4)
        self.spin_gamma.setSingleStep(0.01)
        self.spin_gamma.setRange(0.0, 0.9999)
        self.spin_gamma.setValue(GAMMA)
        self.spin_gamma.valueChanged.connect(self.on_hyperparams_changed)

        self.spin_eps_start = QDoubleSpinBox()
        self.spin_eps_start.setDecimals(4)
        self.spin_eps_start.setSingleStep(0.05)
        self.spin_eps_start.setRange(0.0, 1.0)
        self.spin_eps_start.setValue(EPS_START)
        self.spin_eps_start.valueChanged.connect(self.on_hyperparams_changed)

        self.spin_eps_min = QDoubleSpinBox()
        self.spin_eps_min.setDecimals(4)
        self.spin_eps_min.setSingleStep(0.01)
        self.spin_eps_min.setRange(0.0, 1.0)
        self.spin_eps_min.setValue(EPS_MIN)
        self.spin_eps_min.valueChanged.connect(self.on_hyperparams_changed)

        self.spin_eps_decay = QDoubleSpinBox()
        self.spin_eps_decay.setDecimals(6)
        self.spin_eps_decay.setSingleStep(0.0001)
        self.spin_eps_decay.setRange(0.9, 0.999999)
        self.spin_eps_decay.setValue(EPS_DECAY)
        self.spin_eps_decay.valueChanged.connect(self.on_hyperparams_changed)

        self.lbl_status = QLabel()
        self.lbl_status.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.lbl_status.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.lbl_status.setFont(QFont("Arial", 10))
        self.lbl_status.setMinimumHeight(170)

        self.lbl_plot_title = QLabel("Avg reward per 500 steps")
        self.lbl_plot_title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.btn_run.clicked.connect(self.toggle_run)
        self.btn_step.clicked.connect(self.single_step)
        self.btn_undo.clicked.connect(self.undo_last_step)
        self.btn_reset.clicked.connect(self.reset_world)
        self.btn_save_q.clicked.connect(self.save_q_table)
        self.btn_load_q.clicked.connect(self.load_q_table)
        self.btn_view_q.clicked.connect(self.open_q_table_viewer)
        self.btn_reset_q.clicked.connect(self.reset_q_table)

        for b in (
            self.btn_run,
            self.btn_step,
            self.btn_undo,
            self.btn_reset,
            self.btn_save_q,
            self.btn_load_q,
            self.btn_view_q,
            self.btn_reset_q,
        ):
            b.setMinimumHeight(26)

        controls = QVBoxLayout()

        top_buttons = QHBoxLayout()
        top_buttons.setSpacing(6)
        top_buttons.addWidget(self.btn_run)
        top_buttons.addWidget(self.btn_step)
        top_buttons.addWidget(self.btn_undo)
        top_buttons.addWidget(self.btn_reset)
        controls.addLayout(top_buttons)

        controls.addSpacing(6)
        controls.addWidget(QLabel("Policy"))
        controls.addWidget(self.combo_auto)
        controls.addWidget(self.chk_auto_restart)

        controls.addSpacing(6)
        controls.addWidget(QLabel("Hyperparameters"))
        hyper_form = QFormLayout()
        hyper_form.setContentsMargins(0, 0, 0, 0)
        hyper_form.setHorizontalSpacing(8)
        hyper_form.setVerticalSpacing(4)
        hyper_form.addRow("ALPHA:", self.spin_alpha)
        hyper_form.addRow("GAMMA:", self.spin_gamma)
        hyper_form.addRow("EPS_START:", self.spin_eps_start)
        hyper_form.addRow("EPS_MIN:", self.spin_eps_min)
        hyper_form.addRow("EPS_DECAY:", self.spin_eps_decay)
        controls.addLayout(hyper_form)
        controls.addWidget(self.chk_eps_greedy)

        controls.addSpacing(6)
        qtable_buttons = QHBoxLayout()
        qtable_buttons.setSpacing(6)
        qtable_buttons.addWidget(self.btn_save_q)
        qtable_buttons.addWidget(self.btn_load_q)
        qtable_buttons.addWidget(self.btn_view_q)
        qtable_buttons.addWidget(self.btn_reset_q)
        controls.addLayout(qtable_buttons)

        controls.addSpacing(10)
        controls.addWidget(self.lbl_status)
        controls.addSpacing(6)
        controls.addWidget(self.lbl_plot_title)
        controls.addWidget(self.reward_plot)
        controls.addSpacing(10)
        controls.addStretch(1)

        root = QHBoxLayout()
        root.addWidget(self.view)
        root.addLayout(controls)

        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

        self.on_hyperparams_changed()
        self.update_status_panel()

    def update_status_panel(self):
        mode = self.combo_auto.currentText()
        done_text = "yes" if self.world.done else "no"
        info = self.world.last_info

        exploration_rate = self.world.agent.epsilon if self.world.agent.use_epsilon_greedy else 0.0

        self.lbl_status.setText(
            f"steps: {self.world.steps}\n"
            f"alpha: {self.world.agent.alpha:.3f}   gamma: {self.world.agent.gamma:.3f}\n"
            f"epsilon: {self.world.agent.epsilon:.3f} (min {self.world.agent.epsilon_min:.2f}, decay {self.world.agent.epsilon_decay:.6f})\n"
            f"exploration rate: {exploration_rate * 100.0:.1f}%\n"
            f"policy: {mode}\n"
            f"auto-restart: {self.chk_auto_restart.isChecked()}\n"
            f"states learned: {len(self.world.agent.q_table)}\n"
            f"score: {self.world.score:.2f}\n"
            f"ma500(r): {self.world.reward_ma:.3f}\n\n"
            f"last action: {ACTION_NAMES.get(self.world.last_action, '-')}\n"
            f"last reward: {self.world.last_reward:.2f}\n"
            f"pellet eaten: {info.pellet} | invalid: {info.invalid}\n"
            f"caught: {info.caught} | done: {done_text}\n"
            f"pellets collected: {self.world.collected_pellets}"
        )

    def on_toggle_eps_greedy(self, checked: bool):
        self.world.agent.use_epsilon_greedy = bool(checked)
        self.update_status_panel()

    def on_hyperparams_changed(self):
        eps_min = float(self.spin_eps_min.value())
        if self.spin_eps_start.value() < eps_min:
            self.spin_eps_start.blockSignals(True)
            self.spin_eps_start.setValue(eps_min)
            self.spin_eps_start.blockSignals(False)

        self.world.agent.set_hyperparams(
            alpha=float(self.spin_alpha.value()),
            gamma=float(self.spin_gamma.value()),
            eps_start=float(self.spin_eps_start.value()),
            eps_min=float(self.spin_eps_min.value()),
            eps_decay=float(self.spin_eps_decay.value()),
            set_current_eps_to_start=False,
        )
        self.update_status_panel()

    def _one_step(self):
        mode = self.combo_auto.currentText()

        if mode == "Q-Learning":
            self.world.step_learning()
        else:
            action = self.world.choose_auto_action(mode)
            self.world.step_manual(action)

        self.update_status_panel()
        self.view.update()
        if self.world.reward_ma_dirty:
            self.reward_plot.refresh()
            self.world.reward_ma_dirty = False

        if self.world.done and self.timer.isActive():
            if self.chk_auto_restart.isChecked():
                randomize_positions = self.combo_auto.currentText() == "Q-Learning"
                self.world.reset(clear_learning_curve=False, randomize_positions=randomize_positions)
                self.view.update()
                self.update_status_panel()
            else:
                self.timer.stop()
                self.btn_run.setText("Run")

    def on_tick(self):
        mode = self.combo_auto.currentText()
        loops = STEPS_PER_TICK if mode == "Q-Learning" else 1
        for _ in range(loops):
            self._one_step()
            if self.world.done:
                break

    def toggle_run(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_run.setText("Run")
            return

        if self.world.done:
            randomize_positions = self.combo_auto.currentText() == "Q-Learning"
            self.world.reset(clear_learning_curve=False, randomize_positions=randomize_positions)
            self.view.update()
            self.update_status_panel()

        self.space_timer.stop()
        self.timer.start()
        self.btn_run.setText("Pause")

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Space:
            if not event.isAutoRepeat() and (not self.timer.isActive()):
                self.single_step()
                if not self.space_timer.isActive():
                    self.space_timer.start()
            return True

        if event.type() == QEvent.KeyRelease and event.key() == Qt.Key_Space:
            if not event.isAutoRepeat():
                self.space_timer.stop()
            return True

        return super().eventFilter(obj, event)

    def single_step(self):
        if self.timer.isActive():
            return
        self._one_step()

    def undo_last_step(self):
        if self.timer.isActive():
            return

        if not self.world.undo_last_step():
            return

        self.update_status_panel()
        self.view.update()
        self.reward_plot.refresh(force=True)
        if self.q_table_dialog is not None:
            self.q_table_dialog.refresh()

    def reset_world(self):
        was_running = self.timer.isActive()
        self.timer.stop()
        self.space_timer.stop()

        self.world = PacmanWorld()
        self.world.agent.use_epsilon_greedy = bool(self.chk_eps_greedy.isChecked())
        self.world.agent.set_hyperparams(
            alpha=float(self.spin_alpha.value()),
            gamma=float(self.spin_gamma.value()),
            eps_start=float(self.spin_eps_start.value()),
            eps_min=float(self.spin_eps_min.value()),
            eps_decay=float(self.spin_eps_decay.value()),
            set_current_eps_to_start=True,
        )

        self.view.world = self.world
        self.reward_plot.world = self.world
        if self.q_table_dialog is not None:
            self.q_table_dialog.world = self.world
            self.q_table_dialog.refresh()

        self.view.update()
        self.update_status_panel()
        self.reward_plot.refresh(force=True)
        self.world.reward_ma_dirty = False

        self.btn_run.setText("Run")
        if was_running:
            self.timer.start()
            self.btn_run.setText("Pause")

    def save_q_table(self):
        was_running = self.timer.isActive()
        self.timer.stop()
        self.space_timer.stop()

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Q-table",
            self.qtable_path,
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            if was_running:
                self.timer.start()
                self.btn_run.setText("Pause")
            return

        try:
            self.world.agent.save(path)
            self.qtable_path = path
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

        if was_running:
            self.timer.start()
            self.btn_run.setText("Pause")

    def load_q_table(self):
        was_running = self.timer.isActive()
        self.timer.stop()
        self.space_timer.stop()

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Q-table",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            if was_running:
                self.timer.start()
                self.btn_run.setText("Pause")
            return

        try:
            self.world.agent.load(path)
            self.world.clear_undo()
            self.qtable_path = path

            self.chk_eps_greedy.setChecked(bool(self.world.agent.use_epsilon_greedy))
            self.spin_alpha.setValue(float(self.world.agent.alpha))
            self.spin_gamma.setValue(float(self.world.agent.gamma))
            self.spin_eps_start.setValue(float(max(self.world.agent.epsilon, self.world.agent.epsilon_min)))
            self.spin_eps_min.setValue(float(self.world.agent.epsilon_min))
            self.spin_eps_decay.setValue(float(self.world.agent.epsilon_decay))

            self.update_status_panel()
            if self.q_table_dialog is not None:
                self.q_table_dialog.refresh()
        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))

        if was_running:
            self.timer.start()
            self.btn_run.setText("Pause")

    def reset_q_table(self):
        was_running = self.timer.isActive()
        self.timer.stop()
        self.space_timer.stop()

        answer = QMessageBox.question(
            self,
            "Reset Q-table",
            "Delete all learned Q-values from memory?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if answer == QMessageBox.Yes:
            self.world.agent.reset_q_table()
            self.world.clear_undo()
            self.update_status_panel()
            if self.q_table_dialog is not None:
                self.q_table_dialog.refresh()

        if was_running:
            self.timer.start()
            self.btn_run.setText("Pause")

    def open_q_table_viewer(self):
        if self.q_table_dialog is None:
            self.q_table_dialog = QTableDialog(self.world, self)
        else:
            self.q_table_dialog.world = self.world
            self.q_table_dialog.refresh()

        self.q_table_dialog.show()
        self.q_table_dialog.raise_()
        self.q_table_dialog.activateWindow()


def _table_to_plain(table: dict[tuple[int, ...], list[float]]) -> dict[tuple[int, ...], list[float]]:
    return {state: list(values) for state, values in table.items()}


def _count_table_to_plain(table: dict[tuple[int, ...], list[int]]) -> dict[tuple[int, ...], list[int]]:
    return {state: list(values) for state, values in table.items()}


def _agent_to_payload(agent: TabularQLearningAgent) -> dict:
    return {
        "q_table": _table_to_plain(agent.q_table),
        "q_table_a": _table_to_plain(agent.q_table_a),
        "q_table_b": _table_to_plain(agent.q_table_b),
        "alpha": float(agent.alpha),
        "gamma": float(agent.gamma),
        "epsilon": float(agent.epsilon),
        "epsilon_min": float(agent.epsilon_min),
        "epsilon_decay": float(agent.epsilon_decay),
        "use_epsilon_greedy": bool(agent.use_epsilon_greedy),
        "episodes_seen": int(agent.episodes_seen),
    }


def _payload_to_agent(agent: TabularQLearningAgent, payload: dict):
    agent.reset_q_table()

    for state, values in payload.get("q_table", {}).items():
        agent.q_table[state] = list(values)
    for state, values in payload.get("q_table_a", {}).items():
        agent.q_table_a[state] = list(values)
    for state, values in payload.get("q_table_b", {}).items():
        agent.q_table_b[state] = list(values)

    agent.alpha = float(payload.get("alpha", agent.alpha))
    agent.gamma = float(payload.get("gamma", agent.gamma))
    agent.epsilon = float(payload.get("epsilon", agent.epsilon))
    agent.epsilon_min = float(payload.get("epsilon_min", agent.epsilon_min))
    agent.epsilon_decay = float(payload.get("epsilon_decay", agent.epsilon_decay))
    agent.use_epsilon_greedy = bool(payload.get("use_epsilon_greedy", agent.use_epsilon_greedy))
    agent.episodes_seen = int(payload.get("episodes_seen", agent.episodes_seen))
    agent._sync_all_combined_states()


def evaluate_policy(agent: TabularQLearningAgent, n_episodes: int) -> tuple[float, float, float]:
    eval_world = PacmanWorld()
    total_reward = 0.0
    wins = 0
    total_pellets = 0

    for _ in range(n_episodes):
        eval_world.agent = agent
        eval_world.reset(clear_learning_curve=False, randomize_positions=USE_DOMAIN_RANDOMIZATION)
        done = False
        caught = False

        while not done:
            action = eval_world.agent.choose_greedy_action(eval_world.get_state())
            reward, done, info = eval_world.step_manual(action)
            total_reward += reward
            if info.caught:
                caught = True

        total_pellets += eval_world.collected_pellets
        if (not caught) and (len(eval_world.pellets) == 0):
            wins += 1

    avg_reward = total_reward / max(1, n_episodes)
    win_rate = wins / max(1, n_episodes)
    avg_pellets = total_pellets / max(1, n_episodes)
    return avg_reward, win_rate, avg_pellets


def _train_episodes_range(
    world: PacmanWorld,
    episodes: int,
    *,
    total_episodes: int,
    episode_offset: int,
    count_table_a: dict[tuple[int, ...], list[int]] | None = None,
    count_table_b: dict[tuple[int, ...], list[int]] | None = None,
) -> list[float]:
    rewards: list[float] = []

    for local_episode in range(1, episodes + 1):
        global_episode = episode_offset + local_episode
        progress = global_episode / max(1, total_episodes)
        if progress < CURRICULUM_STAGE1_FRAC:
            world.ghost_chase_prob = 0.55
            randomize_positions = False
        elif progress < CURRICULUM_STAGE2_FRAC:
            world.ghost_chase_prob = 0.68
            randomize_positions = USE_DOMAIN_RANDOMIZATION
        else:
            world.ghost_chase_prob = 0.80
            randomize_positions = USE_DOMAIN_RANDOMIZATION

        world.reset(clear_learning_curve=False, randomize_positions=randomize_positions)
        done = False
        total_reward = 0.0
        nstep_buffer: deque[tuple[tuple[int, ...], int, float, tuple[int, ...], bool]] = deque()

        def _flush_one_transition():
            if not nstep_buffer:
                return

            n = min(N_STEP_RETURNS, len(nstep_buffer))
            ret = 0.0
            for i in range(n):
                ret += (world.agent.gamma ** i) * nstep_buffer[i][2]

            s0, a0, _r0, _sn0, _d0 = nstep_buffer[0]
            _sn, _a, _r, s_n, done_n = nstep_buffer[n - 1]

            if USE_DOUBLE_Q:
                update_a = random.random() < 0.5
                q_upd = world.agent.q_table_a if update_a else world.agent.q_table_b
                q_eval = world.agent.q_table_b if update_a else world.agent.q_table_a

                if done_n:
                    target = ret
                else:
                    q_upd_next = q_upd[s_n]
                    best_action = max(range(world.agent.n_actions), key=lambda a: q_upd_next[a])
                    target = ret + (world.agent.gamma ** n) * q_eval[s_n][best_action]

                old_q = q_upd[s0][a0]
                q_upd[s0][a0] = old_q + world.agent.alpha * (target - old_q)
                world.agent._sync_combined_state(s0)
                world.agent._sync_combined_state(s_n)

                if update_a and count_table_a is not None:
                    count_table_a.setdefault(s0, [0, 0, 0, 0])[a0] += 1
                if (not update_a) and count_table_b is not None:
                    count_table_b.setdefault(s0, [0, 0, 0, 0])[a0] += 1
            else:
                target = ret if done_n else ret + (world.agent.gamma ** n) * max(world.agent.q_table[s_n])
                old_q = world.agent.q_table[s0][a0]
                world.agent.q_table[s0][a0] = old_q + world.agent.alpha * (target - old_q)
                world.agent._sync_combined_state(s0)
                world.agent._sync_combined_state(s_n)
                if count_table_a is not None:
                    count_table_a.setdefault(s0, [0, 0, 0, 0])[a0] += 1

            nstep_buffer.popleft()

        while not done:
            state = world.get_state()
            action = world.agent.choose_action(state)
            reward, done, _info = world._step_action(action, update_q=False)
            next_state = world.get_state()

            nstep_buffer.append((state, action, reward, next_state, done))
            if len(nstep_buffer) >= N_STEP_RETURNS:
                _flush_one_transition()

            total_reward += reward

        while nstep_buffer:
            _flush_one_transition()

        world.agent.on_episode_end()
        rewards.append(total_reward)

    return rewards


def _train_worker(task: dict) -> dict:
    seed = int(task["seed"])
    random.seed(seed)

    world = PacmanWorld()
    _payload_to_agent(world.agent, task["snapshot"])

    counts_a: dict[tuple[int, ...], list[int]] = {}
    counts_b: dict[tuple[int, ...], list[int]] = {}

    rewards = _train_episodes_range(
        world,
        int(task["episodes"]),
        total_episodes=int(task["total_episodes"]),
        episode_offset=int(task["episode_offset"]),
        count_table_a=counts_a,
        count_table_b=counts_b,
    )

    return {
        "snapshot": _agent_to_payload(world.agent),
        "counts_a": _count_table_to_plain(counts_a),
        "counts_b": _count_table_to_plain(counts_b),
        "episodes": int(task["episodes"]),
        "sum_reward": float(sum(rewards)),
    }


def _merge_worker_result(
    global_agent: TabularQLearningAgent,
    global_counts_a: dict[tuple[int, ...], list[int]],
    global_counts_b: dict[tuple[int, ...], list[int]],
    result: dict,
):
    worker = result["snapshot"]
    worker_qa = worker.get("q_table_a", {})
    worker_qb = worker.get("q_table_b", {})
    worker_counts_a = result.get("counts_a", {})
    worker_counts_b = result.get("counts_b", {})

    def _merge_one(
        global_table: dict[tuple[int, ...], list[float]],
        counts_global: dict[tuple[int, ...], list[int]],
        local_table: dict[tuple[int, ...], list[float]],
        counts_local: dict[tuple[int, ...], list[int]],
    ):
        for state, local_counts in counts_local.items():
            cg = counts_global.setdefault(state, [0, 0, 0, 0])
            lq = local_table.get(state)
            if lq is None:
                continue
            gq = global_table[state]

            for a in range(4):
                ln = int(local_counts[a])
                if ln <= 0:
                    continue
                gn = int(cg[a])
                gq[a] = (gq[a] * gn + float(lq[a]) * ln) / float(gn + ln)
                cg[a] = gn + ln

    if USE_DOUBLE_Q:
        _merge_one(global_agent.q_table_a, global_counts_a, worker_qa, worker_counts_a)
        _merge_one(global_agent.q_table_b, global_counts_b, worker_qb, worker_counts_b)
        global_agent._sync_all_combined_states()
    else:
        worker_q = worker.get("q_table", {})
        _merge_one(global_agent.q_table, global_counts_a, worker_q, worker_counts_a)

    worker_eps = float(worker.get("epsilon", global_agent.epsilon))
    global_agent.epsilon = min(global_agent.epsilon, worker_eps)
    global_agent.episodes_seen += int(result.get("episodes", 0))


def train_agent_single(episodes: int, qtable_path: str):
    world = PacmanWorld()

    if os.path.exists(qtable_path):
        world.agent.load(qtable_path)
        print(f"Loaded Q-table from {qtable_path} with {len(world.agent.q_table)} states")

    report_every = 1000
    rolling_rewards = _train_episodes_range(
        world,
        episodes,
        total_episodes=episodes,
        episode_offset=0,
    )

    for episode in range(report_every, episodes + 1, report_every):
        avg = sum(rolling_rewards[episode - report_every : episode]) / report_every
        print(
            f"Episode {episode:6d} | avg_reward={avg:8.2f} | "
            f"epsilon={world.agent.epsilon:.3f}"
        )

    for episode in range(EVAL_EVERY_EPISODES, episodes + 1, EVAL_EVERY_EPISODES):
        eval_reward, eval_win_rate, eval_pellets = evaluate_policy(world.agent, EVAL_EPISODES)
        print(
            f"  Eval@{episode:6d} | avg_reward={eval_reward:8.2f} | "
            f"win_rate={eval_win_rate * 100.0:5.1f}% | avg_pellets={eval_pellets:6.2f}"
        )

    world.agent.save(qtable_path)
    print(f"Saved Q-table to {qtable_path}")


def train_agent_multiprocess(episodes: int, qtable_path: str, workers: int, chunk_episodes: int):
    world = PacmanWorld()

    if os.path.exists(qtable_path):
        world.agent.load(qtable_path)
        print(f"Loaded Q-table from {qtable_path} with {len(world.agent.q_table)} states")

    workers = max(1, int(workers))
    chunk_episodes = max(50, int(chunk_episodes))

    done = 0
    round_idx = 0
    global_counts_a: dict[tuple[int, ...], list[int]] = {}
    global_counts_b: dict[tuple[int, ...], list[int]] = {}

    ctx = mp.get_context("spawn")

    while done < episodes:
        remaining = episodes - done
        batch_workers = min(workers, int(ceil(remaining / chunk_episodes)))
        snapshot = _agent_to_payload(world.agent)

        tasks = []
        assigned = 0
        for i in range(batch_workers):
            ep_i = min(chunk_episodes, remaining - assigned)
            task = {
                "seed": random.randint(1, 2**31 - 1),
                "snapshot": snapshot,
                "episodes": ep_i,
                "total_episodes": episodes,
                "episode_offset": done + assigned,
            }
            tasks.append(task)
            assigned += ep_i
            if assigned >= remaining:
                break

        with ctx.Pool(processes=len(tasks)) as pool:
            results = pool.map(_train_worker, tasks)

        merged_episodes = 0
        merged_reward_sum = 0.0
        for res in results:
            _merge_worker_result(world.agent, global_counts_a, global_counts_b, res)
            merged_episodes += int(res.get("episodes", 0))
            merged_reward_sum += float(res.get("sum_reward", 0.0))

        done += merged_episodes
        round_idx += 1
        avg_reward = merged_reward_sum / max(1, merged_episodes)
        print(
            f"MP round {round_idx:4d} | done={done:7d}/{episodes:7d} | "
            f"workers={len(tasks)} | avg_reward={avg_reward:8.2f} | epsilon={world.agent.epsilon:.3f}"
        )

        if done > 0 and done % EVAL_EVERY_EPISODES == 0:
            eval_reward, eval_win_rate, eval_pellets = evaluate_policy(world.agent, EVAL_EPISODES)
            print(
                f"  Eval@{done:6d} | avg_reward={eval_reward:8.2f} | "
                f"win_rate={eval_win_rate * 100.0:5.1f}% | avg_pellets={eval_pellets:6.2f}"
            )

    world.agent.save(qtable_path)
    print(f"Saved Q-table to {qtable_path}")


def train_agent(episodes: int, qtable_path: str, workers: int = 1, chunk_episodes: int = 2000):
    if workers <= 1:
        train_agent_single(episodes=episodes, qtable_path=qtable_path)
        return
    train_agent_multiprocess(
        episodes=episodes,
        qtable_path=qtable_path,
        workers=workers,
        chunk_episodes=chunk_episodes,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Pacman Q-learning with GUI and headless training")
    parser.add_argument("--train", type=int, default=0, help="number of training episodes (headless)")
    parser.add_argument("--view", action="store_true", help="start GUI viewer")
    parser.add_argument("--qtable", type=str, default=QTABLE_PATH, help="q-table json path")
    parser.add_argument("--workers", type=int, default=1, help="number of worker processes for training")
    parser.add_argument("--chunk-episodes", type=int, default=2000, help="episodes per worker chunk")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.train > 0:
        train_agent(
            episodes=args.train,
            qtable_path=args.qtable,
            workers=args.workers,
            chunk_episodes=args.chunk_episodes,
        )

    if args.view or args.train == 0:
        app = QApplication(sys.argv)
        win = MainWindow(qtable_path=args.qtable)
        win.show()
        sys.exit(app.exec())


if __name__ == "__main__":
    main()
