import json
import multiprocessing as mp
import os
import random
from collections import defaultdict, deque
from math import ceil

from pacman_globals import (
    ACTIONS,
    ALPHA,
    CURRICULUM_STAGE1_FRAC,
    CURRICULUM_STAGE2_FRAC,
    DIST_BIN_CAP,
    DIST_BINS,
    EPS_BOOST_EVERY_EPISODES,
    EPS_BOOST_VALUE,
    EPS_DECAY,
    EPS_MIN,
    EPS_START,
    EVAL_EPISODES,
    EVAL_EVERY_EPISODES,
    GAMMA,
    GHOST_DXY_CLIP,
    MA_HISTORY_MAX,
    MA_WINDOW,
    MAX_STEPS_PER_EPISODE,
    MAZE_LAYOUT,
    N_STEP_RETURNS,
    OPPOSITE_ACTION,
    REWARD_ALL_COLLECTED,
    REWARD_CAUGHT,
    REWARD_INVALID,
    REWARD_PELLET,
    REWARD_STEP,
    REWARD_TIMEOUT,
    SHAPING_GHOST_DELTA,
    SHAPING_PELLET_DELTA,
    STATE_DX_GHOST,
    STATE_DY_GHOST,
    STATE_D_GHOST_BIN,
    STATE_D_PELLET_BIN,
    STATE_PELLET_DIR,
    STATE_WALL_BITS,
    StepInfo,
    USE_DOMAIN_RANDOMIZATION,
    USE_DOUBLE_Q,
    USE_REWARD_SHAPING,
)


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
        self._path_dist_origin: tuple[int, int] | None = None
        self._path_distances: dict[tuple[int, int], int] = {}

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
        self._path_dist_origin = None
        self._path_distances = {}

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
            _nxt, moved = self._move(ghost_pos, action)
            if moved:
                walkable_actions.append(action)

        # Should always have at least one valid action if no dead ends exist
        if not walkable_actions:
            # Fallback (should not happen): pick any valid move
            for action in ACTIONS:
                _nxt, moved = self._move(ghost_pos, action)
                if moved:
                    return action
            # Last resort (should really not happen): pick any action
            return random.choice(list(ACTIONS.keys()))

        # Chase phase: pick best action to get closer to pacman
        if random.random() < self.ghost_chase_prob:
            best_dist = float("inf")
            best_actions = []
            for action in walkable_actions:
                nxt, _moved = self._move(ghost_pos, action)
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
        distances = self._distances_from_pacman()
        px, py = self.pacman
        nearest = min(self.pellets, key=lambda p: distances.get(p, 10**9))
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
        distances = self._distances_from_pacman()
        return min(self.ghosts, key=lambda g: distances.get(g, 10**9))

    def _build_distance_map(self, origin: tuple[int, int]) -> dict[tuple[int, int], int]:
        distances: dict[tuple[int, int], int] = {origin: 0}
        queue = deque([origin])
        while queue:
            x, y = queue.popleft()
            base = distances[(x, y)]
            for dx, dy in ACTIONS.values():
                nxt = (x + dx, y + dy)
                if nxt in distances or not self._is_walkable(nxt):
                    continue
                distances[nxt] = base + 1
                queue.append(nxt)
        return distances

    def _distances_from_pacman(self) -> dict[tuple[int, int], int]:
        if self._path_dist_origin != self.pacman:
            self._path_distances = self._build_distance_map(self.pacman)
            self._path_dist_origin = self.pacman
        return self._path_distances

    def _nearest_ghost_distance(self) -> int:
        distances = self._distances_from_pacman()
        return min(distances.get(g, DIST_BIN_CAP) for g in self.ghosts)

    def _nearest_pellet_distance(self) -> int:
        if not self.pellets:
            return 0
        distances = self._distances_from_pacman()
        return min(distances.get(p, DIST_BIN_CAP) for p in self.pellets)

    def _discretize_dist(self, value: int) -> int:
        clipped = max(0, min(DIST_BIN_CAP, int(value)))
        idx = int(clipped / max(1, DIST_BIN_CAP) * DIST_BINS)
        return max(0, min(DIST_BINS - 1, idx))

    def get_state(self) -> tuple[int, int, int, int, int, int]:
        distances = self._distances_from_pacman()
        px, py = self.pacman
        gx, gy = min(self.ghosts, key=lambda g: distances.get(g, 10**9))
        dx_ghost = max(-GHOST_DXY_CLIP, min(GHOST_DXY_CLIP, gx - px))
        dy_ghost = max(-GHOST_DXY_CLIP, min(GHOST_DXY_CLIP, gy - py))

        d_ghost_bin = self._discretize_dist(min(distances.get(g, DIST_BIN_CAP) for g in self.ghosts))
        if self.pellets:
            d_pellet = min(distances.get(p, DIST_BIN_CAP) for p in self.pellets)
        else:
            d_pellet = 0
        d_pellet_bin = self._discretize_dist(d_pellet)
        pellet_dir = self._nearest_pellet_direction()
        wall_bits = self._wall_flags()

        state = [0] * 6
        state[STATE_DX_GHOST] = dx_ghost
        state[STATE_DY_GHOST] = dy_ghost
        state[STATE_D_GHOST_BIN] = d_ghost_bin
        state[STATE_D_PELLET_BIN] = d_pellet_bin
        state[STATE_PELLET_DIR] = pellet_dir
        state[STATE_WALL_BITS] = wall_bits
        return tuple(state)

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
