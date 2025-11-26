# FILE: environment/career_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Optional, List

class CareerPathEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "pygame"], "render_fps": 30}

    def __init__(self, grid_size: int = 10, steps: int = 100, render_mode: Optional[str] = None):
        super().__init__()
        self.grid_size = int(grid_size)
        self.steps = int(steps)          # 🔥 Increased from 60 → 100 steps
        self.render_mode = render_mode

        # neutral square size (2x2 for even grids)
        mid = self.grid_size // 2
        self.neutral_min = mid - 1
        self.neutral_max = mid

        # training tiles (2 per field)
        self.training_tiles = {
            1: [(1, 1), (2, 2)],
            2: [(self.grid_size - 2, 1), (self.grid_size - 3, 2)],
            3: [(1, self.grid_size - 2), (2, self.grid_size - 3)],
            4: [(self.grid_size - 2, self.grid_size - 2), (self.grid_size - 3, self.grid_size - 3)],
        }

        # goals
        self.goal_tiles = {
            1: (0, 0),
            2: (self.grid_size - 1, 0),
            3: (0, self.grid_size - 1),
            4: (self.grid_size - 1, self.grid_size - 1),
        }

        # 🔥 Reduced thresholds so 3 training actions can succeed
        self.skill_thresholds = {
            1: 0.25,
            2: 0.45,
            3: 0.40,
            4: 0.55
        }

        # actions
        self.action_space = spaces.Discrete(6)

        # observation
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0.0, 0.0], dtype=float),
            high=np.array([self.grid_size - 1, self.grid_size - 1, 4, 1.0, 1.0], dtype=float),
            dtype=float
        )

        # 🔥 Reward settings (boosted training)
        self.TRAIN_BASE = 4.0             # Increased train reward
        self.TRAIN_DECAY = 0.5
        self.MAX_TRAINING = 6             # Increased cap
        self.ENTER_TRAIN_SKILL = 0.05     # Bigger instant skill bump
        self.TRAIN_ACTION_SKILL = 0.15    # Bigger skill gain per training

        # other rewards
        self.IDLE_PENALTY = -0.25
        self.RETURN_TO_NEUTRAL_PEN = -1.0
        self.DISTANCE_SHAPING_POS = 0.10
        self.DISTANCE_SHAPING_NEG = -0.03
        self.GOAL_SUCCESS_BASE = 30.0
        self.EARLY_STAR_PAYOUT = 5.0      # ⭐ Small reward for touching goal w/low skill → CONTINUES
        self.NEUTRAL_STEP_PEN = -0.02
        self.IDLE_THRESHOLD = 3

        self.reset()

    # ==== Helpers, same as before ====
    def _pos_tuple(self, pos) -> Tuple[int, int]:
        if isinstance(pos, np.ndarray):
            return int(pos[0]), int(pos[1])
        return int(pos[0]), int(pos[1])

    def _in_neutral(self, x: int, y: int) -> bool:
        return (self.neutral_min <= x <= self.neutral_max) and (self.neutral_min <= y <= self.neutral_max)

    def _get_zone(self, x: int, y: int) -> int:
        if self._in_neutral(x, y):
            return 0
        if x < self.neutral_min and y < self.neutral_min: return 1
        if x > self.neutral_max and y < self.neutral_min: return 2
        if x < self.neutral_min and y > self.neutral_max: return 3
        if x > self.neutral_max and y > self.neutral_max: return 4
        mid = self.grid_size // 2
        if y < mid:
            return 1 if x < mid else 2
        else:
            return 3 if x < mid else 4

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mid = self.grid_size // 2
        self.agent_pos = np.array([mid, mid], dtype=int)
        self.prev_pos = (int(self.agent_pos[0]), int(self.agent_pos[1]))

        self.target_zone = None
        self.locked_zone = None

        self.skill_level = 0.0
        self.training_count = 0
        self.training_done = False
        self.training_stop_step = None

        self.idle_steps = 0
        self.last_action = None

        self.prev_dist = float(self.grid_size * 2)

        self.current_step = 0
        self.last_info = {"result": "In Progress"}
        return self._get_obs(), {}

    # ====================== STEP ======================
    def step(self, action: int):
        self.current_step += 1
        reward = 0.0
        terminated = False
        truncated = False
        info = {"result": "In Progress"}

        ax, ay = self._pos_tuple(self.agent_pos)
        old_pos = (ax, ay)
        old_zone = self._get_zone(ax, ay)

        nx, ny = ax, ay
        moved = False
        if action == 0: ny = max(0, ay - 1); moved = True
        elif action == 1: ny = min(self.grid_size - 1, ay + 1); moved = True
        elif action == 2: nx = max(0, ax - 1); moved = True
        elif action == 3: nx = min(self.grid_size - 1, ax + 1); moved = True
        elif action == 4:
            reward += self.IDLE_PENALTY
        elif action == 5:
            reward += self._training_action_reward(old_zone, old_pos)

        new_pos = (int(nx), int(ny))
        new_zone = self._get_zone(new_pos[0], new_pos[1])

        # lock to zone
        if self.locked_zone is None and old_zone == 0 and new_zone != 0 and moved:
            self.locked_zone = new_zone
            self.target_zone = new_zone
            gx, gy = self.goal_tiles[self.target_zone]
            self.prev_dist = abs(ax - gx) + abs(ay - gy)

        # can't leave locked zone
        if self.locked_zone is not None and new_zone != self.locked_zone and moved:
            reward += -0.05
            new_pos = old_pos
            new_zone = old_zone
        else:
            if moved:
                self.agent_pos = np.array(new_pos, dtype=int)

        # idle penalty
        if not moved or new_pos == old_pos:
            self.idle_steps += 1
        else:
            self.idle_steps = 0

        if self.idle_steps >= self.IDLE_THRESHOLD:
            reward += -0.10 * (self.idle_steps - self.IDLE_THRESHOLD + 1)

        # penalty for returning to neutral after lock
        if self.locked_zone is not None and new_zone == 0:
            reward += self.RETURN_TO_NEUTRAL_PEN

        # neutral penalty
        if new_zone == 0:
            reward += self.NEUTRAL_STEP_PEN

        # step onto training tile → small skill bump
        ax2, ay2 = self._pos_tuple(self.agent_pos)
        if self.target_zone is not None and (ax2, ay2) in self.training_tiles[self.target_zone]:
            if old_pos != (ax2, ay2):
                self.skill_level = min(1.0, self.skill_level + self.ENTER_TRAIN_SKILL)
                reward += 0.10
            reward += 0.02

        # distance shaping
        if self.target_zone is not None:
            gx, gy = self.goal_tiles[self.target_zone]
            dist = abs(ax2 - gx) + abs(ay2 - gy)
            if dist < self.prev_dist:
                reward += self.DISTANCE_SHAPING_POS
            elif dist > self.prev_dist:
                reward += self.DISTANCE_SHAPING_NEG
            self.prev_dist = dist

        # ================= FORCE GOAL-SEEKING WHEN SKILLS READY =================
        skill_ready = (
            self.target_zone is not None and 
            self.skill_level >= self.skill_thresholds[self.target_zone]
)

        if skill_ready:

            # BIG reward for decreasing distance toward the goal
            if dist < self.prev_dist:
                reward += 1.5     # pull strongly toward the star

            # BIG penalty for increasing distance
            elif dist > self.prev_dist:
                reward -= 1.5

            # penalize extra training once threshold reached
            if action == 5:
                reward -= 1.0    # strong penalty for training when skill is enough

            # penalize wandering in non-goal directions
            if new_zone != self.target_zone:
                reward -= 1.0

        # ================= GOAL LOGIC ==================
        if self.target_zone is not None and (ax2, ay2) == tuple(self.goal_tiles[self.target_zone]):

            # 🔥 NEW LOGIC: DO NOT TERMINATE IF SKILL TOO LOW
            if self.skill_level >= self.skill_thresholds[self.target_zone]:
                reward += self.GOAL_SUCCESS_BASE
                info["result"] = "success"
                terminated = True
            else:
                # small positive reward to encourage touching goal
                reward += self.EARLY_STAR_PAYOUT
                info["result"] = "early_touch"
                # 🔥 DO NOT END EPISODE — allow training more
                terminated = False

        # step limit
        if self.current_step >= self.steps:
            truncated = True
            info["result"] = "failure"

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), dict(info)

    # ================= TRAIN LOGIC ==================
    def _training_action_reward(self, zone: int, pos: Tuple[int, int]) -> float:
        if self.target_zone is None:
            return -0.02
        if zone != self.target_zone:
            return -0.02
        if pos not in self.training_tiles[self.target_zone]:
            return -0.01
        if self.training_done:
            return 0.0

        reward = max(0.0, self.TRAIN_BASE - self.TRAIN_DECAY * float(self.training_count))

        self.skill_level = min(1.0, self.skill_level + self.TRAIN_ACTION_SKILL)
        self.training_count += 1

        if self.training_count >= self.MAX_TRAINING:
            self.training_done = True
            self.training_stop_step = self.current_step

        return float(reward)

    def _get_obs(self):
        x, y = self._pos_tuple(self.agent_pos)
        zone = self._get_zone(x, y)
        threshold = float(self.skill_thresholds[self.target_zone]) if self.target_zone is not None else 0.0
        return np.array([x, y, zone, float(self.skill_level), threshold], dtype=float)

    def render(self):
        if self.render_mode == "pygame":
            if not hasattr(self, "_pygame_init"):
                self._pygame_init = True
                from .rendering import render_pygame
                self._screen, self._draw, self._clock = render_pygame(self)
            self._draw()
            self._clock.tick(12)

    def close(self):
        pass
