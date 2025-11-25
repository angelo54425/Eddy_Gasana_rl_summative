# FILE: environment/career_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CareerPathEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "pygame"], "render_fps": 30}

    def __init__(self, grid_size=10, steps=200, render_mode=None, neutral_size=None):
        super().__init__()

        self.grid_size = grid_size
        self.steps = steps
        self.render_mode = render_mode

        # Determine neutral zone size (central square)
        if neutral_size is None:
            self.neutral_size = 2 if (grid_size % 2 == 0) else 3
        else:
            self.neutral_size = neutral_size

        # Training tiles
        self.training_tiles = {
            1: [(1, 3), (3, 1)],
            2: [(6, 1), (8, 3)],
            3: [(1, 6), (3, 8)],
            4: [(6, 8), (8, 6)]
        }

        # Goal tiles for each field
        self.goal_tiles = {
            1: (0, 0),
            2: (grid_size - 1, 0),
            3: (0, grid_size - 1),
            4: (grid_size - 1, grid_size - 1)
        }

        # Skill thresholds
        self.skill_thresholds = {
            1: 0.4,
            2: 0.8,
            3: 0.6,
            4: 0.9
        }

        # Actions
        self.action_space = spaces.Discrete(6)

        # Observations: (x, y, zone, skill, threshold)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0.0, 0.0], dtype=float),
            high=np.array([grid_size - 1, grid_size - 1, 4, 1.0, 1.0], dtype=float),
            dtype=float
        )

        self.reset()

    # -------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0

        # Start centered in neutral zone
        mid = self.grid_size // 2
        self.agent_pos = np.array([mid, mid], dtype=int)
        self.prev_pos = tuple(self.agent_pos)

        # Choose one career path (zone 1–4)
        self.target_zone = int(self.np_random.integers(1, 5))

        self.skill_level = 0.0
        self.locked_zone = None
        self.status = "In Progress"
        self.failed_star = None

        return self._get_obs(), {}

    # -------------------------------------------------------------

    def step(self, action):
        self.current_step += 1
        reward = 0.0
        terminated = False
        truncated = False
        info = {"result": "In Progress"}

        x, y = int(self.agent_pos[0]), int(self.agent_pos[1])
        old_pos = (x, y)
        old_zone = self._get_zone(x, y)

        # -------------------------------
        # ACTIONS
        # -------------------------------
        new_x, new_y = x, y
        if action == 0:   # UP
            new_y = max(0, y - 1)
        elif action == 1: # DOWN
            new_y = min(self.grid_size - 1, y + 1)
        elif action == 2: # LEFT
            new_x = max(0, x - 1)
        elif action == 3: # RIGHT
            new_x = min(self.grid_size - 1, x + 1)
        elif action == 4: # WAIT
            reward -= 0.05
        elif action == 5: # TRAIN
            reward += self._training_reward(old_zone, old_pos)

        new_pos = (new_x, new_y)
        new_zone = self._get_zone(*new_pos)

        # -------------------------------
        # ZONE LOCKING (CANNOT LEAVE FIELD)
        # -------------------------------
        if self.locked_zone is not None:
            if new_zone != self.locked_zone and action in [0,1,2,3]:
                reward -= 0.5   # punish attempts to exit
                new_pos = old_pos
                new_zone = old_zone
            else:
                self.agent_pos = np.array(new_pos)
        else:
            # If entering a field zone for first time → lock it
            if old_zone == 0 and new_zone != 0:
                self.locked_zone = new_zone

            self.agent_pos = np.array(new_pos)

        # -------------------------------
        # Prevent walking into walls
        # -------------------------------
        if new_pos == old_pos and action in [0,1,2,3]:
            reward -= 0.2

        # Oscillation penalty
        if new_pos == self.prev_pos and action in [0,1,2,3]:
            reward -= 0.2

        # Slight penalty for staying too long in neutral
        if new_zone == 0:
            reward -= 0.02

        # Reward for first time entering target zone
        if old_zone != new_zone and new_zone == self.target_zone:
            reward += 1.0

        # GLOBAL STEP PENALTY
        reward -= 0.05

        # -------------------------------
        # GOAL CHECK
        # -------------------------------
        if tuple(self.agent_pos) == self.goal_tiles[self.target_zone]:
            threshold = self.skill_thresholds[self.target_zone]

            if self.skill_level >= threshold:
                reward += 80.0
                info["result"] = "success"
            else:
                reward -= 20.0
                info["result"] = "failure"
                self.failed_star = tuple(self.agent_pos)

            terminated = True

        # End episode if max steps reached
        if self.current_step >= self.steps:
            reward -= 20.0
            truncated = True
            info["result"] = "failure"

        self.prev_pos = old_pos
        self.status = info["result"]

        return self._get_obs(), reward, terminated, truncated, info

    # -------------------------------------------------------------

    def _training_reward(self, zone, pos):
        """Minimal reward to avoid loops. Training is for progress, not reward."""
        if zone == 0:
            return -0.2
        if pos in self.training_tiles[self.target_zone]:
            old_skill = self.skill_level
            self.skill_level = min(1.0, self.skill_level + 0.10)
            return 0.05
        return -0.2

    # -------------------------------------------------------------

    def _in_neutral(self, x, y):
        """Check if (x,y) lies inside central neutral square."""
        mid = self.grid_size // 2
        half = self.neutral_size // 2
        min_x = mid - half
        max_x = min_x + self.neutral_size - 1
        min_y = mid - half
        max_y = min_y + self.neutral_size - 1
        return (min_x <= x <= max_x) and (min_y <= y <= max_y)

    def _get_zone(self, x, y):
        if self._in_neutral(x, y):
            return 0

        mid = self.grid_size // 2
        if x < mid and y < mid:   return 1
        if x >= mid and y < mid:  return 2
        if x < mid and y >= mid:  return 3
        if x >= mid and y >= mid: return 4
        return 0

    # -------------------------------------------------------------

    def _get_obs(self):
        x, y = int(self.agent_pos[0]), int(self.agent_pos[1])
        zone = self._get_zone(x, y)
        threshold = self.skill_thresholds[self.target_zone]
        return np.array([x, y, zone, self.skill_level, threshold], dtype=float)

    # -------------------------------------------------------------

    def render(self):
        if self.render_mode == "pygame":
            if not hasattr(self, "_pygame_init"):
                self._pygame_init = True
                from .rendering import render_pygame
                self._screen, self._draw, self._clock = render_pygame(self)
            self._draw()
            self._clock.tick(5)

    def close(self):
        pass
