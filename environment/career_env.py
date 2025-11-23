import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class CareerPathEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "pygame"], "render_fps": 30}

    def __init__(self, grid_size=11, max_steps=50, render_mode=None):
        super().__init__()

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode

        # ----- CONSTANTS -----
        self.MOVE_PENALTY = -0.01
        self.TRAIN_REWARD = +0.05
        self.SUCCESS_REWARD = +5.0
        self.FAILURE_PENALTY = -1.0
        self.TIMEOUT_PENALTY = -0.5
        self.INVALID_ATTEMPT_PENALTY = -0.5
        self.SKILL_GAIN = 0.05

        # ----- FIELD DIFFICULTY -----
        self.skill_thresholds = {
            0: 0.60,  # Private
            1: 0.90,  # Medical
            2: 0.70,  # Finance
            3: 0.80   # Engineering
        }

        # ----- ACTION SPACE -----
        self.action_space = spaces.Discrete(6)

        # ----- OBSERVATION SPACE -----
        low = np.array([0, 0, 0.0, 0.0, 0.0], dtype=float)
        high = np.array([grid_size - 1, grid_size - 1, 1.0, 1.0, 1.0], dtype=float)
        self.observation_space = spaces.Box(low=low, high=high, dtype=float)

        # ----- ZONE MAP -----
        self.zone_map = self._build_zone_map()

        # ----- STAR & TRAINING CELLS -----
        self.star_cells = {
            0: (0, 0),
            1: (10, 0),
            2: (0, 10),
            3: (10, 10)
        }

        self.train_cells = {
            0: [(1, 2), (2, 1)],
            1: [(7, 2), (8, 1)],
            2: [(1, 8), (2, 7)],
            3: [(7, 8), (8, 7)]
        }

        # Status tracking
        self.failed_star = None
        self.status = "In Progress"
        self.star_animation_scale = 1.0

        # Sound setup
        pygame.mixer.init()
        self.success_sound = None
        self.fail_sound = None
        self.train_sound = None
        try:
            self.success_sound = pygame.mixer.Sound(None)
            self.fail_sound = pygame.mixer.Sound(None)
            self.train_sound = pygame.mixer.Sound(None)
        except:
            pass

        self._pygame_inited = False

        self.reset()

    # ------------------------------------------------------------------
    def _build_zone_map(self):
        zone_map = {}
        mid = self.grid_size // 2

        for x in range(self.grid_size):
            for y in range(self.grid_size):

                if mid - 1 <= x <= mid + 1 and mid - 1 <= y <= mid + 1:
                    zone_map[(x, y)] = -1  # Neutral

                elif x < mid and y < mid:
                    zone_map[(x, y)] = 0

                elif x > mid and y < mid:
                    zone_map[(x, y)] = 1

                elif x < mid and y > mid:
                    zone_map[(x, y)] = 2

                elif x > mid and y > mid:
                    zone_map[(x, y)] = 3

                else:
                    zone_map[(x, y)] = -1

        return zone_map

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps_remaining = self.max_steps

        mid = self.grid_size // 2
        x = self.np_random.integers(mid - 1, mid + 2)
        y = self.np_random.integers(mid - 1, mid + 2)

        self.agent_pos = np.array([x, y], dtype=int)
        self.field_id = -1
        self.skill_level = 0.0
        self.failed_star = None
        self.status = "In Progress"
        self.star_animation_scale = 1.0

        obs = self._get_obs()
        return obs, {}

    # ------------------------------------------------------------------
    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False

        # ----- Movement -----
        if action in [0, 1, 2, 3]:
            self._move_agent(action)
            reward += self.MOVE_PENALTY

        # ----- Training -----
        elif action == 4:
            gain = self.SKILL_GAIN

            if self.field_id in [0, 1, 2, 3]:
                if tuple(self.agent_pos) in self.train_cells[self.field_id]:
                    gain += 0.05
                    if self.train_sound:
                        self.train_sound.play()

            self.skill_level = min(1.0, self.skill_level + gain)
            reward += self.TRAIN_REWARD

        # ----- Attempt Profession -----
        elif action == 5:
            terminated, reward = self._attempt_profession()

        # ----- Time Step -----
        self.steps_remaining -= 1
        if self.steps_remaining <= 0 and not terminated:
            terminated = True
            reward += self.TIMEOUT_PENALTY
            self.status = "Failure ❌"

        # Star animation pulse
        self.star_animation_scale = 1.0 + 0.1 * np.sin(self.steps_remaining)

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    def _move_agent(self, action):
        x, y = self.agent_pos

        if action == 0:
            y = max(0, y - 1)
        elif action == 1:
            y = min(self.grid_size - 1, y + 1)
        elif action == 2:
            x = max(0, x - 1)
        elif action == 3:
            x = min(self.grid_size - 1, x + 1)

        proposed_zone = self.zone_map[(x, y)]

        if self.field_id == -1 and proposed_zone in [0, 1, 2, 3]:
            self.field_id = proposed_zone

        if self.field_id == -1 or proposed_zone == self.field_id:
            self.agent_pos = np.array([x, y], dtype=int)

    # ------------------------------------------------------------------
    def _attempt_profession(self):
        pos = tuple(self.agent_pos)

        # Invalid if not on star
        if self.field_id == -1 or pos != self.star_cells[self.field_id]:
            return False, self.INVALID_ATTEMPT_PENALTY

        threshold = self.skill_thresholds[self.field_id]

        if self.skill_level >= threshold:
            self.status = "Success ✅"
            if self.success_sound:
                self.success_sound.play()
            return True, self.SUCCESS_REWARD

        self.failed_star = pos
        self.status = "Failure ❌"
        if self.fail_sound:
            self.fail_sound.play()
        return True, self.FAILURE_PENALTY

    # ------------------------------------------------------------------
    def _get_obs(self):
        field_norm = (self.field_id + 1) / 4
        threshold = 0.0 if self.field_id == -1 else self.skill_thresholds[self.field_id]

        return np.array([
            float(self.agent_pos[0]),
            float(self.agent_pos[1]),
            field_norm,
            float(self.skill_level),
            float(threshold)
        ], dtype=float)

    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode == "human":
            print(f"Pos={self.agent_pos}, Field={self.field_id}, Skill={self.skill_level:.2f}, Steps={self.steps_remaining}, Status={self.status}")

        elif self.render_mode == "rgb_array":
            img = np.zeros((self.grid_size * 32, self.grid_size * 32, 3), dtype=np.uint8)
            return img

        elif self.render_mode == "pygame":
            from .rendering import render_pygame

            if not self._pygame_inited:
                self._pygame_inited = True
                self._pygame_screen, self._pygame_draw, self._pygame_clock = render_pygame(self)

            self._pygame_draw()
            self._pygame_clock.tick(5)

    # ------------------------------------------------------------------
    def close(self):
        pass
