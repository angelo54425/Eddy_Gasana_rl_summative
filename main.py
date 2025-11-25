import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import environment
import gymnasium as gym
from stable_baselines3 import DQN
import pygame
import time


def main():

    pygame.init()   # ✅ MUST be called before event handling

    model_path = "models/dqn/dqn_final_model.zip"

    env = gym.make("CareerPathEnv-v0", render_mode="pygame")
    obs, info = env.reset()

    model = DQN.load(model_path)

    # ✅ FIRST FRAME (forces window to appear on macOS)
    env.render()
    pygame.display.flip()

    running = True
    clock = pygame.time.Clock()

    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        env.render()
        pygame.display.flip()

        clock.tick(5)

        if terminated or truncated:
            time.sleep(1)
            obs, info = env.reset()
            env.render()
            pygame.display.flip()

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
