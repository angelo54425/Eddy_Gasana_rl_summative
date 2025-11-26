import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import environment
import gymnasium as gym
import pygame
import time



def main():
    env = gym.make("CareerPathEnv-v0", render_mode="pygame")
    obs, info = env.reset()
    env.render()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            obs, info = env.reset()
            env.render()
            time.sleep(1)

    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()
