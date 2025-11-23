import gymnasium as gym
import environment   # triggers registration
import pygame

env = gym.make("CareerPathEnv-v0", render_mode="pygame")
obs, info = env.reset()

# First render to initialize window
env.render()

running = True
while running:

    # Handle window close button
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Take a random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    # Draw updated frame
    env.render()

    # If episode ends, reset
    if terminated or truncated:
        obs, info = env.reset()
        env.render()

env.close()
pygame.quit()
