import gymnasium as gym
import environment   # important: triggers registration

env = gym.make("CareerPathEnv-v0")

# Reset environment
obs, info = env.reset()
print("Initial observation:", obs)

# Take one random step
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

print("After one step:")
print("Observation:", obs)
print("Reward:", reward)
print("Terminated:", terminated)
print("Truncated:", truncated)

env.close()
