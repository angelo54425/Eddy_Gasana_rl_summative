import sys, os
# ✅ Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import environment
import gymnasium as gym

def main():
    env = gym.make("CareerPathEnv-v0")
    obs, info = env.reset()
    print("Initial observation:", obs)

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print("After one step:", obs, reward)

    env.close()

if __name__ == "__main__":
    main()
