import sys, os, inspect
import gymnasium as gym

# --- FORCE ADD PROJECT ROOT ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
print("Added to sys.path:", ROOT)

# re-import after fixing path
from environment.career_env import CareerPathEnv

env = gym.make("CareerPathEnv-v0")

print("\n=== Gym Loaded Env File ===")
print(inspect.getfile(env.unwrapped.__class__))

print("\n=== Direct Import File ===")
print(inspect.getfile(CareerPathEnv))
