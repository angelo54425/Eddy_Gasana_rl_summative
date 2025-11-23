from gymnasium.envs.registration import register
from .career_env import CareerPathEnv

register(
    id="CareerPathEnv-v0",
    entry_point="environment.career_env:CareerPathEnv",
)
