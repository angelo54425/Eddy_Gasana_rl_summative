from gymnasium.envs.registration import register

print(">>> environment package loaded")

register(
    id="CareerPathEnv-v0",
    entry_point="environment.career_env:CareerPathEnv",
)
