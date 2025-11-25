from gymnasium.envs.registration import register

print(">>> REGISTERING CareerPathEnv-v0 <<<")

register(
    id="CareerPathEnv-v0",
    entry_point="environment.career_env:CareerPathEnv",
)
