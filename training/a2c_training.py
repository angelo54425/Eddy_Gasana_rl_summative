# FILE: training/a2c_training.py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import environment
import gymnasium as gym

try:
    in_registry = "CareerPathEnv-v0" in gym.registry
except Exception:
    in_registry = False
if not in_registry:
    from gymnasium.envs.registration import register
    register(
        id="CareerPathEnv-v0",
        entry_point="environment.career_env:CareerPathEnv",
    )

from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

def make_env():
    env = gym.make("CareerPathEnv-v0")
    return env

def main():
    model_dir = "models/a2c"
    os.makedirs(model_dir, exist_ok=True)

    # Hyperparameters (from plan)
    learning_rate = 7e-4
    n_steps = 5
    gamma = 0.99
    total_timesteps = 300_000

    env = make_env()
    env = Monitor(env)

    eval_env = make_env()
    eval_env = Monitor(eval_env)

    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        verbose=1,
        tensorboard_log="results/a2c_tb"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path="results/a2c_eval",
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    print("Starting A2C training...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(os.path.join(model_dir, "a2c_final_model"))
    print("A2C model saved.")

    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
