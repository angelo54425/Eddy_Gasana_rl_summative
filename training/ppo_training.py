# FILE: training/ppo_training.py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure environment package is imported (and registered)
import environment
import gymnasium as gym

# fallback registration if import order didn't run package __init__
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

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import torch as _torch

def make_env(render_mode=None):
    # Create single environment for training (headless)
    env = gym.make("CareerPathEnv-v0", render_mode=render_mode)
    return env

def main():
    run_name = "ppo_run"
    model_dir = "models/ppo"
    os.makedirs(model_dir, exist_ok=True)

    # Hyperparameters (recommended defaults from tuning plan)
    policy_kwargs = dict(net_arch=[256, 256])
    learning_rate = 3e-4
    n_steps = 2048
    batch_size = 64
    gamma = 0.99
    clip_range = 0.2
    total_timesteps = 500_000  # change if you want

    env = make_env()
    env = Monitor(env)

    # evaluation env (no rendering)
    eval_env = make_env()
    eval_env = Monitor(eval_env)

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        clip_range=clip_range,
        verbose=1,
        tensorboard_log="results/ppo_tb"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path="results/ppo_eval",
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    print("Starting PPO training...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    path = os.path.join(model_dir, "ppo_final_model")
    model.save(path)
    print(f"PPO model saved to: {path}")

    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
