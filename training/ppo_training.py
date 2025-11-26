# training/ppo_training.py
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import environment
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback

import logging
logging.basicConfig(level=logging.INFO)


def make_env():
    import environment  # noqa
    return gym.make("CareerPathEnv-v0")


def main():
    model_dir = "models/ppo"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs("runs/ppo", exist_ok=True)
    os.makedirs("models/ppo/checkpoints", exist_ok=True)

    base_env = Monitor(make_env())
    env = DummyVecEnv([lambda: base_env])

    # small reward threshold example; adjust later based on eval metrics
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=float(os.environ.get("PPO_STOP_REWARD", 120.0)), verbose=1)
    eval_env = make_env()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path="results/ppo_logs",
        eval_freq=5_000,
        deterministic=True,
        callback_on_new_best=stop_callback
    )

    checkpoint = CheckpointCallback(save_freq=25_000, save_path=str(Path(model_dir) / "checkpoints"), name_prefix="ppo_ckpt")

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=8,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs={"net_arch": [256, 256]},
        verbose=1,
        tensorboard_log="runs/ppo"
    )

    total_timesteps = int(os.environ.get("PPO_TIMESTEPS", 150_000))
    model.learn(total_timesteps=total_timesteps, callback=[eval_callback, checkpoint])
    model.save(str(Path(model_dir) / "ppo_final_model"))
    env.close()
    eval_env.close()
    print("PPO training finished. Model saved to models/ppo/ppo_final_model.zip")


if __name__ == "__main__":
    main()
