# training/a2c_training.py
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import environment
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

import logging
logging.basicConfig(level=logging.INFO)


def make_env():
    import environment  # noqa
    return gym.make("CareerPathEnv-v0")


def main():
    model_dir = "models/a2c"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs("runs/a2c", exist_ok=True)
    os.makedirs("models/a2c/checkpoints", exist_ok=True)

    base_env = Monitor(make_env())
    env = DummyVecEnv([lambda: base_env])

    checkpoint = CheckpointCallback(save_freq=20_000, save_path=str(Path(model_dir) / "checkpoints"), name_prefix="a2c_ckpt")

    model = A2C(
        policy="MlpPolicy",
        env=env,
        learning_rate=7e-4,
        n_steps=5,
        gamma=0.995,
        vf_coef=0.5,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [256, 256]},
        verbose=1,
        tensorboard_log="runs/a2c"
    )

    total_timesteps = int(os.environ.get("A2C_TIMESTEPS", 150_000))
    model.learn(total_timesteps=total_timesteps, callback=checkpoint)
    model.save(str(Path(model_dir) / "a2c_final_model"))
    env.close()
    print("A2C training finished. Model saved to models/a2c/a2c_final_model.zip")


if __name__ == "__main__":
    main()
