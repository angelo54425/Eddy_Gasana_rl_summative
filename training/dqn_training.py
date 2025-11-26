# training/dqn_training.py
import os
import sys
from pathlib import Path

# ensure project root is on sys.path (works when running from repo root or from inside training/)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ensure environment package is imported and registered
import environment
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import logging

logging.basicConfig(level=logging.INFO)


def make_env():
    # import here to ensure registration runs
    import environment  # noqa: F401
    return gym.make("CareerPathEnv-v0")


def main():
    model_dir = "models/dqn"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs("logs/dqn", exist_ok=True)
    os.makedirs("models/dqn/checkpoints", exist_ok=True)

    env = make_env()
    env = Monitor(env)

    checkpoint = CheckpointCallback(
        save_freq=25_000,
        save_path=str(Path(model_dir) / "checkpoints"),
        name_prefix="dqn_ckpt"
    )

    # tuned hyperparameters (good defaults for grid world)
    model = DQN(
        policy="MlpPolicy",
        env=env,
        buffer_size=150_000,
        learning_rate=1e-4,
        batch_size=64,
        gamma=0.99,
        learning_starts=2_000,
        train_freq=4,
        target_update_interval=1_000,
        exploration_fraction=0.4,
        exploration_final_eps=0.02,
        verbose=1,
        tensorboard_log="logs/dqn"
    )

    # shorter default for quick iteration, increase for final run
    total_timesteps = int(os.environ.get("DQN_TIMESTEPS", 150_000))
    model.learn(total_timesteps=total_timesteps, callback=checkpoint)
    model.save(str(Path(model_dir) / "dqn_final_model"))
    env.close()
    print("DQN training finished. Model saved to models/dqn/dqn_final_model.zip")


if __name__ == "__main__":
    main()
