import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import environment
import gymnasium as gym
# Ensure the custom environment is registered. Some import orders or
# environments can cause the registration to be missed; if so, register
# it here as a fallback so `gym.make` can find it.
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
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import os


def main():

    # Ensure model directory exists
    model_dir = "models/dqn"
    os.makedirs(model_dir, exist_ok=True)

    # Create environment
    env = gym.make("CareerPathEnv-v0")
    env = Monitor(env)

    # DQN Model Configuration
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0005,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        verbose=1
    )

    # Evaluation callback (optional but useful)
    eval_env = gym.make("CareerPathEnv-v0")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path="results/dqn_logs",
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    # Train model
    model.learn(
        total_timesteps=150000,   # You can increase later
        callback=eval_callback
    )

    # Save final model
    model_path = f"{model_dir}/dqn_final_model"
    model.save(model_path)

    print("✅ Training complete!")
    print(f"✅ Model saved to: {model_path}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
