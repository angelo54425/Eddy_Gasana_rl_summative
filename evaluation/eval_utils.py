# evaluation/eval_utils.py
import os
import json
import csv
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN, PPO, A2C
import torch
import torch.nn as nn

# -------- REINFORCE policy architecture (must match training) --------
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=(128,128)):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # logits

# -------- Evaluation runner --------
def make_env():
    # import environment package (ensures registration)
    import environment
    return gym.make("CareerPathEnv-v0")

def eval_model_sb3(model, n_episodes=50, render=False, render_mode=None):
    """
    Evaluate a Stable-Baselines3 model (DQN/PPO/A2C).
    Returns dict of metrics and list of episode rewards and lengths/results.
    """
    env = make_env()
    if render and render_mode:
        env = gym.make("CareerPathEnv-v0", render_mode=render_mode)
    rewards = []
    lengths = []
    successes = 0
    detailed = []  # list of (total_reward, length, result)

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        length = 0
        result = "incomplete"
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(int(action))
            total += r
            length += 1
            if terminated or truncated:
                result = info.get("result", "incomplete")
                if result == "success":
                    successes += 1
                break
        rewards.append(total)
        lengths.append(length)
        detailed.append({"reward": total, "length": length, "result": result})

    env.close()
    metrics = {
        "mean_reward": float(np.mean(rewards)),
        "median_reward": float(np.median(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": float(successes) / float(n_episodes),
        "n_episodes": n_episodes
    }
    return metrics, detailed

def load_sb3_model(alg_name, model_path, env=None):
    alg = alg_name.lower()
    if alg == "dqn":
        return DQN.load(model_path, env=env)
    if alg == "ppo":
        return PPO.load(model_path, env=env)
    if alg == "a2c":
        return A2C.load(model_path, env=env)
    raise ValueError(f"Unknown SB3 algorithm: {alg_name}")

def eval_reinforce(model_path, n_episodes=50, device=None):
    env = make_env()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    policy = PolicyNet(obs_dim, action_dim).to(device)
    state = torch.load(model_path, map_location=device)
    policy.load_state_dict(state)
    policy.eval()

    rewards = []
    lengths = []
    successes = 0
    detailed = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total = 0.0
        length = 0
        result = "incomplete"
        while True:
            x = torch.tensor(obs, dtype=torch.float32).to(device)
            logits = policy(x.unsqueeze(0)).squeeze(0)
            probs = torch.softmax(logits, dim=-1).cpu().detach().numpy()
            action = int(np.argmax(probs))
            obs, r, terminated, truncated, info = env.step(action)
            total += r
            length += 1
            if terminated or truncated:
                result = info.get("result", "incomplete")
                if result == "success":
                    successes += 1
                break
        rewards.append(total)
        lengths.append(length)
        detailed.append({"reward": total, "length": length, "result": result})

    env.close()
    metrics = {
        "mean_reward": float(np.mean(rewards)),
        "median_reward": float(np.median(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": float(successes) / float(n_episodes),
        "n_episodes": n_episodes
    }
    return metrics, detailed

# -------- Saving utilities --------
def save_metrics_json(metrics, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

def save_detailed_csv(detailed_list, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    keys = ["reward", "length", "result"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in detailed_list:
            writer.writerow(row)
