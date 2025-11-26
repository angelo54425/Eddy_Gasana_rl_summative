# evaluation/eval_utils.py
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import environment
import json
import csv
import numpy as np
import gymnasium as gym


from stable_baselines3 import DQN, PPO, A2C
import torch
import torch.nn as nn


# Simple PolicyNet for REINFORCE evaluation if used
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
        return self.net(x)

def make_env():
    import environment  # ensure registration
    return gym.make("CareerPathEnv-v0")

def _score_result(info):
    """Given env.info/result string classify as graded success:
       returns one of: 'success_full', 'success_partial', 'failure'
    """
    res = info.get("result", "incomplete")
    if res == "success":
        return "success_full"
    if res == "early_star" or res == "failure" or res == "incomplete":
        # partial if star reached but insufficient skill can be labeled via info if set
        # We expect env to set 'result' to 'failure' and maybe include 'pos' in info on termination.
        # Evaluate more thoroughly: if 'pos' in info and env.skill < threshold it's partial.
        # But env may not provide skill; evaluator will treat 'failure' with pos==goal as partial.
        if info.get("pos") is not None:
            return "success_partial"
        return "failure"
    return "failure"

def eval_model_sb3(model, n_episodes=50, render=False, render_mode=None):
    env = make_env()
    if render and render_mode:
        env = gym.make("CareerPathEnv-v0", render_mode=render_mode)
    rewards = []
    lengths = []
    counts = {"success_full":0, "success_partial":0, "failure":0}
    detailed = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total = 0.0
        length = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(int(action))
            total += r
            length += 1
            if terminated or truncated:
                # classify result: check info first; if not enough, use logic
                result = info.get("result", "incomplete")
                # For safety, if info['result']=='failure' but pos==goal, mark partial
                if result == "success":
                    label = "success_full"
                elif result == "failure" and info.get("pos") is not None:
                    label = "success_partial"
                else:
                    label = "failure"
                counts[label] += 1
                detailed.append({"reward": total, "length": length, "result": label})
                break
        rewards.append(total)
        lengths.append(length)
    env.close()

    metrics = {
        "mean_reward": float(np.mean(rewards)),
        "median_reward": float(np.median(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "success_full_rate": float(counts["success_full"]) / n_episodes,
        "success_partial_rate": float(counts["success_partial"]) / n_episodes,
        "failure_rate": float(counts["failure"]) / n_episodes,
        "counts": counts,
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
    counts = {"success_full":0, "success_partial":0, "failure":0}
    detailed = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total = 0.0
        length = 0
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
                    label = "success_full"
                elif result == "failure" and info.get("pos") is not None:
                    label = "success_partial"
                else:
                    label = "failure"
                counts[label] += 1
                detailed.append({"reward": total, "length": length, "result": label})
                break
        rewards.append(total)
        lengths.append(length)

    env.close()
    metrics = {
        "mean_reward": float(np.mean(rewards)),
        "median_reward": float(np.median(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "success_full_rate": float(counts["success_full"]) / n_episodes,
        "success_partial_rate": float(counts["success_partial"]) / n_episodes,
        "failure_rate": float(counts["failure"]) / n_episodes,
        "counts": counts,
        "n_episodes": n_episodes
    }
    return metrics, detailed

# Saving helpers
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
