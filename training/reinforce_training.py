# training/reinforce_training.py
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import environment
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Simple policy net (same as eval)
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=(128, 128)):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden:
            layers.append(nn.Linear(last, h)); layers.append(nn.ReLU()); last = h
        layers.append(nn.Linear(last, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def compute_returns(rewards, gamma):
    R = 0.0
    out = []
    for r in reversed(rewards):
        R = r + gamma * R
        out.insert(0, R)
    return out

def make_env():
    import environment  # noqa
    return gym.make("CareerPathEnv-v0")

def main():
    model_dir = "models/reinforce"
    os.makedirs(model_dir, exist_ok=True)

    env = make_env()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    lr = float(os.environ.get("REINFORCE_LR", 1e-3))
    gamma = float(os.environ.get("REINFORCE_GAMMA", 0.99))
    total_episodes = int(os.environ.get("REINFORCE_EPISODES", 2000))
    batch_episodes = int(os.environ.get("REINFORCE_BATCH_EP", 4))
    hidden = (128, 128)
    save_every = int(os.environ.get("REINFORCE_SAVE_EVERY", 200))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PolicyNet(obs_dim, action_dim, hidden).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    episode = 0
    running_rewards = []

    t0 = time.time()
    while episode < total_episodes:
        batch_obs = []
        batch_acts = []
        batch_rets = []
        batch_rewards = []

        for _ in range(batch_episodes):
            obs, _ = env.reset()
            done = False
            ep_rewards = []
            ep_obs = []
            ep_acts = []

            while True:
                x = torch.tensor(obs, dtype=torch.float32).to(device)
                logits = policy(x.unsqueeze(0)).squeeze(0)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
                next_obs, reward, terminated, truncated, info = env.step(int(action))

                ep_obs.append(obs)
                ep_acts.append(action)
                ep_rewards.append(reward)

                obs = next_obs
                if terminated or truncated:
                    break

            returns = compute_returns(ep_rewards, gamma)
            batch_obs.extend(ep_obs)
            batch_acts.extend(ep_acts)
            batch_rets.extend(returns)
            batch_rewards.append(sum(ep_rewards))
            running_rewards.append(sum(ep_rewards))
            episode += 1

        obs_tensor = torch.tensor(np.array(batch_obs), dtype=torch.float32).to(device)
        acts_tensor = torch.tensor(np.array(batch_acts), dtype=torch.int64).to(device)
        rets_tensor = torch.tensor(np.array(batch_rets), dtype=torch.float32).to(device)
        rets_tensor = (rets_tensor - rets_tensor.mean()) / (rets_tensor.std() + 1e-8)

        logits = policy(obs_tensor)
        logp_all = torch.log_softmax(logits, dim=-1)
        selected_logp = logp_all[range(len(acts_tensor)), acts_tensor]
        loss = - (selected_logp * rets_tensor).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 10 == 0:
            print(f"[REINFORCE] Ep {episode:4d} | Batch avg {np.mean(batch_rewards):.3f} | Recent100 {np.mean(running_rewards[-100:]):.3f} | loss {loss.item():.4f}")

        if episode % save_every == 0:
            save_path = Path(model_dir) / f"reinforce_ep{episode}.pth"
            torch.save(policy.state_dict(), save_path)
            print("Saved REINFORCE policy to", save_path)

    torch.save(policy.state_dict(), Path(model_dir) / "reinforce_final.pth")
    print("REINFORCE training complete in", time.time() - t0, "seconds")
    env.close()

if __name__ == "__main__":
    main()
