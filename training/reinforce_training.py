# FILE: training/reinforce_training.py
import os, sys, time, math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import environment
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ensure registration
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

# Simple policy network
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=(128,128)):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, action_dim))  # logits
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # logits

def compute_returns(rewards, gamma):
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def main():
    model_dir = "models/reinforce"
    os.makedirs(model_dir, exist_ok=True)

    env = gym.make("CareerPathEnv-v0")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Hyperparams (from plan)
    lr = 1e-3
    gamma = 0.99
    total_episodes = 10000            # number of episodes
    batch_episodes = 4                # how many episodes per update (small)
    hidden = (128, 128)
    save_every = 500                  # save every N episodes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PolicyNet(obs_dim, action_dim, hidden).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    running_rewards = deque(maxlen=100)
    episode = 0
    start = time.time()

    while episode < total_episodes:
        batch_obs = []
        batch_acts = []
        batch_rets = []
        batch_lens = []
        batch_rewards = []

        # collect batch_episodes episodes
        for ep in range(batch_episodes):
            obs, _ = env.reset()
            obs = torch.tensor(obs, dtype=torch.float32).to(device)
            done = False
            rewards = []
            log_probs = []
            actions = []
            observations = []

            while True:
                logits = policy(obs.unsqueeze(0)).squeeze(0)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
                logp = dist.log_prob(torch.tensor(action).to(device))

                next_obs, reward, terminated, truncated, info = env.step(int(action))
                observations.append(obs.cpu().numpy())
                actions.append(action)
                rewards.append(reward)
                log_probs.append(logp)

                obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
                if terminated or truncated:
                    break

            returns = compute_returns(rewards, gamma)
            # store
            batch_obs.extend(observations)
            batch_acts.extend(actions)
            batch_rets.extend(returns)
            batch_lens.append(len(rewards))
            batch_rewards.append(sum(rewards))
            running_rewards.append(sum(rewards))
            episode += 1

        # Convert to tensors
        obs_tensor = torch.tensor(np.array(batch_obs), dtype=torch.float32).to(device)
        acts_tensor = torch.tensor(np.array(batch_acts), dtype=torch.int64).to(device)
        rets_tensor = torch.tensor(np.array(batch_rets), dtype=torch.float32).to(device)

        # Normalize returns
        rets_tensor = (rets_tensor - rets_tensor.mean()) / (rets_tensor.std() + 1e-8)

        # Compute loss: negative log lik * returns
        logits = policy(obs_tensor)
        logp_all = torch.log_softmax(logits, dim=-1)
        selected_logp = logp_all[range(len(acts_tensor)), acts_tensor]
        loss = - (selected_logp * rets_tensor).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 10 == 0:
            avg_reward = np.mean(batch_rewards)
            avg_recent = np.mean(running_rewards) if running_rewards else 0.0
            print(f"Episode {episode:5d} | Avg batch reward {avg_reward:.3f} | Recent100 {avg_recent:.3f} | loss {loss.item():.4f}")

        if episode % save_every == 0:
            save_path = os.path.join(model_dir, f"reinforce_ep{episode}.pth")
            torch.save(policy.state_dict(), save_path)
            print("Saved policy to", save_path)

    # final save
    torch.save(policy.state_dict(), os.path.join(model_dir, "reinforce_final.pth"))
    print("Training complete. Time elapsed:", time.time() - start)

if __name__ == "__main__":
    main()
