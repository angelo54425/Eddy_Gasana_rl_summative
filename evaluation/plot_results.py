# evaluation/plot_results.py
import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt

RESULT_DIR = "results"
RAW_DIR = os.path.join(RESULT_DIR, "raw_rewards")
PLOTS_DIR = os.path.join(RESULT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_metrics(alg):
    fp = os.path.join(RESULT_DIR, f"{alg}_metrics.json")
    if not os.path.exists(fp):
        return None
    with open(fp, "r") as f:
        return json.load(f)

def load_rewards_csv(alg):
    fp = os.path.join(RAW_DIR, f"{alg}_rewards.csv")
    if not os.path.exists(fp):
        return None
    rewards = []
    results = []
    with open(fp, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rewards.append(float(r["reward"]))
            results.append(r["result"])
    return rewards, results

def plot_mean_rewards(algorithms):
    means = []
    errs = []
    labels = []
    for alg in algorithms:
        m = load_metrics(alg)
        if m is None:
            continue
        labels.append(alg.upper())
        means.append(m["mean_reward"])
        errs.append(m["std_reward"])
    plt.figure(figsize=(8,6))
    plt.bar(labels, means, yerr=errs, capsize=6)
    plt.ylabel("Mean Reward")
    plt.title("Mean Reward per Algorithm")
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "mean_reward_bar.png")
    plt.savefig(out)
    plt.close()
    print("Saved", out)

def plot_success_rates(algorithms):
    labels = []
    rates = []
    for alg in algorithms:
        m = load_metrics(alg)
        if m is None:
            continue
        labels.append(alg.upper())
        rates.append(m["success_rate"])
    plt.figure(figsize=(8,6))
    plt.bar(labels, rates)
    plt.ylabel("Success Rate")
    plt.title("Success Rate per Algorithm")
    plt.ylim(0,1)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "success_rate_bar.png")
    plt.savefig(out)
    plt.close()
    print("Saved", out)

def plot_reward_distributions(algorithms):
    plt.figure(figsize=(10,6))
    data = []
    labels = []
    for alg in algorithms:
        lr = load_rewards_csv(alg)
        if lr is None:
            continue
        rewards, _ = lr
        data.append(rewards)
        labels.append(alg.upper())
    if not data:
        print("No data for distributions.")
        return
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel("Episode Reward")
    plt.title("Reward Distributions")
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "reward_distributions.png")
    plt.savefig(out)
    plt.close()
    print("Saved", out)

def plot_reward_curves(algorithms, window=5):
    plt.figure(figsize=(10,6))
    for alg in algorithms:
        lr = load_rewards_csv(alg)
        if lr is None:
            continue
        rewards, _ = lr
        # smooth
        rewards = np.array(rewards)
        if len(rewards) < 2:
            continue
        smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(smooth, label=alg.upper())
    plt.legend()
    plt.ylabel("Episode Reward (smoothed)")
    plt.xlabel("Episode index")
    plt.title("Reward Curves")
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "reward_curves.png")
    plt.savefig(out)
    plt.close()
    print("Saved", out)

if __name__ == "__main__":
    algs = ["dqn","ppo","a2c","reinforce"]
    plot_mean_rewards(algs)
    plot_success_rates(algs)
    plot_reward_distributions(algs)
    plot_reward_curves(algs)
    print("Plots saved to results/plots/")
