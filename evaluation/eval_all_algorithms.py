# evaluation/eval_all_algorithms.py
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import environment


import argparse
from pathlib import Path

from evaluation.eval_utils import (
    load_sb3_model, eval_model_sb3, eval_reinforce,
    save_metrics_json, save_detailed_csv
)

MODEL_PATHS = {
    "dqn": "models/dqn/dqn_final_model.zip",
    "ppo": "models/ppo/ppo_final_model.zip",
    "a2c": "models/a2c/a2c_final_model.zip",
    "reinforce": "models/reinforce/reinforce_final.pth",
}

RESULT_DIR = "results"
RAW_DIR = os.path.join(RESULT_DIR, "raw_rewards")
PLOTS_DIR = os.path.join(RESULT_DIR, "plots")


def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def evaluate_all(n_episodes=50):
    ensure_dirs()

    for alg in ["dqn", "ppo", "a2c"]:
        path = MODEL_PATHS[alg]
        if not Path(path).exists():
            print(f"Skipping {alg}: no model at {path}")
            continue

        model = load_sb3_model(alg, path)
        metrics, detailed = eval_model_sb3(model, n_episodes=n_episodes)

        print(f"{alg.upper()} metrics:", metrics)
        save_metrics_json(metrics, f"{RESULT_DIR}/{alg}_metrics.json")
        save_detailed_csv(detailed, f"{RAW_DIR}/{alg}_rewards.csv")

    rein_path = MODEL_PATHS["reinforce"]
    if Path(rein_path).exists():
        metrics, detailed = eval_reinforce(rein_path, n_episodes=n_episodes)
        print("REINFORCE metrics:", metrics)
        save_metrics_json(metrics, f"{RESULT_DIR}/reinforce_metrics.json")
        save_detailed_csv(detailed, f"{RAW_DIR}/reinforce_rewards.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    evaluate_all(n_episodes=args.episodes)
