# evaluation/eval_all_algorithms.py
import os
import argparse
from pathlib import Path
from evaluation.eval_utils import (
    load_sb3_model, eval_model_sb3, eval_reinforce,
    save_metrics_json, save_detailed_csv
)

# model paths (edit if necessary)
MODEL_PATHS = {
    "dqn": "models/dqn/dqn_final_model.zip",
    "ppo": "models/ppo/ppo_final_model.zip",
    "a2c": "models/a2c/a2c_final_model.zip",
    "reinforce": "models/reinforce/reinforce_final.pth"
}

RESULT_DIR = "results"
RAW_DIR = os.path.join(RESULT_DIR, "raw_rewards")
PLOTS_DIR = os.path.join(RESULT_DIR, "plots")

def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

def evaluate_all(n_episodes=50):
    ensure_dirs()

    # DQN, PPO, A2C via SB3
    for alg in ["dqn", "ppo", "a2c"]:
        model_path = MODEL_PATHS[alg]
        if not Path(model_path).exists():
            print(f"Warning: {alg} model not found at {model_path}, skipping.")
            continue
        print(f"Loading {alg} model from {model_path}")
        # load with env argument None (eval runner will create env)
        model = load_sb3_model(alg, model_path)
        metrics, detailed = eval_model_sb3(model, n_episodes=n_episodes)
        print(f"{alg.upper()} metrics:", metrics)
        save_metrics_json(metrics, os.path.join(RESULT_DIR, f"{alg}_metrics.json"))
        save_detailed_csv(detailed, os.path.join(RAW_DIR, f"{alg}_rewards.csv"))

    # REINFORCE
    rpath = MODEL_PATHS["reinforce"]
    if Path(rpath).exists():
        print("Evaluating REINFORCE...")
        metrics, detailed = eval_reinforce(rpath, n_episodes=n_episodes)
        print("REINFORCE metrics:", metrics)
        save_metrics_json(metrics, os.path.join(RESULT_DIR, "reinforce_metrics.json"))
        save_detailed_csv(detailed, os.path.join(RAW_DIR, "reinforce_rewards.csv"))
    else:
        print(f"Warning: Reinforce model not found at {rpath}, skipping.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50, help="Number of eval episodes per algorithm")
    args = parser.parse_args()
    evaluate_all(n_episodes=args.episodes)
    print("Evaluation complete. Results saved to results/ (raw CSVs + metrics JSON).")
