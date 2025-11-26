# 🎓 Career Path Reinforcement Learning Environment  
**Student:** Eddy Gasana  
**GitHub:** https://github.com/angelo54425/Eddy_Gasana_rl_summative.git  
**Video Recording:** [*Demo video*](https://www.loom.com/share/615985209d4d49d4b555827266bb12c6 )

---

## 📌 Project Overview  
This project implements a custom **Gymnasium reinforcement learning environment** called **CareerPathEnv-v0**, where an agent simulates a student navigating four career fields:

- Private  
- Medical  
- Finance  
- Engineering  

The agent begins in a neutral zone, selects a career path by entering a zone, trains on skill tiles, and must reach the goal star with enough skill.  
The environment is rendered using **Pygame**, and multiple reinforcement learning algorithms (DQN, PPO, A2C, REINFORCE) were trained and evaluated.

---

## 📁 Project Structure  
Eddy_Gasana_rl_summative/
│
├── environment/
│ ├── career_env.py
│ ├── rendering.py
│ └── init.py
│
├── training/
│ ├── dqn_training.py
│ ├── ppo_training.py
│ ├── a2c_training.py
│ └── reinforce_training.py
│
├── evaluation/
│ ├── eval_utils.py
│ └── eval_all_algorithms.py
│
├── tests/
│ ├── test_pygame.py
│ └── debug_run.py
│
├── models/
├── results/
│ ├── raw_rewards/
│ └── plots/
│
├── main.py
└── README.md

yaml
Copy code

---

## 🧩 Environment Description  

### Grid Layout  
- **10×10 grid**, divided into four colored quadrants.  
- **Neutral center zone** where the agent starts.  
- **Training tiles** inside each quadrant.  
- **Goal star** at each quadrant’s corner.

### Zone Locking  
Once an agent enters a career zone, it becomes **locked** into that career path.

---

## 🧠 Agent  
Represents a student choosing and progressing through a career.  
The agent must:
1. Choose a zone (career)  
2. Train on valid tiles  
3. Reach the goal with enough skill  

---

## 🎮 Action Space  
`Discrete(6)`

| Action | Meaning |
|--------|---------|
| 0 | Up |
| 1 | Down |
| 2 | Left |
| 3 | Right |
| 4 | Wait |
| 5 | Train |

---

## 👁 Observation Space  
A 5-element vector:

[x, y, zone, skill_level, required_threshold]

yaml
Copy code

---

## 🏆 Reward Structure  

| Event | Reward |
|-------|--------|
| Step penalty | −0.05 |
| Idle/wait | Small negative |
| Invalid move / wall hit | −0.01 to −0.20 |
| Enter zone for the first time | +1.0 |
| Training (valid tile) | +3 → decays |
| Training (invalid tile) | −0.20 |
| Touch star early | Small positive, continue |
| Reach goal with enough skill | +80 |
| Reach goal without skill | Low reward or partial |
| Timeout | −20 |

Reward shaping strongly encourages **training before goal-seeking**.

---

## 🖥 Environment Visualization  
Rendered using **Pygame**:  
- Colored quadrants  
- Training tiles (arrows)  
- Goal star  
- Agent (black circle)  
- Sidebar showing skill, threshold, zone, and steps  

---

## 🏗 System Analysis & Design  

### 🔹 DQN (Deep Q-Network)  
- SB3 **MlpPolicy**  
- Experience replay  
- Target network updates  
- ε-greedy exploration  
- Works best with this environment due to stable gradients and temporal reward scoping.

**Key Hyperparameters:**  
learning_rate: 2.5e-4
gamma: 0.995
buffer_size: 200000
batch_size: 128
train_freq: 4
target_update_interval: 1000
exploration_fraction: 0.40
exploration_final_eps: 0.02
learning_starts: 2000

yaml
Copy code

---

### 🔹 PPO  
- Clipped surrogate objective  
- GAE advantages  
- Good for continuous tasks, less so for sparse reward grids  
- Network: `[256, 256]` for both policy and value

---

### 🔹 A2C  
- Synchronous actor-critic  
- Entropy regularization  
- Stronger exploration but less consistent than DQN

---

### 🔹 REINFORCE  
- Pure Monte-Carlo policy gradient  
- Highest variance  
- Weakest performance in this environment  

---

## 📊 Implementation Summary  

### DQN Results  
mean_reward: 33.54
success_full_rate: 0.60
success_partial_rate: 0.20
failure_rate: 0.20

shell
Copy code

### PPO Results  
mean_reward: -427.2
success_full_rate: 0.00
failure_rate: 1.00

shell
Copy code

### A2C Results  
mean_reward: 3.64
success_full_rate: 0.04

shell
Copy code

### REINFORCE Results  
mean_reward: -417.9
failure_rate: 1.00

yaml
Copy code

---

## 📉 Results Discussion  

### Cumulative Rewards  
- **DQN outperforms all methods**, achieving 60% full success.  
- **PPO and REINFORCE collapse** due to sparse / structured rewards.  
- **A2C is modest** but not competitive with DQN.

### Training Stability  
- DQN stable due to replay buffer & target updates  
- PPO unstable due to large batch/epoch requirements  
- A2C moderate  
- REINFORCE volatile  

### Episodes to Converge  
- DQN: converges quickly after ~20k–40k steps  
- Others: no stable convergence without heavy tuning  

### Generalization  
DQN generalizes well within the same grid layout even from randomized starting states.

---

## ▶️ How to Run  

### Demo (random agent + Pygame)  
```bash
python3 tests/test_pygame.py
Run the trained DQN agent
bash
Copy code
python3 main.py
Train algorithms
bash
Copy code
python3 training/dqn_training.py
python3 training/ppo_training.py
python3 training/a2c_training.py
python3 training/reinforce_training.py
Evaluate all algorithms
bash
Copy code
python3 evaluation/eval_all_algorithms.py --episodes 50
🚀 Future Improvements
Hyperparameter search (Optuna)

Curriculum learning (progressive task difficulty)

Randomized grid generation

Multi-goal or multi-agent support

📝 Contact
Eddy Gasana
GitHub Repo: https://github.com/angelo54425/Eddy_Gasana_rl_summative.git

yaml
Copy code

---

If you want, I can now:

✅ polish your report  
✅ create a PDF version  
✅ generate your 3-minute video script  
✅ build a project poster or slides  

Just tell me what you want next.
