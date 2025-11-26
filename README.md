# 🎓 Career Path Reinforcement Learning Environment  
*A custom RL environment simulating a student's journey through different career fields.*

---

## 📌 Overview  
This project implements a fully custom **Reinforcement Learning environment** where an agent represents a student navigating toward a successful career path. The grid-world environment includes four career zones — **Private, Medical, Finance, Engineering** — each with training tiles and a final goal (star). The agent must choose a career, train sufficiently, and reach its goal with enough skill.

The project includes:

- A complete **Gymnasium custom environment**  
- A **Pygame renderer** for visualization  
- Training scripts for **DQN, PPO, A2C, and REINFORCE**  
- Evaluation utilities & metrics  
- Result visualizations (reward curves)  
- A runnable agent demo (`main.py`)

---

## 🗂 Project Structure


---

## 🧠 Environment Description

### 🟦 Grid Layout  
- A **10×10 grid**  
- **4 career zones** (colored)  
- A central neutral zone  
- Training tiles specific to each career  
- A final star goal per career  
- Once a zone is entered → it becomes **locked**

---

### 🎮 Action Space — `Discrete(6)`

| Action | Meaning     |
|--------|-------------|
| 0 | Up    |
| 1 | Down  |
| 2 | Left  |
| 3 | Right |
| 4 | Wait  |
| 5 | Train |

---

### 👁 Observation Space (`shape = (5,)`)

---

## 🏆 Reward Structure

- **−0.05** per step  
- **−0.02** idle penalty  
- **−0.20** invalid movement  
- **+1.0** when entering chosen career zone  
- **+0.05 → +3** for training (scaled)  
- **Small reward** for touching star early (episode continues)  
- **+80** reaching the final star with enough skill  
- **−20** timeout penalty  

This encourages exploration, focused training, and efficient reaching of the career goal.

---

## 🤖 Algorithms Implemented

### **Deep Q-Network (DQN)**  
- Replay memory  
- Target network  
- ε-greedy exploration  
- Tuned for sparse rewards  
- **Best-performing model**

### **PPO (Proximal Policy Optimization)**  
- Clipped objective  
- GAE  
- Stable Actor-Critic architecture

### **A2C (Advantage Actor Critic)**  
- Shared network  
- Entropy regularization  

### **REINFORCE (Monte-Carlo Policy Gradient)**  
- Baseline-free  
- Fully stochastic policy  
- Struggles with sparse reward design  

---

## 📈 Evaluation Results

| Algorithm | Mean Reward | Success Full | Partial | Failure |
|----------|-------------|--------------|---------|---------|
| **DQN** | 33.5 | **60%** | 20% | 20% |
| **A2C** | 3.64 | 4% | 16% | 80% |
| **PPO** | -427 | 0% | 0% | 100% |
| **REINFORCE** | -417 | 0% | 0% | 100% |

➡ **DQN is the clear top performer**

---

## ▶ Run Demo (Random Agent)
```bash
python3 tests/test_pygame.py

Let me know if you want:

✔ Shields.io badges  
✔ A GIF of the environment  
✔ A “Future Work” section  
✔ A contributors section  

I can add any of these!
