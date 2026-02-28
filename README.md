# EECS590 – Mini Project 3

**Author:** Davis Payne
**Course:** EECS 590 – Reinforcement Learning
**Institution:** North Dakota University System

---

## Overview

This repository contains the code, plots, and supporting materials for **Mini Project 3**, which covers two problems in deep reinforcement learning.

---

## Problem 1: Cadmium Rod Control in a Nuclear Reactor

Implements tabular reinforcement learning (SARSA(λ) and Q-learning) to control cadmium rod insertion in a simulated nuclear reactor. The agent learns to maintain reactivity within a safe productive zone while avoiding meltdown.

**Key topics:**
- MDP formulation (state/action space, reward, discount factor)
- On-policy vs off-policy learning
- Eligibility traces and the bias–variance trade-off
- Linear function approximation with RBF features (Challenge)

**Script:** `scripts/Davis_Payne_Mini_Project3_Problem1.py`

---

## Problem 2: Superhuman Atari via PPO and Saliency Analysis

Trains a **Proximal Policy Optimisation (PPO)** agent with an Actor-Critic CNN on `BreakoutNoFrameskip-v4` using the Arcade Learning Environment (ALE). The agent is trained for 10 million steps with 8 parallel environments on a T4 GPU (Google Colab).

Saliency analysis is performed using:
- **Perturbation-based saliency** — measures log π(a|f) sensitivity to P×P patch occlusion
- **Gradient-based saliency** — computes |∂ log π(a|f)/∂f| via backpropagation
- **Adversarial perturbation** — finds minimal L∞ δ to flip the greedy action

**Script:** `scripts/Davis_Payne_Mini_Project3_Problem2.py`
**Colab Notebook:** `Davis_Payne_Mini_Project3_Problem2_Colab.ipynb`

---

## Repository Structure

```
├── scripts/
│   ├── Davis_Payne_Mini_Project3_Problem1.py   # Nuclear reactor control (SARSA / Q-learning)
│   └── Davis_Payne_Mini_Project3_Problem2.py   # Atari PPO + saliency analysis
├── plots/                                       # Generated training and saliency figures
├── images/                                      # Additional images and screenshots
├── Davis_Payne_Mini_Project3_Problem2_Colab.ipynb  # Self-contained Colab notebook
└── README.md
```

---

## Dependencies

```bash
pip install torch torchvision gymnasium[atari] gymnasium[accept-rom-license] \
            ale-py opencv-python matplotlib numpy
```

---

## Running the Code

**Problem 1 (local CPU):**
```bash
python scripts/Davis_Payne_Mini_Project3_Problem1.py
```

**Problem 2 – quick smoke test (~2 min):**
```bash
python scripts/Davis_Payne_Mini_Project3_Problem2.py --smoke-test
```

**Problem 2 – full training on Colab (10M steps, 8 envs, T4 GPU):**
Upload `Davis_Payne_Mini_Project3_Problem2_Colab.ipynb` to Google Colab, set runtime to **T4 GPU**, and run all cells.
