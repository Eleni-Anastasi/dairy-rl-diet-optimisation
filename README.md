# Reinforcement Learning for Dairy Cow Feed Optimisation

This repository contains the implementation of my Master's thesis on applying **Deep Reinforcement Learning (RL)** to optimise dairy cow feeding strategies. The project investigates whether RL agents can learn feed compositions that simultaneously **maximize economic return** and **reduce enteric methane (CH₄) emissions**, while satisfying nutritional constraints.

Two state-of-the-art continuous control algorithms are implemented:

- **Proximal Policy Optimization (PPO)**
- **Soft Actor-Critic (SAC)**

The RL agents interact with a custom **OpenAI Gym/Gymnasium environment**, which is connected to a mechanistic dairy cow simulation model that predicts milk production, methane emissions, nutrient utilisation, and feed efficiency.

---

# Project Objectives

The objective is to determine an optimal daily feed ration for a **Mode 3 dairy cow (milking, non-pregnant)** by balancing multiple competing goals:

- Maximise feed margin (milk revenue − feed costs)
- Minimise enteric methane emissions
- Satisfy nutritional requirements (DVE, OEB, fresh matter intake)

This problem represents a continuous optimisation task where improving one objective often negatively impacts another.

---

# Environment

The custom reinforcement learning environment is implemented in:

```
env_dairy_mode3.py
```

Each environment step simulates one feeding decision and returns the resulting animal performance.

## Observation Space

The agent observes:

- Feed margin
- DVE requirement
- DVE intake
- OEB intake
- Fresh matter intake
- Milk production
- Methane emissions

## Action Space

Actions represent the proportions of feed ingredients (dry matter basis) used to compose the daily ration.

The environment automatically enforces practical feeding constraints and evaluates each proposed diet using the dairy cow simulation model.

---


## Reward Formulation

The optimisation objective is formulated as a weighted combination of economic performance and environmental impact:

$$
R_t = (1-\lambda)\cdot FeedMargin_{norm,t} - \lambda \cdot Methane_{norm,t}
$$

where both feed margin and methane are normalised.

The value of λ controls the objective:

| Lambda | Objective |
|---|---|
| 0.0 | Feed margin maximisation |
| 0.5 | Combined objective |
| 1.0 | Methane minimisation |


A higher λ gives more importance to methane reduction, while a lower λ gives more importance to feed margin.

---

# Algorithms

Implemented algorithms:
- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)

Main training scripts:
```python
ppo.py
sac.py
```

---

# Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running Training

Train PPO:

```bash
python ppo.py
```

Train SAC:

```bash
python sac.py
```

During training, the scripts periodically evaluate the current policy and save:

```text
eval_history.csv
```

The best-performing model snapshot and final trained model are automatically saved.


# Repository Structure

```text
Core files:
├── libraries/
│   └── simulationLibrary.py  # Core dairy cow hybrid state-space model
├── env_dairy_mode3.py        # RL environment
├── ppo.py                    # PPO training
├── sac.py                    # SAC training
├── requirements.txt
├── README.md

Additional experiment and plotting scripts:
├── plot_diet_composition_ppo.py
├── plot_ppo_methane_min.py
├── ppo_methane_min_5seeds.py
├── sac_methane_5seeds.py
```




# Technologies

- Python
- PyTorch
- Stable-Baselines3
- Gymnasium
- NumPy
- Pandas
- Matplotlib

---

# Research Focus

This project combines techniques from:

- Reinforcement Learning
- Deep Learning
- Optimisation
- Precision Livestock Farming
- Sustainable Agriculture
- Simulation Modelling

---




