# RL for Dairy Cow Feed Optimisation

This project applies Reinforcement Learning (RL) — specifically PPO (Proximal Policy Optimisation) and SAC (Soft Actor-Critic) — to optimise the daily feed ration of a dairy cow in Mode 3 (milking, non-pregnant). The goal is to find diet compositions that maximise the feed margin (milk revenue minus feed cost) while minimising enteric methane (CH₄) emissions.

---


# Environment

The environment is implemented in:

```python
env_dairy_mode3.py
```

The state space includes:
- Feed margin
- DVE requirement
- DVE intake
- OEB intake
- Fresh matter intake
- Milk production
- Methane emissions


Actions correspond to diet ingredient proportions on a dry matter basis.

# Reward Formulation

The reward function balances economic and environmental objectives using a weighted formulation:

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





