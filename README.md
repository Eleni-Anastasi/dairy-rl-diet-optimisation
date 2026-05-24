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
- DVE requirement/intake
- OEB intake
- Milk production
- Methane emissions

Actions correspond to diet ingredient proportions on a dry matter basis.

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

Additional experiments:
```python
ppo_methane_min_5seeds.py
sac_methane_5seeds.py
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

# Repository Structure

├── simulationLibrary.py          # Core dairy cow hybrid state-space model (f3, g3, parameters)
├── env_dairy_mode3.py            # Gymnasium RL environment wrapping the cow model
├── ppo.py                        # PPO training script (Stable-Baselines3)
├── sac.py                        # SAC training script (Stable-Baselines3)
├── plot_diet_composition_ppo.py  # Plot learned diet composition over episode
├── plot_ppo_methane_min.py       # Plot methane minimisation results for PPO
├── requirements.txt              # Python dependencies
└── README.md





