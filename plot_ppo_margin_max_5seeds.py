# plot_ppo_margin_max_stacked.py

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path("ppo_results_margin_max_minmax")
SEEDS = [42, 123, 999, 2024, 7]

def load_csv(path):
    data = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8")
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    return data

paths = [BASE_DIR / f"eval_history_seed_{s}.csv" for s in SEEDS]
runs = [load_csv(p) for p in paths]

timesteps = runs[0]["timesteps"]

reward = np.vstack([r["mean_reward"] for r in runs])
margin = np.vstack([r["mean_margin"] for r in runs])
methane = np.vstack([r["mean_methane"] for r in runs])

def stats(x):
    return x.mean(axis=0), x.min(axis=0), x.max(axis=0)

reward_mean, reward_min, reward_max = stats(reward)
margin_mean, margin_min, margin_max = stats(margin)
methane_mean, methane_min, methane_max = stats(methane)

fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)

axes[0].plot(timesteps, reward_mean)
axes[0].fill_between(timesteps, reward_min, reward_max, alpha=0.3)
axes[0].set_ylabel("Mean eval Reward")
axes[0].set_title("PPO training progress – Feed margin maximisation (λ = 0)")
axes[0].grid(True)


axes[1].plot(timesteps, margin_mean)
axes[1].fill_between(timesteps, margin_min, margin_max, alpha=0.3)
axes[1].set_ylabel("Feed margin (€/day)")
axes[1].grid(True)


axes[2].plot(timesteps, methane_mean)
axes[2].fill_between(timesteps, methane_min, methane_max, alpha=0.3)
axes[2].set_ylabel("Methane (g/day)")
axes[2].set_xlabel("Training steps")
axes[2].grid(True)


plt.tight_layout()
plt.savefig("ppo_margin_max_training.png", dpi=300)
plt.show()


print("\nBEST RESULTS per seed based on maximum reward")
best_rows = []

for r in runs:
    idx = np.argmax(r["mean_reward"])
    best_rows.append(r[idx])

best_reward = np.array([row["mean_reward"] for row in best_rows])
best_margin = np.array([row["mean_margin"] for row in best_rows])
best_methane = np.array([row["mean_methane"] for row in best_rows])
best_milk = np.array([row["mean_milk"] for row in best_rows])
best_conc = np.array([row["mean_conc_pct"] for row in best_rows])

print(f"Reward: {best_reward.mean():.3f} ± {best_reward.std():.3f}")
print(f"Feed margin: {best_margin.mean():.3f} ± {best_margin.std():.3f}")
print(f"Methane: {best_methane.mean():.3f} ± {best_methane.std():.3f}")
print(f"Milk: {best_milk.mean():.3f} ± {best_milk.std():.3f}")
print(f"Concentrate %: {best_conc.mean():.3f} ± {best_conc.std():.3f}")