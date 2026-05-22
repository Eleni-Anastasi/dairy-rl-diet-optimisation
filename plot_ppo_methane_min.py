import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
RESULTS_DIR = "ppo_results_methane_min"
SEEDS = [42, 123, 999, 2024, 7]


# ---------------------------------------------------
# LOAD CSV FILES
# ---------------------------------------------------
files = sorted(glob.glob(os.path.join(RESULTS_DIR, "eval_history_seed_*.csv")))

dfs = []
for f in files:
    df = pd.read_csv(f)
    seed = os.path.basename(f).split("_")[-1].replace(".csv", "")
    df["seed"] = int(seed)
    dfs.append(df)

print(f"Loaded {len(dfs)} seeds")


# ---------------------------------------------------
# EXTRACT BEST KPI PER SEED
# ---------------------------------------------------
best_rows = []

for df in dfs:
    seed = int(df["seed"].iloc[0])
    best_row = df.loc[df["mean_reward"].idxmax()]

    best_rows.append({
        "seed": seed,
        "reward": best_row["mean_reward"],
        "margin": best_row["mean_margin"],
        "methane": best_row["mean_methane"],
        "milk": best_row["mean_milk"],
        "conc": best_row["mean_conc_pct"],
        "timesteps": best_row["timesteps"],
    })

best_df = pd.DataFrame(best_rows)

print("\nPer-seed best results:\n")
print(best_df)

print("\nSummary statistics:\n")
for col in ["margin", "methane", "milk", "conc"]:
    print(
        f"{col:10s} mean = {best_df[col].mean():.3f} | "
        f"std = {best_df[col].std():.3f}"
    )



timesteps = dfs[0]["timesteps"].values

reward = np.array([df["mean_reward"].values for df in dfs])
margin = np.array([df["mean_margin"].values for df in dfs])
methane = np.array([df["mean_methane"].values for df in dfs])

reward_mean = reward.mean(axis=0)
reward_min = reward.min(axis=0)
reward_max = reward.max(axis=0)

margin_mean = margin.mean(axis=0)
margin_min = margin.min(axis=0)
margin_max = margin.max(axis=0)

methane_mean = methane.mean(axis=0)
methane_min = methane.min(axis=0)
methane_max = methane.max(axis=0)



def millions_formatter(x, pos):
    return f"{x / 1e6:.1f}M"


# ---------------------------------------------------
# PLOT
# ---------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)

axes[0].plot(timesteps, reward_mean, linewidth=2)
axes[0].fill_between(timesteps, reward_min, reward_max, alpha=0.3)
axes[0].set_ylabel("Mean eval reward")
axes[0].grid(True, alpha=0.3)

axes[1].plot(timesteps, margin_mean, linewidth=2)
axes[1].fill_between(timesteps, margin_min, margin_max, alpha=0.3)
axes[1].set_ylabel("Feed margin (€/day)")
axes[1].set_xlabel("Training steps")
axes[1].grid(True, alpha=0.3)

axes[2].plot(timesteps, methane_mean, linewidth=2)
axes[2].fill_between(timesteps, methane_min, methane_max, alpha=0.3)
axes[2].set_ylabel("Methane (g/day)")
axes[2].grid(True, alpha=0.3)

for ax in axes:
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(millions_formatter))

fig.suptitle("PPO training progress — Methane minimisation")

plt.tight_layout()

plt.savefig(
    "ppo_methane_min_training.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()