import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env_dairy_mode3 import DairyCowEnvMode3

HORIZON = 83

labels = [
    "Fresh grass",
    "Grass silage",
    "Maize silage",
    "Maize grain",
    "Soybean meal",
    "Rape seed meal",
    "Beet pulp",
    "Brewer grain",
    "Wheat",
]

colors = [
    (0.0, 0.4470, 0.7410),
    (0.8500, 0.3250, 0.0980),
    (0.9290, 0.6940, 0.1250),
    (0.4940, 0.1840, 0.5560),
    (0.4660, 0.6740, 0.1880),
    (0.3010, 0.7450, 0.9330),
    (0.6350, 0.0780, 0.1840),
    (0.0, 1.0, 1.0),
    (1.0, 1.0, 0.0),
]

def rollout_diet(model_path, lambda_ch4, seed):
    env_kwargs = dict(
        lambda_ch4=lambda_ch4,
        oeb_min=0.0,
        oeb_max=250.0,
        fm_limit=100,
        alpha=0.3,
        log_violations=False,
    )

    env = DairyCowEnvMode3(horizon=HORIZON, seed=seed, **env_kwargs)
    model = PPO.load(model_path, env=env)

    obs, info = env.reset()
    terminated = False
    truncated = False

    days = []
    u_history = []

    day = 0
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        days.append(day + 1)
        u_history.append(info["u"].copy())

        day += 1
        if day >= HORIZON:
            break

    env.close()
    return np.array(days), np.array(u_history)


# ----------------------------------------------------
# Best seeds you selected
# λ = 0   -> seed 42
# λ = 0.75 -> seed 123
# Replace paths if needed
# ----------------------------------------------------
model_lambda0 = "ppo_lambda_sweep_results/lambda_0.0/seed_42/best_model.zip"
model_lambda075 = "ppo_lambda_sweep_results/lambda_0.75/seed_123/best_model.zip"

days0, u0 = rollout_diet(model_lambda0, lambda_ch4=0.0, seed=42)
days75, u75 = rollout_diet(model_lambda075, lambda_ch4=0.75, seed=123)

# ----------------------------------------------------
# Plot side by side, same style as your original
# ----------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

axes[0].stackplot(days0, u0.T, labels=labels, colors=colors)
axes[0].set_xlabel("Days in milk")
axes[0].set_ylabel("Ingredient fraction of the diet (%)")
axes[0].set_title(r"Diet composition – RL policy ($\lambda = 0$)")
axes[0].set_ylim(0, 100)
axes[0].set_yticks(range(0, 101, 10))

axes[1].stackplot(days75, u75.T, labels=labels, colors=colors)
axes[1].set_xlabel("Days in milk")
axes[1].set_title(r"Diet composition – RL policy ($\lambda = 0.75$)")
axes[1].set_ylim(0, 100)
axes[1].set_yticks(range(0, 101, 10))

# One legend for the full figure
fig.legend(labels, loc="upper right", bbox_to_anchor=(1.12, 0.98))

plt.tight_layout()
plt.savefig("diet_comparison_lambda0_vs_075.png", dpi=300, bbox_inches="tight")
plt.show()