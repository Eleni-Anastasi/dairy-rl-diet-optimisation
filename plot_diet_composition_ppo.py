import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from env_dairy_mode3 import DairyCowEnvMode3


# ============================================================
# SETTINGS
# ============================================================

HORIZON = 83

RESULTS_DIR = "ppo_results_methane_min" # change

BEST_SEED = 42

MODEL_PATH = os.path.join(
    RESULTS_DIR,
    f"best_model_seed_{BEST_SEED}.zip"
)

VECNORM_PATH = os.path.join(
    RESULTS_DIR,
    f"vecnormalize_seed_{BEST_SEED}.pkl"
)


# ============================================================
# LABELS + COLORS
# ============================================================

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


# ============================================================
# CREATE ENV
# ============================================================

env_kwargs = dict(
    horizon=HORIZON,
    lambda_ch4=0.0,
    oeb_min=0.0,
    oeb_max=250.0,
    fm_limit=100,
    alpha=0.3,
    log_violations=False,
)

env = make_vec_env(
    DairyCowEnvMode3,
    n_envs=1,
    seed=BEST_SEED,
    env_kwargs=env_kwargs,
)

env = VecNormalize.load(VECNORM_PATH, env)

env.training = False
env.norm_reward = False


# ============================================================
# LOAD MODEL
# ============================================================

model = PPO.load(MODEL_PATH)


# ============================================================
# ROLLOUT
# ============================================================

obs = env.reset()

days = []
u_history = []

done = False
day = 0

while not done and day < HORIZON:

    action, _ = model.predict(obs, deterministic=True)

    obs, reward, dones, infos = env.step(action)

    info = infos[0]

    days.append(day + 1)
    u_history.append(info["u"].copy())

    done = bool(dones[0])
    day += 1

u_history = np.array(u_history)

env.close()


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(8, 6))

plt.stackplot(
    days,
    u_history.T,
    labels=labels,
    colors=colors
)

plt.xlabel("Days in milk")
plt.ylabel("Ingredient fraction of the diet (%)")

plt.title(
    "Diet composition – PPO feed methane\n"
    
)

plt.ylim(0, 100)
plt.yticks(range(0, 101, 10))

plt.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5)
)

plt.tight_layout()

plt.savefig(
    "diet_composition_methane_min.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()