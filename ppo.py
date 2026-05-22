# ppo_train.py

import csv
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from env_dairy_mode3 import DairyCowEnvMode3, CONC_IDXS


# ============================================================
# TRAINING SETTINGS
# ============================================================

SEED = 42
HORIZON = 83
TOTAL_STEPS = 3_000_000

EVAL_EVERY = 50_000
EVAL_EPISODES = 5
EVAL_HORIZON = 83

# File used to save observation/reward normalization statistics
NORM_PATH = "vecnormalize_ppo.pkl"


# ============================================================
# ENVIRONMENT SETTINGS
# ============================================================

ENV_KWARGS = dict(
    lambda_ch4=0.0,        # methane penalty weight in the reward
    oeb_min=0.0,
    oeb_max=200.0,
    fm_limit=100,
    alpha=0.1,             # diet smoothing parameter
    log_violations=False,
)


# ============================================================
# PPO HYPERPARAMETERS
# ============================================================

PPO_KWARGS = dict(
    learning_rate=1e-4,
    n_steps=4096,
    batch_size=128,
    n_epochs=10,
    gamma=0.95,
    gae_lambda=0.95,
    clip_range=0.1,
    ent_coef=0.05,
    vf_coef=0.5,
)


def save_eval_history(history, out_csv="eval_history.csv"):
    """Save evaluation results collected during training."""

    if not history:
        print("No eval history to save.")
        return

    keys = [
        "timesteps",
        "mean_reward",
        "mean_margin",
        "mean_methane",
        "mean_milk",
        "mean_conc_pct",
        "steps",
    ]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow({k: row.get(k, "") for k in keys})

    print(f"Saved eval history to: {out_csv}")


def make_eval_env(seed: int, horizon: int):
    """Create a separate evaluation environment."""

    env_kwargs = dict(ENV_KWARGS)
    env_kwargs["horizon"] = horizon

    return make_vec_env(
        env_id=DairyCowEnvMode3,
        n_envs=1,
        seed=seed,
        env_kwargs=env_kwargs,
        monitor_dir=None,
    )


def quick_eval(model, seed: int, episodes=EVAL_EPISODES, horizon=EVAL_HORIZON):
    """
    Evaluate the current policy and return average performance metrics.
    """

    env = make_eval_env(seed + 1000, horizon)

    # Use the same normalization statistics as during training
    env = VecNormalize.load(NORM_PATH, env)
    env.training = False
    env.norm_reward = False

    rets, margins, methanes, milks, concs = [], [], [], [], []
    total_steps = 0

    for _ in range(episodes):
        obs = env.reset()
        done = False
        t = 0

        while not done and t < horizon:
            # Deterministic actions are used for stable evaluation
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

            info = infos[0]

            rets.append(float(rewards[0]))
            margins.append(float(info.get("feed_margin", np.nan)))
            methanes.append(float(info.get("methane", np.nan)))
            milks.append(float(info.get("milk", np.nan)))

            # Concentrate percentage is calculated from the diet vector
            u = np.array(info["u"], dtype=float)
            concs.append(float(u[CONC_IDXS].sum()))

            done = bool(dones[0])
            t += 1
            total_steps += 1

    env.close()

    def mean(x):
        arr = np.array(x, dtype=float)
        return float(np.nanmean(arr)) if arr.size else float("nan")

    return {
        "mean_reward": mean(rets),
        "mean_margin": mean(margins),
        "mean_methane": mean(methanes),
        "mean_milk": mean(milks),
        "mean_conc_pct": mean(concs),
        "steps": total_steps,
    }


class EvalCallback(BaseCallback):
    """Evaluate the model periodically and save the best snapshot."""

    def __init__(self, seed: int, verbose: int = 0, best_path="ppo_mode3_best"):
        super().__init__(verbose)
        self.seed = seed
        self.best = None
        self.history = []
        self.best_path = best_path

    def _on_step(self) -> bool:
        if self.num_timesteps % EVAL_EVERY == 0:

            # Save normalization statistics before evaluating
            self.model.get_env().save(NORM_PATH)

            kpi = quick_eval(
                self.model,
                seed=self.seed,
                episodes=EVAL_EPISODES,
                horizon=EVAL_HORIZON,
            )

            kpi["timesteps"] = int(self.num_timesteps)
            self.history.append(kpi)

            # Keep the best model according to mean evaluation reward
            if self.best is None or kpi["mean_reward"] > self.best["mean_reward"]:
                self.best = kpi
                self.model.save(self.best_path)

            print(
                f"[{self.num_timesteps:>7}] "
                f"evalR={kpi['mean_reward']:.3f}  "
                f"margin={kpi['mean_margin']:.2f}  "
                f"CH4={kpi['mean_methane']:.1f}  "
                f"milk={kpi['mean_milk']:.2f}  "
                f"conc%={kpi['mean_conc_pct']:.1f}"
            )

        return True


def main():
    """Train PPO on the dairy cow feeding environment."""

    train_env_kwargs = dict(ENV_KWARGS)
    train_env_kwargs["horizon"] = HORIZON

    vec = make_vec_env(
        env_id=DairyCowEnvMode3,
        n_envs=1,
        seed=SEED,
        env_kwargs=train_env_kwargs,
    )

    # Normalize observations and rewards to improve training stability
    vec = VecNormalize(
        vec,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )

    vec.save(NORM_PATH)

    model = PPO(
        "MlpPolicy",
        vec,
        seed=SEED,
        verbose=0,
        device="cpu",
        **PPO_KWARGS,
    )

    cb = EvalCallback(seed=SEED, best_path="ppo_mode3_best")

    model.learn(total_timesteps=TOTAL_STEPS, callback=cb)

    # Save final model and final normalization statistics
    model.save("ppo_mode3_final")
    vec.save(NORM_PATH)

    print("Best snapshot:", cb.best)

    save_eval_history(cb.history, out_csv="eval_history.csv")

    vec.close()


if __name__ == "__main__":
    main()