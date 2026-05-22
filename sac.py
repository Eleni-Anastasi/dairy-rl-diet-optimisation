# sac_train.py

import os
import csv
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecCheckNan, VecNormalize
from stable_baselines3.common.vec_env import SubprocVecEnv

from env_dairy_mode3 import DairyCowEnvMode3, CONC_IDXS


# ============================================================
# TRAINING SETTINGS
# ============================================================

SEED = 43


HORIZON = 83

TOTAL_STEPS = 2_000_000

# Run evaluation every N training steps
EVAL_EVERY = 50_000
EVAL_EPISODES = 5
EVAL_HORIZON = 83

# File storing normalization statistics
NORM_STATS_PATH = "vecnormalize_sac.pkl"


# ============================================================
# ENVIRONMENT SETTINGS
# ============================================================

ENV_KWARGS = dict(
    lambda_ch4=0.75,      # methane penalty weight
    oeb_min=0.0,
    oeb_max=200.0,
    fm_limit=100,
    alpha=0.1,            # diet smoothing parameter
    log_violations=False,
)


# ============================================================
# SAC HYPERPARAMETERS
# ============================================================

SAC_KWARGS = dict(
    learning_rate=1e-4,
    buffer_size=1_000_000,
    learning_starts=50_000,   # allow replay buffer to fill first
    batch_size=256,
    tau=0.005,
    gamma=0.95,
    train_freq=1,
    gradient_steps=1,
    ent_coef="auto",
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=0,
)


# ============================================================
# SAVE EVALUATION RESULTS
# ============================================================

def save_eval_history(history, out_csv="eval_history.csv"):
    """Save evaluation metrics collected during training."""

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
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()

        for row in history:
            w.writerow({k: row.get(k, "") for k in keys})

    print(f"Saved eval history to: {out_csv}")


# ============================================================
# TRAINING ENVIRONMENT
# ============================================================

def make_train_env(seed: int, horizon: int):
    """
    Create training environment with:
    - NaN checking
    - observation normalization
    - reward normalization
    """

    env_kwargs = dict(ENV_KWARGS)
    env_kwargs["horizon"] = horizon

    vec = make_vec_env(
        env_id=DairyCowEnvMode3,
        n_envs=1,
        seed=seed,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=env_kwargs,
        monitor_dir=None,
    )

    # Detect invalid values during training
    vec = VecCheckNan(vec, raise_exception=True)

    # Normalize observations and rewards for training stability
    vec = VecNormalize(
        vec,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )

    return vec


# ============================================================
# EVALUATION ENVIRONMENT
# ============================================================

def make_eval_env(seed: int, horizon: int, norm_stats_path: str):
    """
    Create evaluation environment using the same normalization
    statistics as training.
    """

    env_kwargs = dict(ENV_KWARGS)
    env_kwargs["horizon"] = horizon

    eval_vec = make_vec_env(
        env_id=DairyCowEnvMode3,
        n_envs=1,
        seed=seed,
        env_kwargs=env_kwargs,
        monitor_dir=None,
    )

    eval_vec = VecCheckNan(eval_vec, raise_exception=True)

    if not os.path.exists(norm_stats_path):
        raise FileNotFoundError(
            f"VecNormalize stats not found at '{norm_stats_path}'."
        )

    # Load training normalization statistics
    eval_vec = VecNormalize.load(norm_stats_path, eval_vec)

    # Disable statistics updates during evaluation
    eval_vec.training = False

    # Report true rewards during evaluation
    eval_vec.norm_reward = False

    return eval_vec


# ============================================================
# QUICK EVALUATION
# ============================================================

def quick_eval(
    model,
    seed: int,
    norm_stats_path: str,
    episodes=EVAL_EPISODES,
    horizon=EVAL_HORIZON,
) -> dict:
    """
    Deterministic evaluation of the current SAC policy.
    """

    env = make_eval_env(seed + 1000, horizon, norm_stats_path)

    rets, margins, methanes, milks, concs = [], [], [], [], []
    total_steps = 0

    for _ in range(episodes):

        obs = env.reset()
        done = False
        t = 0

        while (not done) and (t < horizon):

            # Deterministic actions for stable evaluation
            action, _ = model.predict(obs, deterministic=True)

            obs, rewards, dones, infos = env.step(action)

            info0 = infos[0]

            rets.append(float(rewards[0]))
            margins.append(float(info0.get("feed_margin", np.nan)))
            methanes.append(float(info0.get("methane", np.nan)))
            milks.append(float(info0.get("milk", np.nan)))

            # Calculate concentrate percentage from diet vector
            u = np.array(info0["u"], float)
            concs.append(float(u[CONC_IDXS].sum()))

            t += 1
            total_steps += 1

            done = bool(dones[0])

    env.close()

    def m(x):
        a = np.array(x, float)
        return float(np.nanmean(a)) if a.size else float("nan")

    return {
        "mean_reward": m(rets),
        "mean_margin": m(margins),
        "mean_methane": m(methanes),
        "mean_milk": m(milks),
        "mean_conc_pct": m(concs),
        "steps": total_steps,
    }


# ============================================================
# EVALUATION CALLBACK
# ============================================================

class EvalCallback(BaseCallback):
    """
    Periodically evaluates the model and saves the best snapshot.
    """

    def __init__(
        self,
        seed: int,
        norm_stats_path: str,
        verbose: int = 0,
        best_path="sac_mode3_best",
    ):
        super().__init__(verbose)

        self.seed = seed
        self.best = None
        self.history = []
        self.best_path = best_path
        self.norm_stats_path = norm_stats_path

    def _on_step(self) -> bool:

        if self.num_timesteps % EVAL_EVERY == 0:

            # Save latest normalization statistics
            self.training_env.save(self.norm_stats_path)

            kpi = quick_eval(
                self.model,
                seed=self.seed,
                norm_stats_path=self.norm_stats_path,
                episodes=EVAL_EPISODES,
                horizon=EVAL_HORIZON,
            )

            kpi["timesteps"] = int(self.num_timesteps)

            self.history.append(kpi)

            # Save best model based on evaluation reward
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


# ============================================================
# TRAINING
# ============================================================

def main():
    """Train SAC on the dairy cow feeding environment."""

    train_vec = make_train_env(SEED, HORIZON)

    model = SAC(
        policy="MlpPolicy",
        env=train_vec,
        seed=SEED,
        device="cpu",
        **SAC_KWARGS,
    )

    cb = EvalCallback(
        seed=SEED,
        norm_stats_path=NORM_STATS_PATH,
        best_path="sac_mode3_best",
    )

    # Save initial normalization statistics
    train_vec.save(NORM_STATS_PATH)

    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=cb,
    )

    # Save final model and normalization statistics
    model.save("sac_mode3_final")
    train_vec.save(NORM_STATS_PATH)

    print("Best snapshot:", cb.best)

    save_eval_history(
        cb.history,
        out_csv="eval_history.csv",
    )

    train_vec.close()


if __name__ == "__main__":
    main()