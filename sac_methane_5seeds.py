# sac_methane_5seeds.py

import os
import csv
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from env_dairy_mode3 import DairyCowEnvMode3, CONC_IDXS


SEEDS = [42, 123, 999]

HORIZON = 83
TOTAL_STEPS = 5_000_000
EVAL_EVERY = 50_000
EVAL_EPISODES = 5
EVAL_HORIZON = 83

LAMBDA_CH4 = 1.0 # change to 0.0, 0.5 ...

RESULTS_DIR = "sac_results_methane" # change
os.makedirs(RESULTS_DIR, exist_ok=True)


ENV_KWARGS = dict(
    lambda_ch4=LAMBDA_CH4,
    oeb_min=0.0,
    oeb_max=200.0,
    fm_limit=100,
    alpha=0.1,
    log_violations=False,
)


SAC_KWARGS = dict(
    learning_rate=1e-4,
    buffer_size=1_000_000,
    learning_starts=50_000,
    batch_size=256,
    tau=0.005,
    gamma=0.95,
    train_freq=1,
    gradient_steps=1,
    ent_coef="auto",
    verbose=0,
)


def save_eval_history(history, path):
    keys = [
        "timesteps",
        "mean_reward",
        "mean_margin",
        "mean_methane",
        "mean_milk",
        "mean_conc_pct",
        "steps",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow({k: row.get(k, "") for k in keys})


def make_env(seed):
    env_kwargs = dict(ENV_KWARGS)
    env_kwargs["horizon"] = HORIZON

    return make_vec_env(
        env_id=DairyCowEnvMode3,
        n_envs=4,
        seed=seed,
        env_kwargs=env_kwargs,
    )


def quick_eval(model, seed, norm_path):
    env = make_env(seed + 1000)

    env = VecNormalize.load(norm_path, env)
    env.training = False
    env.norm_reward = False

    rets, margins, methanes, milks, concs = [], [], [], [], []
    total_steps = 0

    for _ in range(EVAL_EPISODES):
        obs = env.reset()
        done = False
        t = 0

        while not done and t < EVAL_HORIZON:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

            info = infos[0]

            rets.append(float(rewards[0]))
            margins.append(float(info.get("feed_margin", np.nan)))
            methanes.append(float(info.get("methane", np.nan)))
            milks.append(float(info.get("milk", np.nan)))

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
    def __init__(self, seed, run_dir, verbose=0):
        super().__init__(verbose)
        self.seed = seed
        self.run_dir = run_dir
        self.norm_path = os.path.join(run_dir, f"vecnormalize_seed_{seed}.pkl")
        self.best = None
        self.history = []

    def _on_step(self):
        if self.num_timesteps % EVAL_EVERY == 0:
            self.model.get_env().save(self.norm_path)

            kpi = quick_eval(
                self.model,
                seed=self.seed,
                norm_path=self.norm_path,
            )

            kpi["timesteps"] = int(self.num_timesteps)
            self.history.append(kpi)

            if self.best is None or kpi["mean_reward"] > self.best["mean_reward"]:
                self.best = kpi
                self.model.save(os.path.join(self.run_dir, f"best_model_seed_{self.seed}"))

            print(
                f"[SAC lambda={LAMBDA_CH4} | seed={self.seed}] "
                f"[{self.num_timesteps:>7}] "
                f"evalR={kpi['mean_reward']:.3f} "
                f"margin={kpi['mean_margin']:.2f} "
                f"CH4={kpi['mean_methane']:.1f} "
                f"milk={kpi['mean_milk']:.2f} "
                f"conc%={kpi['mean_conc_pct']:.1f}"
            )

        return True


def run_one(seed):
    print("\n" + "=" * 70)
    print(f"Training SAC | lambda={LAMBDA_CH4} | seed={seed}")
    print("=" * 70)

    env = make_env(seed)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    norm_path = os.path.join(RESULTS_DIR, f"vecnormalize_seed_{seed}.pkl")
    env.save(norm_path)

    model = SAC(
        "MlpPolicy",
        env,
        seed=seed,
        device="cpu",
        **SAC_KWARGS,
    )

    callback = EvalCallback(seed=seed, run_dir=RESULTS_DIR)

    model.learn(total_timesteps=TOTAL_STEPS, callback=callback)

    model.save(os.path.join(RESULTS_DIR, f"final_model_seed_{seed}"))
    env.save(os.path.join(RESULTS_DIR, f"vecnormalize_seed_{seed}.pkl"))

    out_csv = os.path.join(RESULTS_DIR, f"eval_history_seed_{seed}.csv")
    save_eval_history(callback.history, out_csv)

    print(f"Saved: {out_csv}")
    print("Best snapshot:", callback.best)

    env.close()


def main():
    for seed in SEEDS:
        run_one(seed)


if __name__ == "__main__":
    main()