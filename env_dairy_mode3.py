from __future__ import annotations
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box


from libraries.simulationLibrary import DefineParameters, DefineSignals, f3, g3

np.seterr(all="warn")

# ----------------------------- Constants -----------------------------
# Baseline diet 
BASELINE_U = np.array([9, 35.6, 18.2, 18.6, 11.2, 3.7, 3.7, 0, 0], dtype=float)

# Per-ingredient soft upper bounds
UMAX = np.array([100, 100, 100, 20, 20, 20, 20, 20, 10], dtype=float)


# Indices of concentrates 
CONC_IDXS = [3, 4, 5, 6, 7, 8]
CONC_MAX  = 40.0   

TOTAL_DM  = 100.0  

# -------------------------- Hard constraint sum=100 -----------------------------
def enforce_sum100(u_raw: np.ndarray, total: float = TOTAL_DM) -> np.ndarray:
    """Hard constraint: sum(u) == total and u >= 0."""
    u = np.asarray(u_raw, float).reshape(-1)
    u = np.clip(u, 0.0, None) # Forces no negative ingredients. Anything < 0 becomes 0.
    s = u.sum()
    if s <= 1e-12:
        u = enforce_sum100(BASELINE_U, total)
    else:
        u *= (total / s)
    return np.maximum(u, 0.0)

# ---------------------------- Mode-3 ---------------------------
def _constant_b_mode3() -> dict:
    _, _, b0 = DefineParameters()
    b = dict(b0)
    b["s"]  = 1186
    b["q1"] = 1
    b["q2"] = 0
    b["q3"] = 2
    b["c1"] = 0
    b["c2"] = 0
    return b

# ====================================================================
#                             Environment
# ====================================================================

class DairyCowEnvMode3(Env):
    """
    One step = one day.
    Action a in [0,1]^9 -> scaled by UMAX (for magnitude) -> normalized to sum 100%.
    Observation (7): [feed_margin, DVE_req, DVE_intake, OEB_intake, FM_weight, milk, methane]
    Reward:
        r= margin_norm - λ * methane_norm
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        horizon: int = 83,
        seed: int | None = None,

        # Main trade-off
        lambda_ch4: float = 1.0,

        oeb_min: float = 0.0,
        oeb_max: float = 200.0,
        dve_multiplier: float = 1.2,    # DVE_intake ≤ dve_multiplier * DVE_req
        fm_limit: float | None = 100.0, # None disables FM feasibility

        # Day-to-day smoothing of diet
        alpha: float = 0.5,

        # Logs
        log_violations: bool = True,

    ):
        super().__init__()

        # Model params & mode
        self.p, self.ops_base, _ = DefineParameters()
        self.ds = 0.0
        self.b  = _constant_b_mode3()

        # RL spaces
        self.action_space = Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32)
        self.observation_space = Box(
            low=-1e12*np.ones(7, np.float32),
            high=+1e12*np.ones(7, np.float32),
            dtype=np.float32
        )

        # Episode control
        self.horizon = int(horizon)
        self.rng = np.random.default_rng(seed)
        self._k = 0

      
        # Reward config
        self.lambda_ch4 = float(lambda_ch4)

        # Min-max ranges for reward scaling
        self.margin_min = 10.0
        self.margin_max = 15.0

        self.methane_min = 400.0
        self.methane_max = 500.0  
        #self.milk_ref = 40.0

        # Feasibility knobs
        self.oeb_min = float(oeb_min)
        self.oeb_max = float(oeb_max)
        self.dve_multiplier = float(dve_multiplier)
        self.fm_limit = None if fm_limit is None else float(fm_limit)

        # Smoothing & logging
        self.alpha = float(alpha)
        self.log_violations = bool(log_violations)

        # State
        nx = int(self.ops_base.get("nx", 0))
        self._x = np.zeros(nx, dtype=float)


        self.umax = UMAX.astype(float).copy()
        self.conc_idxs = list(CONC_IDXS)
        self.conc_max = float(CONC_MAX)

        # Baseline diet (normalized to 100)
        self._last_u = enforce_sum100(BASELINE_U, TOTAL_DM)

    
    @staticmethod
    def _y_to_obs(y: np.ndarray) -> tuple[np.ndarray, dict]:
        # g3: [y1,y2,y3, milk, methane, feed_margin, DVE_req, DVE_intake, OEB_intake, FM_weight]
        milk        = float(y[3])
        methane     = float(y[4])
        feed_margin = float(y[5])
        DVE_req     = float(y[6])
        DVE_intake  = float(y[7])
        OEB_intake  = float(y[8])
        FM_weight   = float(y[9])
        obs = np.array(
            [feed_margin, DVE_req, DVE_intake, OEB_intake, FM_weight, milk, methane],
            dtype=np.float32
        )
        metrics = dict(feed_margin=feed_margin, dve_req=DVE_req, dve_int=DVE_intake,
                       oeb_int=OEB_intake, fm_weight=FM_weight, milk=milk, methane=methane)
        return obs, metrics

    
    def _violations(self, u: np.ndarray, m: dict) -> list[str]:
        v = []
        # per-ingredient soft max 
        over = np.clip(u - self.umax, 0.0, None)
        if float(over.sum()) > 1e-9:
            v.append("ing_over")

        # concentrates total soft cap
        conc_sum = float(u[self.conc_idxs].sum())
        if conc_sum > self.conc_max:
            v.append("conc_over")

        # OEB bounds 
        if m["oeb_int"] < self.oeb_min:
            v.append("oeb_low")
        if m["oeb_int"] > self.oeb_max:
            v.append("oeb_high")

        # FM soft cap
        if (self.fm_limit is not None) and (m["fm_weight"] > self.fm_limit):
            v.append("fm_high")

        # DVE_intake <= 1.2 * DVE_required
        if m["dve_int"] > self.dve_multiplier * m["dve_req"]:
            v.append("dve_high")

        return v

    

    def _minmax(self, x: float, x_min: float, x_max: float, clip: bool = True) -> float:
        z = (x - x_min) / (x_max - x_min)

        if clip:
            z = np.clip(z, 0.0, 1.0)

        return float(z)


    def _reward(self, m: dict) -> float:
        u = m.get("u", None)

        viols = self._violations(u, m) if u is not None else []

        if len(viols) > 0:
            
            return -1.0 - 0.2 * len(viols)

        # ---------------- Min-max normalization ----------------
        margin_norm = self._minmax(
            float(m["feed_margin"]),
            self.margin_min,
            self.margin_max,
            clip=True
        )

        methane_norm = self._minmax(
            float(m["methane"]),
            self.methane_min,
            self.methane_max,
            clip=True
        )

        # ---------------- Reward ----------------
        r = (1- self.lambda_ch4) * margin_norm - self.lambda_ch4 * methane_norm


        # r = - methane_ratio

        # r = methane_ratio / (milk_ratio)

        return float(r)


    # ----------------------------- Gym API ---------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._k = 0

        # Ensure Mode-3 
        self.b = _constant_b_mode3()

        # Initial internal state
        ops_init = dict(self.ops_base)
        ops_init["N"]  = 1
        ops_init["ts"] = 1
        _, x_mat, _, _ = DefineSignals(ops_init, np.zeros((9,)))
        self._x = np.asarray(x_mat[:, 0], dtype=float).copy()

        # Start from normalized baseline
        self._last_u = enforce_sum100(BASELINE_U, TOTAL_DM)

        # One advance + outputs
        self._x = f3(self._x, self._last_u, self.ds, self.p)
        y = np.asarray(g3(self._x, self._last_u, self.ds, self.p, self.b), float)
        obs, metrics = self._y_to_obs(y)

        
        info = {"u": self._last_u.copy(), **metrics}
        return obs, info

    def step(self, action: np.ndarray):
        self._k += 1

        # 1) Normalized action (0..1) -> raw proportions via UMAX guidance
        a = np.asarray(action, float)
        if self.log_violations and ((a < 0).any() or (a > 1).any()):
            print(f"[warn] raw normalized action out of [0,1]: {a}")
        a = np.clip(a, 0.0, 1.0)

        # map to plausible magnitudes, then enforce hard constraint Σu=100
        u_target = enforce_sum100(a * self.umax, TOTAL_DM)

        # 2) Smooth day-to-day ration change
        u = (1.0 - self.alpha) * self._last_u + self.alpha * u_target
        u = enforce_sum100(u, TOTAL_DM)
        self._last_u = u

        # 3) Simulate one day
        x_next = f3(self._x, u, self.ds, self.p)
        y = np.asarray(g3(x_next, u, self.ds, self.p, self.b), dtype=float)

        if not np.isfinite(y).all():
            # Terminate on numerical failure
            obs = np.zeros(7, dtype=np.float32)
            reward = -1e6
            info = {"u": u.copy(), "note": "non-finite outputs; episode aborted"}
            return obs, reward, True, False, info

        # 4) Normal path
        self._x = x_next   # x_next becomes the new internal cow state
        obs, metrics = self._y_to_obs(y)
        metrics["u"] = u.copy()

        reward = self._reward(metrics)
        terminated = (self._k >= self.horizon)
        truncated = False
        info = {"u": u.copy(), **metrics}
        return obs, float(reward), terminated, truncated, info





