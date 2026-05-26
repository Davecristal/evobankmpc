import numpy as np
from evobankmpc.mpc.planner import ConstantSpeed


class AdaptivePlanner:
    def __init__(self, track, config):
        self.track = track
        self.cfg = config
        self.last_mode = "nominal"
        self.recovery_count = 0

    def compute_risk_score(self, features):
        mu_term = np.clip((1.0 - features["mu_mean"]) / self.cfg["mu_norm"], 0.0, 1.0)
        spread_term = np.clip(features["mu_spread"] / self.cfg["spread_norm"], 0.0, 1.0)
        err_term = np.clip(features["topk_err_mean"] / self.cfg["err_norm"], 0.0, 1.0)
        std_term = np.clip(features["topk_err_std"] / self.cfg["err_std_norm"], 0.0, 1.0)
        kappa_term = np.clip(abs(features["curvature_local"]) / self.cfg["kappa_norm"], 0.0, 1.0)
        drop_term = np.clip(features["recent_drop_score"] / self.cfg["drop_norm"], 0.0, 1.0)
        risk = (
            self.cfg["w_mu"] * mu_term
            + self.cfg["w_spread"] * spread_term
            + self.cfg["w_err"] * err_term
            + self.cfg["w_std"] * std_term
            + self.cfg["w_kappa"] * kappa_term
            + self.cfg["w_drop"] * drop_term
        )
        return float(np.clip(risk, 0.0, 1.0))

    def select_mode(self, risk_score):
        if risk_score >= self.cfg["cautious_thr"]:
            self.last_mode = "cautious"
            self.recovery_count = self.cfg["recovery_steps"]
        elif self.recovery_count > 0:
            self.last_mode = "recovery"
            self.recovery_count -= 1
        else:
            self.last_mode = "nominal"
        return self.last_mode

    def compute_plan_mu_and_scale(self, features, risk_score, mode):
        mu_mean = features["mu_mean"]
        if mode == "nominal":
            mu_plan = np.clip(mu_mean * self.cfg["nominal_mu_scale"], 0.5, 1.0)
            scale = np.clip(
                self.cfg["nominal_scale"] - self.cfg["risk_gain_nominal"] * risk_score,
                self.cfg["scale_floor_nominal"], 1.0
            )
        elif mode == "cautious":
            mu_plan = np.clip(mu_mean * self.cfg["cautious_mu_scale"], 0.5, 1.0)
            scale = np.clip(
                self.cfg["cautious_scale"] - self.cfg["risk_gain_cautious"] * risk_score,
                self.cfg["scale_floor_cautious"], self.cfg["scale_cap_cautious"]
            )
        else:
            mu_plan = np.clip(mu_mean * self.cfg["recovery_mu_scale"], 0.5, 1.0)
            scale = np.clip(
                self.cfg["recovery_scale"] - self.cfg["risk_gain_recovery"] * risk_score,
                self.cfg["scale_floor_recovery"], self.cfg["scale_cap_recovery"]
            )
        return float(mu_plan), float(scale)

    def plan(self, x0, v0, projidx, N, Ts, features):
        risk_score = self.compute_risk_score(features)
        mode = self.select_mode(risk_score)
        mu_plan, scale = self.compute_plan_mu_and_scale(features, risk_score, mode)
        xref, projidx_new, v_ref = ConstantSpeed(
            x0=x0[:2], v0=v0, track=self.track, N=N, Ts=Ts, projidx=projidx,
            curr_mu=mu_plan, scale=scale,
        )
        info = {
            "risk_score": risk_score,
            "mode": mode,
            "mu_plan": mu_plan,
            "scale": scale,
            "features": dict(features),
        }
        return xref, projidx_new, v_ref, info
