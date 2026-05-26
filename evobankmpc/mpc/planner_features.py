import numpy as np


def _model_mu(model):
    return float((model.Df + model.Dr) / (9.81 * model.mass))


def compute_planner_features(
    avg_errors_phys,
    best_indices,
    model_bank,
    mu_preds,
    raceline_kappa,
    closest_idx,
    current_time,
    friction_start,
    vx_current,
):
    topk_errors = np.asarray(avg_errors_phys[best_indices], dtype=np.float64)
    topk_mus = np.asarray([_model_mu(model_bank[i]) for i in best_indices], dtype=np.float64)

    recent_drop_score = 0.0
    if len(mu_preds) >= 6:
        recent_drop_score = max(0.0, float(mu_preds[-6]) - float(mu_preds[-1]))

    time_since_drop = max(0.0, current_time - friction_start) if current_time > friction_start else 0.0

    return {
        "mu_mean": float(np.mean(topk_mus)),
        "mu_spread": float(np.std(topk_mus)),
        "topk_err_mean": float(np.mean(topk_errors)),
        "topk_err_std": float(np.std(topk_errors)),
        "curvature_local": float(raceline_kappa[closest_idx]),
        "recent_drop_score": float(recent_drop_score),
        "time_since_drop": float(time_since_drop),
        "vx_current": float(vx_current),
    }
