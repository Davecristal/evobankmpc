import os
import sys
import csv
import json
import time
import math
import argparse
import tempfile
import subprocess
from typing import Dict, Any, List, Optional

try:
    import optuna
except Exception:
    print("[ERROR] optuna is not installed in current environment.")
    raise


SCENARIOS = ["const_decay", "sudden_5s", "sudden_10s"]
SCENARIO_WEIGHTS = {
    "const_decay": 0.30,
    "sudden_5s": 0.30,
    "sudden_10s": 0.40,
}
FIXED_SMOOTHING_MU = 15


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def safe_float(x: Any, default: float = 1e9) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_runner_path(arg_runner: Optional[str]) -> str:
    candidates = []
    if arg_runner:
        candidates.append(arg_runner)

    here = os.path.abspath(os.path.dirname(__file__))
    cwd = os.path.abspath(os.getcwd())

    candidates.extend([
        os.path.join(cwd, "evobankmpc", "mpc", "run_nmpc_orca_evobankmpc.py"),
        os.path.join(here, "run_nmpc_orca_evobankmpc.py"),
    ])

    for cand in candidates:
        if cand and os.path.exists(cand):
            return os.path.abspath(cand)

    raise FileNotFoundError(
        "Cannot find runner file. Please pass --runner explicitly, e.g.\n"
        "  --runner evobankmpc/mpc/run_nmpc_orca_evobankmpc.py"
    )


def default_base_params() -> Dict[str, Any]:
    # Centered around your current ETHZMobil-friendly baseline.
    return {
        "adaptive_sampling_std": 0.0552024586409393,
        "update_interval": 58,
        "mu_alpha": 0.2136510350649864,
        "smoothing_mu_over_mod": 5,
        "v_factor": 0.9620558928042692,
        "cost_r_acc_scale": 0.3481098753375238,
        "cost_r_steer_scale": 1.73838830057911,
        "ey_weight_base": 0.3425566679883475,
        "ey_weight_curvature_gain": 8.152533830931153,
    }


def sample_params(trial: "optuna.trial.Trial", args: argparse.Namespace) -> Dict[str, Any]:
    base = default_base_params()
    base["adaptive_sampling_std"] = args.base_adaptive_sampling_std
    base["update_interval"] = args.base_update_interval
    base["mu_alpha"] = args.base_mu_alpha
    base["smoothing_mu_over_mod"] = args.base_smoothing_mu_over_mod
    base["v_factor"] = args.base_v_factor
    base["cost_r_acc_scale"] = args.base_cost_r_acc_scale
    base["cost_r_steer_scale"] = args.base_cost_r_steer_scale
    base["ey_weight_base"] = args.base_ey_weight_base
    base["ey_weight_curvature_gain"] = args.base_ey_weight_curvature_gain

    params = {}

    # Keep search narrow and practical.
    params["adaptive_sampling_std"] = trial.suggest_float(
        "adaptive_sampling_std",
        max(0.040, base["adaptive_sampling_std"] - 0.012),
        min(0.080, base["adaptive_sampling_std"] + 0.012),
    )

    params["update_interval"] = trial.suggest_int(
        "update_interval",
        max(35, safe_int(base["update_interval"]) - 12),
        min(80, safe_int(base["update_interval"]) + 12),
    )

    # As discussed, smaller mu_alpha often helps ETHZMobil const_decay stability.
    params["mu_alpha"] = trial.suggest_float(
        "mu_alpha",
        max(0.12, base["mu_alpha"] - 0.08),
        min(0.28, base["mu_alpha"] + 0.03),
    )

    # Slightly larger over-mod smoothing often helps post-change oscillation.
    params["smoothing_mu_over_mod"] = trial.suggest_int(
        "smoothing_mu_over_mod",
        max(4, safe_int(base["smoothing_mu_over_mod"])),
        min(10, safe_int(base["smoothing_mu_over_mod"]) + 4),
    )

    params["v_factor"] = trial.suggest_float(
        "v_factor",
        max(0.90, base["v_factor"] - 0.03),
        min(1.00, base["v_factor"] + 0.02),
    )

    params["cost_r_acc_scale"] = trial.suggest_float(
        "cost_r_acc_scale",
        max(0.18, base["cost_r_acc_scale"] * 0.70),
        min(0.60, base["cost_r_acc_scale"] * 1.35),
    )

    params["cost_r_steer_scale"] = trial.suggest_float(
        "cost_r_steer_scale",
        max(0.80, base["cost_r_steer_scale"] * 0.70),
        min(2.60, base["cost_r_steer_scale"] * 1.25),
    )

    if args.optimize_ey:
        params["ey_weight_base"] = trial.suggest_float(
            "ey_weight_base",
            max(0.20, base["ey_weight_base"] * 0.75),
            min(0.55, base["ey_weight_base"] * 1.35),
        )
        params["ey_weight_curvature_gain"] = trial.suggest_float(
            "ey_weight_curvature_gain",
            max(4.0, base["ey_weight_curvature_gain"] * 0.70),
            min(11.0, base["ey_weight_curvature_gain"] * 1.25),
        )
    else:
        params["ey_weight_base"] = base["ey_weight_base"]
        params["ey_weight_curvature_gain"] = base["ey_weight_curvature_gain"]

    params["smoothing_mu"] = FIXED_SMOOTHING_MU
    return params


def build_runner_cmd(
    runner_path: str,
    scenario: str,
    params: Dict[str, Any],
    seed: int,
    output_json: str,
    n_models: int,
) -> List[str]:
    cmd = [
        sys.executable,
        runner_path,
        "--track", "ETHZMobil",
        "--friction_style", scenario,
        "--seed", str(seed),
        "--no_video",
        "--output_file", output_json,
        "--n_models", str(n_models),
        "--adaptive_sampling_std", str(params["adaptive_sampling_std"]),
        "--update_interval", str(params["update_interval"]),
        "--smoothing_mu", str(params["smoothing_mu"]),
        "--mu_alpha", str(params["mu_alpha"]),
        "--smoothing_mu_over_mod", str(params["smoothing_mu_over_mod"]),
        "--v_factor", str(params["v_factor"]),
        "--cost_r_acc_scale", str(params["cost_r_acc_scale"]),
        "--cost_r_steer_scale", str(params["cost_r_steer_scale"]),
        "--ey_weight_base", str(params["ey_weight_base"]),
        "--ey_weight_curvature_gain", str(params["ey_weight_curvature_gain"]),
    ]
    return cmd


def run_one_case(
    runner_path: str,
    scenario: str,
    params: Dict[str, Any],
    seed: int,
    timeout_sec: int,
    n_models: int,
) -> Dict[str, Any]:
    fd, tmp_json = tempfile.mkstemp(prefix="ethzmobil_joint_", suffix=".json")
    os.close(fd)

    cmd = build_runner_cmd(
        runner_path=runner_path,
        scenario=scenario,
        params=params,
        seed=seed,
        output_json=tmp_json,
        n_models=n_models,
    )

    t0 = time.time()
    ok = False
    return_code = -999
    stderr_text = ""
    stdout_text = ""
    metrics = {}

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        elapsed = time.time() - t0
        return_code = proc.returncode
        stdout_text = proc.stdout[-4000:]
        stderr_text = proc.stderr[-4000:]

        if proc.returncode == 0 and os.path.exists(tmp_json):
            metrics = read_json(tmp_json)
            ok = True
        else:
            ok = False
    except subprocess.TimeoutExpired as e:
        elapsed = time.time() - t0
        stderr_text = "TIMEOUT after {} s".format(timeout_sec)
        stdout_text = getattr(e, "stdout", "") or ""
        ok = False
    except Exception as e:
        elapsed = time.time() - t0
        stderr_text = repr(e)
        ok = False
    finally:
        try:
            if os.path.exists(tmp_json):
                os.remove(tmp_json)
        except Exception:
            pass

    return {
        "ok": ok,
        "elapsed_sec": elapsed,
        "return_code": return_code,
        "stdout_tail": stdout_text,
        "stderr_tail": stderr_text,
        "metrics": metrics,
        "cmd": cmd,
    }


def compute_case_score(scenario: str, metrics: Dict[str, Any], ok: bool) -> float:
    if (not ok) or (not metrics):
        return 1e6

    total_lap_time = safe_float(metrics.get("total_lap_time"), 999.0)
    violation_time = safe_float(metrics.get("violation_time"), 999.0)
    mean_deviation = safe_float(metrics.get("mean_deviation"), 999.0)
    mu_rmse = safe_float(metrics.get("mu_rmse"), 1.0)
    mu_rmse_post = safe_float(metrics.get("mu_rmse_post"), mu_rmse)
    df_rmse_post = safe_float(metrics.get("df_rmse_post"), safe_float(metrics.get("df_rmse"), 1.0))
    dr_rmse_post = safe_float(metrics.get("dr_rmse_post"), safe_float(metrics.get("dr_rmse"), 1.0))

    # Main objective: prioritize safe, stable, transferable performance.
    score = total_lap_time
    score += 30.0 * violation_time
    score += 2.0 * mean_deviation
    score += 4.0 * mu_rmse_post
    score += 2.0 * df_rmse_post
    score += 2.0 * dr_rmse_post

    # Soft penalties for clearly bad regimes.
    if violation_time > 0.50:
        score += 30.0 * (violation_time - 0.50)
    if total_lap_time > 19.8:
        score += 10.0 * (total_lap_time - 19.8)

    # sudden_10s is the hardest case, give it a little extra pressure.
    if scenario == "sudden_10s":
        score += 3.0 * mu_rmse_post
        if violation_time > 0.35:
            score += 20.0 * (violation_time - 0.35)

    return score


def flatten_trial_row(
    trial_number: int,
    objective_value: float,
    params: Dict[str, Any],
    case_results: Dict[str, Dict[str, Any]],
    n_models_record: int,
) -> Dict[str, Any]:
    row = {
        "trial": trial_number,
        "objective": objective_value,
        "n_models_record": n_models_record,
        "adaptive_sampling_std": params["adaptive_sampling_std"],
        "update_interval": params["update_interval"],
        "smoothing_mu": params["smoothing_mu"],
        "mu_alpha": params["mu_alpha"],
        "smoothing_mu_over_mod": params["smoothing_mu_over_mod"],
        "v_factor": params["v_factor"],
        "cost_r_acc_scale": params["cost_r_acc_scale"],
        "cost_r_steer_scale": params["cost_r_steer_scale"],
        "ey_weight_base": params["ey_weight_base"],
        "ey_weight_curvature_gain": params["ey_weight_curvature_gain"],
    }

    for sc in SCENARIOS:
        result = case_results[sc]
        metrics = result.get("metrics", {})
        row[sc + "_ok"] = int(bool(result.get("ok", False)))
        row[sc + "_elapsed_sec"] = safe_float(result.get("elapsed_sec"), -1.0)
        row[sc + "_score"] = safe_float(result.get("score"), 1e9)
        row[sc + "_total_lap_time"] = safe_float(metrics.get("total_lap_time"), 1e9)
        row[sc + "_violation_time"] = safe_float(metrics.get("violation_time"), 1e9)
        row[sc + "_mean_deviation"] = safe_float(metrics.get("mean_deviation"), 1e9)
        row[sc + "_mean_cost"] = safe_float(metrics.get("mean_cost"), 1e9)
        row[sc + "_mu_rmse"] = safe_float(metrics.get("mu_rmse"), 1e9)
        row[sc + "_mu_rmse_post"] = safe_float(metrics.get("mu_rmse_post"), 1e9)
        row[sc + "_df_rmse"] = safe_float(metrics.get("df_rmse"), 1e9)
        row[sc + "_dr_rmse"] = safe_float(metrics.get("dr_rmse"), 1e9)
        row[sc + "_df_rmse_post"] = safe_float(metrics.get("df_rmse_post"), 1e9)
        row[sc + "_dr_rmse_post"] = safe_float(metrics.get("dr_rmse_post"), 1e9)

    return row


def append_csv_row(csv_path: str, row: Dict[str, Any]) -> None:
    file_exists = os.path.exists(csv_path)
    fieldnames = list(row.keys())
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def make_objective(
    runner_path: str,
    args: argparse.Namespace,
    csv_path: str,
):
    def objective(trial: "optuna.trial.Trial") -> float:
        params = sample_params(trial, args)
        case_results = {}
        raw_scores = []

        for sc in SCENARIOS:
            result = run_one_case(
                runner_path=runner_path,
                scenario=sc,
                params=params,
                seed=args.seed,
                timeout_sec=args.timeout_sec,
                n_models=args.n_models,
            )
            score = compute_case_score(sc, result.get("metrics", {}), result.get("ok", False))
            result["score"] = score
            case_results[sc] = result
            raw_scores.append(score)

        weighted_avg = 0.0
        for sc in SCENARIOS:
            weighted_avg += SCENARIO_WEIGHTS[sc] * case_results[sc]["score"]
        worst_case = max(raw_scores)
        objective_value = weighted_avg + args.worst_case_weight * worst_case

        row = flatten_trial_row(
            trial_number=trial.number,
            objective_value=objective_value,
            params=params,
            case_results=case_results,
            n_models_record=args.n_models,
        )
        append_csv_row(csv_path, row)

        trial.set_user_attr("params", params)
        for sc in SCENARIOS:
            trial.set_user_attr(sc + "_score", case_results[sc]["score"])
            trial.set_user_attr(sc + "_ok", bool(case_results[sc]["ok"]))

        print("-" * 88)
        print("Trial {:03d} | objective={:.6f}".format(trial.number, objective_value))
        print("params:", json.dumps(params, ensure_ascii=False))
        for sc in SCENARIOS:
            metrics = case_results[sc].get("metrics", {})
            print(
                "  {:>10s} | score={:.4f} | lap={:.3f} | viol={:.3f} | dev={:.5f} | mu_post={:.5f} | time={:.1f}s | ok={}".format(
                    sc,
                    case_results[sc]["score"],
                    safe_float(metrics.get("total_lap_time"), 999.0),
                    safe_float(metrics.get("violation_time"), 999.0),
                    safe_float(metrics.get("mean_deviation"), 999.0),
                    safe_float(metrics.get("mu_rmse_post"), safe_float(metrics.get("mu_rmse"), 999.0)),
                    safe_float(case_results[sc].get("elapsed_sec"), -1.0),
                    case_results[sc].get("ok", False),
                )
            )
        return objective_value

    return objective


def parse_args() -> argparse.Namespace:
    base = default_base_params()
    parser = argparse.ArgumentParser(description="ETHZMobil joint Bayesian optimization (fixed smoothing_mu=15)")
    parser.add_argument("--runner", type=str, default=None, help="Path to runner .py file")
    parser.add_argument("--study_name", type=str, default="ethzmobil_joint_fixed15")
    parser.add_argument("--n_trials", type=int, default=60)
    parser.add_argument("--seed", type=int, default=3846)
    parser.add_argument("--n_models", type=int, default=10000, help="Recorded into csv/json only")
    parser.add_argument("--timeout_sec", type=int, default=3600)
    parser.add_argument("--optimize_ey", action="store_true", help="Also optimize ey weights")
    parser.add_argument("--worst_case_weight", type=float, default=0.35)
    parser.add_argument("--output_dir", type=str, default="opt_results/ethzmobil_joint_fixed15")

    parser.add_argument("--base_adaptive_sampling_std", type=float, default=base["adaptive_sampling_std"])
    parser.add_argument("--base_update_interval", type=int, default=base["update_interval"])
    parser.add_argument("--base_mu_alpha", type=float, default=base["mu_alpha"])
    parser.add_argument("--base_smoothing_mu_over_mod", type=int, default=base["smoothing_mu_over_mod"])
    parser.add_argument("--base_v_factor", type=float, default=base["v_factor"])
    parser.add_argument("--base_cost_r_acc_scale", type=float, default=base["cost_r_acc_scale"])
    parser.add_argument("--base_cost_r_steer_scale", type=float, default=base["cost_r_steer_scale"])
    parser.add_argument("--base_ey_weight_base", type=float, default=base["ey_weight_base"])
    parser.add_argument("--base_ey_weight_curvature_gain", type=float, default=base["ey_weight_curvature_gain"])

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner_path = get_runner_path(args.runner)

    ensure_dir(args.output_dir)
    csv_path = os.path.join(args.output_dir, "all_trials.csv")
    db_path = os.path.join(args.output_dir, args.study_name + ".db")
    best_json_path = os.path.join(args.output_dir, "best_params.json")
    summary_json_path = os.path.join(args.output_dir, "study_summary.json")

    print("=" * 80)
    print("ETHZMobil joint optimization (fixed smoothing_mu = {})".format(FIXED_SMOOTHING_MU))
    print("Runner: {}".format(runner_path))
    print("Scenarios: {}".format(SCENARIOS))
    print("Scenario weights: {}".format(SCENARIO_WEIGHTS))
    print("Seed: {}, N_MODELS(record): {}, Trials: {}".format(args.seed, args.n_models, args.n_trials))
    print("Output dir: {}".format(os.path.abspath(args.output_dir)))
    print("Single CSV: {}".format(os.path.abspath(csv_path)))
    print("=" * 80)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        storage="sqlite:///{}".format(os.path.abspath(db_path)),
        load_if_exists=True,
    )

    objective = make_objective(
        runner_path=runner_path,
        args=args,
        csv_path=csv_path,
    )

    study.optimize(objective, n_trials=args.n_trials)

    best_payload = {
        "study_name": args.study_name,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "fixed_smoothing_mu": FIXED_SMOOTHING_MU,
        "runner": runner_path,
        "n_models_record": args.n_models,
        "seed": args.seed,
    }
    save_json(best_json_path, best_payload)

    summary_payload = {
        "study_name": args.study_name,
        "best_value": study.best_value,
        "best_trial_number": study.best_trial.number,
        "n_trials_total": len(study.trials),
        "best_params": study.best_params,
        "fixed_smoothing_mu": FIXED_SMOOTHING_MU,
        "output_csv": os.path.abspath(csv_path),
        "db_path": os.path.abspath(db_path),
        "runner": runner_path,
        "n_models_record": args.n_models,
        "seed": args.seed,
    }
    save_json(summary_json_path, summary_payload)

    print("\nOptimization finished.")
    print("Best value: {:.6f}".format(study.best_value))
    print("Best trial: {}".format(study.best_trial.number))
    print("Best params:")
    print(json.dumps(study.best_params, indent=2, ensure_ascii=False))
    print("Saved:")
    print("  {}".format(os.path.abspath(csv_path)))
    print("  {}".format(os.path.abspath(best_json_path)))
    print("  {}".format(os.path.abspath(summary_json_path)))
    print("  {}".format(os.path.abspath(db_path)))


if __name__ == "__main__":
    main()
