import argparse
import csv
import json
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import optuna
from optuna.trial import Trial

SCENARIOS: List[str] = ["const_decay", "sudden_5s", "sudden_10s"]
EXPECTED_LAPS = {
    "const_decay": 4,
    "sudden_5s": 3,
    "sudden_10s": 3,
}
SCENARIO_WEIGHTS = {
    "const_decay": 0.3,
    "sudden_5s": 0.3,
    "sudden_10s": 0.4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETHZMobil joint Bayesian optimization (fixed output files)")
    parser.add_argument("--runner", type=str, default=None,
                        help="Path to run_nmpc_orca_evobankmpc_evobankmpc.py. Defaults to same directory.")
    parser.add_argument("--n_trials", type=int, default=60, help="Number of Optuna trials for this launch")
    parser.add_argument("--seed", type=int, default=3846, help="Simulation seed used in every scenario")
    parser.add_argument("--n_models", type=int, default=5000, help="Model-bank size override passed to runner")
    parser.add_argument("--timeout_sec", type=int, default=5400, help="Timeout per scenario run")
    parser.add_argument("--study_name", type=str, default="ethzmobil_joint")
    parser.add_argument("--optimize_ey", action="store_true",
                        help="Also optimize ey_weight_base / ey_weight_curvature_gain")
    parser.add_argument("--results_dir", type=str, default="optuna_joint_results",
                        help="Directory for fixed CSV/JSON outputs")
    parser.add_argument("--storage", type=str, default="",
                        help="Optional Optuna storage URL. Leave empty to auto-use sqlite in results_dir.")
    parser.add_argument("--fresh_study", action="store_true",
                        help="Start a fresh in-memory study for this launch only (still appends CSVs).")
    return parser.parse_args()


def ensure_results_dir(results_dir: str) -> str:
    path = os.path.abspath(results_dir)
    os.makedirs(path, exist_ok=True)
    return path


def make_file_names(study_name: str, results_dir: str) -> Tuple[str, str, str, str]:
    return (
        os.path.join(results_dir, f"optuna_results_{study_name}.csv"),
        os.path.join(results_dir, f"detailed_results_{study_name}.csv"),
        os.path.join(results_dir, f"run_summary_{study_name}.csv"),
        os.path.join(results_dir, f"best_params_{study_name}.json"),
    )


def get_runner_path(arg_runner: Optional[str]) -> str:
    if arg_runner:
        return arg_runner
    return str(Path(__file__).resolve().parent / "run_nmpc_orca_evobankmpc.py")


def build_search_space(trial: Trial, optimize_ey: bool) -> Dict[str, float]:
    params: Dict[str, float] = {
        "adaptive_sampling_std": trial.suggest_float("adaptive_sampling_std", 0.03, 0.12, log=True),
        "update_interval": trial.suggest_int("update_interval", 45, 100),
        "smoothing_mu": trial.suggest_int("smoothing_mu", 5, 20, step=5),
        "mu_alpha": trial.suggest_float("mu_alpha", 0.10, 0.30),
        "smoothing_mu_over_mod": trial.suggest_int("smoothing_mu_over_mod", 5, 10, step=5),
        "v_factor": trial.suggest_float("v_factor", 0.90, 0.98),
        "cost_r_acc_scale": trial.suggest_float("cost_r_acc_scale", 0.20, 0.90, log=True),
        "cost_r_steer_scale": trial.suggest_float("cost_r_steer_scale", 1.0, 2.2, log=True),
    }
    if optimize_ey:
        params["ey_weight_base"] = trial.suggest_float("ey_weight_base", 0.20, 1.20, log=True)
        params["ey_weight_curvature_gain"] = trial.suggest_float("ey_weight_curvature_gain", 4.0, 15.0, log=True)
    return params


def run_one_scenario(
    runner: str,
    scenario: str,
    seed: int,
    n_models: int,
    params: Dict[str, float],
    timeout_sec: int,
) -> Dict:
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
        output_file = f.name

    cmd = [
        "python", runner,
        "--track", "ETHZMobil",
        "--friction_style", scenario,
        "--n_models", str(n_models),
        "--seed", str(seed),
        "--adaptive_sampling_std", str(params["adaptive_sampling_std"]),
        "--update_interval", str(params["update_interval"]),
        "--smoothing_mu", str(params["smoothing_mu"]),
        "--mu_alpha", str(params["mu_alpha"]),
        "--smoothing_mu_over_mod", str(params["smoothing_mu_over_mod"]),
        "--v_factor", str(params["v_factor"]),
        "--cost_r_acc_scale", str(params["cost_r_acc_scale"]),
        "--cost_r_steer_scale", str(params["cost_r_steer_scale"]),
        "--no_video",
        "--no_plots",
        "--output_file", output_file,
    ]
    if "ey_weight_base" in params:
        cmd.extend(["--ey_weight_base", str(params["ey_weight_base"])])
    if "ey_weight_curvature_gain" in params:
        cmd.extend(["--ey_weight_curvature_gain", str(params["ey_weight_curvature_gain"])])

    try:
        subprocess.run(cmd, check=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        return {"_status": "timeout"}
    except subprocess.CalledProcessError:
        return {"_status": "failed"}

    try:
        with open(output_file, "r") as f:
            data = json.load(f)
    except Exception:
        data = {"_status": "failed_to_load"}
    finally:
        try:
            os.unlink(output_file)
        except OSError:
            pass

    data["_status"] = data.get("_status", "ok")
    return data


def scenario_objective(scenario: str, results: Dict) -> float:
    if results.get("_status") != "ok":
        return 1e6

    lap_times = results.get("lap_times", [])
    total_lap_time = float(results.get("total_lap_time", 999.0))
    violation_time = float(results.get("violation_time", 999.0))
    mean_deviation = float(results.get("mean_deviation", 999.0))
    mean_cost = float(results.get("mean_cost", 999.0))
    mu_rmse_post = float(results.get("mu_rmse_post", results.get("mu_rmse", 999.0)))

    missing_laps = max(0, EXPECTED_LAPS[scenario] - len(lap_times))
    incomplete_penalty = 80.0 * missing_laps

    return float(
        total_lap_time
        + 25.0 * violation_time
        + 12.0 * mean_deviation
        + 0.10 * mean_cost
        + 20.0 * mu_rmse_post
        + incomplete_penalty
    )


def save_detailed_results(detailed_csv: str, launch_id: str, trial_number: int, params: Dict[str, float],
                          per_scenario: Dict[str, Dict], objective_value: float, seed: int, n_models: int) -> None:
    file_exists = os.path.isfile(detailed_csv)
    with open(detailed_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            headers = [
                "launch_id", "trial_number", "objective_value", "seed", "n_models",
                "adaptive_sampling_std", "update_interval", "smoothing_mu", "mu_alpha",
                "smoothing_mu_over_mod", "v_factor", "cost_r_acc_scale", "cost_r_steer_scale",
                "ey_weight_base", "ey_weight_curvature_gain",
                "const_total_lap_time", "const_violation_time", "const_mean_deviation", "const_mu_rmse_post", "const_runtime_total_sec",
                "s5_total_lap_time", "s5_violation_time", "s5_mean_deviation", "s5_mu_rmse_post", "s5_runtime_total_sec",
                "s10_total_lap_time", "s10_violation_time", "s10_mean_deviation", "s10_mu_rmse_post", "s10_runtime_total_sec",
            ]
            writer.writerow(headers)

        c = per_scenario.get("const_decay", {})
        s5 = per_scenario.get("sudden_5s", {})
        s10 = per_scenario.get("sudden_10s", {})
        writer.writerow([
            launch_id, trial_number, objective_value, seed, n_models,
            params.get("adaptive_sampling_std"), params.get("update_interval"), params.get("smoothing_mu"), params.get("mu_alpha"),
            params.get("smoothing_mu_over_mod"), params.get("v_factor"), params.get("cost_r_acc_scale"), params.get("cost_r_steer_scale"),
            params.get("ey_weight_base", ""), params.get("ey_weight_curvature_gain", ""),
            c.get("total_lap_time", ""), c.get("violation_time", ""), c.get("mean_deviation", ""), c.get("mu_rmse_post", c.get("mu_rmse", "")), c.get("runtime_total_sec", ""),
            s5.get("total_lap_time", ""), s5.get("violation_time", ""), s5.get("mean_deviation", ""), s5.get("mu_rmse_post", s5.get("mu_rmse", "")), s5.get("runtime_total_sec", ""),
            s10.get("total_lap_time", ""), s10.get("violation_time", ""), s10.get("mean_deviation", ""), s10.get("mu_rmse_post", s10.get("mu_rmse", "")), s10.get("runtime_total_sec", ""),
        ])


def append_run_summary(run_summary_csv: str, launch_id: str, args: argparse.Namespace, study: optuna.Study, best_json: str) -> None:
    file_exists = os.path.isfile(run_summary_csv)
    best = study.best_trial
    with open(run_summary_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "launch_id", "timestamp", "study_name", "n_trials_this_launch", "seed", "n_models",
                "runner", "best_value", "best_trial_number", "best_params_json"
            ])
        writer.writerow([
            launch_id, datetime.now().isoformat(timespec="seconds"), args.study_name, args.n_trials, args.seed, args.n_models,
            get_runner_path(args.runner), best.value, best.number, best_json
        ])


def main() -> None:
    args = parse_args()
    results_dir = ensure_results_dir(args.results_dir)
    runner = get_runner_path(args.runner)
    if not os.path.exists(runner):
        raise FileNotFoundError(f"Runner not found: {runner}")

    results_csv, detailed_csv, run_summary_csv, best_json = make_file_names(args.study_name, results_dir)
    launch_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def objective(trial: Trial) -> float:
        params = build_search_space(trial, args.optimize_ey)
        per_scenario: Dict[str, Dict] = {}
        total_score = 0.0
        for scenario in SCENARIOS:
            results = run_one_scenario(
                runner=runner,
                scenario=scenario,
                seed=args.seed,
                n_models=args.n_models,
                params=params,
                timeout_sec=args.timeout_sec,
            )
            per_scenario[scenario] = results
            total_score += SCENARIO_WEIGHTS[scenario] * scenario_objective(scenario, results)

        save_detailed_results(detailed_csv, launch_id, trial.number, params, per_scenario, total_score, args.seed, args.n_models)

        print(f"Trial {trial.number}: objective={total_score:.6f}")
        for scenario in SCENARIOS:
            r = per_scenario[scenario]
            print(
                f"  {scenario}: status={r.get('_status')} total={r.get('total_lap_time')} "
                f"violation={r.get('violation_time')} mu_post={r.get('mu_rmse_post', r.get('mu_rmse'))}"
            )
        return total_score

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        study.trials_dataframe().to_csv(results_csv, index=False)
        best = study.best_trial
        payload = {
            "launch_id": launch_id,
            "best_value": best.value,
            "best_params": best.params,
            "seed": args.seed,
            "n_models": args.n_models,
            "runner": runner,
            "weights": SCENARIO_WEIGHTS,
            "scenarios": SCENARIOS,
            "results_csv": results_csv,
            "detailed_csv": detailed_csv,
        }
        with open(best_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"✅ Updated best params saved to {best_json}")

    if args.fresh_study:
        study = optuna.create_study(
            direction="minimize",
            study_name=args.study_name,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        )
    else:
        storage = args.storage or f"sqlite:///{os.path.join(results_dir, args.study_name)}.db"
        study = optuna.create_study(
            direction="minimize",
            study_name=args.study_name,
            storage=storage,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        )

    print("=" * 72)
    print("ETHZMobil joint optimization")
    print(f"Runner: {runner}")
    print(f"Scenarios: {SCENARIOS}")
    print(f"Scenario weights: {SCENARIO_WEIGHTS}")
    print(f"Seed: {args.seed}, N_MODELS: {args.n_models}, Trials this launch: {args.n_trials}")
    print(f"Results dir: {results_dir}")
    print(f"Detailed CSV (append): {detailed_csv}")
    print(f"Study CSV (overwrite from study state): {results_csv}")
    print("=" * 72)

    study.optimize(objective, n_trials=args.n_trials, callbacks=[callback], show_progress_bar=True)
    append_run_summary(run_summary_csv, launch_id, args, study, best_json)

    print("\nOptimization finished.")
    print(f"Best value: {study.best_trial.value:.6f}")
    print(f"Best params json: {best_json}")
    print(f"Detailed CSV: {detailed_csv}")
    print(f"Run summary CSV: {run_summary_csv}")


if __name__ == "__main__":
    main()
