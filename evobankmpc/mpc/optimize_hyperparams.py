import subprocess
import json
import tempfile
import os
import optuna
from optuna.trial import Trial
from datetime import datetime
import csv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HYBRID_SCRIPT = os.path.join(SCRIPT_DIR, 'run_nmpc_orca_evobankmpc.py')

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_FILE = f'optuna_results_ETHZMobil_{TIMESTAMP}.csv'
DETAILED_RESULTS_FILE = f'detailed_results_ETHZMobil_{TIMESTAMP}.csv'

# time boundarys for each lap
# LAP_TIME_BOUNDS = [7.85, 7.85, 9.20, 9.20] # ETHZ
LAP_TIME_BOUNDS = [6.00, 5.80, 6.75] # ETHZMobil
LAP_VIOLATION_PENALTY = 50.0

def save_detailed_results(trial_number, params, results, objective_value):
    file_exists = os.path.isfile(DETAILED_RESULTS_FILE)
    with open(DETAILED_RESULTS_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            headers = [
                'trial_number', 'objective_value',
                'adaptive_sampling_std', 'update_interval',
                'smoothing_mu', 'mu_alpha', 'smoothing_mu_over_mod',
                'v_factor',
                'cost_r_acc_scale', 'cost_r_steer_scale',
                'ey_weight_base', 'ey_weight_curvature_gain',
                'seed',
                'lap_times', 'total_lap_time', 'violation_time',
                'mean_deviation', 'mean_cost', 'mu_rmse_post'
            ]
            writer.writerow(headers)
        lap_times_str = ','.join(f"{t:.3f}" for t in results.get('lap_times', []))
        total_lap_time = results.get('total_lap_time', 0.0)
        violation_time = results.get('violation_time', 0.0)
        mean_deviation = results.get('mean_deviation', 0.0)
        mean_cost = results.get('mean_cost', 0.0)
        mu_rmse_post = results.get('mu_rmse_post', 0.0)

        writer.writerow([
            trial_number, objective_value,
            params['adaptive_sampling_std'], params['update_interval'],
            params['smoothing_mu'], params['mu_alpha'], params['smoothing_mu_over_mod'],
            params['v_factor'],
            params['cost_r_acc_scale'], params['cost_r_steer_scale'],
            params['ey_weight_base'], params['ey_weight_curvature_gain'],
            params['seed'],
            lap_times_str, total_lap_time, violation_time,
            mean_deviation, mean_cost, mu_rmse_post
        ])

def objective(trial: Trial):
    adaptive_sampling_std = trial.suggest_float('adaptive_sampling_std', 0.05, 0.35, log=True)
    update_interval = trial.suggest_int('update_interval', 50, 180)
    
    smoothing_mu = trial.suggest_int('smoothing_mu', 10, 40, step=5)          
    mu_alpha = trial.suggest_float('mu_alpha', 0.05, 0.25, log=True)         
    smoothing_mu_over_mod = trial.suggest_int('smoothing_mu_over_mod', 5, 20, step=5)  
    
    v_factor = trial.suggest_float('v_factor', 0.88, 1.0)
    cost_r_acc_scale = trial.suggest_float('cost_r_acc_scale', 0.3, 1.5, log=True)
    cost_r_steer_scale = trial.suggest_float('cost_r_steer_scale', 0.6, 1.8, log=True)
    ey_weight_base = trial.suggest_float('ey_weight_base', 0.3, 5.0, log=True)
    ey_weight_curvature_gain = trial.suggest_float('ey_weight_curvature_gain', 3.0, 20.0, log=True)

    seed = trial.suggest_int('seed', 1, 5000)

    params = {
        'adaptive_sampling_std': adaptive_sampling_std,
        'update_interval': update_interval,
        'smoothing_mu': smoothing_mu,
        'mu_alpha': mu_alpha,
        'smoothing_mu_over_mod': smoothing_mu_over_mod,
        'v_factor': v_factor,
        'cost_r_acc_scale': cost_r_acc_scale,
        'cost_r_steer_scale': cost_r_steer_scale,
        'ey_weight_base': ey_weight_base,
        'ey_weight_curvature_gain': ey_weight_curvature_gain,
        'seed': seed,
    }

    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
        output_file = f.name

    cmd = [
        'python', HYBRID_SCRIPT,
        '--seed', str(seed),
        '--adaptive_sampling_std', str(adaptive_sampling_std),
        '--update_interval', str(update_interval),
        '--smoothing_mu', str(smoothing_mu),
        '--mu_alpha', str(mu_alpha),
        '--smoothing_mu_over_mod', str(smoothing_mu_over_mod),
        '--v_factor', str(v_factor),
        '--cost_r_acc_scale', str(cost_r_acc_scale),
        '--cost_r_steer_scale', str(cost_r_steer_scale),
        '--ey_weight_base', str(ey_weight_base),
        '--ey_weight_curvature_gain', str(ey_weight_curvature_gain),
        '--no_video',
        '--no_plots',
        '--output_file', output_file
    ]

    try:
        subprocess.run(cmd, check=True, timeout=5400)
    except subprocess.TimeoutExpired:
        print(f"Trial {trial.number} timeout after 5400s")
        return float('inf')
    except subprocess.CalledProcessError as e:
        print(f"Trial {trial.number} failed with error: {e}")
        return float('inf')

    try:
        with open(output_file, 'r') as f:
            results = json.load(f)
    except Exception as e:
        print(f"Failed to load results: {e}")
        return float('inf')
    finally:
        os.unlink(output_file)

    lap_times = results.get('lap_times', [])
    total_lap_time = results.get('total_lap_time', 40.0)
    mean_deviation = results.get('mean_deviation', 0.0)
    mean_cost = results.get('mean_cost', 0.0)
    mu_rmse = results.get('mu_rmse', 0.0)
    mu_rmse_post = results.get('mu_rmse_post', 0.0)
    violation_time = results.get('violation_time', 0.0)

    if len(lap_times) < 3:
        print(f"⚠️ Trial {trial.number}: Only {len(lap_times)} laps completed, discarded")
        return 1000.0

    lap_penalty = 0.0
    for i, bound in enumerate(LAP_TIME_BOUNDS):
        if lap_times[i] > bound:
            excess = lap_times[i] - bound
            lap_penalty += excess * LAP_VIOLATION_PENALTY

    # total_lap_penalty = max(0, total_lap_time - 33.8) * 20.0 # ETHZ
    total_lap_penalty = max(0, total_lap_time - 18.38) * 30.0 # ETHZMobil

    # lap_weight = [1.05, 1.05, 1.0, 1.0] # ETHZ
    # weighted_lap_time = sum(w * t for w, t in zip(lap_weight, lap_times[:3])) * 1.05 # ETHZ
    lap_weight = [1.05, 1.05, 1.0] # ETHZMobil
    weighted_lap_time = sum(w * t for w, t in zip(lap_weight, lap_times[:3])) * 1.05 # ETHZMobil



    deviation_penalty = mean_deviation * 15.0
    cost_penalty = mean_cost * 0.5
    # mu_penalty = mu_rmse_post * 40.0
    mu_penalty = (0.3 * mu_rmse + 0.7 * mu_rmse_post) * 40.0
    violation_penalty = violation_time * 50.0 

    objective_value = (weighted_lap_time + lap_penalty + total_lap_penalty +
                       deviation_penalty + cost_penalty + mu_penalty + violation_penalty)

    print(f"Trial {trial.number}: lap_times={lap_times}, total={total_lap_time:.2f}, "
          f"violation={violation_time:.2f}, dev={mean_deviation:.4f}, mu_post={mu_rmse_post:.4f}, obj={objective_value:.4f}")

    save_detailed_results(trial.number, params, results, objective_value)
    return objective_value

def callback(study, trial):
    df = study.trials_dataframe()
    df.to_csv(RESULTS_FILE, index=False)
    print(f"✅ Results saved to {RESULTS_FILE}")
    best_trial = study.best_trial
    print(f"🏆 Current best trial: {best_trial.number}, value: {best_trial.value:.6f}")
    for key, value in best_trial.params.items():
        print(f"      {key}: {value}")

def main():
    if not os.path.exists(HYBRID_SCRIPT):
        print(f"can not find script {HYBRID_SCRIPT}")
        return

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    )

    n_trials = 15  
    print(f"Starting optimization with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=True)

    print("\n" + "="*60)
    print("Optimization completed. Final best trial:")
    best_trial = study.best_trial
    print(f"  Value: {best_trial.value:.6f}")
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    study.trials_dataframe().to_csv(RESULTS_FILE, index=False)
    print(f"All results saved to {RESULTS_FILE}")
    print(f"Detailed results saved to {DETAILED_RESULTS_FILE}")

if __name__ == '__main__':
    main()