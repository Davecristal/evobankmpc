""" evobankmpc average-runs version.
    Based on run_nmpc_orca_evobankmpc.py
    - Removes plotting/video generation
    - Removes runtime statistics
    - Runs the full experiment multiple times internally and averages results
    - Keeps command-line hyperparameters and main control logic
"""

import numpy as np
import os
import copy
import json
import argparse
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

from evobankmpc.params import ORCA
from evobankmpc.models import Dynamic
from evobankmpc.tracks import ETHZ, ETHZMobil
from evobankmpc.mpc.planner import ConstantSpeed
from evobankmpc.mpc.nmpc import setupNLP
from evobankmpc.mpc.constraints import Boundary
from evobankmpc.mpc.evaluate_models_jax import make_jax_evaluator

try:
    from evobankmpc.iterative.value_fn import ValueFn
    from evobankmpc.iterative.safe_set import SafeSet
    from evobankmpc.utils.trainer_jax import Trainer
    ITERATIVE_AVAILABLE = True
except ImportError:
    ITERATIVE_AVAILABLE = False


def calculate_mean_of_indices(tuples_list):
    lists = [t[0] for t in tuples_list]
    numbers = [t[1:] for t in tuples_list]
    lists_mean = np.mean(lists, axis=0)
    numbers_mean = np.mean(numbers, axis=0)
    return np.concatenate((lists_mean, numbers_mean))


parser = argparse.ArgumentParser(description='EvoBank-MPC with adaptive bank (10-run average, no plots/runtime)')
parser.add_argument('--n_runs', type=int, default=10, help='Number of runs to average')
parser.add_argument('--output_file', type=str, default=None, help='Optional JSON file to save averaged results')
# MLP guide params (kept for compatibility)
parser.add_argument('--mlp_switch_threshold', type=float, default=0.05, help='Unused')
parser.add_argument('--hysteresis_steps', type=int, default=1, help='Unused')
# Adaptive bank params
parser.add_argument('--adaptive_sampling_std', type=float, default=0.08, help='Adaptive bank sampling std ratio')
parser.add_argument('--new_model_error_factor', type=float, default=1.1, help='New model error initialization factor')
parser.add_argument('--update_interval', type=int, default=150, help='Adaptive bank update interval (steps)')
parser.add_argument('--friction_update_delay', type=int, default=40, help='Steps to delay update after friction')
parser.add_argument('--stabilization_steps', type=int, default=1000, help='Steps before first update')
# Iterative learning params
parser.add_argument('--enable_iterative', action='store_true', default=False, help='Enable iterative learning')
parser.add_argument('--value_update_interval', type=int, default=3)
parser.add_argument('--value_model_type', type=str, default='nn')
parser.add_argument('--value_hidden_dims', type=int, default=80)
parser.add_argument('--value_max_epoch', type=int, default=50)
parser.add_argument('--value_weight', type=float, default=0.068)
# Mu estimation params
parser.add_argument('--smoothing_mu', type=int, default=20)
parser.add_argument('--mu_alpha', type=float, default=0.08)
parser.add_argument('--smoothing_mu_over_mod', type=int, default=10)
# Planning/cost params
parser.add_argument('--v_factor', type=float, default=0.915)
parser.add_argument('--cost_r_acc_scale', type=float, default=0.5)
parser.add_argument('--cost_r_steer_scale', type=float, default=0.68)
# Adaptive ey penalty params
parser.add_argument('--ey_weight_base', type=float, default=2.0)
parser.add_argument('--ey_weight_curvature_gain', type=float, default=10.0)
# kept compatibility params
parser.add_argument('--dynamic_threshold_std_ratio', type=float, default=0.6, help='Unused')
parser.add_argument('--use_advanced_mu_estimator', action='store_true', default=False, help='Unused')
parser.add_argument('--mu_conf_scale', type=float, default=2.0, help='Unused')
parser.add_argument('--mu_fast_window', type=int, default=5, help='Unused')
parser.add_argument('--mu_slow_alpha', type=float, default=0.1, help='Unused')
parser.add_argument('--mu_bias_lr', type=float, default=0.0, help='Unused')
parser.add_argument('--mu_smoothing_window', type=int, default=3, help='Unused')
args = parser.parse_args()

# =========================
# Manual track/scenario block
# =========================
# Option 1: ETHZ
TRACK_NAME = 'ETHZ'
SIM_TIME = 36.0
# const_decay: 14-14.8; sudden14.3: 14-14.8; sudden3.3: 3-3.8
FRICTION_START = 3
FRICTION_END = 3.8
EFFECTIVE_WINDOW_START = 14.0
EFFECTIVE_WINDOW_END = 17.0
PROJIDX_THRESHOLD = 656
track = ETHZ(reference='optimal', longer=True)
def update_friction(Df, Dr, curr_time, style='sudden'):
    if style == 'const_decay':
        if curr_time > 14.3:
            Df -= Df/2600.
            Dr -= Dr/2600.
    elif style == 'sudden':
        if curr_time > 3.3 and curr_time < 3.5:
        # if curr_time > 14.3 and curr_time < 14.5:
            Df -= Df/22.
            Dr -= Dr/22.
    elif style == 'no_change':
        return Df, Dr
    return Df, Dr

# Option 2: ETHZMobil
# TRACK_NAME = 'ETHZMobil'
# SIM_TIME = 24.0
# # const_decay: 4.8-5.4; sudden5s: 4.8-5.3; sudden10s: 9.7-10.3
# FRICTION_START = 4.8
# FRICTION_END = 5.4
# EFFECTIVE_WINDOW_START = 5.0
# EFFECTIVE_WINDOW_END = 8.0
# PROJIDX_THRESHOLD = 440

# track = ETHZMobil(reference='optimal', longer=True)

# def update_friction(Df, Dr, curr_time, style='const_decay'):
#     if style == 'const_decay':
#         if curr_time > 5:
#             Df -= Df/2600.
#             Dr -= Dr/2600.
#     elif style == 'sudden':
#         if curr_time > 5 and curr_time < 5.2:
#         # if curr_time > 10 and curr_time < 10.2:
#             Df -= Df/22.
#             Dr -= Dr/22.
#     elif style == 'no_change':
#         return Df, Dr
#     return Df, Dr


class ExponentialSmoother:
    def __init__(self, alpha=0.08):
        self.alpha = alpha
        self.value = None
    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value


def compute_curvature(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-6)**1.5
    return curvature


def find_closest_point(x, y, raceline):
    x_refs = raceline[0]
    y_refs = raceline[1]
    distances = np.sqrt((x_refs - x)**2 + (y_refs - y)**2)
    idx = np.argmin(distances)
    return idx, x_refs[idx], y_refs[idx], distances[idx]




ALL_TRYS = []
all_run_results = []
for TRYING in range(args.n_runs):
    
    # Parameters
    Ts = 0.02
    HORIZON = 20
    COST_Q = np.diag([1, 1])
    COST_P = np.diag([0, 0])
    COST_R = np.diag([5/1000 * args.cost_r_acc_scale, 1 * args.cost_r_steer_scale])
    TRACK_CONS = False

    N_MODELS = 5000 
    LookBack_W = 10
    mu_init = 1.0
    HYSTERESIS_STEPS = args.hysteresis_steps

    ENABLE_ADAPTIVE_BANK = True
    UPDATE_INTERVAL = args.update_interval
    N_REPLACE = 50
    ADAPTIVE_SAMPLING_STD_RATIO = args.adaptive_sampling_std
    PRESERVE_RANDOM_RATIO = 0.3
    FRICTION_UPDATE_DELAY = args.friction_update_delay
    NEW_MODEL_ERROR_FACTOR = args.new_model_error_factor
    STABILIZATION_STEPS = args.stabilization_steps

    # ===== Stability-oriented anti-collapse settings =====
    ANCHOR_RATIO = 0.4          
    TOPK_PARENT = 5             
    SWITCH_MARGIN = 0.93        
    SWITCH_CONFIRM_STEPS = 3    
    PROTECT_COST_THRESHOLD = 1200.0
    PROTECT_CONSEC_STEPS = 3
    PROTECT_HOLD_STEPS = 40

    smoothing_mu = args.smoothing_mu
    mu_alpha = args.mu_alpha
    smoothing_mu_over_mod = args.smoothing_mu_over_mod

    params_veh = ORCA(control='pwm')
    true_model = Dynamic(**params_veh)
    model = Dynamic(**params_veh)

    mu_smoother = ExponentialSmoother(alpha=mu_alpha)

    EY_WEIGHT_QUANTIZED = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
    def quantize_ey_weight(ey_weight):
        return min(EY_WEIGHT_QUANTIZED, key=lambda x: abs(x - ey_weight))

    nlp_cache = {}
    def get_or_build_nlp(model_idx, quantized_weight, model_params, model_obj):
        key = (model_idx, quantized_weight)
        if key not in nlp_cache:
            nlp_cache[key] = setupNLP(HORIZON, Ts, COST_Q, COST_P, COST_R,
                                      model_params, model_obj, track,
                                      track_cons=TRACK_CONS, ey_weight=quantized_weight)
        return nlp_cache[key]

    def invalidate_nlp_cache():
        nlp_cache.clear()

    raceline_x, raceline_y = track.raceline
    dx = np.diff(raceline_x)
    dy = np.diff(raceline_y)
    segment_lengths = np.sqrt(dx**2 + dy**2)
    raceline_s = np.concatenate([[0], np.cumsum(segment_lengths)])
    raceline_kappa = compute_curvature(raceline_x, raceline_y)

    if args.enable_iterative and ITERATIVE_AVAILABLE:
        raceline_psi = np.arctan2(dy, dx)
        raceline_psi = np.append(raceline_psi, raceline_psi[-1])
        def cartesian_to_frenet_track(x, y, psi):
            distances = (raceline_x - x)**2 + (raceline_y - y)**2
            idx = np.argmin(distances)
            x_ref = raceline_x[idx]; y_ref = raceline_y[idx]; yaw_ref = raceline_psi[idx]
            normal = np.array([-np.sin(yaw_ref), np.cos(yaw_ref)])
            ey = (x - x_ref) * normal[0] + (y - y_ref) * normal[1]
            s = raceline_s[idx]
            epsi = psi - yaw_ref
            epsi = np.arctan2(np.sin(epsi), np.cos(epsi))
            return s, ey, epsi
    else:
        def cartesian_to_frenet_track(x, y, psi):
            return 0.0, 0.0, 0.0

    variation_dict = {
        'Br': 0.2, 'Cr': 0.1, 'Dr': 0.5,
        'Bf': 0.2, 'Cf': 0.1, 'Df': 0.5,
    }
    MODEL_BANK = []
    MODEL_PARAMS = []
    for i in range(N_MODELS):
        param_variation = params_veh.copy()
        for param_name, var in variation_dict.items():
            if param_name in param_variation:
                param_variation[param_name] *= (1 + var * np.random.randn())
        MODEL_PARAMS.append(param_variation)
        MODEL_BANK.append(Dynamic(**param_variation))

    model_params_array = np.zeros((N_MODELS, 6))
    for i, p in enumerate(MODEL_PARAMS):
        model_params_array[i, 0] = p['Bf']
        model_params_array[i, 1] = p['Cf']
        model_params_array[i, 2] = p['Df']
        model_params_array[i, 3] = p['Br']
        model_params_array[i, 4] = p['Cr']
        model_params_array[i, 5] = p['Dr']
    MODEL_PARAMS_ARRAY = jnp.array(model_params_array, dtype=jnp.float64)

    ANCHOR_COUNT = int(N_MODELS * ANCHOR_RATIO)
    ANCHOR_INDICES = set(range(ANCHOR_COUNT))

    vehicle_params = {
        'mass': params_veh['mass'], 'lf': params_veh['lf'], 'lr': params_veh['lr'],
        'Iz': params_veh['Iz'], 'Cm1': params_veh['Cm1'], 'Cm2': params_veh['Cm2'],
        'Cr0': params_veh['Cr0'], 'Cr2': params_veh['Cr2'],
    }
    jax_evaluator = make_jax_evaluator(vehicle_params)

    nlp_initial = setupNLP(HORIZON, Ts, COST_Q, COST_P, COST_R,
                           params_veh, true_model, track,
                           track_cons=TRACK_CONS, ey_weight=0.0)

    if args.enable_iterative and ITERATIVE_AVAILABLE:
        track_length = np.sum(np.sqrt(dx**2 + dy**2))
        class ValueTrainConfig:
            model_type = args.value_model_type
            n_dim = 1
            n_context = 4
            hidden_dims = args.value_hidden_dims
            layer_num = 3
            lr = 1e-3
            batchsize = 256
            max_epoch = args.value_max_epoch
            silent = True
            n_bins = 8
            activation = 'gelu'
            n_sample = 20
            pe_level = 0
        value_train_config = ValueTrainConfig()

        class IterativeMainConfig:
            value_dim = 4
            state_dim = 6
            save_dir = "./"
            half_width = 4.0
            ss_hull_num_reduction = 10
            ss_hull_precompile_len = 1000
            ss_arr_max_len = 200000
            ss_select_max_len = 1000
            ss_loop_extend_sec = 2.0
            n_steps = HORIZON
            sim_time_step = Ts
            init_vel = 5.0
            ss_size = 4
            ss_ref_step_interval = 3.0
            ss_relaxation = 0.03
            n_samples = 512
            s_frame_max = track_length
            track = track
            print_line = False
            map_inds = None
            state_predictor = None
        iterative_config = IterativeMainConfig()
        value_fn = ValueFn(None, value_train_config, iterative_config)
        safe_set = SafeSet(iterative_config, None, track)
        lap_data_frenet = []
        lap_data_remaining = []
        lap_start_indices = [0]
    else:
        value_fn = None
        safe_set = None
        lap_data_frenet = None
        lap_data_remaining = None

    error_windows_phys = np.zeros((N_MODELS, LookBack_W))
    window_count_phys = 0
    update_restore_counter = 0
    current_model_idx = 0
    last_update_time = -1e9
    violation_total = 0.0
    violation_eps = 1e-6

    pending_model_idx = -1
    pending_count = 0
    protect_steps = 0
    high_cost_counter = 0

    def update_model_bank(current_time_step, avg_errors_phys, current_model_idx, friction_active,
                          temp_std_ratio=None, aggressive=False):
        global MODEL_BANK, MODEL_PARAMS, model_params_array, MODEL_PARAMS_ARRAY, error_windows_phys, update_restore_counter, HYSTERESIS_STEPS
        n_replace_local = N_REPLACE * 2 if aggressive else N_REPLACE
        n_replace_local = min(n_replace_local, N_MODELS // 4)

        sorted_indices = np.argsort(avg_errors_phys)[::-1]
        replace_indices = []

        candidate_replace_pool = [idx for idx in sorted_indices
                                  if idx not in ANCHOR_INDICES and idx != current_model_idx]

        for idx in candidate_replace_pool:
            replace_indices.append(idx)
            if len(replace_indices) >= n_replace_local:
                break

        if len(replace_indices) == 0:
            return

        topk = np.argsort(avg_errors_phys)[:TOPK_PARENT]
        topk_err = avg_errors_phys[topk]
        weights = 1.0 / (topk_err + 1e-8)
        weights = weights / np.sum(weights)

        sampling_std = temp_std_ratio if temp_std_ratio is not None else ADAPTIVE_SAMPLING_STD_RATIO

        for ridx in replace_indices:
            parent_idx = int(np.random.choice(topk, p=weights))
            parent_params = MODEL_PARAMS[parent_idx].copy()

            new_param = parent_params.copy()
            for param_name, _ in variation_dict.items():
                if param_name in new_param:
                    new_value = new_param[param_name] * (1.0 + sampling_std * np.random.randn())
                    new_value = max(0.05, new_value)
                    new_param[param_name] = new_value

            new_model = Dynamic(**new_param)
            MODEL_BANK[ridx] = new_model
            MODEL_PARAMS[ridx] = new_param
            model_params_array[ridx, 0] = new_param['Bf']
            model_params_array[ridx, 1] = new_param['Cf']
            model_params_array[ridx, 2] = new_param['Df']
            model_params_array[ridx, 3] = new_param['Br']
            model_params_array[ridx, 4] = new_param['Cr']
            model_params_array[ridx, 5] = new_param['Dr']
            error_windows_phys[ridx, :] = avg_errors_phys[parent_idx] * NEW_MODEL_ERROR_FACTOR

        MODEL_PARAMS_ARRAY = jnp.array(model_params_array, dtype=jnp.float64)
        invalidate_nlp_cache()
        update_restore_counter = 50
        print(f"[{current_time_step}]  (aggressive={aggressive}): replaced {len(replace_indices)} models")


    n_steps = int(SIM_TIME / Ts)
    n_states = model.n_states
    n_inputs = model.n_inputs
    horizon = HORIZON

    states = np.zeros((n_states, n_steps+1))
    cost_history = []
    dstates = np.zeros((n_states, n_steps+1))
    inputs = np.zeros((n_inputs, n_steps))
    time = np.linspace(0, n_steps, n_steps+1) * Ts

    hstates = np.zeros((n_states, horizon+1))
    hstates2 = np.zeros((n_states, horizon+1))

    model_switches = []
    model_mses = []
    chosen_models = []

    projidx = 0
    x_init = np.zeros(n_states)
    x_init[0], x_init[1] = track.x_init, track.y_init
    x_init[2] = track.psi_init
    x_init[3] = track.vx_init
    dstates[0,0] = x_init[3]
    states[:,0] = x_init
    print('starting at ({:.1f},{:.1f})'.format(x_init[0], x_init[1]))

    uprev = np.zeros(n_inputs)
    ref_speeds = []
    Drs = []; Dfs = []; Drs_preds = []; Dfs_preds = []
    MUs = []; MU_preds = []
    deviation = []
    lap_times = [0.,0.,0.,0.,0.]
    laps_completed = 0
    ind_best_KM = []

    
    for idt in range(n_steps - horizon):
        x0 = states[:, idt]
        model.Df, model.Dr = update_friction(model.Df, model.Dr, idt * Ts)
        params_veh['Df'], params_veh['Dr'] = model.Df, model.Dr

        safe_v_factor = args.v_factor

        if idt > LookBack_W+1 and len(MU_preds) > 0:
            xref, projidx, v = ConstantSpeed(x0[:2], x0[3], track=track,
                                             N=horizon, Ts=Ts, projidx=projidx,
                                             curr_mu=MU_preds[-1], scale=safe_v_factor)
        else:
            xref, projidx, v = ConstantSpeed(x0[:2], x0[3], track=track,
                                             N=horizon, Ts=Ts, projidx=projidx)
        ref_speeds.append(v)
        deviation.append(find_closest_point(x0[0], x0[1], track.raceline)[-1])

        if projidx > PROJIDX_THRESHOLD:
            if laps_completed > 0:
                lap_times[laps_completed] = idt * Ts
                lap_end_idx = idt
                if args.enable_iterative and ITERATIVE_AVAILABLE:
                    start_idx = lap_start_indices[-1] if laps_completed > 0 else 0
                    end_idx = lap_end_idx
                    lap_frenet = []
                    lap_rem = []
                    for i in range(start_idx, end_idx+1):
                        x_c, y_c, psi_c = states[0,i], states[1,i], states[2,i]
                        vx_c = states[3,i]
                        s, ey, epsi = cartesian_to_frenet_track(x_c, y_c, psi_c)
                        lap_frenet.append([s, ey, epsi, vx_c])
                        remaining = (lap_end_idx - i) * Ts
                        lap_rem.append(remaining)
                    lap_data_frenet.append(np.array(lap_frenet, dtype=np.float32))
                    lap_data_remaining.append(np.array(lap_rem, dtype=np.float32).reshape(-1,1))
            else:
                lap_times[laps_completed] = idt * Ts
            laps_completed += 1
            projidx = 0
            if args.enable_iterative and ITERATIVE_AVAILABLE:
                lap_start_indices.append(idt+1)
            if laps_completed >= 3:
                break

        distances = np.sqrt((raceline_x - x0[0])**2 + (raceline_y - x0[1])**2)
        closest_idx = np.argmin(distances)
        kappa = raceline_kappa[closest_idx]
        ey_weight = args.ey_weight_base * (1.0 + args.ey_weight_curvature_gain * np.abs(kappa))
        ey_weight = np.clip(ey_weight, 0.1, 50.0)
        quantized_w = quantize_ey_weight(ey_weight)
        nlp = get_or_build_nlp(current_model_idx, quantized_w, MODEL_PARAMS[current_model_idx], MODEL_BANK[current_model_idx])

        umpc, fval, xmpc, violation = nlp.solve(x0, xref[:2,:], uprev)
        if fval > 2000:
            print(f"⚠️ MPC solving failed (cost={fval:.2f}), Return to the initial model")
            current_model_idx = 0
            nlp_fallback = get_or_build_nlp(0, quantized_w, params_veh, true_model)
            umpc, fval, xmpc, violation = nlp_fallback.solve(x0, xref[:2,:], uprev)

        cost_history.append(fval)
        inputs[:, idt] = umpc[:, 0]
        uprev = inputs[:, idt]

        x_next, dxdt_next = model.sim_continuous(states[:, idt], inputs[:, idt].reshape(-1,1), [0, Ts])
        states[:, idt+1] = x_next[:, -1]
        dstates[:, idt+1] = dxdt_next[:, -1]

        Drs.append(model.Dr)
        Dfs.append(model.Df)

        Ain, bin = Boundary(np.array(x_next[:2, -1]), track)
        if np.any(Ain @ x_next[:2, -1] > bin.flatten() + violation_eps):
            violation_total += Ts

        if idt > 0:
            curr_state_jax = jnp.array(states[:, idt], dtype=jnp.float64)
            curr_input_jax = jnp.array(inputs[:, idt], dtype=jnp.float64)
            true_next = states[:4, idt+1]
            current_time = idt * Ts
            if FRICTION_START < current_time < FRICTION_END + 2.0:
                effective_window = max(3, LookBack_W // 3)
            else:
                effective_window = LookBack_W

            next_states_batch = jax_evaluator(curr_state_jax, curr_input_jax, MODEL_PARAMS_ARRAY, Ts)
            next_states_batch_np = np.array(next_states_batch)
            errors_phys = np.mean((next_states_batch_np[:, :4] - true_next) ** 2, axis=1)
            error_windows_phys = np.roll(error_windows_phys, -1, axis=1)
            error_windows_phys[:, -1] = errors_phys
            window_count_phys = min(window_count_phys + 1, LookBack_W)

            if window_count_phys >= LookBack_W:
                avg_errors_phys = np.mean(error_windows_phys[:, -effective_window:], axis=1)
                new_model_idx = int(np.argmin(avg_errors_phys))

                current_err = avg_errors_phys[current_model_idx]
                new_err = avg_errors_phys[new_model_idx]

                if new_model_idx != current_model_idx and new_err < SWITCH_MARGIN * current_err:
                    if pending_model_idx == new_model_idx:
                        pending_count += 1
                    else:
                        pending_model_idx = new_model_idx
                        pending_count = 1

                    if pending_count >= SWITCH_CONFIRM_STEPS:
                        current_model_idx = new_model_idx
                        model_switches.append(idt)
                        model_mses.append(avg_errors_phys[new_model_idx])
                        chosen_models.append(current_model_idx)
                        pending_model_idx = -1
                        pending_count = 0
                else:
                    pending_model_idx = -1
                    pending_count = 0

            if ENABLE_ADAPTIVE_BANK and window_count_phys >= LookBack_W and idt > LookBack_W:
                current_time = idt * Ts
                skip_update = idt < STABILIZATION_STEPS
                friction_active = (FRICTION_START < current_time < FRICTION_END) or (current_time > FRICTION_END and current_time < FRICTION_END + FRICTION_UPDATE_DELAY * Ts)
                effective_update_interval = UPDATE_INTERVAL
                effective_std_ratio = ADAPTIVE_SAMPLING_STD_RATIO
                aggressive = False

                if current_time <= FRICTION_END + 1.5:
                    skip_update = True
                elif current_time > FRICTION_END + 1.5 and current_time < FRICTION_END + 5.0:
                    effective_update_interval = 35
                    effective_std_ratio = 0.18
                    aggressive = True
                elif current_time > FRICTION_END + 5.0 and current_time < FRICTION_END + 10.0:
                    effective_update_interval = 55
                    effective_std_ratio = 0.12
                    aggressive = True

                if not skip_update and protect_steps == 0 and not friction_active and (idt - last_update_time) >= effective_update_interval:
                    update_model_bank(idt, avg_errors_phys, current_model_idx, friction_active,
                                      temp_std_ratio=effective_std_ratio, aggressive=aggressive)
                    last_update_time = idt

        if update_restore_counter > 0:
            update_restore_counter -= 1
            if update_restore_counter == 0:
                HYSTERESIS_STEPS = args.hysteresis_steps

        if idt <= LookBack_W:
            init_Dr = mu_init * params_veh['mass'] * 9.8 * params_veh['lr'] / (params_veh['lf'] + params_veh['lr'])
            init_Df = mu_init * params_veh['mass'] * 9.8 * params_veh['lf'] / (params_veh['lf'] + params_veh['lr'])
            Drs_preds.append(init_Dr)
            Dfs_preds.append(init_Df)
            MUs.append((model.Df + model.Dr) / (9.81 * params_veh['mass']))
            MU_preds.append(mu_init)
        else:
            MUs.append((model.Df + model.Dr) / (9.81 * params_veh['mass']))
            if window_count_phys >= LookBack_W:
                best_indices = np.argsort(avg_errors_phys)[:smoothing_mu_over_mod]
                best_Dr = np.mean([MODEL_BANK[i].Dr for i in best_indices])
                best_Df = np.mean([MODEL_BANK[i].Df for i in best_indices])
            else:
                best_Dr = MODEL_BANK[current_model_idx].Dr
                best_Df = MODEL_BANK[current_model_idx].Df
            Drs_preds.append(best_Dr)
            Dfs_preds.append(best_Df)
            if len(Drs_preds) >= smoothing_mu:
                avg_Dr = np.mean(Drs_preds[-smoothing_mu:])
                avg_Df = np.mean(Dfs_preds[-smoothing_mu:])
            else:
                avg_Dr = np.mean(Drs_preds)
                avg_Df = np.mean(Dfs_preds)
            mu_raw = (avg_Dr + avg_Df) / (9.81 * params_veh['mass'])
            mu_smoothed = mu_smoother.update(mu_raw)
            mu_final = mu_smoothed * 0.95
            MU_preds.append(mu_final)

        if args.enable_iterative and ITERATIVE_AVAILABLE and laps_completed >= args.value_update_interval:
            if laps_completed % args.value_update_interval == 0 and len(lap_data_frenet) > 0:
                X_train = np.concatenate(lap_data_frenet, axis=0)
                y_train = np.concatenate(lap_data_remaining, axis=0)
                train_data = np.concatenate([X_train, y_train], axis=1)
                train_safe_set = [train_data]
                value_fn.train_valuefn(train_safe_set, iterative_config, None, max_epoch=args.value_max_epoch)
                safe_set.add_lap(y_train.flatten(), X_train)
                if safe_set.get_new_safe_set():
                    safe_set.update_ss_arr()
                    if safe_set.xSS_hull_equations is not None:
                        invalidate_nlp_cache()
                lap_data_frenet = []
                lap_data_remaining = []

        hstates[:,0] = x0
        hstates2[:,0] = x0
        for idh in range(horizon):
            x_next_pred, _ = model.sim_continuous(hstates[:, idh], umpc[:, idh].reshape(-1,1), [0, Ts])
            hstates[:, idh+1] = x_next_pred[:, -1]
            hstates2[:, idh+1] = xmpc[:, idh+1]

        current_mse = np.mean(error_windows_phys[current_model_idx])*10000 if current_model_idx>=0 and window_count_phys>0 else 0.0
        print(f"iter: {idt}, model: {current_model_idx}, cost: {fval:.5f}, mse*10000: {current_mse:.3e}")

    for i in range(len(lap_times)-1,0,-1):
        if lap_times[i] != 0.:
            lap_times[i] -= lap_times[i-1]

    print("lap times:", lap_times[:laps_completed], "violation:", violation_total, "mean deviation:", np.mean(deviation))
    mean_deviation = float(np.mean(deviation)) if len(deviation) > 0 else 0.0
    mean_cost = float(np.mean(cost_history)) if len(cost_history) > 0 else 0.0

    results = {
        'lap_times': lap_times[:laps_completed],
        'mean_deviation': mean_deviation,
        'mean_cost': mean_cost,
        'total_lap_time': float(np.sum(lap_times[:laps_completed])),
        'violation_time': float(violation_total),
    }
    lap_times_full = [0., 0., 0., 0., 0.]
    for i, v in enumerate(results['lap_times']):
        if i < len(lap_times_full):
            lap_times_full[i] = float(v)
    triple = (lap_times_full, float(results['violation_time']), float(results['mean_deviation']))
    ALL_TRYS.append(triple)
    all_run_results.append(results)
    print(triple)
    print(ALL_TRYS)

ALL_TRYS_MEAN = calculate_mean_of_indices(ALL_TRYS)
print(ALL_TRYS_MEAN)

if args.output_file:
    out = {
        'n_runs': int(args.n_runs),
        'all_trys': ALL_TRYS,
        'avg_vector': ALL_TRYS_MEAN.tolist(),
        'per_run_results': all_run_results,
    }
    with open(args.output_file, 'w') as f:
        json.dump(out, f)
    print(f"Average summary saved to {args.output_file}")
