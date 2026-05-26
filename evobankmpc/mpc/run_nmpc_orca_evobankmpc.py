""" 
    Enhanced with adaptive friction smoothing, value function boosting, and safe set.
    Adapted to work reliably on both ETHZ and ETHZMobil tracks via manual comment toggle.
    MLP is disabled because it does not improve performance on these tracks.
    Added curvature‑adaptive lateral error penalty.
    Friction estimation uses Top‑K best models + moving average + exponential smoothing (same as pure physical).
    NLP caching for speed (0.06s per step). Adaptive bank update with aggressive mode after friction drop.
"""

import time as tm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pylab as pylab
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
except ImportError as e:
    print(f"Warning: iterative learning modules not available: {e}")
    ITERATIVE_AVAILABLE = False

params = {
    'legend.fontsize': 'xx-large',
    'axes.labelsize': 'xx-large',
    'axes.titlesize':'xx-large',
    'xtick.labelsize':'xx-large',
    'ytick.labelsize':'xx-large'
}
pylab.rcParams.update(params)
plt.rcParams['text.usetex'] = False

# ============================================================================
parser = argparse.ArgumentParser(description='Hybrid MPC with adaptive bank (MLP disabled)')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--no_video', action='store_true', help='Disable video saving')
parser.add_argument('--no_plots', action='store_true', help='Disable plot saving')
parser.add_argument('--output_file', type=str, default=None, help='JSON file to save results')
# MLP 
parser.add_argument('--mlp_switch_threshold', type=float, default=0.05, help='Unused')
parser.add_argument('--hysteresis_steps', type=int, default=1, help='Unused')
# adaptive bank
parser.add_argument('--adaptive_sampling_std', type=float, default=0.08, help='Adaptive bank sampling std ratio')
parser.add_argument('--new_model_error_factor', type=float, default=1.1, help='New model error initialization factor')
parser.add_argument('--update_interval', type=int, default=150, help='Adaptive bank update interval (steps)')
parser.add_argument('--friction_update_delay', type=int, default=40, help='Steps to delay update after friction')
parser.add_argument('--stabilization_steps', type=int, default=1000, help='Steps before first update')
# value function boosting
parser.add_argument('--enable_iterative', action='store_true', default=False,
                    help='Enable iterative learning (value function & safe set)')
parser.add_argument('--value_update_interval', type=int, default=3,
                    help='Update value function every N laps')
parser.add_argument('--value_model_type', type=str, default='nn',
                    help='Value function model type: nn / nf / bnn')
parser.add_argument('--value_hidden_dims', type=int, default=80,
                    help='Value MLP hidden dims')
parser.add_argument('--value_max_epoch', type=int, default=50,
                    help='Value training epochs per update')
parser.add_argument('--value_weight', type=float, default=0.068,
                    help='Weight of value function prediction in model selection')
# friction estimation
parser.add_argument('--smoothing_mu', type=int, default=20,
                    help='Moving average window for mu estimation')
parser.add_argument('--mu_alpha', type=float, default=0.08,
                    help='Exponential smoothing alpha for mu')
parser.add_argument('--smoothing_mu_over_mod', type=int, default=10,
                    help='Number of top models to average for mu')
# cost function weights
parser.add_argument('--v_factor', type=float, default=0.915, help='Planning speed factor (0.8-1.0)')
parser.add_argument('--cost_r_acc_scale', type=float, default=0.5, help='Scale of acceleration penalty in COST_R')
parser.add_argument('--cost_r_steer_scale', type=float, default=0.68, help='Scale of steering penalty in COST_R')

parser.add_argument('--ey_weight_base', type=float, default=2.0, help='Base lateral error penalty weight')
parser.add_argument('--ey_weight_curvature_gain', type=float, default=10.0, help='Curvature gain for adaptive weight')

parser.add_argument('--dynamic_threshold_std_ratio', type=float, default=0.6, help='Unused')

parser.add_argument('--use_advanced_mu_estimator', action='store_true', default=False,
                    help='Unused, kept for compatibility')
parser.add_argument('--mu_conf_scale', type=float, default=2.0, help='Unused')
parser.add_argument('--mu_fast_window', type=int, default=5, help='Unused')
parser.add_argument('--mu_slow_alpha', type=float, default=0.1, help='Unused')
parser.add_argument('--mu_bias_lr', type=float, default=0.0, help='Unused')

parser.add_argument('--mu_smoothing_window', type=int, default=3, help='Unused, kept for compatibility')
args = parser.parse_args()


np.random.seed(args.seed)

# ============================================================================
script_wall_start = tm.time()

# ============================================================================
# ----------------------------------------------------------------------------
# 1: ETHZ
# TRACK_NAME = 'ETHZ'
# SIM_TIME = 36.0
# # const_decay: 14-50; sudden14.3: 14-14.8; sudden3.3: 3-3.8
# FRICTION_START = 14.0
# FRICTION_END = 50
# EFFECTIVE_WINDOW_START = 14.0
# EFFECTIVE_WINDOW_END = 17.0
# PROJIDX_THRESHOLD = 656   
# track = ETHZ(reference='optimal', longer=True)
# def update_friction(Df,Dr,curr_time,style='const_decay') :
#     if style == 'const_decay' :
#         if curr_time > 14.3:
#             Df -= Df/2600.
#             Dr -= Dr/2600.
#     elif style == 'sudden' :
#         # if curr_time > 3.3 and curr_time < 3.5: # early sudden
#         if curr_time > 14.3 and curr_time < 14.5:
#             Df -= Df/22.
#             Dr -= Dr/22.
#     elif style == 'no_change' :
#         return Df, Dr
#     return Df, Dr

# 2: ETHZMobil
TRACK_NAME = 'ETHZMobil'
SIM_TIME = 24.0
# const_decay: 4.8-50; sudden5s: 4.9-5.2; sudden10s: 9.7-10.3
FRICTION_START = 4.8
FRICTION_END = 50
EFFECTIVE_WINDOW_START = 5.0
EFFECTIVE_WINDOW_END = 8.0
PROJIDX_THRESHOLD = 440
track = ETHZMobil(reference='optimal', longer=True)
def update_friction(Df,Dr,curr_time,style='const_decay'):
    if style == 'const_decay' :
        if curr_time > 5:
            Df -= Df/2600.
            Dr -= Dr/2600.
    elif style == 'sudden' :
        if curr_time > 5 and curr_time < 5.2:
        # if curr_time > 10 and curr_time < 10.2:
            Df -= Df/22.
            Dr -= Dr/22.
    elif style == 'no_change' :
        return Df, Dr
    return Df, Dr
# ----------------------------------------------------------------------------

# ============================================================================
SAVE_VIDEO = not args.no_video
SAVE_PLOTS = not args.no_plots
SAMPLING_TIME = 0.02
HORIZON = 20
COST_Q = np.diag([1, 1])
COST_P = np.diag([0, 0])
COST_R = np.diag([5/1000 * args.cost_r_acc_scale, 1 * args.cost_r_steer_scale])
TRACK_CONS = False   


N_MODELS = 20000
LookBack_W = 10
v_factor = args.v_factor
mu_init = 1.0

# ---------- MLP  ----------
USE_MLP = False
TIER_RATIO = 1.1
MLP_SWITCH_THRESHOLD = args.mlp_switch_threshold
MIN_PHYS_ERR = 1e-8
HYSTERESIS_STEPS = args.hysteresis_steps
DISABLE_MLP_IN_FRICTION = True

ENABLE_ADAPTIVE_BANK = True
UPDATE_INTERVAL = args.update_interval
N_REPLACE = 50
ADAPTIVE_SAMPLING_STD_RATIO = args.adaptive_sampling_std
PRESERVE_RANDOM_RATIO = 0.3
FRICTION_UPDATE_DELAY = args.friction_update_delay
DYNAMIC_STD_ENABLE = False
NEW_MODEL_ERROR_FACTOR = args.new_model_error_factor
STABILIZATION_STEPS = args.stabilization_steps


smoothing_mu = args.smoothing_mu          
mu_alpha = args.mu_alpha                  
smoothing_mu_over_mod = args.smoothing_mu_over_mod   


params_veh = ORCA(control='pwm')
true_model = Dynamic(**params_veh)
model = Dynamic(**params_veh)

# ============================================================================
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

mu_smoother = ExponentialSmoother(alpha=mu_alpha)

# ============================================================================
EY_WEIGHT_QUANTIZED = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]

def quantize_ey_weight(ey_weight):
    return min(EY_WEIGHT_QUANTIZED, key=lambda x: abs(x - ey_weight))

nlp_cache = {}

def get_or_build_nlp(model_idx, quantized_weight, model_params, model_obj, track):
    key = (model_idx, quantized_weight)
    if key not in nlp_cache:
        nlp_cache[key] = setupNLP(HORIZON, SAMPLING_TIME, COST_Q, COST_P, COST_R,
                                  model_params, model_obj, track,
                                  track_cons=TRACK_CONS, ey_weight=quantized_weight)
    return nlp_cache[key]

def invalidate_nlp_cache():
    global nlp_cache
    nlp_cache.clear()
    print("🔄 NLP cache cleared due to model bank update.")

# ============================================================================
raceline_x, raceline_y = track.raceline
dx = np.diff(raceline_x)
dy = np.diff(raceline_y)
segment_lengths = np.sqrt(dx**2 + dy**2)
raceline_s = np.concatenate([[0], np.cumsum(segment_lengths)])
def compute_curvature(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-6)**1.5
    return curvature
raceline_kappa = compute_curvature(raceline_x, raceline_y)

# ============================================================================
def find_closest_point(x, y, raceline):
    x_refs = raceline[0]
    y_refs = raceline[1]
    distances = np.sqrt((x_refs - x)**2 + (y_refs - y)**2)
    idx = np.argmin(distances)
    return idx, x_refs[idx], y_refs[idx], distances[idx]

# ============================================================================
if args.enable_iterative and ITERATIVE_AVAILABLE:
    print("✅ Iterative learning enabled, will collect data every lap.")
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

# ============================================================================
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

vehicle_params = {
    'mass': params_veh['mass'], 'lf': params_veh['lf'], 'lr': params_veh['lr'],
    'Iz': params_veh['Iz'], 'Cm1': params_veh['Cm1'], 'Cm2': params_veh['Cm2'],
    'Cr0': params_veh['Cr0'], 'Cr2': params_veh['Cr2'],
}
jax_evaluator = make_jax_evaluator(vehicle_params)

# ----------------------------------------------------------------------------
print("NLP on-demand caching ready (no pre-building).")
nlp_initial = setupNLP(HORIZON, SAMPLING_TIME, COST_Q, COST_P, COST_R,
                       params_veh, true_model, track,
                       track_cons=TRACK_CONS, ey_weight=0.0)

if args.enable_iterative and ITERATIVE_AVAILABLE:
    raceline_x, raceline_y = track.raceline
    dx = np.diff(raceline_x); dy = np.diff(raceline_y)
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
        random_seed = args.seed
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
        sim_time_step = SAMPLING_TIME
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

# ============================================================================
def update_model_bank(current_time_step, avg_errors_phys, current_model_idx, friction_active,
                      temp_std_ratio=None, aggressive=False):
    global MODEL_BANK, MODEL_PARAMS, model_params_array, MODEL_PARAMS_ARRAY, error_windows_phys, HYSTERESIS_STEPS

    n_replace_local = N_REPLACE * 4 if aggressive else N_REPLACE
    n_replace_local = min(n_replace_local, N_MODELS // 2)

    sorted_indices = np.argsort(avg_errors_phys)[::-1]
    replace_indices = []
    n_preserve = int(N_MODELS * PRESERVE_RANDOM_RATIO)
    preserve_set = set(np.random.choice(N_MODELS, n_preserve, replace=False))
    for idx in sorted_indices:
        if idx != current_model_idx and idx not in preserve_set:
            replace_indices.append(idx)
        if len(replace_indices) >= n_replace_local:
            break
    if len(replace_indices) == 0:
        return

    best_idx = np.argmin(avg_errors_phys)
    best_params = MODEL_PARAMS[best_idx].copy()
    best_err = avg_errors_phys[best_idx]

    std_dict = {k: var for k, var in variation_dict.items()}
    sampling_std = temp_std_ratio if temp_std_ratio is not None else ADAPTIVE_SAMPLING_STD_RATIO
    for k in std_dict:
        std_dict[k] = sampling_std

    for ridx in replace_indices:
        new_param = best_params.copy()
        for param_name, std in std_dict.items():
            if param_name in new_param:
                new_value = new_param[param_name] * (1 + std * np.random.randn())
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
        error_windows_phys[ridx, :] = best_err * NEW_MODEL_ERROR_FACTOR
    MODEL_PARAMS_ARRAY = jnp.array(model_params_array, dtype=jnp.float64)
    invalidate_nlp_cache()
    print(f"[{current_time_step}]  (aggressive={aggressive}): replaced {len(replace_indices)} models (best model index {best_idx})")
    global update_restore_counter
    update_restore_counter = 50

# ============================================================================
Ts = SAMPLING_TIME
n_steps = int(SIM_TIME / Ts)
n_states = model.n_states
n_inputs = model.n_inputs
horizon = HORIZON

states = np.zeros((n_states, n_steps+1))
cost_history = []
dstates = np.zeros((n_states, n_steps+1))
inputs = np.zeros((n_inputs, n_steps))
time = np.linspace(0, n_steps, n_steps+1) * Ts

Ffy = np.zeros(n_steps+1); Frx = np.zeros(n_steps+1); Fry = np.zeros(n_steps+1)
Ffy_preds = np.zeros(n_steps+1); Frx_preds = np.zeros(n_steps+1); Fry_preds = np.zeros(n_steps+1)
hstates = np.zeros((n_states, horizon+1))
hstates2 = np.zeros((n_states, horizon+1))
Hs0 = []; Hs1 = []; Hs0_2 = []; Hs1_2 = []

model_switches = []
model_mses = []
chosen_models = []

error_windows_phys = np.zeros((N_MODELS, LookBack_W))
window_count_phys = 0

model_switch_counter = 0
current_best_model = 0
update_restore_counter = 0

violation_total = 0.0
violation_eps = 1e-6

projidx = 0
x_init = np.zeros(n_states)
x_init[0], x_init[1] = track.x_init, track.y_init
x_init[2] = track.psi_init
x_init[3] = track.vx_init
dstates[0,0] = x_init[3]
states[:,0] = x_init
print('starting at ({:.1f},{:.1f})'.format(x_init[0], x_init[1]))

current_model_idx = 0
uprev = np.zeros(n_inputs)

ref_speeds = []
Drs = []; Dfs = []; Drs_preds = []; Dfs_preds = []
MUs = []; MU_preds = []
deviation = []
lap_times = [0.,0.,0.,0.,0.]
laps_completed = 0

ind_best_KM = []
last_update_time = -1e9

# ============================================================================
setup_wall_end = tm.time()
step_wall_times = []
nlp_solve_times = []
loop_wall_start = tm.time()

# ============================================================================
for idt in range(n_steps - horizon):
    iter_wall_start = tm.time()
    x0 = states[:, idt]

    model.Df, model.Dr = update_friction(model.Df, model.Dr, idt * Ts)
    params_veh['Df'], params_veh['Dr'] = model.Df, model.Dr

    if idt > LookBack_W+1 and len(MU_preds) > 0:
        xref, projidx, v = ConstantSpeed(x0[:2], x0[3], track=track,
                                         N=horizon, Ts=Ts, projidx=projidx,
                                         curr_mu=MU_preds[-1], scale=v_factor)
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


    distances = np.sqrt((raceline_x - x0[0])**2 + (raceline_y - x0[1])**2)
    closest_idx = np.argmin(distances)
    kappa = raceline_kappa[closest_idx]
    ey_weight = args.ey_weight_base * (1.0 + args.ey_weight_curvature_gain * np.abs(kappa))
    ey_weight = np.clip(ey_weight, 0.1, 50.0)

    quantized_w = quantize_ey_weight(ey_weight)
    nlp = get_or_build_nlp(current_model_idx, quantized_w,
                           MODEL_PARAMS[current_model_idx], MODEL_BANK[current_model_idx],
                           track)

    solve_start = tm.time()
    umpc, fval, xmpc, violation = nlp.solve(x0, xref[:2,:], uprev)

    if fval > 1000:
        print(f"⚠️ MPC failed (cost={fval:.2f}), Return to the initial model")
        current_model_idx = 0
        nlp_fallback = get_or_build_nlp(0, quantized_w,
                                        params_veh, true_model,
                                        track)
        umpc, fval, xmpc, violation = nlp_fallback.solve(x0, xref[:2,:], uprev)
    solve_end = tm.time()
    nlp_solve_times.append(solve_end - solve_start)

    cost_history.append(fval)
    inputs[:, idt] = umpc[:, 0]
    uprev = inputs[:, idt]

    x_next, dxdt_next = model.sim_continuous(states[:, idt], inputs[:, idt].reshape(-1,1), [0, Ts])
    states[:, idt+1] = x_next[:, -1]
    dstates[:, idt+1] = dxdt_next[:, -1]
    Ffy[idt+1], Frx[idt+1], Fry[idt+1] = model.calc_forces(states[:, idt], inputs[:, idt])
    Ffy_preds[idt+1], Frx_preds[idt+1], Fry_preds[idt+1] = MODEL_BANK[current_model_idx].calc_forces(states[:, idt], inputs[:, idt], return_slip=False)

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
            effective_window = max(1, LookBack_W // 5)
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
            new_model_idx = np.argmin(avg_errors_phys)
            if new_model_idx != current_model_idx:
                current_model_idx = new_model_idx
                model_switches.append(idt)
                model_mses.append(avg_errors_phys[new_model_idx])
                chosen_models.append(current_model_idx)

        if ENABLE_ADAPTIVE_BANK and window_count_phys >= LookBack_W and idt > LookBack_W:
            current_time = idt * Ts
            if idt < STABILIZATION_STEPS:
                skip_update = True
            else:
                skip_update = False
            friction_active = (FRICTION_START < current_time < FRICTION_END) or \
                              (current_time > FRICTION_END and current_time < FRICTION_END + FRICTION_UPDATE_DELAY * Ts)

            effective_update_interval = UPDATE_INTERVAL
            effective_std_ratio = ADAPTIVE_SAMPLING_STD_RATIO
            aggressive = False
            if current_time > FRICTION_END and current_time < FRICTION_END + 5.0:
                effective_update_interval = 20
                effective_std_ratio = 0.35
                aggressive = True
            elif current_time > FRICTION_END and current_time < FRICTION_END + 10.0:
                effective_update_interval = 40
                effective_std_ratio = 0.25
                aggressive = True

            if not skip_update and not friction_active and (idt - last_update_time) >= effective_update_interval:
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

        window = smoothing_mu
        if len(Drs_preds) >= window:
            avg_Dr = np.mean(Drs_preds[-window:])
            avg_Df = np.mean(Dfs_preds[-window:])
        else:
            avg_Dr = np.mean(Drs_preds)
            avg_Df = np.mean(Dfs_preds)

        mu_raw = (avg_Dr + avg_Df) / (9.81 * params_veh['mass'])
        mu_smoothed = mu_smoother.update(mu_raw)
        mu_final = mu_smoothed * 0.95
        MU_preds.append(mu_final)

    if args.enable_iterative and ITERATIVE_AVAILABLE and laps_completed >= args.value_update_interval:
        if laps_completed % args.value_update_interval == 0 and len(lap_data_frenet) > 0:
            print(f"🎓 Training value function at lap {laps_completed} with {len(lap_data_frenet)} laps of data.")
            X_train = np.concatenate(lap_data_frenet, axis=0)
            y_train = np.concatenate(lap_data_remaining, axis=0)
            train_data = np.concatenate([X_train, y_train], axis=1)
            train_safe_set = [train_data]
            value_fn.train_valuefn(train_safe_set, iterative_config, None, max_epoch=args.value_max_epoch)
            safe_set.add_lap(y_train.flatten(), X_train)
            if safe_set.get_new_safe_set():
                safe_set.update_ss_arr()
                if safe_set.xSS_hull_equations is not None:
                    A_hull = np.array(safe_set.xSS_hull_equations[:, :-1])
                    b_hull = -np.array(safe_set.xSS_hull_equations[:, -1:])
                    print(f"🛡️ Safe set updated. New convex hull with {safe_set.xSS_hull_equations.shape[0]} facets.")
                    invalidate_nlp_cache()
            lap_data_frenet = []
            lap_data_remaining = []

    hstates[:,0] = x0
    hstates2[:,0] = x0
    for idh in range(horizon):
        x_next_pred, _ = model.sim_continuous(hstates[:, idh], umpc[:, idh].reshape(-1,1), [0, Ts])
        hstates[:, idh+1] = x_next_pred[:, -1]
        hstates2[:, idh+1] = xmpc[:, idh+1]
    Hs0.append(hstates[0].copy())
    Hs1.append(hstates[1].copy())
    Hs0_2.append(hstates2[0].copy())
    Hs1_2.append(hstates2[1].copy())

    iter_wall_end = tm.time()
    step_wall_times.append(iter_wall_end - iter_wall_start)
    current_mse = np.mean(error_windows_phys[current_model_idx])*10000 if current_model_idx>=0 and window_count_phys>0 else 0.0
    print(f"iter: {idt}, model: {current_model_idx}, cost: {fval:.5f}, solve_time: {solve_end-solve_start:.2f}, "
          f"step_time: {iter_wall_end-iter_wall_start:.2f}, mse*10000: {current_mse:.3e}")

loop_wall_end = tm.time()

# ============================================================================
if SAVE_PLOTS:
    media_dir = "EvoBank-MPC"
    os.makedirs(media_dir, exist_ok=True)

    plt.figure(figsize=(6.4, 2.4))
    time_steps = int(SIM_TIME / Ts)
    model_usage = np.ones((N_MODELS, time_steps))
    thickness = 12
    for t in range(time_steps):
        if t in np.array(model_switches):
            idx = model_switches.index(t)
            if chosen_models[idx] >= 0:
                row = chosen_models[idx]
                for offset in range(-thickness//2, thickness//2+1):
                    r = row + offset
                    if 0 <= r < N_MODELS:
                        model_usage[r, t:] = 0
                if idx > 0 and chosen_models[idx-1] >= 0:
                    prev = chosen_models[idx-1]
                    for offset in range(-thickness//2, thickness//2+1):
                        r = prev + offset
                        if 0 <= r < N_MODELS:
                            model_usage[r, t:] = 1
    plt.imshow(model_usage, aspect='auto', cmap='binary', extent=[0, SIM_TIME, -0.5, N_MODELS-0.5], interpolation='none')
    cbar = plt.colorbar()
    cbar.set_ticks([0,1]); cbar.set_ticklabels(['Chosen','Not Chosen'])
    plt.xlabel(r'Time [$\mathrm{s}$]'); plt.ylabel('Model Index')
    plt.tight_layout()
    plt.savefig(media_dir+'/Switching_Grid_evo.png', dpi=400)

    vel = np.sqrt(dstates[0,:]**2 + dstates[1,:]**2)
    plt.figure(figsize=(6.4,2.4))
    plt.plot(time[:n_steps-horizon], ref_speeds, color="#E5AE1C", lw=4, label='Reference')
    plt.plot(time[:n_steps-horizon], vel[:n_steps-horizon], color="#0B67B2", lw=4, label='Actual')
    plt.xlabel(r'Time [$\mathrm{s}$]'); plt.ylabel(r'Speed [$\frac{ \mathrm{m} }{\mathrm{s}}$]')
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(media_dir+'/Speeds_evo.png', dpi=400, bbox_inches="tight")

    plt.figure()
    plt.plot(time[:n_steps-horizon], inputs[0,:n_steps-horizon], color="#0B67B2", lw=4)
    plt.xlabel(r'Time [$\mathrm{s}$]'); plt.ylabel('PWM duty cycle [-]')
    plt.grid(True); plt.tight_layout()
    plt.savefig(media_dir+'/Acc_evo.png', dpi=400, bbox_inches="tight")

    plt.figure()
    plt.plot(time[:n_steps-horizon], inputs[1,:n_steps-horizon], color="#0B67B2", lw=4)
    plt.xlabel(r'Time [$\mathrm{s}$]'); plt.ylabel(r'Steering [$\mathrm{rad}$]')
    plt.grid(True); plt.tight_layout()
    plt.savefig(media_dir+'/steering_evo.png', dpi=400, bbox_inches="tight")

    plt.figure()
    plt.plot(time[:n_steps-horizon], states[2,:n_steps-horizon], color="#0B67B2", lw=4)
    plt.xlabel(r'Time [$\mathrm{s}$]'); plt.ylabel(r'Orientation [$\mathrm{rad}$]')
    plt.grid(True); plt.tight_layout()
    plt.savefig(media_dir+'/orientation_evo.png', dpi=400, bbox_inches="tight")

    plt.figure()
    plot_len = min(len(time[:n_steps-horizon]), len(Dfs), len(Dfs_preds))
    plt.plot(time[:plot_len], Dfs[:plot_len], color="#0B67B2", lw=4, linestyle="--", label='Ground Truth Df')
    plt.plot(time[:plot_len], Drs[:plot_len], color="#D44A1C", lw=4, linestyle="--", label='Ground Truth Dr')
    plt.plot(time[:plot_len], Dfs_preds[:plot_len], color="#0B67B2", lw=4, linestyle="-", label='Predicted Df')
    plt.plot(time[:plot_len], Drs_preds[:plot_len], color="#D44A1C", lw=4, linestyle="-", label='Predicted Dr')
    plt.xlabel(r'Time [$\mathrm{s}$]'); plt.ylabel(r'$\mu$ $\mathrm{N}$ [$\mathrm{N}$]')
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(media_dir+'/Ds_evo.png', dpi=400, bbox_inches="tight")

    plt.figure(figsize=(6.4,2.4))
    plot_len = min(len(time[:n_steps-horizon]), len(MUs), len(MU_preds))
    plt.plot(time[:plot_len], MUs[:plot_len], color="#E5AE1C", lw=4, label=r"Ground Truth")
    plt.plot(time[:plot_len], MU_preds[:plot_len], color="#0B67B2", lw=4, label=r"Predicted")
    plt.grid(True)
    plt.xlabel(r'Time [$\mathrm{s}$]')
    plt.ylabel(r'$\mu$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(media_dir+'/MUs_evo.png', dpi=400, bbox_inches="tight")

    np.save(media_dir+'/Time.npy', time[:n_steps-horizon])
    np.save(media_dir+'/MUs.npy', MUs)
    np.save(media_dir+'/MU_preds.npy', MU_preds)

    fig_track = track.plot(color='k', grid=False)
    ax = plt.gca()
    points = np.array([states[0, :n_steps-horizon], states[1, :n_steps-horizon]]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = colors.Normalize(vmin=vel[:n_steps-horizon].min(), vmax=vel[:n_steps-horizon].max())
    custom_cmap = LinearSegmentedColormap.from_list("custom_speed", ['navy','blue','orange','yellow'])
    lc = LineCollection(segments, cmap=custom_cmap, norm=norm, lw=1.5, alpha=0.5)
    lc.set_array(vel[:n_steps-horizon-1])
    ax.add_collection(lc)
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    cbar = plt.colorbar(sm, orientation='vertical', ax=ax)
    cbar.set_label(r'Speed [${ \mathrm{m} }/{\mathrm{s}}$]')
    ax.set_axis_off()
    plt.axis('equal')
    plt.tight_layout(pad=0)
    plt.subplots_adjust(left=0, right=0.8, top=1, bottom=0)
    plt.savefig(media_dir+'/Traj_Velocity_evo.png', dpi=400, bbox_inches="tight")

    if SAVE_VIDEO:
        fig_track_video = track.plot(color='k', grid=False)
        ax_video = plt.gca()
        points_video = np.array([states[0, :], states[1, :]]).T.reshape(-1,1,2)
        segments_video = np.concatenate([points_video[:-1], points_video[1:]], axis=1)
        lc_video = LineCollection(segments_video, cmap=custom_cmap, norm=norm, lw=1.5, alpha=0.5)
        lc_video.set_array(vel[:-1])
        ax_video.add_collection(lc_video)
        H = .1; W = .05
        dims = np.array([[-H/2.,-W/2.],[-H/2.,W/2.],[H/2.,W/2.],[H/2.,-W/2.],[-H/2.,-W/2.]])
        LnP, = ax_video.plot([], [], 'red', alpha=0.8, label='Current pose')
        LnH, = ax_video.plot([], [], '-g', marker='o', markersize=.5, lw=0.5, label="ground truth")
        LnH2, = ax_video.plot([], [], '-b', marker='o', markersize=.5, lw=0.5, label="prediction")
        sm_video = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
        cbar_video = plt.colorbar(sm_video, orientation='vertical', ax=ax_video)
        cbar_video.set_label(r'Speed [${ \mathrm{m} }/{\mathrm{s}}$]')
        ax_video.set_axis_off()
        plt.axis('equal')
        plt.legend()

        def update_frame(idt):
            new_points = np.array([states[0, :idt+1], states[1, :idt+1]]).T.reshape(-1,1,2)
            new_segments = np.concatenate([new_points[:-1], new_points[1:]], axis=1)
            lc_video.set_segments(new_segments)
            lc_video.set_array(vel[:idt])
            LnP.set_data(states[0, idt] + dims[:,0]*np.cos(states[2, idt]) - dims[:,1]*np.sin(states[2, idt]),
                         states[1, idt] + dims[:,0]*np.sin(states[2, idt]) + dims[:,1]*np.cos(states[2, idt]))
            LnH.set_data(Hs0[idt], Hs1[idt])
            LnH2.set_data(Hs0_2[idt], Hs1_2[idt])
            ax_video.set_title(f"Frame {idt}")
            return lc_video, LnP, LnH, LnH2

        fps = 17
        interval = 1000 / fps
        frame_numbers = range(0, n_steps-horizon, 3)
        ani = animation.FuncAnimation(fig_track_video, update_frame, frames=frame_numbers,
                                      interval=interval, blit=True)
        video_path = f"{media_dir}/traj_video.mp4"
        ani.save(video_path, fps=fps, extra_args=['-vcodec', 'libx264', '-b:v', '2000k', '-preset', 'ultrafast'])
        print(f"🎥 Smooth video saved as {video_path}")

# ============================================================================
for i in range(len(lap_times)-1,0,-1):
    if lap_times[i] != 0.:
        lap_times[i] -= lap_times[i-1]
print("lap times:", lap_times[:laps_completed], "violation:", violation_total, "mean deviation:", np.mean(deviation))

mean_cost = np.mean(cost_history)

if len(Dfs) > 0 and len(Dfs_preds) > 0:
    df_rmse = np.sqrt(np.mean((np.array(Dfs) - np.array(Dfs_preds))**2))
    dr_rmse = np.sqrt(np.mean((np.array(Drs) - np.array(Drs_preds))**2))
    mu_rmse = np.sqrt(np.mean((np.array(MUs) - np.array(MU_preds))**2))
    time_array = np.array(time[:len(Dfs)])
    post_idx = np.where(time_array > FRICTION_END)[0]
    if len(post_idx) > 0:
        df_rmse_post = np.sqrt(np.mean((np.array(Dfs)[post_idx] - np.array(Dfs_preds)[post_idx])**2))
        dr_rmse_post = np.sqrt(np.mean((np.array(Drs)[post_idx] - np.array(Drs_preds)[post_idx])**2))
        mu_rmse_post = np.sqrt(np.mean((np.array(MUs)[post_idx] - np.array(MU_preds)[post_idx])**2))
    else:
        df_rmse_post = dr_rmse_post = mu_rmse_post = 0.0
else:
    df_rmse = dr_rmse = mu_rmse = 0.0
    df_rmse_post = dr_rmse_post = mu_rmse_post = 0.0


def summarize_runtime_stats(values):
    if len(values) == 0:
        return {
            'mean': 0.0,
            'median': 0.0,
            'p95': 0.0,
            'max': 0.0,
            'min': 0.0,
        }
    arr = np.array(values, dtype=float)
    return {
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'p95': float(np.percentile(arr, 95)),
        'max': float(np.max(arr)),
        'min': float(np.min(arr)),
    }

setup_runtime_sec = float(setup_wall_end - script_wall_start)
loop_runtime_sec = float(loop_wall_end - loop_wall_start)
total_runtime_sec = float(tm.time() - script_wall_start)
step_runtime_stats = summarize_runtime_stats(step_wall_times)
nlp_runtime_stats = summarize_runtime_stats(nlp_solve_times)
simulated_duration_sec = float((n_steps - horizon) * Ts)
steps_per_second = float((len(step_wall_times) / loop_runtime_sec) if loop_runtime_sec > 0 else 0.0)
simulated_seconds_per_wall_second = float(steps_per_second * 0.02)

print("========== Runtime Summary ==========")
print(f"setup_runtime_sec: {setup_runtime_sec:.4f}")
print(f"loop_runtime_sec: {loop_runtime_sec:.4f}")
print(f"total_runtime_sec: {total_runtime_sec:.4f}")
print(f"step_mean_sec: {step_runtime_stats['mean']:.6f}, step_median_sec: {step_runtime_stats['median']:.6f}, step_p95_sec: {step_runtime_stats['p95']:.6f}")
print(f"nlp_mean_sec: {nlp_runtime_stats['mean']:.6f}, nlp_median_sec: {nlp_runtime_stats['median']:.6f}, nlp_p95_sec: {nlp_runtime_stats['p95']:.6f}")
print(f"steps_per_second: {steps_per_second:.3f}")
print(f"simulated_seconds_per_wall_second: {simulated_seconds_per_wall_second:.3f}")

results = {
    'lap_times': lap_times[:laps_completed],
    'mean_deviation': float(np.mean(deviation)),
    'mean_cost': float(mean_cost),
    'total_lap_time': float(np.sum(lap_times[:laps_completed])),
    'violation_time': float(violation_total),
    'df_rmse': float(df_rmse),
    'dr_rmse': float(dr_rmse),
    'mu_rmse': float(mu_rmse),
    'df_rmse_post': float(df_rmse_post),
    'dr_rmse_post': float(dr_rmse_post),
    'mu_rmse_post': float(mu_rmse_post),
    'runtime_setup_sec': float(setup_runtime_sec),
    'runtime_loop_sec': float(loop_runtime_sec),
    'runtime_total_sec': float(total_runtime_sec),
    'runtime_step_mean_sec': float(step_runtime_stats['mean']),
    'runtime_step_median_sec': float(step_runtime_stats['median']),
    'runtime_step_p95_sec': float(step_runtime_stats['p95']),
    'runtime_step_max_sec': float(step_runtime_stats['max']),
    'runtime_nlp_mean_sec': float(nlp_runtime_stats['mean']),
    'runtime_nlp_median_sec': float(nlp_runtime_stats['median']),
    'runtime_nlp_p95_sec': float(nlp_runtime_stats['p95']),
    'runtime_nlp_max_sec': float(nlp_runtime_stats['max']),
    'steps_per_second': float(steps_per_second),
    'simulated_duration_sec': float(simulated_duration_sec),
    'simulated_seconds_per_wall_second': float(simulated_seconds_per_wall_second),
}

if args.output_file:
    with open(args.output_file, 'w') as f:
        json.dump(results, f)

    runtime_txt_file = os.path.splitext(args.output_file)[0] + '_runtime.txt'
    with open(runtime_txt_file, 'w') as f:
        f.write('========== Runtime Summary ==========\n')
        f.write(f"setup_runtime_sec: {setup_runtime_sec:.6f}\n")
        f.write(f"loop_runtime_sec: {loop_runtime_sec:.6f}\n")
        f.write(f"total_runtime_sec: {total_runtime_sec:.6f}\n")
        f.write(f"step_mean_sec: {step_runtime_stats['mean']:.6f}\n")
        f.write(f"step_median_sec: {step_runtime_stats['median']:.6f}\n")
        f.write(f"step_p95_sec: {step_runtime_stats['p95']:.6f}\n")
        f.write(f"step_max_sec: {step_runtime_stats['max']:.6f}\n")
        f.write(f"nlp_mean_sec: {nlp_runtime_stats['mean']:.6f}\n")
        f.write(f"nlp_median_sec: {nlp_runtime_stats['median']:.6f}\n")
        f.write(f"nlp_p95_sec: {nlp_runtime_stats['p95']:.6f}\n")
        f.write(f"nlp_max_sec: {nlp_runtime_stats['max']:.6f}\n")
        f.write(f"steps_per_second: {steps_per_second:.6f}\n")
        f.write(f"simulated_duration_sec: {simulated_duration_sec:.6f}\n")
        f.write(f"simulated_seconds_per_wall_second: {simulated_seconds_per_wall_second:.6f}\n")
    print(f"Runtime summary saved to {runtime_txt_file}")

print(json.dumps(results))