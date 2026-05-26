"""	Look-back and Look-ahead Adaptive MPC.
"""

__author__ = 'Maitham AL-Sunni'
__email__ = 'maitham@cmu.edu'

import time as tm
import numpy as np
import casadi
import _pickle as pickle
import matplotlib
from matplotlib.lines import lineStyles
matplotlib.use('Agg')  # Set non-GUI backend before importing pyplot
# matplotlib.use("pgf")  # Uses a LaTeX-compatible backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import subprocess
from matplotlib.gridspec import GridSpec
from joblib import Parallel, delayed
from evobankmpc.params import ORCA
from evobankmpc.models import Dynamic
from evobankmpc.tracks import ETHZ, ETHZMobil
from evobankmpc.mpc.planner import ConstantSpeed
from evobankmpc.mpc.nmpc import setupNLP
from evobankmpc.mpc.constraints import Boundary
import multiprocessing as mp
from evobankmpc.mpc.evaluate_models_vectorized import evaluate_models_vectorized
import os
import imageio
import copy
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
import matplotlib.pylab as pylab


params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)
plt.rcParams['text.usetex'] = True


#####################################################################
# Settings

# Make it "True"/"False" if you want to save a video/photo.
SAVE_VIDEO = True

# GENERAL
SIM_TIME = 36
SAMPLING_TIME = 0.02

# MPC
HORIZON = 20
COST_Q = np.diag([1, 1])
COST_P = np.diag([0, 0])
COST_R = np.diag([5/1000, 1])
TRACK_CONS = False

# LLA-MPC BANK AND MU ESTIMATION/SMOOTHING
N_MODELS = 5000  # number of models in the bank
LookBack_W = 10  # Number of steps to accumulate error
smoothing_mu = 20 # moving avg for MUs
smoothing_mu_over_mod = 10 # getting the avg MUs for the best N models
v_factor = .9 # For Plannning
mu_init = 1.
mu_alpha = 0.08
MODEL_BANK = []
MODEL_PARAMS = []


# load vehicle parameters
params = ORCA(control='pwm')
true_model = Dynamic(**params)
model = Dynamic(**params)

# Initializations #1
error_windows = np.zeros((N_MODELS, LookBack_W))
window_count = 0
laps_completed = 0
lap_times = [0.,0.,0.,0.,0.]
Drs = []
Dfs = []
Drs_preds = []
Dfs_preds = []
MUs = []
MU_preds = []
violation_total = []
deviation = []

# Initializations #2
Ts = SAMPLING_TIME
n_steps = int(SIM_TIME/Ts)
n_states = model.n_states
n_inputs = model.n_inputs
horizon = HORIZON

class ExponentialSmoother:
    def __init__(self, alpha=0.3):  # Alpha controls smoothness (0.1-0.3 recommended)
        self.alpha = alpha
        self.smooth_value = None

    def update(self, new_value):
        if self.smooth_value is None:
            self.smooth_value = new_value  # Initialize with first value
        else:
            self.smooth_value = self.alpha * new_value + (1 - self.alpha) * self.smooth_value
        return self.smooth_value  # Return smoothed value

def find_closest_point(x, y, raceline):
    # Get raceline coordinates
    x_refs = raceline[0]
    y_refs = raceline[1]
    # Calculate distances to all points
    distances = np.sqrt((x_refs - x) ** 2 + (y_refs - y) ** 2)
    # Find index of minimum distance
    idx = np.argmin(distances)
    return idx, x_refs[idx], y_refs[idx], distances[idx]

def update_friction(Df,Dr,curr_time,style='sudden') :
    if style is 'const_decay' :
        if curr_time > 14.3: # For ETHZ
        # if curr_time > 5: # For ETHZMobil
            Df -= Df/2600.
            Dr -= Dr/2600.
    elif style is 'sudden' :
        # if curr_time > 3.3 and curr_time < 3.5: # For ETHZ
        if curr_time > 14.3 and curr_time < 14.5: # For ETHZ

        # if curr_time > 5 and curr_time < 5.2: # For ETHZMobil
        # if curr_time > 10 and curr_time < 10.2: # For ETHZMobil
            Df -= Df/22.
            Dr -= Dr/22.
    elif style is 'no_change' :
        return Df, Dr
    return Df, Dr


# Define variation percentages for each parameter
variation_dict = {
    # 'mass': 0.15,        # 20% variation
    # 'lf': 0.15,          # 10% variation
    # 'lr': 0.15,          # 10% variation
    # 'Cm1': 0.15,         # 15% variation
    # 'Cm2': 0.15,         # 15% variation
    # 'Cr0': 0.15,         # 15% variation
    # 'Cr2': 0.15,         # 15% variation
    'Br': 0.2,          # 15% variation
    'Cr': 0.1,         # 15% variation
    'Dr': 0.5,          # 15% variation
    'Bf': 0.2,          # 15% variation
    'Cf': 0.1,          # 15% variation
    'Df': 0.5,          # 15% variation
}

# Create model bank with parameter variations
for i in range(N_MODELS):
    param_variation = params.copy()

    for param_name, variation_percentage in variation_dict.items():
        if param_name in param_variation:
            param_variation[param_name] *= (1 + variation_percentage * np.random.randn())

    MODEL_PARAMS.append(param_variation)
    MODEL_BANK.append(Dynamic(**param_variation))

Bfs_pass = np.array([m.Bf for m in MODEL_BANK])
Cfs_pass = np.array([m.Cf for m in MODEL_BANK])
Dfs_pass = np.array([m.Df for m in MODEL_BANK])
Brs_pass = np.array([m.Br for m in MODEL_BANK])
Crs_pass = np.array([m.Cr for m in MODEL_BANK])
Drs_pass = np.array([m.Dr for m in MODEL_BANK])

params_pass = (Bfs_pass, Cfs_pass, Dfs_pass, Brs_pass, Crs_pass, Drs_pass)

# tune it!!!!!!!!!!!
# 0.2 for sudden is good (but osc.). 0.03 is smooth but slower. 0.06 is faster a lil bit.
smoother = ExponentialSmoother(alpha=mu_alpha)

#####################################################################
# load track

TRACK_NAME = 'ETHZ'
track = ETHZ(reference='optimal', longer=True)


# Setup NLP solvers for all models

print("Setting up NLP solvers for all models...")
nlp_bank = []
for i in range(N_MODELS):
    print("setting up NLP # {}".format(i))
    nlp = setupNLP(horizon, Ts, COST_Q, COST_P, COST_R,
                  MODEL_PARAMS[i],
                  MODEL_BANK[i],
                  track, track_cons=TRACK_CONS)
    nlp_bank.append(nlp)
print("NLP setup complete.")

# just begainning

nlp_initial = setupNLP(horizon, Ts, COST_Q, COST_P, COST_R, params, true_model, track, track_cons=TRACK_CONS)

# nlp_initial = nlp_bank[0]

#####################################################################
# closed-loop simulation

# initialize
ref_speeds = []

states = np.zeros([n_states, n_steps+1])
dstates = np.zeros([n_states, n_steps+1])
inputs = np.zeros([n_inputs, n_steps])
time = np.linspace(0, n_steps, n_steps+1)*Ts

Ffy = np.zeros([n_steps+1])
Frx = np.zeros([n_steps+1])
Fry = np.zeros([n_steps+1])

Ffy_preds = np.zeros([n_steps+1])
Frx_preds = np.zeros([n_steps+1])
Fry_preds = np.zeros([n_steps+1])

hstates = np.zeros([n_states,horizon+1])
hstates2 = np.zeros([n_states,horizon+1])

Hs0 = []
Hs1 = []

Hs0_2 = []
Hs1_2 = []

# Arrays to store model performance metrics
model_switches = []  # Store timesteps when model switches occur
model_mses = []     # Store MSE values at switches
chosen_models = []  # Store which model was chosen

projidx = 0
x_init = np.zeros(n_states)
x_init[0], x_init[1] = track.x_init, track.y_init
x_init[2] = track.psi_init
x_init[3] = track.vx_init
dstates[0,0] = x_init[3]
states[:,0] = x_init
print('starting at ({:.1f},{:.1f})'.format(x_init[0], x_init[1]))


media_dir = "LLA-MPC"
os.makedirs(media_dir, exist_ok=True)

# dynamic plot
H = .1
W = .05
dims = np.array([[-H/2.,-W/2.],[-H/2.,W/2.],[H/2.,W/2.],[H/2.,-W/2.],[-H/2.,-W/2.]])


# Initialize model selection variables
current_model_idx = 0  # Start with nominal model
uprev = np.zeros(n_inputs)  # Initialize uprev


# main simulation loop
for idt in range(n_steps-horizon):
    print("checkiter", id(model))
    x0 = states[:,idt]

    #  "const_decay" or "sudden" or "no_change"
    model.Df, model.Dr = update_friction(model.Df, model.Dr, idt * Ts)
    params['Df'], params['Dr'] = model.Df, model.Dr

    # planner based on BayesOpt
    if idt > LookBack_W+1:
        xref, projidx, v = ConstantSpeed(x0=x0[:2], v0=x0[3], track=track, N=horizon, Ts=Ts, projidx=projidx,
                                         curr_mu=MU_pred, scale=v_factor)
    else:
        xref, projidx, v = ConstantSpeed(x0=x0[:2], v0=x0[3], track=track, N=horizon, Ts=Ts, projidx=projidx)

    ref_speeds.append(v)
    deviation.append(find_closest_point(x0[0], x0[1], track.raceline)[-1])

    if projidx > 656:
    # if projidx > 440:
        if laps_completed > 0:
            lap_times[laps_completed] = idt * Ts
            print(lap_times)
        else:
            lap_times[laps_completed] = idt * Ts
            print(lap_times)
        laps_completed += 1
        projidx = 0

    start = tm.time()

    if idt <= LookBack_W:
        nlp = nlp_initial
    else:
        nlp = nlp_bank[current_model_idx]

    umpc, fval, xmpc, violation = nlp.solve(x0=x0, xref=xref[:2,:], uprev=uprev)

    inputs[:,idt] = umpc[:,0]
    uprev = inputs[:,idt]  # Update uprev for next iteration

    # update current position with numerical integration (exact model)
    x_next, dxdt_next = model.sim_continuous(states[:,idt], inputs[:,idt].reshape(-1,1), [0, Ts])

    # Uncomment if violation time is needed to be calculated
    # Ain, bin = Boundary(np.array(x_next[:2, -1]), track)
    # flag = (np.array(Ain @ np.array(x_next[:2, -1])).T > np.array(np.array([bin[0][0], bin[1][0]]))).any()
    # violation_total.append(flag * 0.02)

    states[:,idt+1] = x_next[:,-1]
    dstates[:,idt+1] = dxdt_next[:,-1]
    Ffy[idt+1], Frx[idt+1], Fry[idt+1] = model.calc_forces(states[:,idt], inputs[:,idt])
    Ffy_preds[idt+1], Frx_preds[idt+1], Fry_preds[idt+1] = MODEL_BANK[current_model_idx].calc_forces(states[:,idt], inputs[:,idt], return_slip=False)

    Drs.append(model.Dr)
    Dfs.append(model.Df)

    if idt <= LookBack_W:
        Drs_preds.append(mu_init * params['mass'] * 9.8 * params['lr'] / (params['lf'] + params['lr']))
        Dfs_preds.append(mu_init * params['mass'] * 9.8 * params['lf'] / (params['lf'] + params['lr']))
        MUs.append(copy.deepcopy((model.Df + model.Dr) / (9.81 * params['mass'])))
        MU_preds.append(copy.deepcopy(mu_init))
    else:
        MUs.append(copy.deepcopy((model.Df + model.Dr) / (9.81 * params['mass'])))

        bestKDr = []
        bestKDf = []
        for best_ind in ind_best_KM:
            bestKDr.append(copy.deepcopy(MODEL_BANK[best_ind].Dr))
            bestKDf.append(copy.deepcopy(MODEL_BANK[best_ind].Df))
        Drs_preds.append(np.mean(bestKDr))
        Dfs_preds.append(np.mean(bestKDf))
        MU_pred = copy.deepcopy(np.mean(np.array(Drs_preds)[-smoothing_mu:])  + np.mean(np.array(Dfs_preds)[-smoothing_mu:]) ) / (9.81 * params['mass'])

        # .95 workes perfectly.
        MU_preds.append(smoother.update(MU_pred)*.95)

    # In your main loop:
    if idt > 0:
        # Calculate errors for all models at once
        errors = np.mean((evaluate_models_vectorized(MODEL_BANK, N_MODELS, states[:, idt], inputs[:, idt], Ts, params_pass) - states[0:4, idt + 1]) ** 2, axis=1)

        # Update error windows using numpy operations
        error_windows = np.roll(error_windows, -1, axis=1)
        error_windows[:, -1] = errors
        window_count = min(window_count + 1, LookBack_W)

        # Make decision if we have enough data points
        if window_count >= LookBack_W:
            avg_errors = np.mean(error_windows, axis=1)
            new_model_idx = np.argmin(avg_errors)
            ind_best_KM = avg_errors.argsort()[:smoothing_mu_over_mod]

            if new_model_idx != current_model_idx:
                current_model_idx = new_model_idx
                model_switches.append(idt)
                model_mses.append(avg_errors[new_model_idx])
                chosen_models.append(current_model_idx)

    # forward sim to predict over the horizon
    hstates[:,0] = x0
    hstates2[:,0] = x0
    for idh in range(horizon):
        x_next, dxdt_next = model.sim_continuous(hstates[:,idh], umpc[:,idh].reshape(-1,1), [0, Ts])
        hstates[:,idh+1] = x_next[:,-1]
        hstates2[:,idh+1] = xmpc[:,idh+1]

    Hs0.append(copy.deepcopy(hstates[0]))
    Hs1.append(copy.deepcopy(hstates[1]))

    Hs0_2.append(copy.deepcopy(hstates2[0]))
    Hs1_2.append(copy.deepcopy(hstates2[1]))

    end = tm.time()
    print("iter: {}, model: {}, cost: {:.5f}, time: {:.2f}, mse*10000: {:.5f}".format(
        idt,
        current_model_idx,
        fval,
        end - start,
        np.mean(error_windows[current_model_idx])*10000 if window_count > 0 else 0.0
    ))

#####################################################################
# SAVE FIGURES AND VIDEOS

# Create binary grid visualization of model switching
plt.figure(figsize=(6.4, 2.4))

# Create time bins
time_steps = int(SIM_TIME / Ts)
model_usage = np.ones((N_MODELS, time_steps))

# Line thickness (number of rows to fill)
thickness = 12  # Adjust this value to make lines thicker or thinner

# Fill in the grid
for t in range(time_steps):
    if t in np.array(model_switches):
        idx = model_switches.index(t)
        # Fill current model and adjacent rows
        for offset in range(-thickness // 2, thickness // 2 + 1):
            row = chosen_models[idx] + offset
            if 0 <= row < N_MODELS:  # Make sure we stay within bounds
                model_usage[row, t:] = 0

        # Clear previous model's usage
        if idx > 0:
            prev_model = chosen_models[idx - 1]
            for offset in range(-thickness // 2, thickness // 2 + 1):
                row = prev_model + offset
                if 0 <= row < N_MODELS:
                    model_usage[row, t:] = 1

# Plot the binary grid
plt.imshow(model_usage,
           aspect='auto',
           cmap='binary',
           extent=[0, SIM_TIME, -0.5, N_MODELS - 0.5],
           interpolation='none')

# Customize colorbar
cbar = plt.colorbar()
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['Chosen', 'Not Chosen'])

plt.xlabel(r'Time [$\mathrm{s}$]')
plt.ylabel('Model Index')
plt.tight_layout()
plt.savefig(media_dir + '/Switching_Grid.png', dpi=400)

# plot speed
plt.figure(figsize=(6.4, 2.4))
vel = np.sqrt(dstates[0,:]**2 + dstates[1,:]**2)
plt.plot(time[:n_steps-horizon], ref_speeds,color="#E5AE1C",linewidth=4, label='Reference')
plt.plot(time[:n_steps-horizon], vel[:n_steps-horizon],color="#0B67B2",linewidth=4, label='Actual')
plt.xlabel(r'Time [$\mathrm{s}$]')
plt.ylabel(r'Speed [$\frac{ \mathrm{m} }{\mathrm{s}}$]')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(media_dir+'/Speeds.png', dpi=400, bbox_inches="tight")


# # plot acceleration
plt.figure()
plt.plot(time[:n_steps-horizon], inputs[0,:n_steps-horizon],color="#0B67B2",linewidth=4)
plt.xlabel(r'Time [$\mathrm{s}$]')
plt.ylabel('PWM duty cycle [-]')
plt.grid(True)
plt.tight_layout()
plt.savefig(media_dir+'/Acc.png', dpi=400, bbox_inches="tight")


# plot mus
plt.figure(figsize=(6.4, 2.4))
plt.plot(time[:n_steps-horizon],MUs,color="#E5AE1C",linewidth=4, label=r"Ground Truth")
plt.plot(time[:n_steps-horizon],MU_preds,color="#0B67B2",linewidth=4,  label=r"Predicted")
plt.grid(True)
plt.xlabel(r'Time [$\mathrm{s}$]')
plt.ylabel(r'$\mu$')
plt.legend()
plt.tight_layout()
plt.savefig(media_dir+'/MUs.png', dpi=400, bbox_inches="tight")

# np.save(media_dir+'/Time.npy', time[:n_steps-horizon])
# np.save(media_dir+'/MUs.npy', MUs)
# np.save(media_dir+'/MU_preds.npy', MU_preds)

# plot steering angle
plt.figure()
plt.plot(time[:n_steps-horizon], inputs[1,:n_steps-horizon],color="#0B67B2",linewidth=4)
plt.xlabel(r'Time [$\mathrm{s}$]')
plt.ylabel(r'Steering [$\mathrm{rad}$]')
plt.grid(True)
plt.tight_layout()
plt.savefig(media_dir+'/steering.png', dpi=400, bbox_inches="tight")


# plot inertial heading
plt.figure()
plt.plot(time[:n_steps-horizon], states[2,:n_steps-horizon],color="#0B67B2",linewidth=4)
plt.xlabel(r'Time [$\mathrm{s}$]')
plt.ylabel(r'Orientation [$\mathrm{rad}$]')
plt.grid(True)
plt.tight_layout()
plt.savefig(media_dir+'/orientation.png', dpi=400, bbox_inches="tight")


plt.figure()
plt.plot(time[:n_steps-horizon], Dfs[:n_steps-horizon],color="#0B67B2",linewidth=4,linestyle="--", label='Ground Truth Df')
plt.plot(time[:n_steps-horizon], Drs[:n_steps-horizon],color="#D44A1C",linewidth=4,linestyle="--", label='Ground Truth Dr')
plt.plot(time[:n_steps-horizon], Dfs_preds[:n_steps-horizon],color="#0B67B2",linewidth=4,linestyle="-", label='Predicted Df')
plt.plot(time[:n_steps-horizon], Drs_preds[:n_steps-horizon],color="#D44A1C",linewidth=4,linestyle="-", label='Predicted Dr')
plt.xlabel(r'Time [$\mathrm{s}$]')
plt.ylabel(r'$\mu$ $\mathrm{N}$ [$\mathrm{N}$]')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(media_dir+'/Ds.png', dpi=400, bbox_inches="tight")


# fig_track = track.plot(color='k', grid=False)
# # fig_track.set_dpi(50)
# plt.plot(track.x_raceline, track.y_raceline, '--k', alpha=0.5, lw=0.5)
# ax = plt.gca()
# LnS, = ax.plot(states[0,0], states[1,0], '#4B0082', label='Trajectory',alpha=0.65)
# xyproj, _ = track.project(x=x_init[0], y=x_init[1], raceline=track.raceline)
# LnP, = ax.plot(states[0,0] + dims[:,0]*np.cos(states[2,0]) - dims[:,1]*np.sin(states[2,0])\
# 		, states[1,0] + dims[:,0]*np.sin(states[2,0]) + dims[:,1]*np.cos(states[2,0]), 'red', alpha=0.8, label='Current pose')
# LnH, = ax.plot(hstates[0], hstates[1], '-g', marker='o', markersize=.5, lw=0.5, color='green', label="ground truth")
# LnH2, = ax.plot(hstates2[0], hstates2[1], '-b', marker='o', markersize=.5, lw=0.5, color='blue', label="prediction")
#
# # Remove axes, ticks, and frame
# ax.set_axis_off()
# plt.axis('equal')
# plt.axis('off')
#
# plt.legend()
#
# def update(idt):
#     if idt == 0:
#         fig_track.tight_layout(pad=0)
#
#
#     ax.set_title(f"Frame {idt}")
#
#     LnS.set_xdata(states[0, :idt + 1])
#     LnS.set_ydata(states[1, :idt + 1])
#
#     LnP.set_xdata(states[0, idt] + dims[:, 0] * np.cos(states[2, idt]) - dims[:, 1] * np.sin(states[2, idt]))
#     LnP.set_ydata(states[1, idt] + dims[:, 0] * np.sin(states[2, idt]) + dims[:, 1] * np.cos(states[2, idt]))
#
#     LnH.set_xdata(Hs0[idt])
#     LnH.set_ydata(Hs1[idt])
#
#     LnH2.set_xdata(Hs0_2[idt])
#     LnH2.set_ydata(Hs1_2[idt])
#
#     return LnS, LnP, LnH, LnH2
#
# if SAVE_VIDEO:
#     fps = 50
#     interval = 1000 / fps  # Convert fps to milliseconds
#     ani = animation.FuncAnimation(fig_track, update, frames=int((n_steps-horizon)/3), interval=interval, blit=True)
#     video_path = f"{media_dir}/traj_video.mp4"
#     # fig_track.tight_layout()
#     ani.save(video_path, fps=fps, extra_args=['-vcodec', 'h264_videotoolbox', '-b:v', '2000k', '-preset', 'ultrafast'])
#     print(f"🎥 Smooth video saved as {video_path}")


# Assuming 'track', 'states', 'dims', 'x_init', 'hstates', 'hstates2', 'vel', 'n_steps', 'horizon' are defined elsewhere
# Custom colormap and normalization
colors_list = ['navy', 'blue', 'orange', 'yellow']
custom_cmap = LinearSegmentedColormap.from_list("custom_speed", colors_list)
norm = colors.Normalize(vmin=np.min(vel[:n_steps-horizon]), vmax=np.max(vel[:n_steps-horizon]))

# Create the track plot
fig_track = track.plot(color='k', grid=False)
ax = plt.gca()
# Create segments for the changing colors based on velocity
points = np.array([states[0, :n_steps-horizon], states[1, :n_steps-horizon]]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, cmap=custom_cmap, norm=norm, linewidth=1.5, alpha=0.5)
lc.set_array(vel[:n_steps-horizon-1])
ax.add_collection(lc)

# Additional lines
LnP, = ax.plot(states[0,0] + dims[:,0]*np.cos(states[2,0]) - dims[:,1]*np.sin(states[2,0]),
               states[1,0] + dims[:,0]*np.sin(states[2,0]) + dims[:,1]*np.cos(states[2,0]), 'red', alpha=0.8, label='Current pose')
LnH, = ax.plot(hstates[0], hstates[1], '-g', marker='o', markersize=.5, lw=0.5, color='green', label="ground truth")
LnH2, = ax.plot(hstates2[0], hstates2[1], '-b', marker='o', markersize=.5, lw=0.5, color='blue', label="prediction")

# Add a colorbar
sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, orientation='vertical')
cbar.set_label(r'Speed [${ \mathrm{m} }/{\mathrm{s}}$]')

# Remove axes, ticks, and frame
ax.set_axis_off()
plt.axis('equal')
plt.axis('off')
plt.legend()

def update(idt):
    if idt == 0:
        fig_track.tight_layout(pad=0)
    ax.set_title(f"Frame {idt}")
    # Update segments for dynamic coloring
    new_points = np.array([states[0, :idt+1], states[1, :idt+1]]).T.reshape(-1, 1, 2)
    new_segments = np.concatenate([new_points[:-1], new_points[1:]], axis=1)
    lc.set_segments(new_segments)
    lc.set_array(vel[:idt])

    LnP.set_xdata(states[0, idt] + dims[:, 0] * np.cos(states[2, idt]) - dims[:, 1] * np.sin(states[2, idt]))
    LnP.set_ydata(states[1, idt] + dims[:, 0] * np.sin(states[2, idt]) + dims[:, 1] * np.cos(states[2, idt]))

    LnH.set_xdata(Hs0[idt])
    LnH.set_ydata(Hs1[idt])

    LnH2.set_xdata(Hs0_2[idt])
    LnH2.set_ydata(Hs1_2[idt])

    return lc, LnP, LnH, LnH2

if SAVE_VIDEO:
    fps = 17
    interval = 1000 / fps  # Convert fps to milliseconds
    # Generate frame numbers skipping every 2 frames
    frame_numbers = range(0, n_steps-horizon,3)  # Adjust step to skip frames
    ani = animation.FuncAnimation(fig_track, update, frames=frame_numbers, interval=interval, blit=True)
    video_path = f"{media_dir}/traj_video.mp4"
    ani.save(video_path, fps=fps, extra_args=['-vcodec', 'h264_videotoolbox', '-b:v', '2000k', '-preset', 'ultrafast'])
    print(f"🎥 Smooth video saved as {video_path}")







# Create the track plot
fig_track = track.plot(color='k', grid=False)
# Create a custom colormap that goes from blue to purple to red to yellow
colors_list = ['navy', 'blue', 'orange','yellow']
custom_cmap = LinearSegmentedColormap.from_list("custom_speed", colors_list)
# Normalize velocity values for colormapping
norm = colors.Normalize(vmin=np.min(vel[:n_steps-horizon]),
                        vmax=np.max(vel[:n_steps-horizon]))
# Plot trajectory with changing colors based on velocity
points = np.array([states[0,:-(horizon)], states[1,:-(horizon)]]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
# Create a LineCollection with the colored segments
lc = LineCollection(segments, cmap=custom_cmap, norm=norm, linewidth=1.5, alpha=0.5)
lc.set_array(vel[:n_steps-horizon-1])
plt.gca().add_collection(lc)
# Add a colorbar
sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, orientation='vertical')
cbar.set_label(r'Speed [${ \mathrm{m} }/{\mathrm{s}}$]')
# Remove axes, ticks, and frame
ax.set_axis_off()
plt.axis('equal')
plt.axis('off')
# Remove the figure border
fig_track.tight_layout(pad=0)
fig_track.subplots_adjust(left=0, right=.8, top=1, bottom=0)
plt.savefig(media_dir+'/Traj_Velocity.png', dpi=400, bbox_inches="tight")


for i in range(len(lap_times)-1,0,-1) :
	if lap_times[i] != 0. :
		lap_times[i] = lap_times[i] - lap_times[i-1]
print(lap_times)
print("lap times:",lap_times, "violation: ",np.sum(violation_total), "mean deviation", np.mean(deviation))