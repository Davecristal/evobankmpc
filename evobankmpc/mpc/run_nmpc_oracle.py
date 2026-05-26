"""	Nonlinear MPC using dynamic bicycle model.
"""

fval_history = []
violation_total = []
from evobankmpc.mpc.constraints import Boundary
import copy

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


def find_closest_point(x, y, raceline):
	"""
    Find closest point on raceline to given (x,y) point

    Args:
        x, y: Query point coordinates
        raceline: Track raceline [x_refs, y_refs]

    Returns:
        idx: Index of closest point
        x_closest, y_closest: Coordinates of closest point
        distance: Distance to closest point
    """
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

import time as tm
import numpy as np
import casadi
import _pickle as pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from evobankmpc.params import ORCA
from evobankmpc.models import Dynamic
from evobankmpc.tracks import ETHZ, ETHZMobil
from evobankmpc.mpc.planner import ConstantSpeed
from evobankmpc.mpc.nmpc import setupNLP
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection


import os
import subprocess
import copy

import matplotlib
matplotlib.use('Agg')  # Set non-GUI backend before importing pyplot
# matplotlib.use("pgf")  # Uses a LaTeX-compatible backend

import matplotlib.pylab as pylab

import matplotlib.animation as animation

params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)
plt.rcParams['text.usetex'] = True


#####################################################################
# CHANGE THIS

SAVE_RESULTS = False
TRACK_CONS = False

#####################################################################
# default settings

SIM_TIME = 36
SAMPLING_TIME = 0.02
HORIZON = 20
COST_Q = np.diag([1, 1])
COST_P = np.diag([0, 0])
COST_R = np.diag([5/1000, 1])

v_factor = 1

if not TRACK_CONS:
	SUFFIX = 'NOCONS-'
else:
	SUFFIX = ''

#####################################################################
# load vehicle parameters

params = ORCA(control='pwm')
model = Dynamic(**params)

#####################################################################
# load track

TRACK_NAME = 'ETHZ'
track = ETHZ(reference='optimal', longer=True)

#####################################################################
# extract data

Ts = SAMPLING_TIME
n_steps = int(SIM_TIME/Ts)
n_states = model.n_states
n_inputs = model.n_inputs
horizon = HORIZON

#####################################################################
# define controller

nlp = setupNLP(horizon, Ts, COST_Q, COST_P, COST_R, params, model, track, track_cons=TRACK_CONS)

#####################################################################
# closed-loop simulation

# initialize
states = np.zeros([n_states, n_steps+1])
dstates = np.zeros([n_states, n_steps+1])
inputs = np.zeros([n_inputs, n_steps])
time = np.linspace(0, n_steps, n_steps+1)*Ts
Ffy = np.zeros([n_steps+1])
Frx = np.zeros([n_steps+1])
Fry = np.zeros([n_steps+1])
hstates = np.zeros([n_states,horizon+1])
hstates2 = np.zeros([n_states,horizon+1])

Hs0 = []
Hs1 = []

Hs0_2 = []
Hs1_2 = []

MU_preds = []

projidx = 0
x_init = np.zeros(n_states)
x_init[0], x_init[1] = track.x_init, track.y_init
x_init[2] = track.psi_init
x_init[3] = track.vx_init
dstates[0,0] = x_init[3]
states[:,0] = x_init
print('starting at ({:.1f},{:.1f})'.format(x_init[0], x_init[1]))


media_dir = "Oracle"
os.makedirs(media_dir, exist_ok=True)

# dynamic plot
H = .1
W = .05
dims = np.array([[-H/2.,-W/2.],[-H/2.,W/2.],[H/2.,W/2.],[H/2.,-W/2.],[-H/2.,-W/2.]])

fig_track = track.plot(color='k', grid=False)
fig_track.set_dpi(80)
plt.plot(track.x_raceline, track.y_raceline, '--k', alpha=0.5, lw=0.5)
ax = plt.gca()
LnS, = ax.plot(states[0,0], states[1,0], '#4B0082', label='Trajectory',alpha=0.65)
# LnR, = ax.plot(states[0,0], states[1,0], '-b', marker='o', markersize=.5, lw=0.5, label="reference")
xyproj, _ = track.project(x=x_init[0], y=x_init[1], raceline=track.raceline)
LnP, = ax.plot(states[0,0] + dims[:,0]*np.cos(states[2,0]) - dims[:,1]*np.sin(states[2,0])\
		, states[1,0] + dims[:,0]*np.sin(states[2,0]) + dims[:,1]*np.cos(states[2,0]), 'red', alpha=0.8)
LnH, = ax.plot(hstates[0], hstates[1], '-g', marker='o', markersize=.5, lw=0.5, color='green', label="Ground truth")
LnH2, = ax.plot(hstates2[0], hstates2[1], '-b', marker='o', markersize=.5, lw=0.5, color='blue', label="Prediction")
plt.xlabel(r'$x$ [$\mathrm{m}$]')
plt.ylabel(r'$y$ [$\mathrm{m}$]')
plt.legend()

ref_speeds = []
Drs = []
Dfs = []

laps_completed = 0
lap_times = [0.,0.,0.,0.,0.]

# main simulation loop
for idt in range(n_steps-horizon):

	uprev = inputs[:,idt-1]
	x0 = states[:,idt]
	Drs.append(model.Dr)
	Dfs.append(model.Df)

	model.Df, model.Dr = update_friction(model.Df, model.Dr, idt*Ts)
	params['Df'], params['Dr'] = model.Df, model.Dr

	MU = (model.Df+model.Dr)/(9.81*params['mass'])
	MU_preds.append(copy.deepcopy(MU))

	# planner based on BayesOpt
	if idt > 2 :
		xref, projidx, v = ConstantSpeed(x0=x0[:2], v0=x0[3], track=track, N=horizon, Ts=Ts, projidx=projidx, curr_mu=MU_preds[-1], scale= v_factor )
	else :
		xref, projidx, v = ConstantSpeed(x0=x0[:2], v0=x0[3], track=track, N=horizon, Ts=Ts, projidx=projidx)

	fval_history.append(find_closest_point(x0[0],x0[1], track.raceline)[-1])
	ref_speeds.append(v)

	if projidx > 656:
	# if projidx > 440:
		if laps_completed > 0:
			lap_times[laps_completed] = idt * Ts  # - lap_times[laps_completed - 1]
			print(lap_times)
		else:
			lap_times[laps_completed] = idt * Ts
			print(lap_times)
		laps_completed += 1
		projidx = 0

	# print(xref)
	start = tm.time()
	nlp = setupNLP(horizon, Ts, COST_Q, COST_P, COST_R, params, model, track, track_cons=TRACK_CONS)
	umpc, fval, xmpc, violation = nlp.solve(x0=x0, xref=xref[:2,:], uprev=uprev)

	# print(xmpc)
	end = tm.time()
	inputs[:,idt] = umpc[:,0]
	print("iter: {}, cost: {:.5f}, time: {:.2f}".format(idt, fval, end-start))

	# update current position with numerical integration (exact model)
	x_next, dxdt_next = model.sim_continuous(states[:,idt], inputs[:,idt].reshape(-1,1), [0, Ts])
	Ain, bin = Boundary(np.array(x_next[:2,-1]), track)
	flag = (np.array(Ain@np.array(x_next[:2,-1])).T > np.array(np.array( [bin[0][0], bin[1][0]] ) ) ).any()
	violation_total.append(flag*0.02 )

	states[:,idt+1] = x_next[:,-1]
	dstates[:,idt+1] = dxdt_next[:,-1]
	Ffy[idt+1], Frx[idt+1], Fry[idt+1] = model.calc_forces(states[:,idt], inputs[:,idt])

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


	mean_cost = np.mean(fval_history)
	print(f"Mean cost at iter {idt}: {mean_cost:.3f}")


# plots

# # plot speed
# plt.figure(figsize=(6.4, 2.4))
vel = np.sqrt(dstates[0,:]**2 + dstates[1,:]**2)
# plt.plot(time[:n_steps-horizon], ref_speeds,color="#E5AE1C",linewidth=4, label='Reference')
# plt.plot(time[:n_steps-horizon], vel[:n_steps-horizon],color="#0B67B2",linewidth=4, label='Actual')
# # plt.plot(time[:n_steps-horizon], states[3,:n_steps-horizon], label='vx')
# # plt.plot(time[:n_steps-horizon], states[4,:n_steps-horizon], label='vy')
# plt.xlabel(r'Time [$\mathrm{s}$]')
# plt.ylabel(r'Speed [$\frac{ \mathrm{m} }{\mathrm{s}}$]')
# plt.title('Speeds')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig(media_dir+'/Speeds.png', dpi=1200, bbox_inches="tight")
#


# Create the track plot
fig_track = track.plot(color='k', grid=False)
# Create a custom colormap that goes from blue to purple to red to yellow
# This should match the reference image better
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

# # plot acceleration
# plt.figure()
# plt.plot(time[:n_steps-horizon], inputs[0,:n_steps-horizon],color="#0B67B2",linewidth=4)
# plt.xlabel(r'Time [$\mathrm{s}$]')
# plt.ylabel('PWM duty cycle [-]')
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(media_dir+'/Acc.png', dpi=1200, bbox_inches="tight")


# # plot mus
# plt.figure(figsize=(6.4, 2.4))
# plt.plot(time[:n_steps-horizon],MU_preds,color="#E5AE1C",linewidth=4, label=r"Ground Truth")
# plt.grid(True)
# plt.xlabel(r'Time [$\mathrm{s}$]')
# plt.ylabel(r'$\mu$')
# plt.legend()
# plt.tight_layout()
# plt.savefig(media_dir+'/MUs.png', dpi=400, bbox_inches="tight")
#
# np.save(media_dir+'/Time.npy', time[:n_steps-horizon])
# np.save(media_dir+'/MU_preds.npy', MU_preds)

# # plot steering angle
# plt.figure()
# plt.plot(time[:n_steps-horizon], inputs[1,:n_steps-horizon],color="#0B67B2",linewidth=4)
# plt.xlabel(r'Time [$\mathrm{s}$]')
# plt.ylabel(r'Steering [$\mathrm{rad}$]')
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(media_dir+'/steering.png', dpi=1200, bbox_inches="tight")


# # plot inertial heading
# plt.figure()
# plt.plot(time[:n_steps-horizon], states[2,:n_steps-horizon],color="#0B67B2",linewidth=4)
# plt.xlabel(r'Time [$\mathrm{s}$]')
# plt.ylabel(r'Orientation [$\mathrm{rad}$]')
# plt.grid(True)
# plt.tight_layout()
# # plt.savefig(media_dir+'/orientation.png', dpi=1200, bbox_inches="tight")
#
#
# plt.figure()
# plt.plot(time[:n_steps-horizon], Dfs[:n_steps-horizon],color="#0B67B2",linewidth=4,linestyle="-", label='Ground Truth Df')
# plt.plot(time[:n_steps-horizon], Drs[:n_steps-horizon],color="#D44A1C",linewidth=4,linestyle="-", label='Ground Truth Dr')
# plt.xlabel(r'Time [$\mathrm{s}$]')
# plt.ylabel(r'$\mu$ $\mathrm{N}$ [$\mathrm{N}$]')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig(media_dir+'/Ds.png', dpi=400, bbox_inches="tight")

# fps = 50
# interval = 1000 / fps  # Convert fps to milliseconds
# ani = animation.FuncAnimation(fig_track, update, frames=n_steps-horizon, interval=interval, blit=True)
# video_path = f"{media_dir}/output_video.mp4"
# # fig_track.tight_layout()
# ani.save(video_path, fps=fps, extra_args=['-vcodec', 'h264_videotoolbox', '-b:v', '4000k', '-preset', 'ultrafast'])
# print(f"🎥 Smooth video saved as {video_path}")

# fig_track = track.plot(color='k', grid=False)
# plt.plot(track.x_raceline, track.y_raceline, '--k', alpha=0.5, lw=0.5)
# plt.plot(states[0,:-(horizon)], states[1,:-(horizon)],color="#4B0082",linewidth=3,linestyle="-")
# plt.xlabel(r'$x$ [$\mathrm{m}$]')
# plt.ylabel(r'$y$ [$\mathrm{m}$]')
# # plt.legend()
# plt.tight_layout()
# plt.savefig(media_dir+'/Traj.png', dpi=1200, bbox_inches="tight")


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

SAVE_VIDEO = 1
if SAVE_VIDEO:
    fps = 17
    interval = 1000 / fps  # Convert fps to milliseconds
    # Generate frame numbers skipping every 2 frames
    frame_numbers = range(0, n_steps-horizon,3)  # Adjust step to skip frames
    ani = animation.FuncAnimation(fig_track, update, frames=frame_numbers, interval=interval, blit=True)
    video_path = f"{media_dir}/traj_video.mp4"
    ani.save(video_path, fps=fps, extra_args=['-vcodec', 'h264_videotoolbox', '-b:v', '2000k', '-preset', 'ultrafast'])
    print(f"🎥 Smooth video saved as {video_path}")



for i in range(len(lap_times)-1,0,-1) :
	if lap_times[i] != 0. :
		lap_times[i] = lap_times[i] - lap_times[i-1]
# print(lap_times)
print("lap times:",lap_times, "violation: ",np.sum(violation_total), "mean_dev", np.mean(fval_history))
