#!/usr/bin/env python3

import numpy as np
from evobankmpc.params import ORCA
from evobankmpc.models import Dynamic
from evobankmpc.tracks import ETHZ
import argparse
import os

def generate_data(track_name='ETHZ', n_steps=150000, sim_time=None, Ts=0.02):
    params = ORCA(control='pwm')
    model = Dynamic(**params)
    track = ETHZ(reference='optimal', longer=True)

    if sim_time is None:
        n_steps = int(n_steps)
    else:
        n_steps = int(sim_time / Ts)

    states = np.zeros((6, n_steps+1))
    inputs = np.zeros((2, n_steps))
    states[:,0] = [track.x_init, track.y_init, track.psi_init, track.vx_init, 0, 0]

    for i in range(n_steps):
        r = np.random.rand()
        t = i * Ts
        if r < 0.5:     
            pwm = np.random.uniform(-0.1, 1.0)
            steer = np.random.uniform(-0.35, 0.35)
        elif r < 0.7:    
            pwm = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
            steer = 0.3 * np.sin(2 * np.pi * 0.8 * t)
        elif r < 0.85:   
            pwm = 1.0 if np.random.rand() > 0.5 else -0.1
            steer = 0.3 if np.random.rand() > 0.5 else -0.3
        else:          
            if i > 0:
         
                params['Df'] *= 0.9999
                params['Dr'] *= 0.9999
                model.Df, model.Dr = params['Df'], params['Dr']
            pwm = 0.5 + 0.2 * np.sin(2 * np.pi * 0.3 * t)
            steer = 0.2 * np.sin(2 * np.pi * 0.5 * t)
        inputs[0, i] = pwm
        inputs[1, i] = steer

        x_next, _ = model.sim_continuous(states[:, i], inputs[:, i:i+1], [0, Ts])
        states[:, i+1] = x_next[:, -1]

    delta_states = states[:, 1:] - states[:, :-1]  # (6, n_steps)

    psi = states[2, :-1]
    vx = states[3, :-1]
    vy = states[4, :-1]
    omega = states[5, :-1]
    steer = inputs[1, :]
    pwm = inputs[0, :]

    ay = vy + vx * omega   

    lf = params['lf']; lr = params['lr']
    alpha_f = steer - np.arctan2((lf * omega + vy), np.abs(vx) + 1e-6)
    alpha_r = np.arctan2((lr * omega - vy), np.abs(vx) + 1e-6)

    X = np.concatenate([
        states[:, :-1].T,           
        inputs.T,                   
        ay.reshape(-1,1),           
        alpha_f.reshape(-1,1),      
        alpha_r.reshape(-1,1),     
    ], axis=1).astype(np.float32)   

    y = delta_states.T.astype(np.float32)


    mask = np.all(np.abs(y) < 10, axis=1)
    X = X[mask]
    y = y[mask]


    os.makedirs('evobankmpc/data', exist_ok=True)
    np.savez('evobankmpc/data/training_data_large.npz', X=X, y=y)
    print(f"{X.shape[0]} : {X.shape[1]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_steps', type=int, default=150000)
    parser.add_argument('--sim_time', type=float, default=None)
    parser.add_argument('--Ts', type=float, default=0.02)
    args = parser.parse_args()
    generate_data(n_steps=args.n_steps, sim_time=args.sim_time, Ts=args.Ts)