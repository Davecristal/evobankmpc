import jax
import jax.numpy as jnp
from jax import jit, vmap

def make_jax_evaluator(vehicle_params):

    mass = vehicle_params['mass']
    lf = vehicle_params['lf']
    lr = vehicle_params['lr']
    Iz = vehicle_params['Iz']
    Cm1 = vehicle_params['Cm1']
    Cm2 = vehicle_params['Cm2']
    Cr0 = vehicle_params['Cr0']
    Cr2 = vehicle_params['Cr2']

    @jit
    def diffequation(state, input, params):
        Bf, Cf, Df, Br, Cr, Dr = params
        psi = state[2]
        vx = state[3]
        vy = state[4]
        omega = state[5]
        steer = input[1]
        pwm = input[0]

        vmin = 0.05
        vx = jnp.where(vx < vmin, vmin, vx)
        vy = jnp.where(vx < vmin, 0.0, vy)
        omega = jnp.where(vx < vmin, 0.0, omega)
        steer = jnp.where(vx < vmin, 0.0, steer)

        Frx = (Cm1 - Cm2 * vx) * pwm - Cr0 - Cr2 * (vx ** 2)
        alphaf = steer - jnp.arctan2((lf * omega + vy), vx)
        alphar = jnp.arctan2((lr * omega - vy), vx)
        Ffy = Df * jnp.sin(Cf * jnp.arctan(Bf * alphaf))
        Fry = Dr * jnp.sin(Cr * jnp.arctan(Br * alphar))

        dxdt = jnp.zeros(6)
        dxdt = dxdt.at[0].set(vx * jnp.cos(psi) - vy * jnp.sin(psi))
        dxdt = dxdt.at[1].set(vx * jnp.sin(psi) + vy * jnp.cos(psi))
        dxdt = dxdt.at[2].set(omega)
        dxdt = dxdt.at[3].set((Frx - Ffy * jnp.sin(steer)) / mass + vy * omega)
        dxdt = dxdt.at[4].set((Fry + Ffy * jnp.cos(steer)) / mass - vx * omega)
        dxdt = dxdt.at[5].set((Ffy * lf * jnp.cos(steer) - Fry * lr) / Iz)
        return dxdt

    @jit
    def integrate_one(state, input, params, Ts):
        k1 = diffequation(state, input, params)
        k2 = diffequation(state + 0.5 * Ts * k1, input, params)
        k3 = diffequation(state + 0.5 * Ts * k2, input, params)
        k4 = diffequation(state + Ts * k3, input, params)
        return state + (Ts / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    @jit
    def batch_evaluate(state, input, params_batch, Ts):
        state_batch = jnp.tile(state, (params_batch.shape[0], 1))
        step_vmap = vmap(integrate_one, in_axes=(0, None, 0, None))
        return step_vmap(state_batch, input, params_batch, Ts)

    return batch_evaluate


def make_mlp_evaluator(mlp_predictor, vehicle_params):

    mass = vehicle_params['mass']
    lf = vehicle_params['lf']
    lr = vehicle_params['lr']


    @jit
    def evaluate(state, input):

        psi = state[2]
        vx = state[3]
        vy = state[4]
        omega = state[5]
        steer = input[1]
        pwm = input[0]


        vmin = 0.05
        vx_safe = jnp.where(vx < vmin, vmin, vx)
        vy_safe = jnp.where(vx < vmin, 0.0, vy)
        omega_safe = jnp.where(vx < vmin, 0.0, omega)
        steer_safe = jnp.where(vx < vmin, 0.0, steer)


        ay = vx_safe * omega_safe  # or ay = (Fry + Ffy*cos(steer))/mass

        alphaf = steer_safe - jnp.arctan2((lf * omega_safe + vy_safe), vx_safe)
        alphar = jnp.arctan2((lr * omega_safe - vy_safe), vx_safe)

        feat = jnp.concatenate([state, input, jnp.array([ay, alphaf, alphar])])
        feat = feat.reshape(1, -1)   # (1, 11)
        delta = mlp_predictor(feat).reshape(-1)
        return state + delta

    return evaluate