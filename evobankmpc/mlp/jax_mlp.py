import jax.numpy as jnp
from jax import jit
import numpy as np
import pickle


def load_mlp_weights(weights_file='evobankmpc/mlp/mlp_weights.npz'):
    data = np.load(weights_file)
    w1 = data['w1'].T   # (8,64)
    w2 = data['w2'].T   # (64,64)
    w3 = data['w3'].T   # (64,6)
    b1 = data['b1']     # (64,)
    b2 = data['b2']     # (64,)
    b3 = data['b3']     # (6,)
    return [jnp.array(w1), jnp.array(b1),
            jnp.array(w2), jnp.array(b2),
            jnp.array(w3), jnp.array(b3)]

def make_mlp_predictor(weights):
    w1, b1, w2, b2, w3, b3 = weights
    @jit
    def predict(x):
        h = jnp.maximum(0, jnp.dot(x, w1) + b1)   # ReLU
        h = jnp.maximum(0, jnp.dot(h, w2) + b2)
        out = jnp.dot(h, w3) + b3
        return out
    return predict


def load_residual_mlp_weights(weights_file='evobankmpc/mlp/residual_mlp_weights.pkl'):
    with open(weights_file, 'rb') as f:
        weights_list = pickle.load(f)
    return [jnp.array(w) for w in weights_list]

def load_residual_config(config_file='evobankmpc/mlp/residual_config.npy'):
    config = np.load(config_file, allow_pickle=True).item()
    return config

def load_scaler(mean_file='evobankmpc/mlp/scaler_mean.npy', scale_file='evobankmpc/mlp/scaler_scale.npy'):
    mean = np.load(mean_file)
    scale = np.load(scale_file)
    return jnp.array(mean), jnp.array(scale)

def make_residual_mlp_predictor(weights, config, mean, scale):

    input_dim = config['input_dim']
    hidden_dims = config['hidden_dims']
    output_dim = config['output_dim']
    n_blocks = len(hidden_dims)


    idx = 0
    w_init = weights[idx]
    b_init = weights[idx+1]
    idx += 2

    block_weights = []
    for _ in range(n_blocks):
        w_fc1 = weights[idx]
        b_fc1 = weights[idx+1]
        w_fc2 = weights[idx+2]
        b_fc2 = weights[idx+3]
        idx += 4
        block_weights.append((w_fc1, b_fc1, w_fc2, b_fc2))

    w_out = weights[idx]
    b_out = weights[idx+1]

    @jit
    def predict(x):
        x_norm = (x - mean) / scale
        h = jnp.maximum(0, jnp.dot(x_norm, w_init) + b_init)
        for (w_fc1, b_fc1, w_fc2, b_fc2) in block_weights:
            residual = h
            out = jnp.maximum(0, jnp.dot(h, w_fc1) + b_fc1)   # ReLU after first linear
            out = jnp.dot(out, w_fc2) + b_fc2
            out = out + residual
            h = jnp.maximum(0, out)   # ReLU after residual
        out = jnp.dot(h, w_out) + b_out
        return out

    return predict