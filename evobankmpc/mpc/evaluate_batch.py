import numpy as np

def evaluate_batch(args):
    models, current_state, input_val, Ts = args
    predictions = np.empty((len(models), len(current_state)))
    input_reshaped = input_val.reshape(-1, 1)
    t = np.array([0, Ts])

    for i, model in enumerate(models):
        x_next, _ = model.sim_continuous(current_state, input_reshaped, t)
        predictions[i] = x_next[:, -1]
    return predictions