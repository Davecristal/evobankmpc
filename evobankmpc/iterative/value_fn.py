import copy
import os
from functools import partial
from typing import Optional, Tuple

import distrax
import jax
import jax.numpy as jnp
import numpy as np

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, will skip loss plotting.")

from evobankmpc.iterative.model_train import ModelTrain
from evobankmpc.utils.trainer_jax import Trainer
from evobankmpc.iterative.networks import EnsembleBNN


class ValueFn:
    def __init__(self, us, value_config, config) -> None:
        self.us = us
        self.thread_ret = [None, None]
        self.thread = None

        if value_config.model_type == 'bnn':
            self.model = EnsembleBNN(value_config)
        else:
            self.model = ModelTrain(value_config)

        if value_config.model_type == 'nf':
            self.dist_narrow = distrax.MultivariateNormalDiag(jnp.zeros(1), jnp.ones(1) / 10)

        self.value_config = copy.deepcopy(value_config)
        self.config = copy.deepcopy(config)
        self.value_dim = config.value_dim         
        self.state_dim = config.state_dim          
        self.trainer = Trainer('valuefn', self.config.save_dir)

        self.data_range = None

    def load_valuefn(self, safe_set, config):
        self.config = copy.deepcopy(config)
        safe_set = copy.deepcopy(safe_set)
        safe_set_cat = np.concatenate(safe_set)
        self.data_range = np.array([np.min(safe_set_cat, axis=0), np.max(safe_set_cat, axis=0)])
        self.model.flax_train_state, _ = self.trainer.load_state(
            self.model.flax_train_state,
            path=os.path.abspath(self.config.save_dir) + "/valuefn_model_best",
            abs_path=True
        )
        return self.model, self.config

    @partial(jax.jit, static_argnums=(0))
    def get_value(self, flax_train_state, sampled_states, rng_key, data_range):
        data_in = sampled_states[:, :self.config.value_dim]

        if self.value_config.model_type == 'nf':
            data_in = (data_in - data_range[0, :self.config.value_dim]) / (
                data_range[1, :self.config.value_dim] - data_range[0, :self.config.value_dim] + 1e-8
            )

        sampled_value = self.model.test(flax_train_state, data_in, rng_key)

        if self.value_config.model_type == 'nf':
            sampled_value, sampled_value_var = sampled_value
            sampled_value = sampled_value * (data_range[1, -1] - data_range[0, -1]) + data_range[0, -1]
        else:
            pass

        sampled_value = jnp.clip(sampled_value, data_range[0, -1], data_range[1, -1])
        return sampled_value

    def save_model(self, path=None):
        if path is None:
            path = self.config.save_dir
        self.trainer.save_state(self.model.flax_train_state, path=os.path.abspath(path) + "/")

    def train_valuefn(self, safe_set, config, value_learning_lmppi_ret, max_epoch=0):
        self.config = copy.deepcopy(config)
        safe_set = copy.deepcopy(safe_set)
        max_epoch = self.value_config.max_epoch if max_epoch == 0 else max_epoch

        safe_set_cat = np.concatenate(safe_set)
        self.data_range = np.array([np.min(safe_set_cat, axis=0), np.max(safe_set_cat, axis=0)])

        if self.value_config.model_type == 'bnn':
            self.data_range[:, -1] -= self.data_range[0, -1]

        if self.value_config.model_type == 'nf':
            data_out = safe_set_cat[:, -1:]                      
            data_in = safe_set_cat[:, :-1]                      
            data_out, data_in = self._add_boundary_data(safe_set_cat, data_out, data_in)
            data_out = (data_out - self.data_range[0, -1]) / (self.data_range[1, -1] - self.data_range[0, -1] + 1e-8)
            data_in = (data_in - self.data_range[0, :self.config.value_dim]) / (
                self.data_range[1, :self.config.value_dim] - self.data_range[0, :self.config.value_dim] + 1e-8
            )
        elif self.value_config.model_type == 'bnn':
            data_out, data_in = self._compile_bnn_data(safe_set)
        elif self.value_config.model_type == 'nn':
            data_out = safe_set_cat[:, -1:]
            data_in = safe_set_cat[:, :-1]
            data_out = (data_out - self.data_range[0, -1]) / (self.data_range[1, -1] - self.data_range[0, -1] + 1e-8)
            data_in = (data_in - self.data_range[0, :self.config.value_dim]) / (
                self.data_range[1, :self.config.value_dim] - self.data_range[0, :self.config.value_dim] + 1e-8
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.value_config.model_type}")

        self.model.flax_train_state, losses = self.model.train(
            self.model.flax_train_state,
            data_out,
            data_in,
            max_epoch=max_epoch,
            loss_threshold=0.0001
        )
        self.trainer.save_state(self.model.flax_train_state, path=os.path.abspath(self.config.save_dir) + "/")

        if self.us is not None:
            self.us.logline('losses', losses[-1], print_line=self.config.print_line)

        if MATPLOTLIB_AVAILABLE and len(losses) > 0:
            try:
                plt.figure(figsize=(8, 5))
                plt.plot(losses, 'b-', linewidth=2)
                if max(losses) > 1e-3:
                    plt.yscale('log')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Value Function Training Loss')
                plt.grid(True, alpha=0.3)
                save_dir = getattr(self.config, 'save_dir', './')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, 'value_loss.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Value function loss curve saved to {save_path}")
            except Exception as e:
                print(f"Warning: Failed to save loss curve: {e}")
        else:
            if not MATPLOTLIB_AVAILABLE:
                print("Warning: matplotlib not available, cannot save loss curve.")

        return self.model, self.config

    def _add_boundary_data(self, ss_arr, data_out, data_in):
        n_boundary = 100  
        if self.config.value_dim == 4:      
            boundary_data = np.zeros((n_boundary, self.config.value_dim + 1))
            boundary_data[:, 0] = np.random.uniform(self.data_range[0, 0], self.data_range[1, 0], size=n_boundary)
            boundary_data[:, 1] = np.random.uniform(
                self.data_range[0, 1] - self.config.half_width,
                self.data_range[1, 1] + self.config.half_width,
                size=n_boundary
            )
            for ind in range(2, self.config.value_dim):
                boundary_data[:, ind] = np.random.uniform(self.data_range[0, ind], self.data_range[1, ind], size=n_boundary)
            boundary_data[:, -1] = np.min(ss_arr[:, -1])
            data_out = jnp.concatenate([data_out, boundary_data[:, -1:]], axis=0)
            data_in = jnp.concatenate([data_in, boundary_data[:, :self.config.value_dim]], axis=0)
        else:
            pass

        return data_out, data_in

    def _compile_bnn_data(self, safe_set):

        data_outs = []
        data_ins = []
        fix_length = None

        for ind_ss in range(len(safe_set)):
            ss_arr = safe_set[ind_ss]
            data_out = ss_arr[:, -1:]
            data_in = ss_arr[:, :-1]

            if ind_ss == 0:
                fix_length = data_in.shape[0]
            else:
                if data_in.shape[0] < fix_length:
                    pad_len = fix_length - data_in.shape[0]
                    data_in = jnp.concatenate([data_in, jnp.zeros((pad_len, self.config.value_dim))], axis=0)
                    data_out = jnp.concatenate([data_out, jnp.zeros((pad_len, 1))], axis=0)
                elif data_in.shape[0] > fix_length:
                    data_in = data_in[:fix_length]
                    data_out = data_out[:fix_length]

            data_outs.append(data_out)
            data_ins.append(data_in)

        data_outs = jnp.array(data_outs)
        data_ins = jnp.array(data_ins)
        return data_outs, data_ins