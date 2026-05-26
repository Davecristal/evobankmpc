import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
import distrax
from functools import partial
from flax import linen as nn
import flax.training.train_state as flax_TrainState
import tqdm

from evobankmpc.iterative.nsf import NeuralSplineFlow
from evobankmpc.iterative.networks import MLP, BayesianPNN

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


class ModelTrain:

    def __init__(self, config) -> None:
        self.config = config

        self.jrng = jax.random.PRNGKey(config.random_seed)

        self.iterator = range if getattr(config, 'silent', False) else tqdm.trange

        self.x_init = jnp.zeros((config.batchsize, config.n_dim))
        self.x_context = jnp.zeros((config.batchsize, config.n_context))

        if config.model_type == 'nf':
            self.dist_narrow = distrax.MultivariateNormalDiag(jnp.zeros(1), jnp.ones(1) / 10)
            self.dist = distrax.MultivariateNormalDiag(jnp.zeros(1), jnp.ones(1))
            activation = getattr(config, 'activation', 'gelu')
            n_bins = getattr(config, 'n_bins', 8)
            self.model = NeuralSplineFlow(
                n_dim=config.n_dim,
                n_context=config.n_context,
                hidden_dims=[config.hidden_dims, config.hidden_dims],
                n_transforms=config.layer_num,
                activation=activation,
                n_bins=n_bins
            )
            params = self.model.init(self.jrng, self.x_init, self.x_context)

        elif config.model_type == 'nn':
            self.model = MLP(
                out_dims=config.n_dim,
                hidden_dims=config.hidden_dims,
                layer_num=config.layer_num
            )
            params = self.model.init(self.jrng, self.x_context)

        elif config.model_type == 'bnn':
            self.model = BayesianPNN(
                input_dim=config.n_context,
                output_features=config.n_dim,
                hidden_features=config.hidden_dims,
                layer_num=config.layer_num
            )
            params = self.model.init(self.jrng, self.x_context, self.jrng)

        else:
            raise ValueError(f"Unknown model_type: {config.model_type}")

        n_params = sum(x.size for x in jax.tree.leaves(params))
        print(f"{config.model_type} model parameters: {n_params}")

        self.flax_train_state = flax_TrainState.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optax.chain(
                optax.clip_by_global_norm(8),
                optax.adam(learning_rate=config.lr)
            )
        )

        self.epoch_info = np.zeros(8)

    @partial(jax.jit, static_argnums=(0,))
    def train_step_nf(self, state, y_gt, context):
        def loss_fn(params):
            log_prob = state.apply_fn(params, y_gt, context)
            return -log_prob.mean()

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    def train(self, state, data_out, data_in, max_epoch=300, loss_threshold=0.0001):
        losses = []
        n_total = data_in.shape[0]
        batchsize = min(self.config.batchsize, n_total)

        for epoch in self.iterator(max_epoch):

            perm = np.random.permutation(n_total)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_total, batchsize):
                idx = perm[start:start + batchsize]
                y_gt = jnp.asarray(data_out[idx])
                context = jnp.asarray(data_in[idx])

                if self.config.model_type == 'nf':
                    state, loss = self.train_step_nf(state, y_gt, context)
                elif self.config.model_type == 'nn':
                    state, loss = self.model.train_step_nn(state, y_gt, context)
                elif self.config.model_type == 'bnn':
                    state, loss = self.model.train_step_bnn(state, y_gt, context, self.jrng)

                epoch_loss += loss.item()
                n_batches += 1

            mean_loss = epoch_loss / n_batches
            losses.append(mean_loss)

            # if mean_loss < loss_threshold:
            #     break

        return state, losses

    @partial(jax.jit, static_argnums=(0,))
    def test(self, state, data_in, rng_key=None):
        if self.config.model_type == 'nn':
            def test_fn(s, ctx):
                return s.apply_fn(s.params, ctx)

        elif self.config.model_type == 'bnn':
            def test_fn(s, ctx, rng):
                mean, _ = s.apply_fn(s.params, ctx, rng)
                return mean

        elif self.config.model_type == 'nf':
            def test_fn(s, ctx, rng):
                z = self.dist_narrow.sample(seed=rng, sample_shape=(ctx.shape[0] * self.config.n_sample))
                ctx_batch = ctx[None, :, :].repeat(self.config.n_sample, 0).reshape(-1, ctx.shape[-1])
                samples = s.apply_fn(s.params, z, ctx_batch, method=s.apply_fn.sample)
                samples = samples.reshape(self.config.n_sample, -1, samples.shape[-1])
                return samples.mean(axis=0), samples.var(axis=0)

        else:
            raise ValueError(f"Unknown model_type: {self.config.model_type}")

        if rng_key is None:
            rng_key = self.jrng

        if self.config.model_type == 'nn':
            return test_fn(state, data_in)
        elif self.config.model_type == 'bnn':
            return test_fn(state, data_in, rng_key)
        else:  # 'nf'
            return test_fn(state, data_in, rng_key)