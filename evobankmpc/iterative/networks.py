import jax
import optax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from functools import partial
import numpy as np

import flax.training.train_state as flax_TrainState


def sample_gaussian(rng, mean, log_std):
    std = jnp.exp(log_std)
    return mean + std * random.normal(rng, mean.shape)


@jax.vmap
def l1_Loss(results, labels):
    return jnp.abs(results - labels)


class MLP(nn.Module):
    out_dims: int
    hidden_dims: int = 512
    layer_num: int = 5

    @nn.compact
    def __call__(self, z, activation=nn.relu):
        for _ in range(self.layer_num - 1):
            z = activation(nn.Dense(self.hidden_dims, kernel_init=jax.nn.initializers.he_uniform())(z))
        z = nn.Dense(self.out_dims)(z)
        return z

    @partial(jax.jit, static_argnums=(0,))
    def train_step_nn(self, state, y_gt, context):
        def loss_fn(params):
            output = state.apply_fn(params, context)
            return l1_Loss(y_gt, output).mean()

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss


class BayesianPNN(nn.Module):
    input_dim: int
    output_features: int
    hidden_features: int = 256
    layer_num: int = 3

    @nn.compact
    def __call__(self, x, rng, init_max_logvar=0.5, init_min_logvar=10.):
        activation = nn.swish
        x = MLP(
            layer_num=self.layer_num,
            out_dims=self.output_features * 2,
            hidden_dims=self.hidden_features
        )(x, activation)
        mean, log_var = jnp.split(x, 2, axis=-1)

        max_logvar = self.param('max_logvar', lambda rng, shape: jnp.full(shape, init_max_logvar), (1,))
        min_logvar = self.param('min_logvar', lambda rng, shape: jnp.full(shape, init_min_logvar), (1,))
        log_var = max_logvar - jax.nn.softplus(max_logvar - log_var)
        log_var = min_logvar + jax.nn.softplus(log_var - min_logvar)

        return mean, (log_var, max_logvar, min_logvar)

    @partial(jax.jit, static_argnums=(0,))
    def train_step_bnn(self, state, y_gt, context, rng):
        def loss_fn(params, y_gt, context, rng):
            mean, (log_var, max_logvar, min_logvar) = state.apply_fn(params, context, rng)
            inv_var = jnp.exp(-log_var)
            mse_loss = jnp.mean(jnp.mean((mean - y_gt) ** 2 * inv_var, axis=-1), axis=-1)
            var_loss = jnp.mean(jnp.mean(log_var, axis=-1), axis=-1)
            log_var_loss = 0.01 * jnp.sum(max_logvar) - 0.01 * jnp.sum(min_logvar)
            total_loss = mse_loss + var_loss + log_var_loss
            return total_loss, mse_loss

        (_, loss), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, y_gt, context, rng)
        state = state.apply_gradients(grads=grads)
        return state, jnp.sqrt(loss)  


class EnsembleBNN:
    def __init__(self, value_config, rng_seed=0) -> None:
        self.config = value_config
        self.num_ensemble = value_config.num_ensemble
        from evobankmpc.iterative.model_train import ModelTrain
        self.value_model = ModelTrain(value_config)
        self.model = self.value_model.model
        rngs = random.split(jax.random.PRNGKey(rng_seed), self.num_ensemble)
        self.flax_train_state = jax.vmap(self._get_train_state)(rngs)

    def _get_train_state(self, rng):
        rng1, rng2 = random.split(rng)
        params = self.model.init(rng1, jnp.zeros((self.config.batchsize, self.config.n_context)), rng2)
        return flax_TrainState.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optax.adam(learning_rate=self.config.lr),
        )

    def train(self, ensemble_flax_train_state, data_out, data_in, max_epoch=300, loss_threshold=0.0001):
        ensemble_flax_train_state, losses = jax.vmap(
            self.value_model.train, in_axes=(0, 0, 0, None)
        )(ensemble_flax_train_state, data_out, data_in, max_epoch)
        return ensemble_flax_train_state, jnp.mean(jnp.asarray(losses), axis=1)

    @partial(jax.jit, static_argnums=(0,))
    def test(self, ensemble_flax_train_state, data_in, rng_key=None):
        rng_keys = random.split(rng_key, self.num_ensemble)
        predictions = jax.vmap(self.value_model.test, in_axes=(0, None, 0))(
            ensemble_flax_train_state, data_in, rng_keys
        )
        return jnp.mean(jnp.asarray(predictions), axis=0)