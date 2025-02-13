"""Util for eligibility traces."""
from functools import partial
from flax import linen as nn
import jax
import jax.numpy as jnp

from models.neural_networks import FADense
import numpy as np


@partial(jax.jit, static_argnames=('trace_mode'))
def trace_update(grads, z, gamma_lambda, trace_mode: str = 'accumulate', alpha=None, _I=1):
    """Update the eligibility trace. Also compute the gradients if d is given.

    See Sutton & Barto, 1998, p. 275
    Args:
        grads: Immediate gradient of the loss with respect to the model parameters
        z: eligibility trace
        I: acummulated gamma, for episodic tasks
        trace_mode: one of "accumulate", "dutch"
        gamma_lambda: discount factor
        alpha: learning rate, used for dutch traces
    """

    def accumulate(_z, _g):
        """Accumulate trace.

        z ← λz + α I @ ∇f.
        """
        return gamma_lambda * _z + (_I * _g.T).T

    # def acc_tanh(_z, _g):
    #     """Accumulate trace with tanh squashing."""
    #     return jnp.tanh(accumulate(_z, _g))

    def dutch(_z, _g):
        """Dutch trace.

        z ← γλz + α[1 - γλ e.T ∇f] ∇f
        """
        return gamma_lambda * _z + (1 - alpha * gamma_lambda * (_z.T @ _g)) * _g

    return jax.tree.map(locals()[trace_mode], z, grads)


@partial(jax.jit, static_argnames=('trace_mode',))
def compute_updates(z, trace_mode: str = 'accumulate', d=None, dutch_diff=None, alpha=None, grads=None):
    """Compute gradients given the eligibility trace."""
    # Multiply trace with TD-error.
    grads = jax.tree.map(lambda t: (d.T * t.T).T,  z)
    if trace_mode == 'dutch':
        grads = jax.tree.map(lambda _z, _g: _g + alpha * (dutch_diff.T * (_z-_g).T).T, z, grads)
    return grads


def init_trace(params, batch_shape=()):
    """Initialize the eligibility trace: zθ ← 0 (d'-component eligibility trace vector)."""
    return jax.tree.map(lambda x: jnp.zeros(batch_shape+x.shape), params)


class TraceModel(nn.Module):
    """A model to predict elegibility traces."""

    flat_shapes: list
    f_align: bool = True

    def setup(self):
        """Initialize a model for every component in the flattened example trace."""
        self.models = [FADense(np.prod(s), f_align=self.f_align) for s in self.flat_shapes]

    def __call__(self, obs):
        """Flatten the given pytree and predict every component separately using Linear functions."""
        return [m(obs).reshape(s) for m, s in zip(self.models, self.flat_shapes)]
