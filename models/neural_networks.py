"""Neural networks using the flax package."""

import flax.linen as nn
import jax
import jax.numpy as jnp



"""
Neural network structure.
"""


class FADense(nn.Dense):
    """Dense Layer with feedback alignment."""

    f_align: bool = True
    kernel_init: nn.initializers.Initializer = nn.initializers.glorot_normal(in_axis=-1, out_axis=-2)

    @nn.compact
    def __call__(self, x):
        """Make use of randomly initialized Feedback Matrix B when f_align is True."""
        if self.f_align:
            B = self.variable(
                "falign",
                "B",
                self.kernel_init,
                self.make_rng() if self.has_rng("params") else None,
                (jnp.shape(x)[-1], self.features),
                self.param_dtype,
            ).value
        else:
            B = self.param(
                "kernel",
                self.kernel_init,
                (jnp.shape(x)[-1], self.features),
                self.param_dtype,
            )

        def f(mdl, x, B):
            return nn.Dense.__call__(mdl, x)

        def fwd(mdl, x, B):
            """Forward pass with tmp for backward pass."""
            return nn.Dense.__call__(mdl, x), (x, B)

        # f_bwd :: (c, CT b) -> CT a
        def bwd(tmp, y_bar):
            """Backward pass that may use feedback alignment."""
            _x, _B = tmp
            grads = {"params": {"kernel": jnp.einsum("...X,...Y->YX", y_bar, _x)}}
            if self.use_bias:
                grads["params"]["bias"] = jnp.einsum("...X->X", y_bar)
            # if self.f_align:
            #     grads['params']['B'] = jnp.zeros_like(B)
            x_grad = jnp.einsum("YX,...X->...Y", _B, y_bar)
            return (grads, x_grad, jnp.zeros_like(_B))

        fa_grad = nn.custom_vjp(f, forward_fn=fwd, backward_fn=bwd)

        return fa_grad(self, x, B)


class MLP(nn.Module):
    """MLP built with equinox."""

    layers: list
    f_align: bool = False

    @nn.compact
    def __call__(self, x):
        """Call MLP."""
        for i, size in enumerate(self.layers[:-1]):
            x = jax.nn.elu(FADense(size, f_align=self.f_align)(x))
        x = FADense(self.layers[-1], f_align=self.f_align)(x)
        return x


class FAAffine(nn.Module):
    """Affine Layer with feedback alignment."""

    features: int
    f_align: bool = True
    offset: int = 0

    @nn.compact
    def __call__(self, x):
        """Make use of randomly initialized Feedback Matrix B when f_align is True."""
        a = self.param("a", nn.initializers.normal(), (self.features,))
        b = self.param("b", nn.initializers.zeros, (self.features,))

        def s(x):
            return x[..., self.offset : self.features + self.offset]

        def f(mdl, x, a, b):
            return a * s(x) + b

        def fwd(mdl, x, a, b):
            """Forward pass with tmp for backward pass."""
            return a * s(x) + b, (x, a)

        # f_bwd :: (c, CT b) -> CT a
        def bwd(res, y_bar):
            """Backward pass that may use feedback alignment."""
            _x, _a = res
            grads = {"params": {"a": s(_x) * y_bar, "b": y_bar}}
            x_bar = jnp.zeros_like(_x)
            x_bar = x_bar.at[..., self.offset : self.features + self.offset].set(
                y_bar if not self.f_align else y_bar * _a
            )
            return (grads, x_bar, jnp.zeros_like(a), jnp.zeros_like(b))

        fa_grad = nn.custom_vjp(f, forward_fn=fwd, backward_fn=bwd)
        return fa_grad(self, x, a, b)
