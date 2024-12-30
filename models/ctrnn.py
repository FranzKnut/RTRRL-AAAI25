"""CTRNN implementation."""

from dataclasses import field
from functools import partial
import flax.linen as nn
import jax
import jax.interpreters
import jax.numpy as jnp
import jax.random as jrand

from chex import PRNGKey
from flax.linen import nowrap


from typing import Tuple

from models.wirings import make_mask_initializer


def ctrnn_ode(params, h, x):
    """Compute euler integration step or CTRNN ODE."""
    W, tau = params
    # Concatenate input and hidden state
    y = jnp.concatenate([x, h, jnp.ones(x.shape[:-1] + (1,))], axis=-1)
    # This way we only need one FC layer for recurrent and input connections
    u = y @ W.T
    act = jnp.tanh(u)
    # Subtract decay and divide by tau
    return (act - h) / tau


class CTRNNCell(nn.RNNCellBase):
    """Simple CTRNN cell."""

    num_units: int
    dt: float = 1.0
    ode_type: str = "murray"
    wiring: str | None = "random"
    wiring_kwargs: dict = field(default_factory=dict)

    @nn.compact
    def __call__(self, h, x):  # noqa
        """Compute euler integration step or CTRNN ODE."""
        # Define params
        w_shape = (self.num_units, x.shape[-1] + self.num_units + 1)
        W = self.param("W", nn.initializers.lecun_normal(in_axis=-1, out_axis=-2), w_shape)

        if self.wiring is not None:
            mask = self.variable(
                "wiring",
                "mask",
                make_mask_initializer(self.wiring, **self.wiring_kwargs),
                self.make_rng() if self.has_rng("params") else None,
                w_shape,
                int,
            ).value
            W = jax.lax.stop_gradient(mask) * W
        # Compute updates
        tau = self.param("tau", partial(jrand.uniform, minval=1, maxval=5), (self.num_units,))
        df_dt = ctrnn_ode((W, tau), h, x)

        # Euler integration step with dt
        out = jax.tree.map(lambda a, b: a + b * self.dt, h, df_dt)
        return out, out

    @nowrap
    def initialize_carry(self, rng: PRNGKey, input_shape: Tuple[int, ...]):
        """Initialize neuron states."""
        return jnp.zeros(input_shape[:-1] + (self.num_units,))

    @property
    def num_feature_axes(self) -> int:
        """Returns the number of feature axes of the RNN cell."""
        return 1


def rtrl_ctrnn(cell, carry, params, x, ode=ctrnn_ode):
    """Compute jacobian trace update for RTRL."""
    h, jp, jx = carry

    # immediate jacobian (this step)
    df_dw, df_dh, df_dx = jax.jacrev(ode, argnums=[0, 1, 2])((params["W"], params["tau"]), h, x)
    df_dw = {"W": df_dw[0], "tau": df_dw[1]}  # , "b": df_dw[1]

    # dh/dh = d(h + f(h) * dt)/dh = I + df/dh * dt
    dh_dh = df_dh * cell.dt  # + jnp.identity(cell.num_units)

    # jacobian trace (previous step * dh_h)
    comm = jax.tree.map(lambda p: jnp.tensordot(dh_dh, p, axes=1), jp)

    def rtrl_step(p, rec, dh):
        return p + rec + dh * cell.dt

    # Update dh_dw approximation
    dh_dw = jax.tree.map(rtrl_step, jp, comm, df_dw)

    # Update dh_dx approximation
    dh_dx = df_dx + jx

    return dh_dw, dh_dx


def hebbian(pre, post):
    return jnp.outer(post, pre)


def rflo_murray(cell: CTRNNCell, carry, params, x):
    """Compute jacobian trace for RFLO."""
    h, jp, jx = carry
    W, tau = params.values()

    jw = jp["W"]
    jtau = jp["tau"]

    # immediate jacobian (this step)
    v = jnp.concatenate([x, h, jnp.ones(x.shape[:-1] + (1,))], axis=-1)
    u = v @ W.T
    # df_dh = jax.jacfwd(jax.nn.tanh)(u)
    # df_dh = jax.jacrev(jax.nn.tanh)(u)
    df_dh = 1 - jnp.tanh(u) ** 2
    # post = jnp.tanh(u)

    # hebb = hebbian(v, post)

    # Outer product the get Immediate Jacobian
    # M_immediate = jnp.einsum('ij,k', df_dh, v)
    M_immediate = df_dh[..., None] * v[None]

    # Update eligibility traces
    jw += (1 / tau)[:, None] * (M_immediate - jw)
    dh_dtau = ((h - jnp.tanh(u)) / tau) - jtau
    jtau += dh_dtau / tau

    df_dw = {"W": jw, "tau": jtau}
    dh_dx = jx
    # dh_dh = df_dh @ W.T[x.shape[-1]:x.shape[-1]+h.shape[-1]]
    return df_dw, dh_dx  # , hebb


class OnlineCTRNNCell(CTRNNCell):
    """Online CTRNN module."""

    plasticity: str = "rflo"

    @nn.compact
    def __call__(self, carry, x):  # noqa
        def f(mdl, h, x):
            h, *traces = h
            carry, out = CTRNNCell.__call__(mdl, h, x)
            return (carry, *traces), out

        def fwd(mdl, carry, x):
            """Forward pass with tmp for backward pass."""
            out, _ = CTRNNCell.__call__(mdl, carry[0], x)

            _p = mdl.variables["params"]
            if self.plasticity == "rtrl":
                traces = rtrl_ctrnn(self, carry, _p, x)
            elif self.plasticity == "rflo":
                traces = rflo_murray(self, carry, _p, x)
            else:
                raise ValueError(f"Plasticity mode {self.plasticity} not recognized.")
            return ((out, *traces), out), (out, *traces)

        @jax.jit
        def bwd(tmp, y_bar):
            """Backward pass that may use feedback alignment."""
            # carry, jp, jx, hebb = tmp
            carry, jp, jx = tmp
            df_dy = y_bar[-1]
            if self.plasticity == "rflo":
                grads_p = jax.tree.map(lambda t: (df_dy.T * t.T).T, jp)
            else:
                grads_p = jax.tree.map(lambda t: df_dy @ t, jp)
            if len(df_dy.shape) > 1:
                # has batch dim
                grads_p = jax.tree.map(lambda x: jnp.mean(x, axis=0), grads_p)
            # grads_p['W'] += hebb
            grads_x = jnp.einsum("...h,...hi->...i", df_dy, jx)
            carry = jax.tree.map(jnp.zeros_like, tmp)  # [:-1]
            return ({"params": grads_p}, carry, grads_x)

        f_grad = nn.custom_vjp(f, forward_fn=fwd, backward_fn=bwd)
        return f_grad(self, carry, x)

    def initialize_carry(self, rng: PRNGKey, input_shape: Tuple[int, ...]):
        """Initialize the carry with jacobian traces."""
        h = super().initialize_carry(rng, input_shape)
        # jh = jnp.zeros(h.shape[:-1] + (h.shape[-1], h.shape[-1]))
        jx = jnp.zeros(h.shape[:-1] + (h.shape[-1], input_shape[-1]))
        params = self.init(rng, (h, None, None), jnp.zeros(input_shape))
        leading_shape = h.shape[:-1] if self.plasticity == "rflo" else h.shape
        jp = jax.tree.map(lambda x: jnp.zeros(leading_shape + x.shape), params["params"])
        return h, jp, jx


if __name__ == "__main__":
    import numpy as np
    import optax

    import matplotlib.pyplot as plt

    key = jrand.PRNGKey(0)
    key, key_model, key_data, key_train = jrand.split(key, 4)
    x = jnp.linspace(0, 5 * np.pi, 100)[:, None]
    y = jnp.sin(x) + 2

    cell = CTRNNCell(32)
    carry = cell.initialize_carry(jrand.PRNGKey(0), (1,))
    params = cell.init(jrand.PRNGKey(0), carry, x[0])

    def loss_mse(y_hat, _y):
        """MSE loss function."""
        return jax.numpy.mean((y_hat - _y) ** 2)

    def loss_rnn(p, c, __x, __y):
        """RNN loss."""
        c, y_hat = cell.apply(p, c, __x)
        return loss_mse(y_hat, __y)

    jax.grad(loss_rnn)(params, carry, x[0], y[0])

    model = nn.Sequential(
        [
            nn.RNN(cell),
            nn.Dense(1),
        ]
    )

    params = model.init(key_model, x, mutable=True)

    def loss_mlp(p, __x, __y):
        """MLP loss function."""
        y_hat = model.apply(p, __x)
        return loss_mse(y_hat, __y)

    loss_mlp(params, x, y)

    def print_progress(i, loss):
        """Print inside jit."""
        if i % 1000 == 0:
            print(f"Iteration {i} | Loss: {loss:.3f}")

    def train(_loss_fn, _params, data, _key, num_steps=10_000, lr=1e-4, batch_size=64):
        """Train network. We use Stochastic Gradient Descent with a constant learning rate."""
        _x, _y = data
        optimizer = optax.lion(lr)
        opt_state = optimizer.init(_params)

        def step(carry, n):
            __params, _opt_state, _key = carry
            _key, key_batch = jrand.split(_key)
            # batch = jrand.choice(key_batch, jnp.hstack([_x, _y]), (batch_size,))
            current_loss, grads = jax.value_and_grad(_loss_fn)(__params, _x, _y)
            updates, _opt_state = optimizer.update(grads, _opt_state, __params)
            __params = optax.apply_updates(__params, updates)
            jax.debug.callback(print_progress, n, current_loss)
            return (__params, _opt_state, _key), current_loss

        (_params, *_), _losses = jax.lax.scan(step, (_params, opt_state, _key), jnp.arange(num_steps, dtype=np.int32))
        print(f"Final loss: {_losses[-1]:.3f}")
        return _params, _losses

    key, key_train = jrand.split(key_data)
    params, losses = train(loss_mlp, params, (x, y), key_train)

    plt.figure(figsize=(10, 5))

    # Plot the training loss
    plt.subplot(1, 2, 1)
    plt.plot(losses)

    # Plot the trained model output
    plt.subplot(1, 2, 2)
    y_hat = model.apply(params, x)
    plt.plot(x, y, label="target")
    plt.plot(x, y_hat, label="trained")
    plt.legend()
    plt.show()
