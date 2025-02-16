"""Online Actor-Critic algorithm with eligibility traces for continuous actions."""

from dataclasses import asdict, dataclass, field
from functools import partial
from pprint import pprint
from typing import Tuple

import numpy as np
import simple_parsing
from tqdm import trange

from chex import PRNGKey
import distrax
import jax
import jax.numpy as jnp
from jax import random as jrandom
from flax import linen as nn
import optax
from brax.training.acme import running_statistics

from envs.wrappers import VmapWrapper
from logging_util import DummyLogger, log_norms, with_logger
from envs.environments import EnvironmentParams, make_env, print_env_info, render_frames
from models.ctrnn import OnlineCTRNNCell
from models.online_lru import OnlineLRULayer
from traces import compute_updates, init_trace, trace_update
from models.neural_networks import FADense, MLP
from optimizers import OptimizerConfig, get_current_lrs, make_optimizer
from models.jax_util import sigmoid_between, zeros_like_tree

# jax.config.update('jax_debug_nans', True)
# jax.config.update('jax_platform_name', 'cpu')

# Uncomment for faster compilation using persistent cache.
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 1000000)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")


@dataclass(unsafe_hash=True)
class RTRRLParams:
    """Class representing the parameters for the RTRRL algorithm."""

    debug: int | bool = 0  # Enable Debugging, higher levels include more fine-grainde debugging: 2 = Profiling
    seed: int | None = None

    # Training
    episodes: int = 150_000
    steps: int = 1000
    patience: int = 20

    # Validation
    eval_every: int = 100
    eval_steps: int = 1000
    eval_batch_size: int = 10
    render_every_evals: int = 10
    render_start: int = 0
    render_steps: int = 200

    # Logging
    logging: str | None = None
    log_repo: str | None = None
    save_model: bool = False
    log_norms: bool = True
    log_code: bool = False
    log_every: int = 1

    # Environment
    env_params: EnvironmentParams = EnvironmentParams(
        render=False,
        env_name="StatelessCartPoleEasy",
        max_ep_length=1000,
        batch_size=1,
    )

    # Optimizer
    optimizer_params_td: OptimizerConfig = OptimizerConfig(
        opt_name="adam",
        learning_rate=1e-3,
    )
    optimizer_params_rnn: OptimizerConfig = OptimizerConfig(
        opt_name="adam",
        learning_rate=1e-3,
    )

    # RNN
    rnn_model: str | None = "ctrnn"
    hidden_size: int = 32
    gradient_mode: str = "rflo"
    wiring: str = "fully_connected"

    # TD(lambda)
    trace_mode: str = "accumulate"
    gamma: float = 0.99
    lambda_v: float = 0.9
    lambda_pi: float = 0.9
    lambda_rnn: float = 0.9
    eta_pi: float = 1
    eta_f: float = 1
    entropy_rate: float = 0
    eta: float | None = 0

    # Features
    meta_rl: bool = True
    f_align: bool = True
    normalize_reward: bool = True
    normalize_obs: bool = False

    layer_norm: bool = False
    var_scaling: bool = False
    mlp_actor: bool = False
    pass_obs: bool = False
    update_period: float = 1
    dropout_rate: float = 0
    act_magnitude_factor: float = 0
    slow_rnn_factor: float = 0e-2


class TD(nn.Module):
    """TD lambda."""

    a_dim: int
    discrete: bool
    mlp_actor: bool = False
    f_align: bool = False

    def setup(self) -> None:
        """Initialize components."""
        # Actor
        actor_out_dim = self.a_dim if self.discrete else 2 * self.a_dim
        if self.mlp_actor:
            self.actor = MLP([64, actor_out_dim], f_align=self.f_align, name="actor")
        else:
            self.actor = FADense(
                actor_out_dim,
                f_align=self.f_align,
                #  kernel_init=nn.initializers.zeros_init(),
                use_bias=False,
                #  offset=1,
                name="actor",
            )

        # Critic
        self.critic = FADense(
            1,
            # self.critic = MLP([64, 1],
            f_align=self.f_align,
            #   kernel_init=nn.initializers.zeros_init(),
            #   use_bias=False,
            name="critic",
        )


class RNNActorCritic(nn.RNNCellBase):
    """RTRRL cell with shared RNN and linear actor and critic networks."""

    a_dim: int
    discrete: bool
    obs_dim: int = None
    batch_shape: tuple = ()
    hidden_size: int = 32
    rnn_model: str | None = "ctrnn"
    gradient_mode: str = "rflo"
    f_align: bool = True
    act_log_bounds: tuple[float] = field(default_factory=lambda: [-2, 2])
    act_bounds: tuple[float] | None = None
    dropout_rate: float = 0.0
    pass_obs: bool = True
    mlp_actor: bool = True
    pred_obs: bool = False
    layer_norm: bool = True
    wiring: str = "fully_connected"
    wiring_kwargs: dict = field(default_factory=dict)

    @nn.nowrap
    def _make_rnn(self):
        if self.rnn_model == "ctrnn":
            extra_w_kw = {"interneurons": self.hidden_size - (self.a_dim + 1)} if self.wiring == "ncp" else {}
            return OnlineCTRNNCell(
                self.hidden_size,
                # num_modules=self.num_modules,
                **{
                    # 'num_units': self.hidden_size,
                    "plasticity": self.gradient_mode,
                    "wiring": self.wiring,
                    "wiring_kwargs": {**self.wiring_kwargs, **extra_w_kw},
                },
                name="rnn",
            )

        elif self.rnn_model == "lru":
            return OnlineLRULayer(
                self.hidden_size,
                # num_modules=self.num_modules,
                name="rnn",
            )

    def setup(self) -> None:
        """Initialize components."""
        if self.rnn_model:
            self.rnn = self._make_rnn()

        self.td = TD(self.a_dim, self.discrete, self.mlp_actor, self.f_align, name="td")

        if self.pred_obs:
            self.obs = FADense(
                self.obs_dim + 1,  # Predict obs and reward
                f_align=self.f_align,
                #    kernel_init=nn.initializers.zeros_init(),
                #  use_bias=False,
                name="obs",
            )

        # Droput Layer
        if self.dropout_rate:
            self._dropout = nn.Dropout(self.dropout_rate)

        if self.layer_norm:
            self._layer_norm = nn.LayerNorm(use_bias=False, use_scale=False)

    def rnn_step(self, carry, obs, training=True):
        """Step RNN."""
        if not self.rnn_model:
            return obs, carry
        if carry is None:
            # Initialize seed and the carry
            carry = self.initialize_carry(jrandom.PRNGKey(self.random_seed), obs.shape)
        carry, hidden = self.rnn(carry, obs)
        # Dropout
        if self.dropout_rate:
            hidden = self._dropout(hidden, deterministic=not training)
        # Layer Norm
        if self.layer_norm:
            hidden = self._layer_norm(hidden)
        return hidden, carry

    def value(self, hidden, x=None):
        """Compute value from latent."""
        if self.pass_obs:
            hidden = jnp.concatenate([hidden, x], axis=-1)
        return self.td.critic(hidden)

    def action_dist(self, hidden, x=None):
        """Compute action distribution form latent."""
        if self.pass_obs:
            hidden = jnp.concatenate([hidden, x], axis=-1)
        if not self.discrete:
            loc, log_scale = jnp.split(self.td.actor(hidden), 2, axis=-1)
            log_scale = sigmoid_between(log_scale, *self.act_log_bounds)
            if self.act_bounds is not None:
                loc = sigmoid_between(loc, *self.act_bounds)
            dist = distrax.Normal(loc=loc, scale=jax.nn.softplus(log_scale))
        else:
            logits = self.td.actor(hidden)
            dist = distrax.Categorical(logits=logits)
        return dist

    @nn.compact
    def __call__(self, carry, x, key=None):
        """Step RNN and compute actor and critic."""
        # RNN
        hidden, new_carry = self.rnn_step(carry, x)

        # Actor
        actor = self.action_dist(hidden, x)

        if key is not None:
            actor = actor.sample(seed=key).squeeze()

        # Critic
        v_hat = self.value(hidden, x)

        return new_carry, (actor, v_hat, hidden)

    @property
    def num_feature_axes(self) -> int:
        """Returns the number of feature axes of the RNN cell."""
        return 1

    @nn.nowrap
    def initialize_carry(self, rng: PRNGKey, input_shape: Tuple[int, ...]):
        """Initialize the Worldmodel cell carry."""
        if not self.rnn_model:
            return None

        if len(input_shape) > 0:
            self.batch_shape = input_shape[:-1]

        return self._make_rnn().initialize_carry(rng, input_shape)


def train_rtrrl(args: RTRRLParams, logger=DummyLogger()):
    """Online Actor-Critic algorithm with eligibility traces.

    Parameters: trace-decay rates λθ ∈ [0, 1], λw ∈ [0, 1]; step sizes αθ > 0, αw > 0, η > 0
    """
    logger = logger
    env, env_info, eval_env = make_env(args.env_params, make_eval=True)
    eval_env = VmapWrapper(eval_env, batch_size=args.eval_batch_size)
    pprint(args, width=1)
    print_env_info(env_info)

    # ENVIRONMENT --------------------------------------------------------------
    # Initialize S ∈ S (e.g., to s0)
    OBS_SIZE, DISCRETE, ACT_SIZE, obs_mask, act_clip = env_info.values()
    args.seed = args.seed or np.random.randint(1e6)
    key = jrandom.PRNGKey(args.seed)
    logger.log_params(asdict(args))
    key, key_model, key_step, key_init, key_env, _key = jrandom.split(key, 6)
    env_state = env.reset(key_env)

    # Model INIT --------------------------------------------------------------
    model = RNNActorCritic(
        a_dim=ACT_SIZE,
        obs_dim=OBS_SIZE,
        discrete=DISCRETE,
        hidden_size=args.hidden_size,
        gradient_mode=args.gradient_mode,
        f_align=args.f_align,
        wiring=args.wiring,
        dropout_rate=args.dropout_rate,
        pass_obs=args.pass_obs,
        rnn_model=args.rnn_model,
        mlp_actor=args.mlp_actor,
        layer_norm=args.layer_norm,
        act_bounds=jnp.array(act_clip) if act_clip is not None else None,
    )
    initial_input = env_state.obs
    batch_shape = initial_input.shape[:-1]

    # Set up normalization
    obs_rms = reward_rms = None

    def normalize(obs, rms=None):
        if rms is None:
            return obs
        else:
            return running_statistics.normalize(obs, rms)

    if args.normalize_obs:
        obs_rms = running_statistics.init_state(env_state.obs[0])
        obs_rms = running_statistics.update(obs_rms, env_state.obs)
    if args.normalize_reward:
        reward_rms = running_statistics.init_state(env_state.reward[0])
        reward_rms = running_statistics.update(reward_rms, env_state.reward)

    initial_input = normalize(env_state.obs, obs_rms)

    if args.meta_rl:
        initial_input = jnp.concatenate(
            [
                initial_input,
                jnp.zeros(batch_shape + (env.action_size,)),
                normalize(env_state.reward, reward_rms).reshape(batch_shape + (-1,)),
            ],
            axis=-1,
        )

    # Get initial state
    rnn_state = model.initialize_carry(key_init, initial_input.shape)
    h0 = rnn_state  # [0]
    r_bar = jnp.array([0.0])
    initial_I = jnp.ones(batch_shape)

    # Initialize model and make first step
    (rnn_state, (action, v_prev, *_)), params = model.init_with_output(
        key_model, rnn_state, initial_input, key=key_model
    )
    initial_action = action.reshape((*batch_shape, -1))  # .sample(seed=key_step)

    # Initialize eligibility trace
    trace_keys = ["td"]
    rnn_keys = [k for k in params["params"] if "rnn" in k]
    if args.eta_f and args.rnn_model is not None:
        trace_keys += rnn_keys
    z0 = {k: v for k, v in init_trace(params, batch_shape)["params"].items() if k in trace_keys}

    # Initialize optimizer
    param_labels = {"td": "td"}
    if args.rnn_model is not None:
        param_labels = {**param_labels, **{k: "rnn" for k in rnn_keys}}
    optimizer = optax.multi_transform(
        {
            "rnn": make_optimizer(direction="max", config=args.optimizer_params_rnn),
            "td": make_optimizer(direction="max", config=args.optimizer_params_td),
        },
        param_labels,
    )
    critic_lr = args.optimizer_params_td.learning_rate
    opt_state = optimizer.init(params["params"])
    slow_params = params

    # Logging preparation
    logger["best_eval_reward"] = -jnp.inf
    render_every = (args.render_every_evals or 0) * (args.eval_every or 0)
    assert render_every > 0 or not args.env_params.render, (
        "render_every_evals and eval_every must be > 0 if render is True"
    )

    @jax.jit
    def eval_model(_params, key, _obs_rms=None, _reward_rms=None):
        """Evaluate given agent in given environment."""
        if key is None:
            key = jrandom.PRNGKey(0)
        key, key_init = jrandom.split(key)
        env_state = eval_env.reset(key_init)
        # Normalization
        _input = normalize(env_state.obs, _obs_rms).reshape(args.eval_batch_size, -1)
        if args.meta_rl:
            _input = jnp.concatenate(
                [
                    _input,
                    jnp.zeros((_input.shape[0], ACT_SIZE)),
                    normalize(env_state.reward, _reward_rms).reshape(args.eval_batch_size, -1),
                ],
                axis=-1,
            )
        # Initialize RNN states
        rnn_state = model.initialize_carry(key_init, _input.shape)
        # Initialize input for the filter

        def eval_step(_params, carry, _=None):
            """Step function for scan."""
            print("Tracing Eval Step")
            # Unpack carry
            env_state, rnn_state, re_action, _key = carry
            _key, step_key = jrandom.split(_key)
            f_input = normalize(env_state.obs, _obs_rms).reshape(args.eval_batch_size, -1)
            # obs = normalization(state.obs) if normalization is not None else state.obs

            if args.meta_rl:
                r = normalize(env_state.reward, _reward_rms).reshape(args.eval_batch_size, -1)
                f_input = jnp.concatenate([f_input, re_action, r], axis=-1)

            step_key = jrandom.split(step_key, args.eval_batch_size)
            rnn_state, (action, *_) = jax.vmap(partial(model.apply, _params))(rnn_state, f_input, key=step_key)
            # action = action.mean(axis=-1)
            if DISCRETE:
                # action = action.mean(axis=0)
                action = action.squeeze().astype(jnp.int32)
            else:
                action = action.reshape((args.eval_batch_size, -1))
            # Step environments
            env_state = eval_env.step(env_state, action)
            # Assemble next input for the filter
            re_action = jax.nn.one_hot(action, eval_env.action_size) if DISCRETE else action
            carry = env_state, rnn_state, re_action, _key
            return carry, env_state

        # Scan over the number of steps
        _, (env_states) = jax.lax.scan(
            partial(eval_step, _params),
            (env_state, rnn_state, jnp.zeros((args.eval_batch_size, ACT_SIZE)), key),
            length=args.eval_steps,
        )

        # total_reward = jnp.sum(env_states.reward) / jnp.max(jnp.array([jnp.sum(env_states.done), 1])).mean()
        # For episodes that are done early, get the first occurence of done
        ep_until = jnp.where(env_states.done.any(axis=0), env_states.done.argmax(axis=0), env_states.done.shape[0])
        # Compute cumsum and get value corresponding to end of episode per batch.
        ep_rewards = env_states.reward.cumsum(axis=0)[ep_until, jnp.arange(ep_until.shape[-1])].mean()
        return ep_rewards, env_states

    # Set up scan body
    @jax.jit
    def step_fn(_carry, _):
        print("Tracing step_fn")
        (
            params,
            slow_params,
            env_state,
            opt_state,
            action,
            rnn_state,
            z,
            v_prev,
            r_bar,
            _I,
            _obs_rms,
            _reward_rms,
            seed,
        ) = _carry
        seed, action_key, dropout_key = jrandom.split(seed, 3)

        # Step ENV
        action = jnp.clip(action, *jnp.array(act_clip)) if act_clip is not None else action
        if DISCRETE:
            action = action.reshape(batch_shape)
        env_state = env.step(env_state, action)
        if args.normalize_obs:
            _obs_rms = running_statistics.update(_obs_rms, env_state.obs)
        if args.normalize_reward:
            _reward_rms = running_statistics.update(_reward_rms, env_state.reward)
        reward = env_state.reward.reshape(-1)

        # Reset cell state and trace if done
        rnn_state = jax.tree.map(lambda a, b: jax.vmap(jnp.where)(env_state.done, a, b), h0, rnn_state)
        if args.rnn_model:
            old_hidden = rnn_state[0]

        # Make input vector
        f_input = env_state.obs
        if args.meta_rl:
            # Set action to zeros where env is done
            action, _r = jax.tree.map(
                lambda a: jax.vmap(jnp.where)(env_state.done, jnp.zeros_like(a), a), (action, reward)
            )
            if DISCRETE:
                # One-hot encode action
                re_action = jax.nn.one_hot(action, ACT_SIZE)
            else:
                re_action = action
            # Append action and reward to input
            f_input = jnp.concatenate([f_input, re_action, _r.reshape(batch_shape + (-1,))], axis=-1)

        def grads_step(h, i):
            """Entire gradient computation and trace updates. Used to vmap over."""

            # RNN step
            def rnn_step(_params):
                return model.apply(_params, h, i, training=True, rngs={"dropout": dropout_key}, method=model.rnn_step)

            # We use vjp to get a function for computing the rnn gradients later
            hidden, rnn_backwards, rnn_state = jax.vjp(rnn_step, slow_params, has_aux=True)

            @partial(jax.grad, has_aux=True, argnums=[0, 1])
            def td_loss(_params, _hidden):
                # Compute value prediction
                v_hat = model.apply(_params, _hidden, i, method=model.value)

                # Compute action distribution
                action_dist = model.apply(_params, _hidden, i, method=model.action_dist)
                # Sample action from target network
                action = action_dist.sample(seed=action_key)
                # Actor loss is log probability of sampled action
                actor_loss = action_dist.log_prob(action)

                # Add up losses
                loss = actor_loss.mean() * args.eta_pi + v_hat.mean()
                loss_info = {
                    "total_td_loss": loss,
                    "actor_loss": actor_loss.mean() * args.eta_pi,
                    "critic_loss": v_hat.mean(),
                }

                if args.var_scaling and not DISCRETE:
                    # Variance for scaling the loss
                    scale = jnp.clip(jnp.mean(action_dist.scale), max=1)
                    loss *= scale
                else:
                    scale = jnp.ones(())

                return loss, (action, action_dist, scale, v_hat, loss_info)

            (grads_next, hidden_grads), (action, action_dist, actor_scale, v_hat, loss_info) = td_loss(
                slow_params, hidden
            )

            # Compute gradients wrt rnn params
            hidden_grads = rnn_backwards(hidden_grads)[0]
            grads_next = jax.tree.map(lambda x, y: x + y, hidden_grads, grads_next)

            # Loss function for terms outside of TD(lambda)
            @partial(jax.grad, has_aux=True, argnums=[0, 1])
            def non_td_loss(_params, _hidden):
                loss = 0.0
                if args.rnn_model and args.slow_rnn_factor:
                    # Encourage slow changing rnn state
                    loss -= jnp.array(
                        jax.tree.map(lambda a, b: jnp.linalg.norm(a - b) * args.slow_rnn_factor, old_hidden, _hidden)
                    ).mean()

                # # Entropy loss
                action_dist = model.apply(_params, _hidden, i, method=model.action_dist)
                ent = action_dist.entropy().mean()
                loss += ent * args.entropy_rate
                loss_info = {"entropy": ent}

                if not DISCRETE and args.act_magnitude_factor:
                    # Discourage high absolute action values
                    loss_info["magnitude_loss"] = jnp.abs(action_dist.mean()).mean()
                    loss -= loss_info["magnitude_loss"] * args.act_magnitude_factor

                return loss, loss_info

            (non_td_grad, hidden_non_td_grad), non_td_loss_info = non_td_loss(slow_params, hidden)
            loss_info = {**loss_info, **non_td_loss_info}
            # Compute gradients wrt
            hidden_extra_grad = rnn_backwards(hidden_non_td_grad)[0]
            non_td_grads = jax.tree.map(lambda x, y: actor_scale * (x + y), non_td_grad, hidden_extra_grad)
            return rnn_state, non_td_grads, loss_info, action, action_dist, v_hat, grads_next

        # vmap over gradient computation
        rnn_state, non_td_grads, loss_info, action, action_dist, v_hat, grads_next = jax.vmap(grads_step)(
            rnn_state, f_input
        )

        # TD-ERROR ------------------------------------------------------------
        v_targ = reward + args.gamma * v_hat.squeeze() * (1 - env_state.done)
        d = v_targ - r_bar - v_prev.squeeze()
        loss_info["v_targ"] = jnp.mean(v_targ)

        # Combine traces with td-error to compute the updates
        updates = {
            "params": {
                "td": {
                    "critic": compute_updates(
                        z["td"]["critic"],
                        trace_mode=args.trace_mode,
                        d=d,
                        dutch_diff=(v_hat - v_prev),  # Used for dutch traces
                        alpha=critic_lr,
                    ),
                    "actor": compute_updates(z["td"]["actor"], d=d),
                }
            }
        }
        if args.rnn_model:
            if args.eta_f:
                updates["params"]["rnn"] = compute_updates(z["rnn"], d=d * args.eta_f)
            else:
                updates["params"]["rnn"] = zeros_like_tree(non_td_grads["params"]["rnn"])

        # Sum up td and non-td updates
        updates["params"] = jax.tree.map(lambda x, y: x + y, non_td_grads["params"], updates["params"])
        # Take mean over batch
        updates = jax.tree.map(lambda x: jnp.mean(x, axis=0), updates)
        # Step optimizer
        updates, opt_state = optimizer.update(updates["params"], opt_state, params["params"])
        # Apply updates
        params["params"] = optax.apply_updates(params["params"], updates)
        if args.update_period != 1:
            # Polyak averaging of updates
            rnn_slow_params = optax.incremental_update(
                params["params"]["rnn"], slow_params["params"]["rnn"], args.update_period
            )
            slow_params = params
            slow_params["params"]["rnn"] = rnn_slow_params
        else:
            slow_params = params

        # HACK: clip RNN tau
        if args.rnn_model == "ctrnn":
            params["params"]["rnn"]["tau"] = jnp.clip(params["params"]["rnn"]["tau"], min=1.0)
            slow_params["params"]["rnn"]["tau"] = jnp.clip(slow_params["params"]["rnn"]["tau"], min=1.0)

        if args.eta is None:
            # For episodic tasks remember total discounting factors
            # I ← γI or 1 if terminal
            _I = args.gamma * _I * (1 - env_state.done) + env_state.done
        else:
            # # Running average of rewards, for infinite horizon
            # See Sutton & Barto 2017 p. 275 ff.
            r_bar = r_bar + args.eta * jnp.mean(d)

        # Trace updates
        # Reset trace if done
        z = jax.tree.map(lambda a, b: jax.vmap(jnp.where)(env_state.done, a, b), z0, z)

        # Update Traces
        def trace_updates(_grads_next, _z, _i, _lambda_scale_rnn=1):
            z_next = {
                "td": {
                    "actor": trace_update(
                        _grads_next["params"]["td"]["actor"],
                        _z["td"]["actor"],
                        gamma_lambda=args.lambda_pi * args.gamma * _lambda_scale_rnn,
                        _I=_i,
                    ),
                    "critic": trace_update(
                        _grads_next["params"]["td"]["critic"],
                        _z["td"]["critic"],
                        trace_mode=args.trace_mode,
                        gamma_lambda=args.lambda_v * args.gamma,
                        alpha=critic_lr,
                        _I=_i,
                    ),
                }
            }
            if args.rnn_model and args.eta_f:
                if args.lambda_rnn != 0:
                    z_next["rnn"] = trace_update(
                        _grads_next["params"]["rnn"],
                        _z["rnn"],
                        gamma_lambda=args.lambda_rnn * args.gamma * _lambda_scale_rnn,
                        _I=_i,
                    )
                else:
                    z_next["rnn"] = _grads_next["params"]["rnn"]
            return z_next

        z = jax.vmap(trace_updates)(grads_next, z, _I)

        _carry = (
            params,
            slow_params,
            env_state,
            opt_state,
            action.reshape((*batch_shape, -1)),
            rnn_state,
            z,
            v_hat,
            r_bar,
            _I,
            _obs_rms,
            _reward_rms,
            seed,
        )
        return _carry, (env_state.reward, env_state.done, loss_info, v_hat, d)

    carry = (
        params,
        slow_params,
        env_state,
        opt_state,
        initial_action,
        rnn_state,
        z0,
        v_prev,
        r_bar,
        initial_I,
        obs_rms,
        reward_rms,
        key_step,
    )

    # Loop misc
    pbar = trange(args.episodes, mininterval=1)
    steps = jnp.array(range(args.steps))
    all_rewards = []
    all_param_norms = []
    steps_since_best = 0
    # ------------------------------------------------------------------------
    # MAIN LOOP --------------------------------------------------------------
    try:
        # fill queue for delayed updates
        for i in pbar:
            _key, key_step, key_eval = jrandom.split(_key, 3)
            if args.debug == 2 and i == 0:
                lowered = jax.jit(step_fn).lower(carry, 0)
                print(lowered.as_text())
                compiled = lowered.compile()
                pprint(compiled.cost_analysis())
                jax.profiler.start_trace("tmp/jax-trace", create_perfetto_link=True)
            # Scan fixed number of steps
            carry, scan_out = jax.lax.scan(step_fn, carry, steps)
            if args.debug == 2 and i == 0:
                jax.profiler.stop_trace()
            (
                params,
                slow_params,
                env_state,
                opt_state,
                action,
                rnn_state,
                z,
                v_hat,
                r_bar,
                initial_I,
                obs_rms,
                reward_rms,
                seed,
            ) = carry

            reward, dones, loss_info, values, delta = scan_out
            num_episodes = jnp.sum(dones)
            divisor = max(num_episodes, 1)
            avg_r = jnp.sum(reward) / divisor
            avg_d = jnp.sum(delta) / divisor
            avg_r_bar = jnp.sum(r_bar) / divisor
            avg_val = jnp.mean(values)
            # Calculate total steps from batch size and number of steps
            log_steps = (i + 1) * args.steps
            total_steps = log_steps * (args.env_params.batch_size or 1)
            if i % args.log_every == 0:
                # Logging -------------------------------------------------------------
                metrics = {
                    "steps": total_steps,
                    "mean_reward": avg_r,
                    "num_episodes": num_episodes,
                    "mean_delta": avg_d,
                    "mean_r_bar": avg_r_bar,
                    "mean_v": avg_val,
                    **jax.tree.map(jnp.mean, loss_info),
                }
                current_lrs = get_current_lrs(opt_state)
                if args.optimizer_params_td.decay_type:
                    metrics["lr/td"] = current_lrs["lr_td"]
                if args.optimizer_params_rnn.decay_type:
                    metrics["lr/rnn"] = current_lrs["lr_rnn"]

            else:
                metrics = {}

            if args.log_norms and i % args.log_every == 0:
                norms = log_norms({"z": z, "params": params, "slow_params": slow_params})
                metrics = {**metrics, **{"norms/" + k: v for k, v in norms.items()}}
                # metrics = {k: float(v) for k, v in metrics.items()}

            # Print current stats
            pbar.set_description(
                f"{i}, Steps: {log_steps:2.2E}, R: {avg_r:.2f}, Delta: {avg_d:.2f}, Value: {avg_val:.2f}",
                refresh=False,
            )

            if args.eval_every and (i % args.eval_every == 0 or i == args.episodes - 1):  # also eval last
                key_eval, _key = jrandom.split(key_eval)
                # Do not render the first episode, render the last one
                eval_avg, env_states = eval_model(slow_params, key=_key, _obs_rms=obs_rms, _reward_rms=reward_rms)
                metrics["eval/rewards"] = float(eval_avg)
                pbar.write(f"Eval reward: {eval_avg:.2f}")

                # Maybe render and log video
                should_render = args.env_params.render and ((i % render_every == 0 and i > 0) or i == args.episodes - 1)
                if logger is not None and should_render:
                    frames = render_frames(
                        env, env_states.pipeline_state, args.render_start, args.render_start + args.render_steps
                    )
                    if frames:
                        logger.log_video("env/video", np.array(frames), fps=30, caption=f"Reward: {eval_avg:.2f}")

                if eval_avg > logger["best_eval_reward"]:
                    # New best
                    steps_since_best = 0
                    logger["best_eval_reward"] = eval_avg
                    metrics["eval/best_eval_reward"] = eval_avg
                else:
                    # For early stopping
                    steps_since_best += 1

            # Log metrics
            logger.log(metrics, step=log_steps)

            # Early stopping
            if args.patience and steps_since_best >= args.patience:
                print(f"Early stopping patience {args.patience}")
                break
    except Exception as e:
        print("Exception in training loop!")
        raise e
    finally:
        logger["avg_reward"] = jnp.mean(jnp.array(all_rewards)) if all_rewards else 0
        logger.finalize(all_param_norms)

    logger.finalize()
    return logger["best_eval_reward"]


if __name__ == "__main__":
    # Parse hparams from cmd line
    hparams: RTRRLParams = simple_parsing.parse(RTRRLParams, add_config_path_arg=True)

    # Name run
    run_name = hparams.env_params.env_name

    # Run RTRRL
    with_logger(
        train_rtrrl,
        hparams,
        logger_name=hparams.logging,
        project_name="RTRRL",
        run_name=run_name,
        hparams_type=RTRRLParams,
    )
