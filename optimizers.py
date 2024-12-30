"""Util for creating optax optimizers."""

from dataclasses import dataclass, field
import optax


@dataclass(frozen=True)
class OptimizerConfig:
    """Class representing the parameters for an optimizer.

    Attributes:
        opt_name (str): The name of the optimizer.
        learning_rate (float): The learning rate for the optimizer.
        kwargs (dict): Additional keyword arguments for the optimizer.
        decay_type (str): The type of decay for the learning rate.
        lr_kwargs (dict): Additional keyword arguments for the learning rate decay.
        weight_decay (float): The weight decay for the optimizer.
        gradient_clip (float): The value to clip the gradients.
        multi_step (int): number of steps to accumulate.
    """

    opt_name: str = "adam"
    learning_rate: float = 1e-3
    kwargs: dict = field(default_factory=dict, hash=False)
    decay_type: str | None = None
    lr_kwargs: dict = field(default_factory=dict, hash=False)
    weight_decay: float = 0.0
    gradient_clip: float | None = None
    multi_step: int | None = None


def make_optimizer(config=OptimizerConfig(), direction="min") -> optax.GradientTransformation:
    """Make optax optimizer.

    The decorator allows reading scheduled lr from the optimizer state.

    Parameters
    ----------
    learning_rate : float
        initial learning rate
    direction : str, optional
        min or max. Defaults to "min", by default "min"
    opt_name : str, optional
        Name of optimizer, by default 'sgd'
    gradient_clip : int, optional
        Clip gradient norm. Defaults to 0
    lr_decay : int, optional
         Exponential lr decay. Defaults to 1, by default 1
    optimizer_params : dict, optional
        Additional kwargs to the optimizer, by default {}

    Returns
    -------
        optax optimizer
    """
    learning_rate = config.learning_rate
    weight_decay = config.weight_decay
    if direction in ["max", "maximize"]:
        learning_rate = -learning_rate
    else:
        weight_decay = -weight_decay

    # def decay_mask(tree):
    #     mask = jax.tree.map(lambda _: False, tree)  # Initialize all False

    #     # Only set some leaves to true for RNNs
    #     if isinstance(tree, BaseRNNCell):
    #         mask = eqx.tree_at(lambda x: x.w,  # HERE the leaves for decay are selected
    #                            mask, True)
    #     # elif isinstance(tree, Linear):
    #     #     mask = eqx.tree_at(lambda x: x.W,  # HERE the leaves for decay are selected,
    #     #                        mask, True)
    #     return mask

    if config.decay_type == "cosine_warmup":
        """Args:
            init_value: Initial value for the scalar to be annealed.
            peak_value: Peak value for scalar to be annealed at end of warmup.
            warmup_steps: Positive integer, the length of the linear warmup.
            decay_steps: Positive integer, the total length of the schedule. Note that
                this includes the warmup time, so the number of steps during which cosine
                annealing is applied is ``decay_steps - warmup_steps``.
            end_value: End value of the scalar to be annealed.
            exponent: Float. The default decay is ``0.5 * (1 + cos(pi t/T))``,
                where ``t`` is the current timestep and ``T`` is ``decay_steps``.
                The exponent modifies this to be ``(0.5 * (1 + cos(pi * t/T)))
                ** exponent``.
                Defaults to 1.0.
        """
        learning_rate = optax.warmup_cosine_decay_schedule(
            learning_rate * config.lr_kwargs["initial_multiplier"],
            peak_value=learning_rate,
            end_value=learning_rate * config.lr_kwargs["end_multiplier"],
            decay_steps=config.lr_kwargs["decay_steps"],
            warmup_steps=config.lr_kwargs["warmup_steps"],
        )
    if config.decay_type == "cosine":
        """Args:
            init_value: An initial value for the learning rate.
            decay_steps: Positive integer - the number of steps for which to apply
                the decay for.
            alpha: The minimum value of the multiplier used to adjust the
                learning rate. Defaults to 0.0.
            exponent:  The default decay is ``0.5 * (1 + cos(pi * t/T))``, where 
                ``t`` is the current timestep and ``T`` is the ``decay_steps``. The
                exponent modifies this to be ``(0.5 * (1 + cos(pi * t/T))) ** exponent``.
                Defaults to 1.0.

        """
        learning_rate = optax.cosine_decay_schedule(learning_rate, decay_steps=config.lr_kwargs["decay_steps"], alpha=config.lr_kwargs.get("alpha", 0))
    elif config.decay_type == "exponential":
        """Args:
            init_value: the initial learning rate.
            transition_steps: must be positive. See the decay computation above.
            decay_rate: must not be zero. The decay rate.
            transition_begin: must be positive. After how many steps to start annealing
                (before this many steps the scalar value is held fixed at `init_value`).
            staircase: if `True`, decay the values at discrete intervals.
            end_value: the value at which the exponential decay stops. When
                `decay_rate` < 1, `end_value` is treated as a lower bound, otherwise as
                an upper bound. Has no effect when `decay_rate` = 0.
        """
        learning_rate = optax.exponential_decay(
            learning_rate,
            config.lr_kwargs["transition_steps"],
            config.lr_kwargs["decay_rate"],
            config.lr_kwargs.get("warmup_steps", 0),
            config.lr_kwargs.get("staircase", False),
            config.lr_kwargs.get("end_value", None),
        )
    elif config.decay_type is not None:
        raise ValueError(f"Decay type {config.decay_type} unknown.")

    @optax.inject_hyperparams
    def _make_opt(learning_rate):
        # Create optimizer from optax chain
        optimizer = optax.chain(
            # Weight decay
            optax.add_decayed_weights(weight_decay),  # , mask=decay_mask
            # Gradient clipping
            optax.clip_by_global_norm(config.gradient_clip) if config.gradient_clip else optax.identity(),
            # Optimizer
            getattr(optax, config.opt_name)(learning_rate, **config.kwargs),
        )
        if config.multi_step:
            optimizer = optax.MultiSteps(optimizer, every_k_schedule=config.multi_step)
        return optimizer

    return _make_opt(learning_rate=learning_rate)
