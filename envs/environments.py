"""Wrappers to support Brax and Gymnax training."""


from dataclasses import dataclass, field
from typing import Iterable
import gym
from functools import partial
from jax import numpy as jnp
import jax
from brax.envs.base import Env as BraxEnv  # noqa


import gymnax
import brax
import popjym

from envs.wrappers import EpisodeWrapper, FlatObs_BraxWrapper, GymBraxWrapper, GymnaxBraxWrapper, PO_BraxWrapper, PopJymBraxWrapper, RandomizedAutoResetWrapperNaive, VmapWrapper


from . import *  # noqa


@dataclass(frozen=True, eq=True)
class EnvironmentParams:
    """Parameters for gym environments.

    Attributes:
        obs_mask (Union[str, Iterable[int]]): Mask for the observation space.
        gymnax_params (dict): Parameters for the GymNax library.
        init_kwargs (dict): Initialization arguments for the environment.
        env_params (dict): Additional parameters for the environment.
        batch_size (int): Number of parallel environments.
        render (bool): Whether to render the environment during training.
    """

    env_name: str = 'CartPole-v1'
    reward_scaling: int = 1
    obs_mask: str | Iterable[int] | None = None
    init_kwargs: dict = field(default_factory=dict, hash=False)
    env_kwargs: dict = field(default_factory=dict, hash=False)
    max_ep_length: int = 500
    batch_size: int | None = None
    render: bool = True


def print_env_info(env_info):
    """Print infos about an environment. Takes env_info from make_env."""
    OBS_SIZE, DISCRETE, ACT_SIZE, obs_mask, act_clip = env_info.values()
    print('ENV:')
    print(f'obs_size:    {OBS_SIZE}')
    print(f'act_size:    {ACT_SIZE}' + (' (discrete)' if DISCRETE else ' (continuous)'))
    if len(obs_mask) < OBS_SIZE:
        print(f'obs_mask:    {obs_mask}')
    print(f'act_clip:    {act_clip}')
    # print(f'value_size: {VALUE_SIZE}')


def get_obs_mask(base_obs_size: int, obs_mask=None):
    """Get the observation mask from string description.

    obs_mask may take values ['odd', 'even', 'first_half'] or a list of indices.
    """
    if obs_mask:
        if obs_mask == 'odd' or obs_mask == 'even':
            obs_mask = jnp.array([i for i in range(base_obs_size) if i % 2 == (obs_mask == 'odd')], dtype=jnp.int32)
        elif obs_mask == 'first_half':
            obs_mask = jnp.array([i for i in range((base_obs_size+1) // 2)], dtype=jnp.int32)
        obs_mask = jnp.array(obs_mask)
    else:
        obs_mask = jnp.array([i for i in range(base_obs_size)])
    return obs_mask


def get_env_specs(env: gym.Env, obs_mask=None):
    """Infer the sizes for the observation and action space given a mask.

    Parameters
    ----------
    env : gym.Env
        _description_
    obs_mask : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    is_gymnax = hasattr(env, 'observation_space')
    ACT_SIZE = env.action_size
    if is_gymnax:
        env: GymnaxBraxWrapper
        DISCRETE = env.discrete
        act_space = env.action_space
        if act_space.dtype == jnp.float32:
            if ACT_SIZE == 1:
                act_clip = (act_space.low, act_space.high)
            else:
                act_clip = tuple(map(tuple, (act_space.low._value, act_space.high._value)))
        else:
            act_clip = None
    else:
        env: BraxEnv
        # is brax
        DISCRETE = False
        # Assuming brax only takes normalized actions!
        act_clip = None
        act_clip = tuple([-1] * ACT_SIZE), tuple([1] * ACT_SIZE)
    obs_mask = get_obs_mask(env.observation_size, obs_mask)
    OBS_SIZE = len(obs_mask)

    return OBS_SIZE, DISCRETE, ACT_SIZE, obs_mask, act_clip


def make_env(params: EnvironmentParams, debug=0, make_eval=False, use_vmap_wrapper=True):
    """Make brax or gymnax env.

    Parameters
    ----------
    params : EnvironmentParams
        additional params. must have 'env_name', 'batch_size' and 'max_ep_length'. 'obs_mask' is optional.
         Environment name. If starts with 'brax', will use brax env, otherwise gymnax.
    debug: int | bool
        set the debug level.
    make_eval : bool, optional
        Whether to make an eval env, by default False
        If true, eval env without batching is also returned

    Returns
    -----------
    env : envs.Env
        Environment
    env_info : dict
        Dictionary with env info
    """
    env: BraxEnv
    env_name = params.env_name
    if env_name in gymnax.registered_envs:
        # Set params for gymnax envs
        params.env_kwargs['max_steps_in_episode'] = params.max_ep_length

        # create a gym environment
        env, gymnax_params = gymnax.make(env_name, **params.init_kwargs)
        env = GymnaxBraxWrapper(env, params.env_kwargs)
    elif env_name in popjym.registration.REGISTERED_ENVS:
        env, env_params = popjym.make(env_name)
        env = PopJymBraxWrapper(env, params.env_kwargs)
    else:
        if env_name.startswith('brax'):
            # Create entrypoint for brax env
            entry_point = partial(brax.envs.get_environment, env_name="-".join(env_name.split('-')[1:]), **params.env_kwargs)
            if env_name not in gym.envs.registry:
                gym.register(env_name, entry_point=entry_point,
                             order_enforce=False)
        # Create a gym environment wrapped with vmap, AutoReset, and Episode wrappers
        env = gym.make(env_name, autoreset=False, disable_env_checker=debug < 3, **params.init_kwargs)

        if not env_name.startswith('brax'):
            # probably a gym env
            env = GymBraxWrapper(env, params.env_kwargs)

    OBS_SIZE, DISCRETE, ACT_SIZE, obs_mask, act_clip = get_env_specs(env, params.obs_mask)

    # Wrap with the brax wrappers
    env = EpisodeWrapper(env, params.max_ep_length, action_repeat=1)
    env = FlatObs_BraxWrapper(env)
    env = PO_BraxWrapper(env, obs_mask)
    if params.batch_size and use_vmap_wrapper:
        env = VmapWrapper(env, batch_size=params.batch_size)
    # env = EfficientAutoResetWrapper(env)
    env = RandomizedAutoResetWrapperNaive(env)
    env_info = dict(obs_size=OBS_SIZE,
                    discrete=DISCRETE,
                    act_size=ACT_SIZE,
                    obs_mask=obs_mask,
                    act_clip=act_clip)

    if make_eval:
        if env_name in gymnax.registered_envs:
            eval_env, _ = gymnax.make(env_name, **params.init_kwargs)
            eval_env = GymnaxBraxWrapper(eval_env, params.env_kwargs)
        elif env_name in popjym.registration.REGISTERED_ENVS:
            eval_env, _ = popjym.make(env_name)
            eval_env = PopJymBraxWrapper(eval_env, params.env_kwargs)
        else:
            eval_env = gym.make(env_name, disable_env_checker=getattr(params, 'debug', 0) < 3, **params.init_kwargs)
            if not env_name.startswith('brax'):
                eval_env = GymBraxWrapper(eval_env, params.env_kwargs)

        eval_env = EpisodeWrapper(eval_env, params.max_ep_length, action_repeat=1)
        eval_env = FlatObs_BraxWrapper(eval_env)
        eval_env = PO_BraxWrapper(eval_env, obs_mask)
        # eval_env = VmapWrapper(eval_env, batch_size=params.batch_size)
        eval_env = RandomizedAutoResetWrapperNaive(eval_env)
        return env, env_info, eval_env

    return env, env_info


def render_frames(_env: gym.Env, states: list, start_idx: int = None, end_idx: int = None):
    """Render the given states of the environment.

    Parameters
    ----------
    _env : gym.Env
        Environment to render. Can handle Brax, Gymnax and Gym envs.
    states: list
        List of states to render.
    start_idx : int, optional
        start rendering from this index, by default None, means start at 0
    end_idx : int, optional
        render until this index, by default None, means render all

    Returns
    -------
    list[array]
        List of RGB array renderings of the environment at given states.
    """
    if not isinstance(states, list):
        states = [jax.tree.map(lambda x: x[n], states) for n in range(start_idx or 0, end_idx or states.time.shape[0])]

    # Define rendering function for specific envs
    if isinstance(_env.unwrapped, GymnaxBraxWrapper):
        if _env.name in ['CartPole-v1', 'MountainCarContinuous-v0', 'MountainCar-v0', 'Pendulum-v1', 'Acrobot-v1']:
            from gymnax.visualize.vis_gym import get_gym_state
            gym__env = gym.make(_env.name, render_mode="rgb_array").unwrapped

            def render_gym(_env, _state):
                """Taken from gymnax.visualize.vis_gym."""
                gym_state = get_gym_state(_state, _env.name)
                if _env.name == "Pendulum-v1":
                    gym__env.last_u = gym_state[-1]
                gym__env.state = gym_state
                rgb_array = gym__env.render()
                return rgb_array.transpose(2, 0, 1)
        else:
            print("Cannot render env: ", _env.name)
            return []

    elif not hasattr(_env.unwrapped, 'env'):  # is Braxenv
        from brax.io import image

        def render_gym(_env, _state):
            return image.render_array(_env.sys, _state, 256, 256, camera='track').transpose(2, 0, 1)
    else:
        def render_gym(_env, _state):
            _env.unwrapped.env.state = _state
            if _env.name == "Pendulum-v1":
                _env.unwrapped.env.last_u = _state[-1]
            return _env.render().transpose(2, 0, 1)

    frames = []
    for _state in states:
        frames.append(render_gym(_env, _state))

    if isinstance(_env.unwrapped, GymnaxBraxWrapper):
        gym__env.close()

    return frames
