from envs.environments import EnvironmentParams
from rtrrl2 import RTRRLParams, train_rtrrl

# jax.config.update('jax_disable_jit', True)

args = RTRRLParams(
    debug=2,
    env_params=EnvironmentParams(
        env_name='CartPole-v1',
        # env_name='Reacher-misc',
        # env_name='MemoryChain-bsuite',
        # init_kwargs={'size': 8},
        # env_kwargs={'memory_length': 16},
        # env_name='CartpoleContinuousJax-v0',
        # init_kwargs={'task': 'balancing-dv'},
        # env_kwargs={'available_torque': [-1.0, 0.0, +1.0]},
        batch_size=1,
        max_ep_length=500,
        render=False,
        obs_mask=None,  # (2, 3)
    ),
    steps=1,
    episodes=3,
)

train_rtrrl(args)
