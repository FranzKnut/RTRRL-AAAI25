"""Utility functions for JAX models."""

from functools import partial
import os
import json
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
import orbax.checkpoint
from jax.tree_util import tree_map, tree_reduce
import jax.tree_util as jtu


class JAX_RNG:
    """Base class that facilitates jax PRNG managment."""

    def __init__(self, rng) -> None:
        """Set up internal rng."""
        self._rng = rng

    @property
    def rng(self):
        """Split internal rng."""
        self._rng, rng = jrandom.split(self._rng)
        return rng


def symlog(x):
    """Symmetric log."""
    return jnp.sign(x) * jnp.log(jnp.abs(x) + 1)


def sigmoid_between(x, lower, upper):
    """Map input to sigmoid that goes from lower to upper."""
    return (upper - lower) * jax.nn.sigmoid(x) + lower


@jax.jit
def preprocess_img(img):
    """Make grayscale from RGB Image."""
    import dm_pix as pix

    return pix.rgb_to_grayscale(jnp.array(img / 255.0, dtype=jnp.float32))


def tree_norm(tree, **kwargs):
    """Sum of the norm of all elements in the tree."""
    return tree_reduce(lambda x, y: x + jnp.linalg.norm(y, **kwargs), tree, initializer=0)


def leaf_norms(tree):
    """Return Dict of leaf names and their norms."""
    flattened, _ = jtu.tree_flatten_with_path(tree)
    flattened = {jtu.keystr(k): v for k, v in flattened}
    return {k: tree_reduce(lambda x, y: x + jnp.linalg.norm(y), v, initializer=0) for k, v in flattened.items()}


@partial(jax.jit, static_argnames=["batch_size"])
def zeros_like_tree(tree, batch_size=None):
    """Create pytree of zeros with batchsize."""
    if batch_size is not None:
        return tree_map(lambda x: jnp.zeros((batch_size,) + x.shape), tree)
    else:
        return tree_map(lambda x: jnp.zeros_like(x), tree)


def tree_stack(trees):
    """Take a list of trees and stack every corresponding leaf.

    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function. Taken from https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = jax.tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(leaf) for leaf in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


def checkpointing(path, fresh=False, hparams: dict = None):
    """Set up checkpointing at given path.

    Returns:
        params : PyTree
                Restored parameters or None if no checkpoint found or fresh is True.
        save_model : Callable
                Function (PyTree->None) for saving given PyTree
        hparams : dict
                Stores given hyper-parameters alongside model params as json
    """
    path = os.path.abspath(path)
    hparams_file_path = os.path.join(path, "hparams.json")

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    orbax_path = os.path.join(path, "ckpt")

    def save_model(_params):
        return checkpointer.save(orbax_path, _params, force=True)

    restored_params = None
    restored_hparams = "{}"
    print(path, end=": ")
    exists = os.path.exists(path)
    if not exists:
        print("No checkpoint found")
    else:
        if fresh:
            print("Overwriting existing checkpoint")
        else:
            restored_params = checkpointer.restore(orbax_path)
            print("Restored model from checkpoint")
            if os.path.exists(hparams_file_path):
                with open(hparams_file_path) as f:
                    restored_hparams = json.load(f)

    if (not exists or fresh) and hparams is not None:
        os.makedirs(path, exist_ok=True)
        with open(hparams_file_path, "w") as f:
            json.dump(hparams, f)

    return (restored_params, restored_hparams), save_model


def mse_loss(y_hat, y):
    """Mean squared error."""
    return jnp.mean((y - y_hat) ** 2)


def bce_loss(y_hat, y):
    """Binary cross-entropy."""
    return optax.sigmoid_binary_cross_entropy(y_hat, y)


def mae_loss(y_hat, y):
    """Mean absoluted error."""
    return jnp.mean(jnp.abs(y - y_hat))


def make_vmap_model(_model, **kwargs):
    @jax.jit
    def _vmap_model(_p, _input):
        return jax.vmap(jax.tree_util.Partial(_model.apply, _p, **kwargs))(_input)

    return _vmap_model


def make_validate(_model, test_data, **kwargs):
    vmap_model = make_vmap_model(_model, **kwargs)

    @jax.jit
    def _validate(_p):
        y_hat, _ = vmap_model(_p, test_data)
        return mse_loss(y_hat, test_data)

    return _validate
