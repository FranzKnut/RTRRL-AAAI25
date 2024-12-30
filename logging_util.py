"""Utilies for logging."""

import collections
import contextlib
from dataclasses import asdict
import os
import traceback
from typing import Callable
from typing_extensions import override

from PIL import Image
from dacite import from_dict

import plotly.express as px
import flax.linen as nn
import jax.tree_util as jtu
from models.jax_util import leaf_norms, tree_norm, tree_stack


class ExceptionPrinter(contextlib.AbstractContextManager):
    """Hacky way to print exceptions in wandb agent."""

    def __enter__(self):  # noqa
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_val, exc_tb)
        return False


def wandb_wrapper(project_name, func, hparams, params_type):
    """Init wandb and evaluate function."""
    global wandb
    import wandb

    logger = WandbLogger()

    with wandb.init(
        project=project_name, config=hparams, mode="disabled" if hparams.debug else "online", dir="logs/"
    ), ExceptionPrinter():
        # If called by wandb.agent,
        # this config will be set by Sweep Controller
        hparams = from_dict(params_type, update_nested_dict(asdict(hparams), wandb.config))
        if hparams.log_code:
            wandb.run.log_code()

        return func(hparams, logger=logger)


class DummyLogger(dict, object):
    """Dummy Logger that does nothing besides acting as dictionary."""

    def __repr__(self) -> str:
        """Return name of logger."""
        return "DummyLogger"

    def log(self, metrics: dict, step: int = None):
        """Log a dictionary of metrics (per step).

        Parameters
        ----------
        metrics : dict
            Dictonaries of scalar metrics.
        step : int, optional
            Step number, by default framework will use global step.
        """
        pass

    def log_params(self, params_dict):
        """Log the given hyperparameters.

        Parameters
        ----------
        params_dict : dict
            Dict of hyperparameters.
        """
        pass

    def finalize(self, all_param_norms=None):
        """Log additional plots or media.

        Parameters
        ----------
        all_param_norms : TODO
            _description_
        """
        pass

    def save_model(self, model: nn.Module, filename="model.ckpt"):
        """Save an equinox model to file.

        Parameters
        ----------
        model : equinox.Module
            The model to be saved
        filename : str, optional
            by default "model.ckpt"
        """
        pass

    def log_video(self, name: str, frames, step: int = None, fps=4, **kwargs):
        """Save a video given as array.

        Parameters
        ----------
        name : str
            _description_
        frames : array
            leading dimension for frames, then height, width, channels
        step : int, optional
            Step number, by default framework will use global step.
        fps : int, optional
            _description_, by default 4
        """
        pass


def update_nested_dict(d, u):
    """Update nested dict d with values from nested dict u.

    Parameters
    ----------
    d : dict
        Base dict
    u : dict
        Updates
    Returns
    -------
    dict
        d with values overwritten by u
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_nested_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class AimLogger(DummyLogger):
    """Wandb-like interface for aim."""

    def __repr__(self) -> str:
        """Return name of logger."""
        return "DummyLogger"

    @override
    def __init__(self, name, repo=None, hparams=None, run_name=""):
        """Create aim run."""
        global aim
        import aim

        self.run = aim.Run(experiment=name, repo=repo, log_system_params=True)
        if not isinstance(hparams, dict):
            hparams = asdict(hparams)
        self.log_params(hparams)
        self.run.name = run_name + " " + self.run.hash

    @override
    def log(self, metrics, step=None):
        """Loop over scalars and track them with aim."""
        for k, v in metrics.items():
            self.run.track(v, name=k, epoch=None if step is None else int(step))

    @override
    def log_params(self, params_dict):
        """Log the given hyperparameters.

        Parameters
        ----------
        params_dict : dict
            Dict of hyperparameters.
        """
        self.run["hparams"] = params_dict

    def __setitem__(self, key, value):
        """Log scalar for aim."""
        if not isinstance(value, dict):
            # Attempt conversion to float if not a dict
            value = float(value)
        self.run[key] = value

    def __getitem__(self, key):
        """Get value from aim run."""
        return self.run[key]

    @override
    def finalize(self, all_param_norms=None, x_vals=None):
        """Make lineplots for param norms and block until all metrics are logged."""
        if all_param_norms:
            all_param_norms = tree_stack(all_param_norms)
            self.log(
                {
                    f"Params/{k}": aim.Figure(px.line(x=x_vals, y=list(v.values()), title=k, labels=list(v.keys())))
                    for k, v in all_param_norms.items()
                    if v
                }
            )

        self.run.report_successful_finish(block=True)

    @override
    def save_model(self, model, filename="model.ckpt"):
        """Save the model to aim directory using equinox."""
        raise NotImplementedError()
        # run_artifacts_dir = os.path.join('artifacts/aim',  self.run.hash)
        # os.makedirs(run_artifacts_dir, exist_ok=True)
        # model_file = os.path.join(run_artifacts_dir, filename)

    @override
    def log_video(self, name, frames, step=None, fps=30, caption=""):
        """Log a video to wandb."""
        filename = os.path.join("logs/gifs", self.run.hash + ".gif")
        images = [Image.fromarray(frames[i].transpose(1, 2, 0)) for i in range(len(frames))]
        os.makedirs("logs/gifs", exist_ok=True)
        images[0].save(filename, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0)
        self.log({name: aim.Image(filename, caption=caption, format="gif")}, step=step)


class WandbLogger(DummyLogger):
    """Wandb-like interface for aim."""

    @override
    def log(self, metrics, step=None):
        """Log metrics to wandb."""
        wandb.log(metrics, step=step)

    def __setitem__(self, key, value):
        """Log scalar for wandb."""
        wandb.run.summary[key] = value

    def __getitem__(self, key):
        """Get value from aim run."""
        return wandb.run.summary[key]

    @override
    def finalize(self, all_param_norms: dict = None, x_vals=None):
        """Make lineplots for all items in all_param_norms."""
        if all_param_norms:
            all_param_norms = tree_stack(all_param_norms)
            wandb.log(
                {
                    f"Params/{k}": wandb.plot.line_series(
                        xs=x_vals,
                        ys=v.values(),
                        title=k,
                        keys=list(v.keys()),
                    )
                    for k, v in all_param_norms.items()
                }
            )

    @override
    def save_model(self, model, filename="model.ckpt"):
        """Save the model to wandb directory using orbax."""
        raise NotImplementedError()

    @override
    def log_video(self, name, frames, step=None, fps=30, caption=""):
        """Log a video to wandb."""
        wandb.log({name: wandb.Video(frames, fps=fps, caption=caption)}, step=step)


def with_logger(
    func: Callable,
    hparams: dict,
    logger_name: str,
    project_name: str,
    aim_repo: str = None,
    run_name="",
    hparams_type=None,
):
    """Wrap training function with logger."""
    if logger_name == "wandb":

        def pick_fun_and_run(_hparams, logger):
            return func(_hparams, logger=logger)

        return wandb_wrapper(project_name, pick_fun_and_run, hparams, params_type=hparams_type)
    elif logger_name == "aim":
        logger = AimLogger(project_name, repo=aim_repo, hparams=hparams, run_name=run_name)
        return func(hparams, logger=logger)  # TODO: Consider try catch to avoid broken aim repositories
    else:
        return func(hparams)


def calc_norms(norm_params: dict = {}, leaf_norm_params: dict = {}):
    """Compute norms and leaf norms of given dict of pytrees."""
    norms = {k: tree_norm(v) for k, v in norm_params.items()}
    param_norms = {k: leaf_norms(v) for k, v in leaf_norm_params.items()}
    return norms, param_norms


def log_norms(pytree):
    """Compute norms and leaf norms of given pytree."""
    flattened, _ = jtu.tree_flatten_with_path(pytree)
    flattened = {jtu.keystr(k): v for k, v in flattened}
    return calc_norms(flattened)[0]
