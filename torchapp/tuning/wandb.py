from pathlib import Path
import wandb
from rich.console import Console
from rich.pretty import pprint
import math
from ..util import call_func

console = Console()


def get_parameter_config(param) -> dict:
    if param.tune_choices:
        return dict(
            distribution="categorical",
            values=param.tune_choices,
        )

    if param.annotation in [int, float]:
        assert param.tune_min is not None
        assert param.tune_max is not None

        distribution = "log_uniform_values" if param.tune_log else "uniform"
        if param.annotation == int:
            distribution = f"q_{distribution}"

        return dict(
            distribution=distribution,
            min=param.tune_min,
            max=param.tune_max,
        )

    raise NotImplementedError


def get_sweep_config(
    app,
    name: str = None,
    method: str = "random",  # Should be enum
    min_iter: int = None,
    **kwargs,
):
    parameters_config = dict()
    tuning_params = app.tuning_params()

    for key, value in tuning_params.items():
        if key not in kwargs or kwargs[key] is None:
            parameters_config[key] = get_parameter_config(value)

    if method not in ["grid", "random", "bayes"]:
        raise NotImplementedError(f"Cannot interpret sampling method '{method}' using wandb.")

    sweep_config = {
        "name": name,
        "method": method,
        "parameters": parameters_config,
    }
    if app.monitor():
        sweep_config["metric"] = dict(name=app.monitor(), goal=app.goal())

    if min_iter:
        sweep_config["early_terminate"] = dict(type="hyperband", min_iter=min_iter)
    console.print("Configuration for hyper-parameter tuning:", style="bold red")
    pprint(sweep_config)
    return sweep_config


def wandb_tune(
    app,
    sweep_id: str = None,
    name: str = None,
    method: str = "random",  # Should be enum
    runs: int = 1,
    min_iter: int = None,
    **kwargs,
) -> str:
    """
    Performs hyperparameter tuning using 'weights and biases' sweeps.

    Args:
        sweep_id(str, optional): The sweep ID, only necessary if sweep has already been generated for the project, defaults to None.
        name(str, optional): The name of the sweep run. This defaults to the project name with the suffix '-tuning' if left as None.
        method(str, optional): The hyperparameter sweep method, can be 'random' for random,
            'grid' for grid search,
            and 'bayes' for bayes optimisation.
            Defaults to 'random'.
        runs(int, optional): The number of runs. Defaults to 1.
        min_iter(int, optional): The minimum number of iterations if using early termination. If left empty, then early termination is not used.
        **kwargs:

    Returns:
        str: The sweep id. This can be used by other runs in the same sweep.
    """
    # Create a sweep id if it hasn't been given as an argument
    if not sweep_id:
        sweep_config = get_sweep_config(
            app=app,
            name=name,
            method=method,
            min_iter=min_iter,
            **kwargs,
        )
        sweep_id = wandb.sweep(sweep_config, project=name)
        console.print(f"The wandb sweep id is: {sweep_id}", style="bold red")

    def agent_train():
        with wandb.init() as run:
            run_kwargs = dict(kwargs)
            run_kwargs.update(wandb.config)
            if "output_dir" in run_kwargs:
                run_kwargs["output_dir"] = Path(run_kwargs["output_dir"]) / run.name

            console.print("Training with parameters:", style="bold red")
            pprint(run_kwargs)

            return call_func(app.train, **run_kwargs)

    wandb.agent(sweep_id, function=agent_train, count=runs, project=name)

    return sweep_id
