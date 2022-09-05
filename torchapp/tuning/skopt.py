from pathlib import Path

try:
    import skopt
    from skopt.space.space import Real, Integer, Categorical
except:
    raise Exception(
        "No module named 'skopt'. Please install this as an extra dependency or choose a different optimization engine."
    )

from ..util import call_func


def get_optimizer(method):
    method = method.lower()
    if method.startswith("bayes") or method.startswith("gp"):
        return skopt.gp_minimize
    elif method.startswith("random"):
        return skopt.dummy_minimize
    elif method.startswith("forest"):
        return skopt.forest_minimize
    elif method.startswith("gbrt") or method.startswith("gradientboost"):
        return skopt.gbrt_minimize
    raise NotImplementedError(f"Cannot interpret sampling method '{method}' using scikit-optimize.")


def get_param_search_space(param):
    if param.tune_choices:
        return Categorical(categories=param.tune_choices)

    prior = "uniform" if not param.log else "log-uniform"
    if param.annotation == float:
        return Real(param.tune_min, param.tune_max, prior=prior)

    if param.annotation == int:
        return Integer(param.tune_min, param.tune_max, prior=prior)

    raise NotImplementedError("scikit-optimize tuning engine cannot understand param '{name}': {param}")


def skopt_tune(
    app,
    file: str = "",
    name: str = None,
    method: str = "bayes",  # Should be enum
    runs: int = 1,
    seed: int = None,
    **kwargs,
):

    # Get tuning parameters
    tuning_params = app.tuning_params()
    used_tuning_params = {}
    for key, value in tuning_params.items():
        if key not in kwargs or kwargs[key] is None:
            used_tuning_params[key] = value

    # Get search space
    search_space = [get_param_search_space(param) for param in used_tuning_params.values()]

    optimizer = get_optimizer(method)

    def objective(*args):
        run_kwargs = dict(kwargs)

        for key, value in zip(used_tuning_params.keys(), *args):
            run_kwargs[key] = value

        base_output_dir = Path(run_kwargs.get("output_dir", ".")) / name

        run_number = 0
        while True:
            trial_name = f"trial-{run_number}"
            output_dir = base_output_dir / trial_name
            if not base_output_dir.exists():
                break
            run_number += 1

        run_kwargs["output_dir"] = output_dir
        run_kwargs["project_name"] = name
        run_kwargs["run_name"] = trial_name

        # Train
        learner = call_func(app.train, **run_kwargs)
        metric = app.get_best_metric(learner)

        # make negative if the goal is to maximize this metric
        if app.goal()[:3] != "min":
            metric = -metric

        return metric

    # TODO Read file if present

    results = optimizer(objective, search_space, n_calls=runs, random_state=seed)
    if file:
        skopt.dump(results, str(file))

    return results
