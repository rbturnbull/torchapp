from sklearn.base import BaseEstimator, RegressorMixin
from pathlib import Path
from skopt.learning import GradientBoostingQuantileRegressor
from .util import best_model_score



try:
    import skopt
    from skopt.space.space import Real, Integer, Categorical
    from skopt.callbacks import CheckpointSaver
except ImportError as err:
    raise Exception(
        f"Error loading 'skopt'. Please install this as an extra dependency or choose a different optimization engine.\n{err}"
    )


class WrappedGBQR(RegressorMixin, BaseEstimator):
    def __init__(self, **kwargs):
        self.model = GradientBoostingQuantileRegressor(**kwargs)

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)



def get_optimizer(method):
    method = method.lower()
    if method.startswith("bayes") or method.startswith("gp") or not method:
        return skopt.gp_minimize
    elif method.startswith("random"):
        return skopt.dummy_minimize
    elif method.startswith("forest"):
        return skopt.forest_minimize
    elif method.startswith("gbrt") or method.startswith("gradientboost"):
        return skopt.gbrt_minimize
    raise NotImplementedError(f"Cannot interpret sampling method '{method}' using scikit-optimize.")


def get_param_search_space(param, name):
    if param.tune_choices:
        return Categorical(categories=param.tune_choices, name=name)

    prior = "uniform" if not param.tune_log else "log-uniform"
    if param.annotation == float:
        return Real(param.tune_min, param.tune_max, prior=prior, name=name)

    if param.annotation == int:
        return Integer(param.tune_min, param.tune_max, prior=prior, name=name)

    raise NotImplementedError("scikit-optimize tuning engine cannot understand param '{name}': {param}")


class SkoptPlot(object):
    """
    Save current state after each iteration with :class:`skopt.dump`.
    """
    def __init__(self, path:Path, format):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.format = format

    def __call__(self, result):
        from skopt.plots import plot_convergence, plot_evaluations, plot_objective
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        plot_convergence(result)
        plt.savefig(str(self.path/f"convergence.{self.format}"), format=self.format)

        plot_evaluations(result)
        plt.savefig(str(self.path/f"evaluations.{self.format}"), format=self.format)

        if result.models:
            plot_objective(result)
            plt.savefig(str(self.path/f"objective.{self.format}"), format=self.format)


class SkoptObjective():
    def __init__(self, app, kwargs, used_tuning_params, name, base_output_dir):
        self.app = app
        self.kwargs = kwargs
        self.used_tuning_params = used_tuning_params
        self.name = name
        self.base_output_dir = Path(base_output_dir)

    def __call__(self, *args):
        run_kwargs = dict(self.kwargs)

        for key, value in zip(self.used_tuning_params.keys(), *args):
            run_kwargs[key] = value

        run_number = 0
        while True:
            trial_name = f"trial-{run_number}"
            output_dir = self.base_output_dir / trial_name
            if not output_dir.exists():
                break
            run_number += 1

        run_kwargs["output_dir"] = output_dir
        run_kwargs["project_name"] = self.name
        run_kwargs["run_name"] = trial_name

        # Train
        _, trainer = self.app.train(**run_kwargs)
        metric = best_model_score(trainer)

        # make negative if the goal is to maximize this metric
        if self.app.goal()[:3] != "min":
            metric = -metric

        return metric

    def __deepcopy__(self, memo):
        """ Returns None for deepcopy because this shouldn't be copied into the checkpoint. """
        return None


def skopt_tune(
    app,
    file: str = "",
    name: str = None,
    method: str = "",  # Should be enum
    runs: int = 1,
    seed: int = None,
    **kwargs,
):

    # Get tuning parameters
    tuning_params = app.tuning_params()
    used_tuning_params = {}
    for key, value in tuning_params.items():
        # Skip parameters that have been passed as arguments the tune of the app
        if key in app.original_kwargs['tune'] and app.original_kwargs['tune'].get(key) is not None:
            continue

        used_tuning_params[key] = value

    # Get search space
    search_space = [get_param_search_space(param, name=key) for key, param in used_tuning_params.items()]

    optimizer = get_optimizer(method)

    if not name:
        name = f"{app.project_name()}-tuning"
    base_output_dir = Path(kwargs.get("output_dir", ".")) / name

    optimizer_kwargs = dict(n_calls=runs, random_state=seed, callback=[])

    if False:
        optimizer_kwargs["callback"].append(SkoptPlot(base_output_dir, "svg"))

    if file:
        file = Path(file)
        # if a file is given, first try to read from that file the results and then use it as a checkpoint
        # https://scikit-optimize.github.io/stable/auto_examples/interruptible-optimization.html
        if file.exists():
            try:
                checkpoint = skopt.load(file)
                x0 = checkpoint.x_iters
                y0 = checkpoint.func_vals
                optimizer_kwargs['x0'] = x0
                optimizer_kwargs['y0'] = y0
            except Exception as e:
                raise IOError(f"Cannot read scikit-optimize checkpoint file '{file}': {e}")

        checkpoint_saver = CheckpointSaver(str(file), compress=9, store_objective=False)
        optimizer_kwargs['callback'].append( checkpoint_saver )

    objective = SkoptObjective(app, kwargs, used_tuning_params, name, base_output_dir)
    
    if optimizer == skopt.gbrt_minimize:
        optimizer_kwargs['base_estimator'] = WrappedGBQR()

    results = optimizer(objective, search_space, **optimizer_kwargs)

    return results
