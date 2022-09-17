from pathlib import Path

try:
    import optuna
    from optuna.integration import FastAIV2PruningCallback
    from optuna import samplers
except:
    raise Exception(
        "No module named 'optuna'. Please install this as an extra dependency or choose a different optimization engine."
    )

from ..util import call_func


def get_sampler(method, seed=0):
    method = method.lower()
    if method.startswith("tpe"):
        return samplers.TPESampler(seed=seed)
    elif method.startswith("cma"):
        return samplers.CmaEsSampler(seed=seed)
    # elif method.startswith("grid"):
    #     return samplers.GridSampler()
    elif method.startswith("random"):
        return samplers.RandomSampler(seed=seed)

    raise NotImplementedError(f"Cannot interpret sampling method '{method}' using Optuna.")


def suggest(trial, name, param):
    if param.tune_choices:
        return trial.suggest_categorical(name, param.tune_choices)
    elif param.annotation == float:
        return trial.suggest_float(name, param.tune_min, param.tune_max, log=param.tune_log)
    elif param.annotation == int:
        return trial.suggest_int(name, param.tune_min, param.tune_max, log=param.tune_log)

    raise NotImplementedError("Optuna Tuning Engine cannot understand param '{name}': {param}")


def optuna_tune(
    app,
    storage: str = "",
    name: str = None,
    method: str = "tpe",  # Should be enum
    runs: int = 1,
    seed: int = None,
    **kwargs,
):
    output_dir = Path(kwargs.get("output_dir", "."))
    output_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial):
        run_kwargs = dict(kwargs)

        tuning_params = app.tuning_params()

        for key, value in tuning_params.items():
            if key not in kwargs or kwargs[key] is None:
                run_kwargs[key] = suggest(trial, key, value)

        trial_name = f"trial-{trial.number}"

        output_dir = Path(run_kwargs.get("output_dir", "."))
        run_kwargs["output_dir"] = output_dir / trial.study.study_name / trial_name
        run_kwargs["project_name"] = trial.study.study_name
        run_kwargs["run_name"] = trial_name

        # Train
        learner = call_func(app.train, **run_kwargs)

        # Return metric from recorder
        return app.get_best_metric(learner)

    if not storage:
        storage = None
    elif "://" not in storage:
        storage_path = output_dir/f"{storage}.sqlite3"
        storage = f"sqlite:///{storage_path.resolve()}"

    study = optuna.create_study(
        study_name=name,
        storage=storage,
        sampler=get_sampler(method, seed=seed),
        load_if_exists=True,
        direction=app.goal(),
    )
    study.optimize(objective, n_trials=runs)
    return study
