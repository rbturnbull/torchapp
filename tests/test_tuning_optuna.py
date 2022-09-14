from pathlib import Path
from optuna import samplers
import pytest

from .tuning_test_app import TuningTestApp


def test_optuna_tune_default():
    app = TuningTestApp()
    runs = 10
    result = app.tune(engine="optuna", runs=runs, seed=42)
    assert len(result.trials) == runs
    assert result.best_value > 9.93
    assert isinstance(result.sampler, samplers.RandomSampler)
    df = result.trials_dataframe()
    assert "params_a" in df.columns
    assert "params_x" in df.columns
    assert "params_string" in df.columns
    assert "params_switch" in df.columns


def test_optuna_tune_cmaes():
    app = TuningTestApp()
    runs = 10
    result = app.tune(engine="optuna", method="cmaes", runs=runs, seed=42, string="abcdefghij")
    assert len(result.trials) == runs
    assert result.best_value > 6
    assert isinstance(result.sampler, samplers.CmaEsSampler)
    df = result.trials_dataframe()
    assert "params_a" in df.columns
    assert "params_x" in df.columns
    assert "params_string" not in df.columns
    assert "params_switch" in df.columns


def test_optuna_tune_tpe():
    app = TuningTestApp()
    runs = 10
    id = "test_optuna_tune_tpe"
    storage_path = Path(f"{id}.sqlite3")
    if storage_path.exists():
        raise FileExistsError(
            f"The file {storage_path} exists in the current working directory. "
            "This is probably the result of a failed test in the past. "
            "This file will need to be removed before the tests can run again."
        )

    result = app.tune(engine="optuna", id=id, method="tpe", runs=runs, seed=42)
    best_value = result.best_value   
    result_runs = len(result.trials)
    df = result.trials_dataframe()

    assert storage_path.exists()
    storage_path.unlink()

    assert result_runs == runs
    assert best_value > 9.9
    assert isinstance(result.sampler, samplers.TPESampler)

    assert "params_a" in df.columns
    assert "params_x" in df.columns
    assert "params_string" in df.columns
    assert "params_switch" in df.columns
    


def test_get_sampler():
    from torchapp.tuning.optuna import get_sampler

    assert isinstance(get_sampler("tpe"), samplers.TPESampler)
    assert isinstance(get_sampler("cma-es"), samplers.CmaEsSampler)
    assert isinstance(get_sampler("random"), samplers.RandomSampler)
    with pytest.raises(NotImplementedError):
        get_sampler("bayes")
    with pytest.raises(NotImplementedError):
        get_sampler("grid")
