from optuna import samplers
import pytest

from .tuning_test_app import TuningTestApp


def test_optuna_tune_default():
    app = TuningTestApp()
    runs = 120
    result = app.tune(engine="optuna", runs=runs, seed=42)
    assert len(result.trials) == runs
    assert result.best_value > 9.93
    assert result.best_trial.number == 104
    assert isinstance(result.sampler, samplers.RandomSampler)
    df = result.trials_dataframe()
    assert "params_a" in df.columns
    assert "params_x" in df.columns
    assert "params_string" in df.columns


def test_optuna_tune_cmaes():
    app = TuningTestApp()
    result = app.tune(engine="optuna", method="cmaes", runs=15, seed=42, string="abcdefghij")
    assert len(result.trials) == 15
    assert result.best_value > 9.8
    assert result.best_trial.number == 6
    assert isinstance(result.sampler, samplers.CmaEsSampler)
    df = result.trials_dataframe()
    assert "params_a" in df.columns
    assert "params_x" in df.columns
    assert "params_string" not in df.columns


def test_optuna_tune_tpe():
    app = TuningTestApp()
    runs = 100
    result = app.tune(engine="optuna", method="tpe", runs=runs, seed=42)
    assert len(result.trials) == runs
    assert result.best_value > 9.9999
    assert result.best_trial.number == 72
    assert isinstance(result.sampler, samplers.TPESampler)
    df = result.trials_dataframe()
    assert "params_a" in df.columns
    assert "params_x" in df.columns
    assert "params_string" in df.columns


def test_get_sampler():
    from torchapp.tuning.optuna import get_sampler

    assert isinstance(get_sampler("tpe"), samplers.TPESampler)
    assert isinstance(get_sampler("cma-es"), samplers.CmaEsSampler)
    assert isinstance(get_sampler("random"), samplers.RandomSampler)
    with pytest.raises(NotImplementedError):
        get_sampler("bayes")
    with pytest.raises(NotImplementedError):
        get_sampler("grid")
