from pathlib import Path
from optuna import samplers
import pytest
import tempfile

from .tuning_test_app import TuningTestApp


def test_optuna_tune_default():
    app = TuningTestApp()
    runs = 100
    result = app.tune(engine="optuna", runs=runs, seed=42)
    assert len(result.trials) == runs
    # assert result.best_value > 3.9
    assert isinstance(result.sampler, samplers.TPESampler)
    df = result.trials_dataframe()
    assert "params_a" in df.columns
    assert "params_x" in df.columns
    assert "params_string" in df.columns
    assert "params_switch" in df.columns
    assert "params_activation" in df.columns


def test_optuna_tune_random():
    app = TuningTestApp()
    runs = 100
    result = app.tune(engine="optuna", method="random", runs=runs, seed=42)
    assert len(result.trials) == runs
    # assert result.best_value > 3.9
    assert isinstance(result.sampler, samplers.RandomSampler)
    df = result.trials_dataframe()
    assert "params_a" in df.columns
    assert "params_x" in df.columns
    assert "params_string" in df.columns
    assert "params_switch" in df.columns
    assert "params_activation" in df.columns


def test_optuna_tune_cmaes():
    app = TuningTestApp()
    runs = 50
    result = app.tune(engine="optuna", method="cmaes", runs=runs, seed=42, string="abcdefghij")
    assert len(result.trials) == runs
    # assert result.best_value > 6
    assert isinstance(result.sampler, samplers.CmaEsSampler)
    df = result.trials_dataframe()
    assert "params_a" in df.columns
    assert "params_x" in df.columns
    assert "params_string" not in df.columns
    assert "params_switch" in df.columns
    assert "params_activation" in df.columns


def test_optuna_tune_tpe():
    app = TuningTestApp()
    runs = 50
    id = "test_optuna_tune_tpe"
    with tempfile.TemporaryDirectory() as tmp_dir:
        storage_path = Path(tmp_dir).resolve()/f"{id}.sqlite3"
        result = app.tune(engine="optuna", id=id, method="tpe", runs=runs, seed=42, output_dir=tmp_dir)
        # best_value = result.best_value   
        result_runs = len(result.trials)
        df = result.trials_dataframe()

        assert storage_path.exists()
        assert result_runs == runs
        # assert best_value > 5.0
        assert isinstance(result.sampler, samplers.TPESampler)

        assert "params_a" in df.columns
        assert "params_x" in df.columns
        assert "params_string" in df.columns
        assert "params_switch" in df.columns
        assert "params_activation" in df.columns


def test_get_sampler():
    from torchapp.tuning.optuna import get_sampler

    assert isinstance(get_sampler("tpe"), samplers.TPESampler)
    assert isinstance(get_sampler("cma-es"), samplers.CmaEsSampler)
    assert isinstance(get_sampler("random"), samplers.RandomSampler)
    with pytest.raises(NotImplementedError):
        get_sampler("bayes")
    with pytest.raises(NotImplementedError):
        get_sampler("grid")
