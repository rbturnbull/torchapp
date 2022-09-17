import pytest
import skopt
from torchapp.tuning.skopt import get_optimizer
import torchapp as ta
from .tuning_test_app import TuningTestApp

TOLERANCE = 0.1

def test_skopt_tune_random():
    app = TuningTestApp()
    runs = 100
    result = app.tune(engine="skopt", method="random", runs=runs, seed=42)
    assert len(result.func_vals) == runs
    assert result.fun < -4
    assert result.space.n_dims == 5
    assert type(result.space[0][1]).__name__ == 'Real'
    assert type(result.space[1][1]).__name__ == 'Integer'
    assert type(result.space[2][1]).__name__ == 'Categorical'
    assert type(result.space[3][1]).__name__ == 'Categorical'
    assert type(result.space[4][1]).__name__ == 'Categorical'


def test_skopt_tune_bayes():
    app = TuningTestApp()
    runs = 10
    result = app.tune(engine="skopt", method="bayes", runs=runs, seed=42, activation=ta.Activation.ReLU)
    assert len(result.func_vals) == runs
    assert result.fun < -9.0
    assert result.space.n_dims == 4
    assert type(result.space[0][1]).__name__ == 'Real'
    assert type(result.space[1][1]).__name__ == 'Integer'
    assert type(result.space[2][1]).__name__ == 'Categorical'
    assert type(result.space[3][1]).__name__ == 'Categorical'


def test_skopt_tune_bayes_2param():
    app = TuningTestApp()
    runs = 10
    result = app.tune(engine="skopt", method="bayes", runs=runs, switch=True, seed=42, activation=ta.Activation.ReLU, x=4.0)
    assert len(result.func_vals) == runs
    assert result.fun < 1
    assert result.space.n_dims == 2
    assert type(result.space[0][1]).__name__ == 'Integer'
    assert type(result.space[1][1]).__name__ == 'Categorical'


def test_skopt_tune_forest():
    app = TuningTestApp()
    runs = 20
    result = app.tune(engine="skopt", method="forest", runs=runs, seed=42)
    assert len(result.func_vals) == runs
    assert result.fun < 13
    assert result.space.n_dims == 5
    assert type(result.space[0][1]).__name__ == 'Real'
    assert type(result.space[1][1]).__name__ == 'Integer'
    assert type(result.space[2][1]).__name__ == 'Categorical'
    assert type(result.space[3][1]).__name__ == 'Categorical'
    assert type(result.space[4][1]).__name__ == 'Categorical'


def test_skopt_tune_gradientboost():
    app = TuningTestApp()
    runs = 20
    result = app.tune(engine="skopt", method="gradientboost", runs=runs, seed=42)
    assert len(result.func_vals) == runs
    assert result.fun < 52
    assert result.space.n_dims == 5
    assert type(result.space[0][1]).__name__ == 'Real'
    assert type(result.space[1][1]).__name__ == 'Integer'
    assert type(result.space[2][1]).__name__ == 'Categorical'
    assert type(result.space[3][1]).__name__ == 'Categorical'
    assert type(result.space[4][1]).__name__ == 'Categorical'


def test_get_optimizer():
    assert get_optimizer("bayes") == skopt.gp_minimize
    assert get_optimizer("gp") == skopt.gp_minimize
    assert get_optimizer("forest") == skopt.forest_minimize
    assert get_optimizer("random") == skopt.dummy_minimize
    assert get_optimizer("gbrt") == skopt.gbrt_minimize
    assert get_optimizer("gradientboost") == skopt.gbrt_minimize
    with pytest.raises(NotImplementedError):
        get_optimizer("tpe")
    with pytest.raises(NotImplementedError):
        get_optimizer("cma")
