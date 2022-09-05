import pytest
from numpy.testing import assert_allclose
from .tuning_test_app import TuningTestApp
import skopt
from torchapp.tuning.skopt import get_optimizer


def test_skopt_tune_random():
    app = TuningTestApp()
    runs = 100
    result = app.tune(engine="skopt", method="random", runs=runs, seed=42)
    assert len(result.func_vals) == runs
    assert result.fun < -9.9
    assert result.space.n_dims == 3
    assert type(result.space[0][1]).__name__ == 'Real'
    assert type(result.space[1][1]).__name__ == 'Integer'
    assert type(result.space[2][1]).__name__ == 'Categorical'
    for found, desired in zip(result.x, [1.9370031589297412, 10, 'abcdefghij']):
        if isinstance(found, float):
            assert_allclose(found, desired)
        else:
            assert found == desired


def test_skopt_tune_bayes():
    app = TuningTestApp()
    runs = 30
    result = app.tune(engine="skopt", method="bayes", runs=runs, seed=42)
    assert len(result.func_vals) == runs
    assert result.fun < -9.97
    assert result.space.n_dims == 3
    assert type(result.space[0][1]).__name__ == 'Real'
    assert type(result.space[1][1]).__name__ == 'Integer'
    assert type(result.space[2][1]).__name__ == 'Categorical'
    for found, desired in zip(result.x, [1.9370031589297412, 6, 'abcdefghij']):
        if isinstance(found, float):
            assert_allclose(found, desired)
        else:
            assert found == desired


def test_skopt_tune_bayes_2param():
    app = TuningTestApp()
    runs = 40
    result = app.tune(engine="skopt", method="bayes", runs=runs, seed=42, x=4.0)
    assert len(result.func_vals) == runs
    assert result.fun < -5.999
    assert result.space.n_dims == 2
    assert type(result.space[0][1]).__name__ == 'Integer'
    assert type(result.space[1][1]).__name__ == 'Categorical'
    for found, desired in zip(result.x, [1, 'abcdefghij']):
        assert found == desired


def test_skopt_tune_forest():
    app = TuningTestApp()
    runs = 30
    result = app.tune(engine="skopt", method="forest", runs=runs, seed=42)
    assert len(result.func_vals) == runs
    assert result.fun < -9.96
    assert result.space.n_dims == 3
    assert type(result.space[0][1]).__name__ == 'Real'
    assert type(result.space[1][1]).__name__ == 'Integer'
    assert type(result.space[2][1]).__name__ == 'Categorical'
    for found, desired in zip(result.x, [1.9370031589297412, 10, 'abcdefghij']):
        if isinstance(found, float):
            assert_allclose(found, desired)
        else:
            assert found == desired


def test_skopt_tune_gradientboost():
    app = TuningTestApp()
    runs = 30
    result = app.tune(engine="skopt", method="gradientboost", runs=runs, seed=42)
    assert len(result.func_vals) == runs
    assert result.fun < -9.97
    assert result.space.n_dims == 3
    assert type(result.space[0][1]).__name__ == 'Real'
    assert type(result.space[1][1]).__name__ == 'Integer'
    assert type(result.space[2][1]).__name__ == 'Categorical'
    for found, desired in zip(result.x, [1.9535076154752264, 10, 'abcdefghij']):
        if isinstance(found, float):
            assert_allclose(found, desired)
        else:
            assert found == desired


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
