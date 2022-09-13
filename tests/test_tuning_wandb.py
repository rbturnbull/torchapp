import pytest
from unittest.mock import patch, PropertyMock
import math
from torchapp import params
from torchapp.tuning.wandb import get_parameter_config, get_sweep_config
from .tuning_test_app import TuningTestApp


def test_int():
    param = params.Param(annotation=int, tune_min=0, tune_max=100)
    config = get_parameter_config(param)

    assert isinstance(config, dict)
    assert config["min"] == 0
    assert config["max"] == 100
    assert config["distribution"] == "q_uniform"


def test_int_log():
    param = params.Param(annotation=int, log=True, tune_min=1, tune_max=100)
    config = get_parameter_config(param)

    assert isinstance(config, dict)
    assert config["min"] == 1
    assert config["max"] == 100
    assert config["distribution"] == "q_log_uniform_values"


def test_float_log():
    param = params.Param(annotation=float, log=True, tune_min=1.0, tune_max=100.0)
    config = get_parameter_config(param)

    assert isinstance(config, dict)
    assert config["min"] == 1
    assert config["max"] == 100
    assert config["distribution"] == "log_uniform_values"


def test_float():
    param = params.Param(annotation=float, tune_min=1.0, tune_max=100.0)
    config = get_parameter_config(param)

    assert isinstance(config, dict)
    assert config["min"] == 1.0
    assert config["max"] == 100.0
    assert config["distribution"] == "uniform"


def test_string():
    param = params.Param(annotation=str, tune_min=1.0, tune_max=100.0)
    with pytest.raises(NotImplementedError):
        get_parameter_config(param)


def test_get_sweep_config():
    app = TuningTestApp()
    config = get_sweep_config(app=app, name="run_name", method="random", min_iter=None)
    assert config == {
        'name': 'run_name',
        'method': 'random',
        'parameters': {
            'x': {'distribution': 'uniform', 'min': -10.0, 'max': 10.0},
            'a': {'distribution': 'q_uniform', 'min': 1, 'max': 12},
            'string': {'distribution': 'categorical', 'values': ['abcdefghij', 'baby', 'c']},
            'switch': {'distribution': 'categorical', 'values': [False, True]},
        },
        'metric': {'name': 'metric', 'goal': 'maximize'},
    }


def test_get_sweep_config_min_iter():
    app = TuningTestApp()
    config = get_sweep_config(app=app, name="run_name", method="bayes", min_iter=10)
    assert config == {
        'name': 'run_name',
        'method': 'bayes',
        'parameters': {
            'x': {'distribution': 'uniform', 'min': -10.0, 'max': 10.0},
            'a': {'distribution': 'q_uniform', 'min': 1, 'max': 12},
            'string': {'distribution': 'categorical', 'values': ['abcdefghij', 'baby', 'c']},
            'switch': {'distribution': 'categorical', 'values': [False, True]},
        },
        'metric': {'name': 'metric', 'goal': 'maximize'},
        'early_terminate': {'type': 'hyperband', 'min_iter': 10},
    }


def test_get_sweep_config_method_unknonw():
    app = TuningTestApp()
    with pytest.raises(NotImplementedError):
        get_sweep_config(app=app, name="run_name", method="tpe", min_iter=10)


def mock_agent(sweep_id, function, count, project):
    assert sweep_id == "sweep_id"
    with patch('wandb.config', dict()) as mock_config:
        learner = function()
        assert learner.recorder.values[-1][-1] == -9.0

    with patch('wandb.config', dict(x=-1.0)) as mock_config:
        learner = function()
        assert learner.recorder.values[-1][-1] == -19.0

    with patch('wandb.config', dict(x=2.0, string="abcdefghij")) as mock_config:
        learner = function()
        assert learner.recorder.values[-1][-1] == -1.0


@patch('wandb.sweep', lambda *args, **kwargs: "sweep_id")
@patch('wandb.agent', mock_agent)
@patch('wandb.init')
def test_wandb_tune(mock_wandb_init):
    app = TuningTestApp()
    runs = 30
    app.tune(engine="wandb", method="bayes", runs=runs, seed=42)
