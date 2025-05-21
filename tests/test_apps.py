from pathlib import Path
from unittest.mock import patch
import pytest
from click.testing import CliRunner
import torchapp as ta


def test_model_defaults_change():
    class DummyApp(ta.TorchApp):
        def model(self, size: int = ta.Param(default=2)):
            assert size == 2

    DummyApp().model()


def test_model_unimplemented_error():
    with pytest.raises(NotImplementedError):
        ta.TorchApp().model()


def test_dataloaders_unimplemented_error():
    with pytest.raises(NotImplementedError):
        ta.TorchApp().dataloaders()


def test_assert_initialized():
    class DummyApp(ta.TorchApp):
        def __init__(self):
            pass

    with pytest.raises(TorchAppInitializationError):
        DummyApp().cli()


def test_click():
    cli = ta.TorchApp.click()
    runner = CliRunner()
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert '[OPTIONS] COMMAND [ARGS]' in result.output


def test_str():
    class DummyApp(ta.TorchApp):
        pass

    app = DummyApp()
    assert str(app) == "DummyApp"


def test_checkpoint_default():
    with pytest.raises(ValueError):
        ta.TorchApp().checkpoint()


def test_load_checkpoint_path_not_found():
    with pytest.raises(FileNotFoundError):
        ta.TorchApp().load_checkpoint(checkpoint="model3.h5")


def test_goal_empty():
    class DummyApp(ta.TorchApp):
        def monitor(self):
            return None

    app = DummyApp()
    assert app.goal() == "minimize"
