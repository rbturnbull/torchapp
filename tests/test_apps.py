import pytest
from typer.testing import CliRunner
import torchapp as ta


def test_model_defaults_change():
    class DummyApp(ta.TorchApp):
        @ta.method
        def model(self, size: int = ta.Param(default=2)):
            assert size == 2

    DummyApp().model()


def test_model_unimplemented_error():
    with pytest.raises(NotImplementedError):
        ta.TorchApp().model()


def test_data_unimplemented_error():
    with pytest.raises(NotImplementedError):
        ta.TorchApp().data()


def test_main_help():
    app = ta.TorchApp().main_app
    runner = CliRunner()
    result = runner.invoke(app, ['--help'])
    assert result.exit_code == 0
    assert "Show this message and exit." in result.output
    assert 'Usage: predict [OPTIONS]' in result.output


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
