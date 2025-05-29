import re
import pytest
from torchapp.testing import CliRunner
import torchapp as ta


class DummyApp(ta.TorchApp):
    @ta.method
    def model(self, size: int = ta.Param(default=2)):
        assert size == 2

    @ta.method
    def data(self, **kwargs):
        pass


def test_model_defaults_change():
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
    ANSI_ESCAPE = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
    output = ANSI_ESCAPE.sub('', result.output)
    assert "Show this message and exit." in output
    assert 'Usage: ' in output
    assert 'predict [OPTIONS]' in output


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
