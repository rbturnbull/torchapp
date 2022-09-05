from typer.testing import CliRunner
from unittest.mock import patch
from torchapp.main import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Generate a torchapp project" in result.stdout


@patch("torchapp.main.cookiecutter_main")
def test_generate(cookiecutter_main):
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    cookiecutter_main.assert_called_once()
    kwargs = cookiecutter_main.call_args_list[0].kwargs
    assert kwargs['replay'] == False
    assert "cookiecutter" in kwargs['template']
    assert "torchapp" in kwargs['template']


@patch("torchapp.main.cookiecutter_main")
def test_template_path(cookiecutter_main):
    result = runner.invoke(app, ["--template", "path/to/template"])
    assert result.exit_code == 0
    cookiecutter_main.assert_called_once()
    kwargs = cookiecutter_main.call_args_list[0].kwargs
    assert "path/to/template" in kwargs['template']


@patch("torchapp.main.cookiecutter_main")
def test_template_github(cookiecutter_main):
    for gh in ['gh', 'github']:
        result = runner.invoke(app, ["--template", gh])
        assert result.exit_code == 0
        cookiecutter_main.assert_called_once()
        kwargs = cookiecutter_main.call_args_list[0].kwargs
        assert "https://github.com/rbturnbull/torchapp-cookiecutter" == kwargs['template']
        cookiecutter_main.reset_mock()
