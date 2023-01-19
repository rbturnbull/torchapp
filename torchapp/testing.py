import re
import sys
import yaml
import importlib
import pytest
import torch
from typing import get_type_hints
from click.testing import CliRunner
from pathlib import Path
import difflib
from torch import nn
from collections import OrderedDict
from fastai.data.core import DataLoaders
from fastai.learner import Learner
from rich.console import Console
from fastai.torch_core import TensorBase

from .apps import TorchApp

console = Console()

######################################################################
## pytest fixtures
######################################################################


@pytest.fixture
def interactive(request):
    return request.config.getoption("-s") == "no"


######################################################################
## YAML functions from https://stackoverflow.com/a/8641732
######################################################################
class quoted(str):
    pass


def quoted_presenter(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')


yaml.add_representer(quoted, quoted_presenter)


class literal(str):
    pass


def literal_presenter(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(literal, literal_presenter)


def ordered_dict_presenter(dumper, data):
    return dumper.represent_dict(data.items())


yaml.add_representer(OrderedDict, ordered_dict_presenter)

######################################################################
## TorchApp Testing Utils
######################################################################


class TorchAppTestCaseError(Exception):
    pass


def get_diff(a, b):
    a = str(a).splitlines(1)
    b = str(b).splitlines(1)

    diff = difflib.unified_diff(a, b)

    return "\n".join(diff).replace("\n\n", "\n")


def clean_output(output):
    if isinstance(output, (TensorBase, torch.Tensor)):
        output = f"{type(output)} {tuple(output.shape)}"
    output = str(output)
    output = re.sub(r"0[xX][0-9a-fA-F]+", "<HEX>", output)
    return output


def strip_whitespace_recursive(obj):
    if isinstance(obj, str):
        return re.sub(r"\s*", " ", obj, re.MULTILINE)
    if isinstance(obj, dict):
        return {k:strip_whitespace_recursive(v) for k,v in obj.items()}

    return obj


def assert_output(file: Path, interactive: bool, params: dict, output, expected, regenerate: bool = False):
    """
    Tests to see if the output is the same as the expected data and allows for saving a new version of the expected files if needed.

    Args:
        file (Path): The path to the expected file in yaml format.
        interactive (bool): Whether or not to prompt for replacing the expected file.
        params (dict): The dictionary of parameters to store in the expected file.
        output (str): The string representation of the output from the app.
        expected (str): The expected output from the yaml file.
    """
    if expected == output:
        return

    # if expected and output are both strings, check to see if they are equal when normalizing whitespace
    expected_cleaned = strip_whitespace_recursive(expected)
    output_cleaned = strip_whitespace_recursive(output)
    if expected_cleaned == output_cleaned:
        return

    if isinstance(expected, dict) and isinstance(output, dict):
        keys = set(expected.keys()) | set(output.keys())
        diff = {}
        for key in keys:
            diff[key] = get_diff(expected.get(key, ""), output.get(key, ""))
            if diff[key]:
                console.print(diff[key])
    else:
        diff = get_diff(str(expected), str(output))
        console.print(diff)

    if interactive or regenerate:
        # If we aren't automatically regenerating the expected files, then prompt the user
        if not regenerate:
            prompt_response = input(
                f"\nExpected file '{file.name}' does not match test output (see diff above).\n"
                "Should this file be replaced? (y/N) "
            )
            regenerate = prompt_response.lower() == "y"

        if regenerate:
            with open(file, "w") as f:
                output_for_yaml = literal(output) if isinstance(output, str) and "\n" in output else output
                # order the params dictionary if necessary
                if isinstance(params, dict):
                    params = OrderedDict(params)

                data = OrderedDict(params=params, output=output_for_yaml)
                yaml.dump(data, f)
                return

    raise TorchAppTestCaseError(diff)


class TorchAppTestCase:
    """Automated tests for TorchApp classes"""

    app_class = None
    expected_base = None

    def get_expected_base(self) -> Path:
        if not self.expected_base:
            module = importlib.import_module(self.__module__)
            self.expected_base = Path(module.__file__).parent / "expected"

        self.expected_base = Path(self.expected_base)
        return self.expected_base

    def get_expected_dir(self) -> Path:
        """
        Returns the path to the directory where the expected files.

        It creates the directory if it doesn't already exist.
        """
        expected_dir = self.get_expected_base() / self.__class__.__name__
        expected_dir.mkdir(exist_ok=True, parents=True)
        return expected_dir

    def get_app(self) -> TorchApp:
        """
        Returns an instance of the app for this test case.

        It instantiates an object from `app_class`.
        Override `app_class` or this method so the correct app is returned from calling this method.
        """
        # pdb.set_trace()
        assert self.app_class is not None
        app = self.app_class()

        assert isinstance(app, TorchApp)
        return app

    def subtest_dir(self, name: str):
        directory = self.get_expected_dir() / name
        directory.mkdir(exist_ok=True, parents=True)
        return directory

    def subtest_files(self, name: str):
        directory = self.subtest_dir(name)
        files = list(directory.glob("*.yaml"))
        return files

    def subtests(self, app, name: str):
        files = self.subtest_files(name)

        if len(files) == 0:
            pytest.skip(
                f"Skipping test for '{name}' because no expected files were found in:\n" f"{self.subtest_dir(name)}."
            )

        for file in files:
            with open(file) as f:
                file_dict = yaml.safe_load(f) or {}
                params = file_dict.get("params", {})
                output = file_dict.get("output", "")

                yield params, output, file

    def test_model(self, interactive: bool):
        """
        Tests the method of a TorchApp to create a pytorch model.

        The expected output is the string representation of the model created.

        Args:
            interactive (bool): Whether or not failed tests should prompt the user to regenerate the expected files.
        """
        app = self.get_app()
        name = sys._getframe().f_code.co_name
        method_name = name[5:] if name.startswith("test_") else name
        regenerate = False

        if interactive:
            if not self.subtest_files(name):
                prompt_response = input(
                    f"\nNo expected files for '{name}' when testing '{app}'.\n"
                    "Should a default expected file be automatically generated? (y/N) "
                )
                if prompt_response.lower() == "y":
                    regenerate = True
                    directory = self.subtest_dir(name)
                    with open(directory / f"{method_name}_default.yaml", "w") as f:
                        # The output will be autogenerated later
                        data = OrderedDict(params={}, output="")
                        yaml.dump(data, f)

        for params, expected_output, file in self.subtests(app, name):
            model = app.model(**params)
            if model is None:
                model_summary = "None"
            else:
                assert isinstance(model, nn.Module)
                model_summary = str(model)

            assert_output(file, interactive, params, model_summary, expected_output, regenerate=regenerate)

    def test_dataloaders(self, interactive: bool):
        app = self.get_app()
        for params, expected_output, file in self.subtests(app, sys._getframe().f_code.co_name):
            # Make all paths relative to the result of get_expected_dir()
            modified_params = dict(params)
            hints = get_type_hints(app.dataloaders)
            for key, value in hints.items():
                # if this is a union class, then loop over all options
                if not isinstance(value, type) and hasattr(value, "__args__"):  # This is the case for unions
                    values = value.__args__
                else:
                    values = [value]

                for v in values:
                    if key in params and Path in v.__mro__:
                        relative_path = params[key]
                        modified_params[key] = (self.get_expected_dir() / relative_path).resolve()
                        break

            dataloaders = app.dataloaders(**modified_params)

            assert isinstance(dataloaders, DataLoaders)

            batch = dataloaders.train.one_batch()
            dataloaders_summary = OrderedDict(
                type=type(dataloaders).__name__,
                train_size=len(dataloaders.train),
                validation_size=len(dataloaders.valid),
                batch_x_type=type(batch[0]).__name__,
                batch_y_type=type(batch[1]).__name__,
                batch_x_shape=str(batch[0].shape),
                batch_y_shape=str(batch[1].shape),
            )

            assert_output(file, interactive, params, dataloaders_summary, expected_output)

    def perform_subtests(self, interactive: bool, name: str):
        """
        Performs a number of subtests for a method on the app.

        Args:
            interactive (bool): Whether or not the user should be prompted for creating or regenerating expected files.
            name (str): The name of the method to be tested with the string `test_` prepended to it.
        """
        app = self.get_app()
        regenerate = False
        method_name = name[5:] if name.startswith("test_") else name
        method = getattr(app, method_name)

        if interactive:
            if not self.subtest_files(name):
                prompt_response = input(
                    f"\nNo expected files for '{name}' when testing '{app}'.\n"
                    "Should a default expected file be automatically generated? (y/N) "
                )
                if prompt_response.lower() == "y":
                    regenerate = True
                    directory = self.subtest_dir(name)
                    with open(directory / f"{method_name}_default.yaml", "w") as f:
                        # The output will be autogenerated later
                        data = OrderedDict(params={}, output="")
                        yaml.dump(data, f)

        for params, expected_output, file in self.subtests(app, name):
            modified_params = dict(params)
            hints = get_type_hints(method)
            for key, value in hints.items():
                # if this is a union class, then loop over all options
                if not isinstance(value, type) and hasattr(value, "__args__"):  # This is the case for unions
                    values = value.__args__
                else:
                    values = [value]

                for v in values:
                    if key in params and Path in v.__mro__:
                        relative_path = params[key]
                        modified_params[key] = (self.get_expected_dir() / relative_path).resolve()
                        break

            output = clean_output(method(**modified_params))
            assert_output(file, interactive, params, output, expected_output, regenerate=regenerate)

    def test_goal(self, interactive: bool):
        self.perform_subtests(interactive=interactive, name=sys._getframe().f_code.co_name)

    def test_metrics(self, interactive: bool):
        self.perform_subtests(interactive=interactive, name=sys._getframe().f_code.co_name)

    def test_loss_func(self, interactive: bool):
        self.perform_subtests(interactive=interactive, name=sys._getframe().f_code.co_name)

    def test_monitor(self, interactive: bool):
        self.perform_subtests(interactive=interactive, name=sys._getframe().f_code.co_name)

    def test_activation(self, interactive: bool):
        self.perform_subtests(interactive=interactive, name=sys._getframe().f_code.co_name)

    def test_pretrained_location(self, interactive: bool):
        self.perform_subtests(interactive=interactive, name=sys._getframe().f_code.co_name)

    def test_one_batch_output_size(self, interactive: bool):
        self.perform_subtests(interactive=interactive, name=sys._getframe().f_code.co_name)

    def test_one_batch_loss(self, interactive: bool):
        self.perform_subtests(interactive=interactive, name=sys._getframe().f_code.co_name)

    def test_cli(self, interactive: bool):
        app = self.get_app()
        regenerate = False
        name = sys._getframe().f_code.co_name

        if interactive:
            if not self.subtest_files(name):
                prompt_response = input(
                    f"\nNo expected files for '{name}' when testing '{app}'.\n"
                    "Should default expected files be automatically generated? (y/N) "
                )
                if prompt_response.lower() == "y":
                    regenerate = True
                    directory = self.subtest_dir(name)
                    default_commands = ["train", "predict", "show-batch", "tune"]
                    for command in default_commands:
                        with open(directory / f"{command}_help.yaml", "w") as f:
                            data = OrderedDict(
                                params=[command, "--help"], output=""
                            )  # The output will be autogenerated later
                            yaml.dump(data, f)
                    plain_commands = ["bibtex", "bibliography"]
                    for command in plain_commands:
                        with open(directory / f"{command}.yaml", "w") as f:
                            data = OrderedDict(params=[command], output="")  # The output will be autogenerated later
                            yaml.dump(data, f)

        runner = CliRunner()
        for params, expected_output, file in self.subtests(app, name):
            result = runner.invoke(app.cli(), params)
            output = dict(
                stdout=literal(result.stdout),
                exit_code=result.exit_code,
            )

            assert_output(file, interactive, params, output, expected_output, regenerate=regenerate)
