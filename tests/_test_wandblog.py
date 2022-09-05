from torchapp.callbacks import WandbCallbackTime

from fastai.callback.wandb import WandbCallback
import unittest
import sys, os, shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from torchapp.logging import wandb
from torchapp.logging import WandbMixin
import wandb
from torchapp.examples.iris import IrisApp

ARTIFACT_PATH = "tmp/artifact.txt"
ARTIFACT_NAME = "test-artifact"
MODEL_RUN_NAME = "test_run"
TEST_CONFIG = {"bs": 5, "epochs": 7}


def mock_init_offline(dir, project, reinit, config, **kwargs):
    # kwargs['mode'] = 'disabled'
    # run = wandb.init(dir, project, reinit, config, **kwargs
    # )
    # assert run == wandb.run
    return dict(dir=dir, project=project, reinit=reinit, config=config, kwargs=kwargs)


def mock_artifact_add_file(artifact_path):
    assert artifact_path == ARTIFACT_PATH


class MockWandbArtifact:
    def __init__(self, artifact_name, type, **kwargs):
        self.artifact_name = artifact_name
        self.type = type
        self.kwargs = kwargs

    def add_file(self, artifact_path):
        assert str(artifact_path) == ARTIFACT_PATH
        self.upload = "upload"

    def add_reference(self, artifact_path):
        assert str(artifact_path) == ARTIFACT_PATH
        self.upload = "reference"


def mock_wandb_artifact(artifact_name, type, **kwargs):
    return MockWandbArtifact(artifact_name, type, **kwargs)


def mock_log_artifact(artifact):
    assert artifact.artifact_name == ARTIFACT_NAME
    assert artifact.type == "test"
    assert artifact.upload == "upload"


def mock_log_artifact_reference(artifact):
    assert artifact.artifact_name == ARTIFACT_NAME
    assert artifact.type == "test"
    assert artifact.upload == "reference"


class MockWandbModelArtifact(MockWandbArtifact):
    def add_file(self, artifact_path):
        self.artifact_path = artifact_path
        self.upload = "upload"

    def add_reference(self, artifact_path):
        self.artifact_path = artifact_path
        self.upload = "reference"


def mock_log_artifact_model(artifact):
    assert artifact.artifact_name == MODEL_RUN_NAME
    assert artifact.type == "model"
    assert artifact.upload == "upload"
    print(artifact.artifact_path)


def mock_sweep_init():
    wandb.init(mode="disabled")


def mock_train(obj, **kwargs):
    pass


def mock_agent(id, function, count, project):
    pass


SWEEP_METHOD = "random"


def check_sweep_config(sweep_config):
    assert "name" in sweep_config
    assert "method" in sweep_config

    assert "parameters" in sweep_config
    assert "-tuning" in sweep_config["name"]
    assert sweep_config["method"] == SWEEP_METHOD

    assert "batch_size" in sweep_config["parameters"]
    bs_param = sweep_config["parameters"]["batch_size"]
    assert "distribution" in bs_param
    assert bs_param["distribution"] == "q_log_uniform"

    assert ("min" in bs_param) and ("max" in bs_param)
    assert (type(bs_param["min"]) == float) and (type(bs_param["max"]) == float)


def mock_test_sweep(sweep_config, project):
    check_sweep_config(sweep_config)
    return "test-id"


class TestApp(WandbMixin, IrisApp):

    pass


class WandbMixinTest(unittest.TestCase):
    def setUp(self) -> None:
        self.wandb_app = TestApp()
        self.test_config = TEST_CONFIG.copy()
        self.wandb_path = Path("wandb_test")
        self.artifact_path = Path("tmp")

        if self.wandb_path.exists():
            shutil.rmtree(self.wandb_path)
        self.wandb_path.mkdir(exist_ok=True)

        if self.artifact_path.exists():
            shutil.rmtree(self.wandb_path)
        self.artifact_path.mkdir(exist_ok=True)
        return super().setUp()

    def tearDown(self) -> None:
        shutil.rmtree(self.wandb_path)
        shutil.rmtree(self.artifact_path)
        return super().tearDown()

    def test_init(self):

        self.wandb_app.init_run(
            "wandb_test", upload_model=False, mode="disabled", config=self.test_config
        )
        assert self.wandb_app.run is wandb.run
        # print(self.wandb_app.run.project_name)
        # print(self.wandb_app.project_name())
        # assert self.wandb_app.run.project_name == 'test_run'

        # assert self.wandb_app.run.config['bs'] == 5
        # assert self.wandb_app.run.config['epochs'] == 7

    @patch("torchapp.logging.wandb.init", mock_init_offline)
    def test_init_params(self):
        self.wandb_app.init_run(
            "wandb_test",
            project_name="test_run2",
            upload_model=True,
            mode="offline",
            config=self.test_config,
        )
        # assert self.wandb_app.run[0] is wandb.run

        self.assertEqual(self.wandb_app.run["project"], "test_run2")

        self.assertEqual(self.wandb_app.run["config"]["bs"], 5)
        self.assertEqual(self.wandb_app.run["config"]["epochs"], 7)
        self.assertEqual(self.wandb_app.upload_model, True)
        self.assertEqual(self.wandb_app.run["kwargs"]["mode"], "offline")

    def test_log(self):

        self.test_init()
        self.wandb_app.log({"test_param": "test_out"})

    @patch("torchapp.logging.wandb.Artifact", MockWandbArtifact)
    @patch.object(wandb.sdk.wandb_run.Run, "log_artifact", mock_log_artifact)
    def test_log_artifact(self):

        self.test_init()
        with open(self.artifact_path / "artifact.txt", "w") as file:
            file.write("test")
        self.wandb_app.log_artifact(
            self.artifact_path / "artifact.txt", "test-file", "text", upload=True
        )

    @patch("torchapp.logging.wandb.Artifact", MockWandbArtifact)
    @patch.object(wandb.sdk.wandb_run.Run, "log_artifact", mock_log_artifact_reference)
    def test_log_artifact_reference(self):

        self.test_init()
        with open(self.artifact_path / "artifact.txt", "w") as file:
            file.write("test")

        self.wandb_app.log_artifact(
            self.artifact_path / "artifact.txt", "test-artifact", "text", upload=False
        )

    @patch("torchapp.logging.wandb.Artifact", MockWandbModelArtifact)
    @patch.object(wandb.sdk.wandb_run.Run, "log_artifact", mock_log_artifact_model)
    def test_save_model(self):

        self.wandb_app.init_run(
            "wandb_test", upload_model=True, mode="disabled", config=self.test_config
        )
        learner = self.wandb_app.learner(self.wandb_app.dataloaders(), output_dir="./tmp")
        self.wandb_app.save_model(learner, MODEL_RUN_NAME)

    def test_logging_callbacks(self):
        callbacks = self.wandb_app.callbacks()
        # import pdb; pdb.set_trace()
        callback_types = [type(x) for x in callbacks]
        assert WandbCallbackTime in callback_types
        assert WandbCallback in callback_types
        for callback in callbacks:
            if type(callback) == WandbCallbackTime:
                break

        assert type(callback.wandb_callback) == WandbCallback

    @patch("torchapp.logging.wandb.init", mock_sweep_init)
    @patch.object(TestApp, "train", mock_train)
    @patch("torchapp.logging.wandb.agent", mock_agent)
    @patch("torchapp.logging.wandb.sweep", mock_test_sweep)
    def test_sweep_tune(self):
        self.wandb_app.tune(method=SWEEP_METHOD)
