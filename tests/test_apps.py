import re
from types import SimpleNamespace

import pytest
import torch

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


class TrainerSpy:
    def __init__(self):
        self.fit_args = None
        self.fit_kwargs = None

    def fit(self, *args, **kwargs):
        self.fit_args = args
        self.fit_kwargs = kwargs


class CheckpointTrainApp(ta.TorchApp):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Linear(2, 1)
        self.trainer_spy = TrainerSpy()
        self.setup_calls = 0
        self.trainer_calls = 0

    def setup_and_data(self, **kwargs):
        self.setup_calls += 1
        return [(torch.tensor([[1.0, 2.0]]), torch.tensor([[0.0]]))]

    def validation_dataloader(self, **kwargs):
        return None

    def lightning_module(self, **kwargs):
        return SimpleNamespace(model=self.network, input_count=1)

    def trainer(self, **kwargs):
        self.trainer_calls += 1
        return self.trainer_spy


def test_train_loads_pretrained_model_only(tmp_path):
    app = CheckpointTrainApp()
    expected_weight = torch.full_like(app.network.weight, 3.0)
    expected_bias = torch.full_like(app.network.bias, -2.0)
    checkpoint_path = tmp_path / "pretrained.ckpt"
    torch.save(
        {
            "state_dict": {
                "model.weight": expected_weight,
                "model.bias": expected_bias,
                "metric.unrelated": torch.tensor(99),
            },
            "epoch": 12,
            "global_step": 345,
        },
        checkpoint_path,
    )

    app.train(pretrained_checkpoint=checkpoint_path)

    assert torch.equal(app.network.weight, expected_weight)
    assert torch.equal(app.network.bias, expected_bias)
    assert app.trainer_spy.fit_kwargs == {}


def test_train_uses_subclass_pretrained_transfer(tmp_path):
    class CustomTransferApp(CheckpointTrainApp):
        def __init__(self):
            super().__init__()
            self.transfer_checkpoint = None

        def transfer_pretrained_weights(
            self,
            model,
            pretrained_model_state,
            checkpoint=None,
            **kwargs,
        ):
            self.transfer_checkpoint = checkpoint
            model.load_state_dict(pretrained_model_state, strict=False)

    app = CustomTransferApp()
    initial_bias = app.network.bias.detach().clone()
    expected_weight = torch.full_like(app.network.weight, 7.0)
    checkpoint_path = tmp_path / "custom-transfer.ckpt"
    checkpoint = {
        "state_dict": {"model.weight": expected_weight},
        "transfer_metadata": "custom",
    }
    torch.save(checkpoint, checkpoint_path)

    app.train(pretrained_checkpoint=checkpoint_path)

    assert torch.equal(app.network.weight, expected_weight)
    assert torch.equal(app.network.bias, initial_bias)
    assert app.transfer_checkpoint["transfer_metadata"] == "custom"
    assert app.trainer_spy.fit_kwargs == {}


def test_train_delegates_resume_to_lightning(tmp_path):
    app = CheckpointTrainApp()
    initial_state = {name: tensor.detach().clone() for name, tensor in app.network.state_dict().items()}
    checkpoint_path = tmp_path / "resume.ckpt"
    torch.save(
        {
            "state_dict": {
                "model.weight": torch.full_like(app.network.weight, 8.0),
                "model.bias": torch.full_like(app.network.bias, 8.0),
            }
        },
        checkpoint_path,
    )

    app.train(resume_checkpoint=checkpoint_path)

    for name, tensor in app.network.state_dict().items():
        assert torch.equal(tensor, initial_state[name])
    assert app.trainer_spy.fit_kwargs == {"ckpt_path": str(checkpoint_path)}


def test_train_rejects_both_checkpoint_modes_before_setup(tmp_path):
    app = CheckpointTrainApp()
    checkpoint_path = tmp_path / "checkpoint.ckpt"
    checkpoint_path.touch()

    with pytest.raises(
        ValueError,
        match=re.escape(
            "--resume-checkpoint and --pretrained-checkpoint are mutually exclusive. "
            "Use --resume-checkpoint to continue an interrupted run, or "
            "--pretrained-checkpoint to start a new fine-tuning run."
        ),
    ):
        app.train(
            resume_checkpoint=checkpoint_path,
            pretrained_checkpoint=checkpoint_path,
        )

    assert app.setup_calls == 0
    assert app.trainer_calls == 0


@pytest.mark.parametrize(
    ("argument", "message"),
    [
        ("resume_checkpoint", "Resume checkpoint does not exist: "),
        ("pretrained_checkpoint", "Pretrained checkpoint does not exist: "),
    ],
)
def test_train_rejects_missing_checkpoint_paths(tmp_path, argument, message):
    app = CheckpointTrainApp()
    checkpoint_path = tmp_path / "missing.ckpt"

    with pytest.raises(
        FileNotFoundError,
        match=re.escape(f"{message}{checkpoint_path}"),
    ):
        app.train(**{argument: checkpoint_path})

    assert app.setup_calls == 0
    assert app.trainer_calls == 0


@pytest.mark.parametrize(
    ("checkpoint", "message"),
    [
        ({}, "Checkpoint does not contain a Lightning state_dict: "),
        (
            {"state_dict": {"metric.unrelated": torch.tensor(1)}},
            "No model.* parameters were found in checkpoint: ",
        ),
    ],
)
def test_train_rejects_malformed_pretrained_checkpoints(tmp_path, checkpoint, message):
    app = CheckpointTrainApp()
    checkpoint_path = tmp_path / "malformed.ckpt"
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(
        ValueError,
        match=re.escape(f"{message}{checkpoint_path}"),
    ):
        app.train(pretrained_checkpoint=checkpoint_path)

    assert app.trainer_calls == 0


def test_checkpoint_callback_saves_latest_full_state_only():
    weights_checkpoint, full_checkpoint = ta.TorchApp().checkpoint_callback()

    assert weights_checkpoint.save_weights_only is True
    assert not weights_checkpoint.save_last
    assert full_checkpoint.save_weights_only is False
    assert full_checkpoint.save_last is True


def test_train_checkpoint_options_in_cli_help():
    runner = CliRunner()
    result = runner.invoke(
        ta.TorchApp().tools_app,
        ["train", "--help"],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 0
    assert "--resume-checkpoint" in result.output
    assert "--pretrained-checkpoint" in result.output


def test_checkpoint_callback_save_last_weights():
    weights_checkpoint, full_checkpoint = ta.TorchApp().checkpoint_callback(save_last_weights=True)

    assert weights_checkpoint.save_last is True
    assert weights_checkpoint.CHECKPOINT_NAME_LAST != full_checkpoint.CHECKPOINT_NAME_LAST
    assert full_checkpoint.save_last is True
