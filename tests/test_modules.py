from torch import nn
from torchapp.modules import GeneralLightningModule
import torch


def test_save_hyperparameters():
    module = GeneralLightningModule(
        model=nn.Linear(1, 1),
        loss_function=nn.MSELoss(), 
        max_learning_rate=0.1,
        metrics=[("test", None)],
    )
    assert module.hparams.max_learning_rate == 0.1
    assert module.hparams["input_count"] == 1
    assert module.hparams["metrics"] == [("test", None)]
    assert isinstance(module.hparams.model, nn.Linear)
    assert isinstance(module.hparams.loss_function, nn.MSELoss)


def test_save_hyperparameters_kwargs():
    module = GeneralLightningModule(
        model=nn.Linear(1, 1),
        loss_function=nn.MSELoss(), 
        extra="extra",
        max_learning_rate=0.1,
        metrics=[("test", None)],
        reduction=torch.mean,
    )
    assert module.hparams.max_learning_rate == 0.1
    assert module.hparams["input_count"] == 1
    assert isinstance(module.hparams.model, nn.Linear)
    assert isinstance(module.hparams.loss_function, nn.MSELoss)
    assert module.hparams.extra == "extra"
    assert module.hparams.reduction == torch.mean


