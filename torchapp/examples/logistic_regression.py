#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
from torch import nn
from fastai.data.block import DataBlock, TransformBlock
from fastai.data.transforms import ColReader, RandomSplitter, Transform
import torchapp as ta
from torchapp.blocks import BoolBlock, Float32Block
from torchapp.metrics import logit_accuracy, logit_f1



class LogisticRegressionApp(ta.TorchApp):
    """
    Creates a basic app to do logistic regression.
    """

    def dataloaders(
        self,
        csv: Path = ta.Param(help="The path to a CSV file with the data."),
        x: str = ta.Param(default="x", help="The column name of the independent variable."),
        y: str = ta.Param(default="y", help="The column name of the dependent variable."),
        validation_proportion: float = ta.Param(
            default=0.2, help="The proportion of the dataset to use for validation."
        ),
        batch_size: int = ta.Param(
            default=32,
            tune=True,
            tune_min=8,
            tune_max=128,
            log=True,
            help="The number of items to use in each batch.",
        ),
    ):

        datablock = DataBlock(
            blocks=[Float32Block, BoolBlock],
            get_x=ColReader(x),
            get_y=ColReader(y),
            splitter=RandomSplitter(validation_proportion),
        )
        df = pd.read_csv(csv)

        return datablock.dataloaders(df, bs=batch_size)

    def model(self) -> nn.Module:
        """Builds a simple logistic regression model."""
        return nn.Linear(in_features=1, out_features=1, bias=True)

    def loss_func(self):
        return nn.BCEWithLogitsLoss()

    def metrics(self):
        return [logit_accuracy, logit_f1]

    def monitor(self):
        return "logit_f1"


if __name__ == "__main__":
    LogisticRegressionApp.main()
