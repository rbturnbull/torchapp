#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
from torch import nn
from fastai.data.block import DataBlock, TransformBlock
from fastai.data.transforms import ColReader, RandomSplitter, Transform
import torchapp as ta
from torchapp.blocks import BoolBlock, Float32Block
from torchapp.metrics import logit_accuracy, logit_f1


class Normalize(Transform):    
    def __init__(self, mean=None, std=None): 
        self.mean = mean
        self.std = std

    def encodes(self, x): 
        return (x-self.mean) / self.std
    
    def decodes(self, x):
        return x * self.std + self.mean


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
        seed: int = ta.Param(default=42, help="The random seed to use for splitting the data."),
        batch_size: int = ta.Param(
            default=32,
            tune=True,
            tune_min=8,
            tune_max=128,
            log=True,
            help="The number of items to use in each batch.",
        ),
    ):

        df = pd.read_csv(csv)
        datablock = DataBlock(
            blocks=[Float32Block(type_tfms=[Normalize(mean=df[x].mean(), std=df[x].std())]), BoolBlock],
            get_x=ColReader(x),
            get_y=ColReader(y),
            splitter=RandomSplitter(validation_proportion, seed=seed),
        )

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
