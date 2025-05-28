#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
from torch import nn
import torch
import torchapp as ta
from torchapp.metrics import logit_accuracy, logit_f1
import torchapp as ta
from torch.utils.data import DataLoader, Dataset
import lightning as L
from dataclasses import dataclass


# class Normalize():    
#     def __init__(self, mean=None, std=None): 
#         self.mean = mean
#         self.std = std

#     def encodes(self, x): 
#         return (x-self.mean) / self.std
    
#     def decodes(self, x):
#         return x * self.std + self.mean


@dataclass
class LogisticRegressionDataset(Dataset):
    df: pd.DataFrame
    x_columns:list[str]
    y_column:str

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = torch.tensor(row[self.x_columns].values, dtype=torch.float32)
        y = torch.tensor(row[self.y_column], dtype=torch.float32)
        return x, y
    

class LogisticRegressionApp(ta.TorchApp):
    """
    Creates a basic app to do logistic regression.
    """
    @ta.method
    def data(
        self,
        csv: Path = ta.Param(help="The path to a CSV file with the data."),
        x: str = ta.Param(default="x", help="The column name of the independent variable."),
        y: str = ta.Param(default="y", help="The column name of the dependent variable."),
        validation_fraction: float = ta.Param(
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
        validation_df = df.sample(frac=validation_fraction, random_state=seed)
        train_df = df.drop(validation_df.index)
        train_dataset = LogisticRegressionDataset(train_df, [x], y)
        val_dataset = LogisticRegressionDataset(validation_df, [x], y)
        data_module = L.LightningDataModule()
        data_module.train_dataloader = lambda: DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        data_module.val_dataloader = lambda: DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return data_module

    @ta.method
    def model(self) -> nn.Module:
        """Builds a simple logistic regression model."""
        return nn.Linear(in_features=1, out_features=1, bias=True)

    @ta.method
    def loss_func(self):
        return nn.BCEWithLogitsLoss()
    
    @ta.method
    def metrics(self):
        return [logit_accuracy, logit_f1]

    @ta.method
    def monitor(self):
        return "logit_f1"


if __name__ == "__main__":
    LogisticRegressionApp.tools()
