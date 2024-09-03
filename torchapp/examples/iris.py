#!/usr/bin/env python3

from pathlib import Path
import numpy as np
from torch.utils.data import random_split, DataLoader, Dataset
from sklearn.datasets import load_iris
from torch import nn
import torchapp as ta
import torch
from torchmetrics import Metric, Accuracy
import lightning as L

class IrisDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = torch.tensor(row[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']].values, dtype=torch.float32)
        y = torch.tensor(row['target'], dtype=int)
        return x, y
    

class IrisApp(ta.TorchApp):
    """
    A classification app to predict the type of iris from sepal and petal lengths and widths.

    A classic dataset publised in:
        Fisher, R.A. “The Use of Multiple Measurements in Taxonomic Problems” Annals of Eugenics, 7, Part II, 179–188 (1936).
    For more information about the dataset, see:
        https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-plants-dataset
    """
    @ta.method
    def data(self, validation_fraction: float = 0.2, batch_size: int = 32, seed: int = 42):
        iris_data = load_iris(as_frame=True)
        df = iris_data['frame']
        validation_df = df.sample(frac=validation_fraction, random_state=seed)
        train_df = df.drop(validation_df.index)
        train_dataset = IrisDataset(train_df)
        val_dataset = IrisDataset(validation_df)
        data_module = L.LightningDataModule()
        data_module.train_dataloader = lambda: DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        data_module.val_dataloader = lambda: DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return data_module

    # @ta.method
    # def metrics(self) -> list[Metric]:
    #     return [Accuracy()]

    @ta.method
    def model(
        self, 
        hidden_size:int=ta.Param(default=8, tune=True, tune_min=4, tune_max=128, tune_log=True),
    ):
        in_features = 4
        output_categories = 3
        return nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_categories),
        )

    @ta.method
    def loss_function(self):
        return nn.CrossEntropyLoss()

    @ta.method
    def get_bibtex_files(self):
        files = super().get_bibtex_files()
        files.append(Path(__file__).parent / "iris.bib")
        return files

    @ta.method
    def prediction_dataloader(
        self, 
        module, 
        sepal_length:float=None, 
        sepal_width:float=None, 
        petal_length:float=None, 
        petal_width:float=None, 
    ) -> list:
        assert sepal_length is not None
        assert sepal_width is not None
        assert petal_length is not None
        assert petal_width is not None

        x = torch.tensor([[sepal_length, sepal_width, petal_length, petal_width]], dtype=torch.float32)
        return [x]

    @ta.method
    def output_results(
        self, 
        results,
    ):
        print(f"Predicted class: {results[0].argmax().item()}")



if __name__ == "__main__":
    IrisApp.tools()
