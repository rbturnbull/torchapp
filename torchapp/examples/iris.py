#!/usr/bin/env python3

from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import load_iris
from torch import nn
import torchapp as ta
import torch
import pandas as pd
import lightning as L
from dataclasses import dataclass
from torchapp.metrics import accuracy


@dataclass
class IrisDataset(Dataset):
    df: pd.DataFrame
    feature_names: list[str]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = torch.tensor(row[self.feature_names].values, dtype=torch.float32)
        y = torch.tensor(row['target'], dtype=int)
        return x, y


@dataclass
class Standardize():
    mean:torch.Tensor
    std:torch.Tensor

    def __call__(self, x:torch.Tensor|float) -> torch.Tensor|float:
        return (x - self.mean) / self.std

    def reverse(self, x:torch.Tensor|float) -> torch.Tensor|float:
        return x * self.std + self.mean


def standardize_and_get_transform(x:torch.Tensor|pd.Series) -> tuple[torch.Tensor|pd.Series, Standardize]:
    transform = Standardize(mean=x.mean(), std=x.std())
    return transform(x), transform


class IrisApp(ta.TorchApp):
    """
    A classification app to predict the type of iris from sepal and petal lengths and widths.

    A classic dataset publised in:
        Fisher, R.A. “The Use of Multiple Measurements in Taxonomic Problems” Annals of Eugenics, 7, Part II, 179–188 (1936).
    For more information about the dataset, see:
        https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-plants-dataset
    """
    @ta.method
    def setup(self):
        iris_data = load_iris(as_frame=True)
        df = iris_data['frame']
        self.feature_names = iris_data['feature_names']
        self.target_names = iris_data['target_names']
        self.df = df

    @ta.method
    def data(self, validation_fraction: float = 0.2, batch_size: int = 32, seed: int = 42):
        df = self.df

        # Standardize and save the transforms
        self.transforms = {}
        for column in self.feature_names:
            df[column], self.transforms[column] = standardize_and_get_transform(df[column])

        validation_df = df.sample(frac=validation_fraction, random_state=seed)
        train_df = df.drop(validation_df.index)
        train_dataset = IrisDataset(train_df, self.feature_names)
        val_dataset = IrisDataset(validation_df, self.feature_names)
        data_module = L.LightningDataModule()

        data_module.train_dataloader = lambda: DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        data_module.val_dataloader = lambda: DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return data_module

    @ta.method
    def metrics(self):
        return [accuracy]

    @ta.method
    def extra_hyperparameters(self):
        return dict(target_names=self.target_names, transforms=self.transforms)

    @ta.method
    def model(
        self, 
        hidden_size:int=ta.Param(default=8, tune=True, tune_min=4, tune_max=128, tune_log=True),
        intermediate_layers:int=ta.Param(default=1, tune=True, tune_min=0, tune_max=3),
    ):
        in_features = 4
        output_categories = 3

        modules = [nn.Linear(in_features, hidden_size)]
        for _ in range(intermediate_layers):
            modules.append(nn.ReLU())
            modules.append(nn.Linear(hidden_size, hidden_size))

        modules.append(nn.ReLU())
        modules.append(nn.Linear(hidden_size, output_categories))
        return nn.Sequential(*modules)

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
        sepal_length:float=ta.Param(...,help="The sepal length in cm."), 
        sepal_width:float=ta.Param(...,help="The sepal width in cm."), 
        petal_length:float=ta.Param(...,help="The petal length in cm."), 
        petal_width:float=ta.Param(...,help="The petal width in cm."), 
    ) -> list:
        assert sepal_width is not None
        assert sepal_length is not None
        assert petal_width is not None
        assert petal_length is not None

        self.target_names = module.hparams.target_names

        # data must be in the same order as the feature_names
        data = [sepal_length, sepal_width, petal_length, petal_width]
        transformed_data = [transform(x) for x,transform in zip(data, module.hparams.transforms.values())]
        dataset = [torch.tensor(transformed_data, dtype=torch.float32)]
        return DataLoader(dataset, batch_size=1)

    @ta.method
    def output_results(
        self, 
        results,
    ):
        assert results.shape == (3,)
        probabilities = torch.softmax(results, dim=0)
        predicted_class = results.argmax().item()
        predicted_name = self.target_names[predicted_class]
        print(f"Predicted class: {predicted_name} ({probabilities[predicted_class]:.2%})")


if __name__ == "__main__":
    IrisApp.tools()
