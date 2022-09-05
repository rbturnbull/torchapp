#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from fastai.tabular.data import TabularDataLoaders
from fastai.tabular.all import tabular_learner, accuracy, error_rate
from sklearn.datasets import load_iris
import torchapp as ta


class IrisApp(ta.TorchApp):
    """
    A classification app to predict the type of iris from sepal and petal lengths and widths.

    A classic dataset publised in:
        Fisher, R.A. “The Use of Multiple Measurements in Taxonomic Problems” Annals of Eugenics, 7, Part II, 179–188 (1936).
    For more information about the dataset, see:
        https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-plants-dataset
    """

    def dataloaders(
        self,
        batch_size: int = ta.Param(default=32, tune_min=8, tune_max=128, log=True, tune=True),
    ):
        df = load_iris(as_frame=True)

        df["frame"]["target_name"] = np.take(df["target_names"], df["target"])

        return TabularDataLoaders.from_df(
            df["frame"],
            cont_names=df["feature_names"],
            y_names="target_name",
            bs=batch_size,
        )

    def metrics(self) -> list:
        return [accuracy, error_rate]

    def model(self):
        return None

    def build_learner_func(self):
        return tabular_learner

    def get_bibtex_files(self):
        files = super().get_bibtex_files()
        files.append(Path(__file__).parent / "iris.bib")
        return files


if __name__ == "__main__":
    IrisApp.main()
