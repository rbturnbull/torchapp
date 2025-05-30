==========
torchapp
==========

.. image:: https://raw.githubusercontent.com/rbturnbull/torchapp/master/docs/images/torchapp-banner.svg

.. start-badges

|testing badge| |coverage badge| |docs badge| |black badge| |git3moji badge| |torchapp badge|

.. |torchapp badge| image:: https://img.shields.io/badge/Torch-App-B1230A.svg
    :target: https://rbturnbull.github.io/torchapp/

.. |testing badge| image:: https://github.com/rbturnbull/torchapp/actions/workflows/testing.yml/badge.svg
    :target: https://github.com/rbturnbull/torchapp/actions

.. |docs badge| image:: https://github.com/rbturnbull/torchapp/actions/workflows/docs.yml/badge.svg
    :target: https://rbturnbull.github.io/torchapp
    
.. |black badge| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    
.. |coverage badge| image:: https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/rbturnbull/506563cd9b49c8126284e34864c862d0/raw/coverage-badge.json
    :target: https://rbturnbull.github.io/torchapp/coverage/

.. |git3moji badge| image:: https://img.shields.io/badge/git3moji-%E2%9A%A1%EF%B8%8F%F0%9F%90%9B%F0%9F%93%BA%F0%9F%91%AE%F0%9F%94%A4-fffad8.svg
    :target: https://robinpokorny.github.io/git3moji/

.. end-badges

A wrapper for PyTorch projects to create easy command-line interfaces and manage hyper-parameter tuning.

Documentation at https://rbturnbull.github.io/torchapp/

.. start-quickstart

Installation
=======================

The software can be installed using ``pip``

.. code-block:: bash

    pip install torchapp

To install the latest version from the repository, you can use this command:

.. code-block:: bash

    pip install git+https://github.com/rbturnbull/torchapp.git


.. warning::

    Earlier versions of torchapp used fastai but current versions use Lightning. 
    If you used torchapp before, please check your code for compatibility with the new version or restrict to using torch below version 0.4.

Writing an App
=======================

Inherit a class from :code:`TorchApp` to make an app. The parent class includes several methods for training and hyper-parameter tuning. 
The minimum requirement is that you fill out the dataloaders method and the model method.

The :code:`data` method requires that you return a ``LightningDataModule`` object. This is a collection of dataloader objects. 
Typically it contains one dataloader for training and another for testing. For more information see https://lightning.ai/docs/pytorch/stable/data/datamodule.html
You can add parameter values with typing hints in the function signature and these will be automatically added to any mehtod that requires the training data (e.g. ``train``).

The :code:`model` method requires that you return a PyTorch module. Parameters in the function signature will be added to the ``train`` method.

Here's an example for doing training on the Iris dataset, a classic dataset for classification tasks:

.. code-block:: Python
   
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
   

Programmatic Interface
=======================

To use the app in Python, simply instantiate it:

.. code-block:: Python

   app = IrisApp()

Then you can train with the method:

.. code-block:: Python

   app.train(csv=training_csv_path)

This takes the arguments of both the :code:`data` method and the :code:`train` method.

Predictions are made by simply calling the app object.

.. code-block:: Python

    app(data_csv_path)

Command-Line Interface
=======================

Command-line interfaces are created simply by using the Poetry package management tool. Just add line like this in :code:`pyproject.toml` (assuming your package is called ``iris``):

.. code-block:: toml

    iris = "iris.apps:IrisApp.main"
    iris-tools = "iris.apps:IrisApp.tools"

Now we can train with the command line:

.. code-block:: bash

    iris-tools train --csv training_csv_path

All the arguments for the dataloader and the model can be set through arguments in the CLI. To see them run

.. code-block:: bash

    iris-tools train --help

Predictions are made like this:

.. code-block:: bash

    iris --csv data_csv_path

See information for other commands by running:

.. code-block:: bash

    iris-tools --help

Hyperparameter Tuning
=======================

All the arguments in the dataloader and the model can be tuned using a variety of hyperparameter tuning libraries including.

In Python run this:

.. code-block:: python

    app.tune(runs=10)

Or from the command line, run

.. code-block:: bash

    iris-tools tune --runs 10

These commands will connect with W&B and your runs will be visible on the wandb.ai site.

Project Generation
=======================

To use a template to construct a package for your app, simply run:

.. code-block:: bash

    torchapp-generator

.. end-quickstart

Credits
=======================

.. start-credits

torchapp was created created by `Robert Turnbull <https://robturnbull.com>`_ with contributions from Wytamma Wirth, Jonathan Garber and Simone Bae.

Citation details to follow.

Logo elements derived from icons by `ProSymbols <https://thenounproject.com/icon/flame-797130/>`_ and `Philipp Petzka <https://thenounproject.com/icon/parcel-2727677/>`_.

.. end-credits