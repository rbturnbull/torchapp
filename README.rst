==========
torchapp
==========

.. image:: https://raw.githubusercontent.com/rbturnbull/torchapp/master/docs/images/torchapp-banner.svg

.. start-badges

|testing badge| |coverage badge| |docs badge| |black badge| |git3moji badge| |torchapp badge|


.. |torchapp badge| image:: https://img.shields.io/badge/MLOpps-torchapp-B1230A.svg
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


Writing an App
=======================

Inherit a class from :code:`TorchApp` to make an app. The parent class includes several methods for training and hyper-parameter tuning. 
The minimum requirement is that you fill out the dataloaders method and the model method.

The :code:`dataloaders` method requires that you return a fastai Dataloaders object. This is a collection of dataloader objects. 
Typically it contains one dataloader for training and another for testing. For more information see https://docs.fast.ai/data.core.html#DataLoaders
You can add parameter values with typing hints in the function signature and these will be automatically added to the train and show_batch methods.

The :code:`model` method requires that you return a pytorch module. Parameters in the function signature will be added to the train method.

Here's an example for doing logistic regression:

.. code-block:: Python
   
    #!/usr/bin/env python3
    from pathlib import Path
    import pandas as pd
    from torch import nn
    from fastai.data.block import DataBlock, TransformBlock
    from fastai.data.transforms import ColReader, RandomSplitter
    import torchapp as ta
    from torchapp.blocks import BoolBlock


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
                help="The number of items to use in each batch.",
            ),
        ):

            datablock = DataBlock(
                blocks=[TransformBlock, BoolBlock],
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


    if __name__ == "__main__":
        LogisticRegressionApp.main()
   

Programmatic Interface
=======================

To use the app in Python, simply instantiate it:

.. code-block:: Python

   app = LogisticRegressionApp()

Then you can train with the method:

.. code-block:: Python

   app.train(training_csv_path)

This takes the arguments of both the :code:`dataloaders` method and the :code:`train` method. The function signature is modified so these arguments show up in auto-completion in a Jupyter notebook.

Predictions are made by simply calling the app object.

.. code-block:: Python

    app(data_csv_path)

Command-Line Interface
=======================

Command-line interfaces are created simply by using the Poetry package management tool. Just add a line like this in :code:`pyproject.toml`

.. code-block:: toml

    logistic = "logistic.apps:LogisticRegressionApp.main"

Now we can train with the command line:

.. code-block:: bash

    logistic train training_csv_path

All the arguments for the dataloader and the model can be set through arguments in the CLI. To see them run

.. code-block:: bash

    logistic train -h

Predictions are made like this:

.. code-block:: bash

    logistic predict data_csv_path

Hyperparameter Tuning
=======================

All the arguments in the dataloader and the model can be tuned using Weights & Biases (W&B) hyperparameter sweeps (https://docs.wandb.ai/guides/sweeps). In Python, simply run:

.. code-block:: python

    app.tune(runs=10)

Or from the command line, run

.. code-block:: bash

    logistic tune --runs 10

These commands will connect with W&B and your runs will be visible on the wandb.ai site.

Project Generation
=======================

To use a template to construct a package for your app, simply run:

.. code-block:: bash

    torchapp

.. end-quickstart

Credits
=======================

.. start-credits

torchapp was created created by Robert Turnbull with contributions from Jonathan Garber and Simone Bae.

Citation details to follow.

Logo elements derived from icons by `ProSymbols <https://thenounproject.com/icon/flame-797130/>`_ and `Philipp Petzka <https://thenounproject.com/icon/parcel-2727677/>`_.

.. end-credits