=======================
Quickstart
=======================

Writing an App
=======================

Inherit a class from :code:`TorchApp` to make an app. The parent class includes several methods for training and hyper-parameter tuning. 
The minimum requirement is that you fill out the dataloaders method and the model method.

The :code:`dataloaders` method requires that you return a fastai Dataloaders object. This is a collection of dataloader objects. 
Typically it contains one dataloader for training and another for testing. For more information see https://docs.fast.ai/data.core.html#DataLoaders
You can add parameter values with typing hints in the function signature and these will be automatically added to the train and show_batch methods.

The :code:`model` method requires that you return a pytorch module. Parameters in the function signature will be added to the train method.

Here's an example for doing logistic regression:

.. literalinclude :: ../torchapp/examples/logistic_regression.py
   :language: python

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

