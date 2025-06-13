.. torchapp documentation master file, created by
   sphinx-quickstart on Mon Jan 17 15:55:58 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

torchapp
===================================

.. image:: https://raw.githubusercontent.com/rbturnbull/torchapp/master/docs/images/torchapp-banner.svg

.. include:: ../README.rst
   :start-after: start-badges
   :end-before: end-badges

**A Tool for Packaging PyTorch Models as Reusable Applications**

Deploying machine learning models remains a challenge, particularly in scientific contexts where models are often shared only as standalone scripts or Jupyter notebooks—frequently without parameters and in forms that hinder reuse. TorchApp streamlines this process by enabling users to generate fully structured Python packages from PyTorch models, complete with documentation, testing, and interfaces for training, validation, and prediction. Users subclass the TorchApp base class and override methods to define the model and data loaders; the arguments to these methods are automatically exposed via a command-line interface. TorchApp also supports hyperparameter tuning by allowing argument distributions to be specified, and integrates with experiment tracking tools such as Weights & Biases. Optionally, a graphical user interface can be auto-generated using function signatures and type hints. TorchApp applications are easily testable using a built-in testing framework and are readily publishable to PyPI or Conda, making it simple to share deep learning tools with a broader audience.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   setup
   .. parameters
   .. training
   .. tuning
   testing
   vision
   examples
   contributing
   credits



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
