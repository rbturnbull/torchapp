=======================
Hyper-parameter Tuning
=======================

Parameters
==========


Tuning Engines
==============

torchapp can use three optimizer packages as engines to perform the hyper-parameter tuning. 
They are Sweeps by Weights and Biases, Optuna, and scikit-optimize.

Optuna
--------

The ID of the tuning job is used for the Optuna `storage using the RDB backend <https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html#sphx-glr-tutorial-20-recipes-001-rdb-py>`_. 
If the ID is not a URL then the storage will be an SQLite file which uses the ID as the stem of the filename.
