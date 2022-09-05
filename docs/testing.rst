=======================
Testing
=======================

Classes inheriting from ``TorchApp`` can be tested automatically by inheriting from the ``TorchAppTestCase``.
This creates a number of automated tests designed to be used in conjunction with ``pytest``. 
The expected results of the tests are definied in yaml files in a directory named ``expected`` next to the scripts with the testing code 
and in a subdirectory with the class name which is an instance of ``TorchAppTestCase``. Each test has its own subdirectory there. 
There can be multiple expected files for testing the same method to test different combinations of arguments.
The format for the yaml files is:

.. code-block:: yaml

    params: {...}
    output: "..."

The params gives a list of keyword arguments for the TorchApp method being tested and the output is some data in a format required by the testing function.
For example the expected output for the model method is the string representation of the model definied by the params.

If you want to change the expected output for a test then run pytest with the ``-s`` option. This will find failing tests and prompt the user to ask if the expected file should be regenerated.
One way to automatically create expectation files is to add blank files and prompt for adding the expected result. 
The resulting file should be read to make sure that it is correct.

Logistic Regression App Example
===============================

For instance, the LogisticRegression app included as an example in this project is tested as follows in ``tests/test_logistic.py``. 

.. literalinclude :: ../tests/test_logistic.py
   :language: python

The expected values for the tests are in ``tests/expected/TestLogisticRegressionApp``. 
The file for testing the model is in ``tests/expected/TestLogisticRegressionApp/test_model/model.yaml``: 

.. literalinclude :: ../tests/expected/TestLogisticRegressionApp/test_model/model.yaml
   :language: yaml

TorchAppTestCase Reference
==========================

.. autoclass:: torchapp.testing.TorchAppTestCase
    :members:
    :inherited-members:

