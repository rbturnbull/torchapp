=======================
Project Setup
=======================

You can set up a TorchApp project automatically using the built in cookiecutter template or you can add it manually to an existing Python package.

Automatic Generation
=======================

To generate a project using torchapp, then run the command-line utility.

.. code:: bash

    torchapp-generator

This uses `Cookiecutter <https://cookiecutter.readthedocs.io/>`_ to generate a project for you.

This will use the template that comes with your torchapp installation. If you wish to use the most up-to-date template, then run

.. code:: bash

    torchapp-generator gh

This will use the template from the torchapp-cookiecutter repository on Github. If you wish to create your own template, feel free to fork that repository and use the new URL as the command-line argument.

The dependency management is handled using Poetry. Install Poetry using the `instructions <https://python-poetry.org/docs/#installation>`_.
Then to install the dependencies, run:

.. code:: bash

    poetry install

Enter the virtual environment with:

.. code:: bash

    poetry env activate

The command-line utility should be installed to your path automatically in that environment.

Automatic Generation Reference
------------------------------

.. click:: torchapp.main:app
   :prog: torchapp
   :nested: full


Manual Setup
=======================

You can use TorchApp in an existing project, by following these steps.

If you are using the poetry dependency management system, the install the app like this:

.. code:: bash

    poetry add git+https://github.com/rbturnbull/torchapp.git

If using pip then:

.. code:: bash

    pip install git+https://github.com/rbturnbull/torchapp.git


Then in your code (in a file usually named ``apps.py``) subclass TorchApp and implement at least the ``data`` and the ``model`` methods.

.. code:: python

    import torchapp as ta

    class MyApp(ta.TorchApp):

        @ta.method
        def data(self):
            ...
        
        @ta.method
        def model(self):
            ...

If you are using a file as the main script then instantiate the app and call the main function:

.. code:: python

    if __name__ == "__main__":
        MyApp().main()

If you wish to include the app in a Python package, it is easiest to use the Poetry dependency management system. 
In the ``pyproject.toml`` file add the main method of your app to the scripts section like this:

.. code:: toml

    [tool.poetry.scripts]
    myapp = "path.to.script:MyApp.main"
    myapp-tools = "path.to.script:MyApp.tools"

For example, if the name of the executable was going to be ``logistic`` and the path to the file from the base directory was ``logistic/apps.py`` 
and the subclass of TorchApp was called ``LogisticApp``, then the following should be added to ``pyproject.toml``:

.. code:: toml

    [tool.poetry.scripts]
    logistic = "logistic.apps:LogisticApp.main"
    logistic-tools = "logistic.apps:LogisticApp.tools"


Pre-commit Hooks
=======================

To set up black code formatting with a pre-commit hook, run:

.. code:: bash

    pre-commit install
    
Coverage Badge
=======================

To set up the automatic coverage badge on Github, you need to create a Github authorization token (https://github.com/settings/tokens/new)
and give it permission to modify gists. Then add this token as the secret in ``Settings/Secrets/Actions`` with the variable name:  ``GIST_SECRET``.

