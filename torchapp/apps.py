from contextlib import nullcontext
from pathlib import Path
from types import MethodType
from typing import List, Optional, Union, Dict
import inspect
import hashlib
from appdirs import user_cache_dir
import torch
from torch import nn
from fastai.learner import Learner, load_learner
from fastai.data.core import DataLoaders
from fastai.callback.schedule import fit_one_cycle

# from fastai.distributed import distrib_ctx
from fastai.callback.tracker import SaveModelCallback
from fastai.callback.progress import CSVLogger
import click
import typer
from typer.main import get_params_convertors_ctx_param_name_from_function
from rich.console import Console
from rich.traceback import install
from rich.table import Table
from rich.box import SIMPLE



install()
console = Console()

from .citations import Citable
from .util import copy_func, call_func, change_typer_to_defaults, add_kwargs
from .params import Param
from .callbacks import TorchAppWandbCallback, TorchAppMlflowCallback
from .download import cached_download

bibtex_dir = Path(__file__).parent / "bibtex"


class TorchAppInitializationError(Exception):
    pass


class TorchApp(Citable):
    torchapp_initialized = False
    fine_tune = False

    def __init__(self):
        super().__init__()

        # Make deep copies of methods so that we can change the function signatures dynamically
        self.fit = self.copy_method(self.fit)
        self.train = self.copy_method(self.train)
        self.dataloaders = self.copy_method(self.dataloaders)
        self.model = self.copy_method(self.model)
        self.pretrained_location = self.copy_method(self.pretrained_location)
        self.show_batch = self.copy_method(self.show_batch)
        self.tune = self.copy_method(self.tune)
        self.pretrained_local_path = self.copy_method(self.pretrained_local_path)
        self.learner_kwargs = self.copy_method(self.learner_kwargs)
        self.learner = self.copy_method(self.learner)
        self.__call__ = self.copy_method(self.__call__)
        self.validate = self.copy_method(self.validate)
        self.callbacks = self.copy_method(self.callbacks)
        self.one_batch_output = self.copy_method(self.one_batch_output)
        self.one_batch_output_size = self.copy_method(self.one_batch_output_size)
        self.one_batch_loss = self.copy_method(self.one_batch_loss)
        self.loss_func = self.copy_method(self.loss_func)
        self.metrics = self.copy_method(self.metrics)
        self.lr_finder = self.copy_method(self.lr_finder)
        self.inference_dataloader = self.copy_method(self.inference_dataloader)
        self.output_results = self.copy_method(self.output_results)

        # Add keyword arguments to the signatures of the methods used in the CLI
        add_kwargs(to_func=self.learner_kwargs, from_funcs=[self.metrics, self.loss_func])
        add_kwargs(to_func=self.learner, from_funcs=[self.learner_kwargs, self.dataloaders, self.model])
        add_kwargs(to_func=self.train, from_funcs=[self.learner, self.fit, self.callbacks])
        add_kwargs(to_func=self.show_batch, from_funcs=self.dataloaders)
        add_kwargs(to_func=self.tune, from_funcs=self.train)
        add_kwargs(to_func=self.pretrained_local_path, from_funcs=self.pretrained_location)
        add_kwargs(
            to_func=self.__call__,
            from_funcs=[self.pretrained_local_path, self.inference_dataloader, self.output_results],
        )
        add_kwargs(to_func=self.validate, from_funcs=[self.pretrained_local_path, self.dataloaders])
        add_kwargs(to_func=self.one_batch_output, from_funcs=self.learner)
        add_kwargs(to_func=self.one_batch_loss, from_funcs=self.learner)
        add_kwargs(to_func=self.lr_finder, from_funcs=self.learner)
        add_kwargs(to_func=self.one_batch_output_size, from_funcs=self.one_batch_output)

        # Make copies of methods to use just for the CLI
        self.train_cli = self.copy_method(self.train)
        self.show_batch_cli = self.copy_method(self.show_batch)
        self.tune_cli = self.copy_method(self.tune)
        self.pretrained_local_path_cli = self.copy_method(self.pretrained_local_path)
        self.infer_cli = self.copy_method(self.__call__)
        self.validate_cli = self.copy_method(self.validate)
        self.lr_finder_cli = self.copy_method(self.lr_finder)

        # Remove params from defaults in methods not used for the cli
        change_typer_to_defaults(self.fit)
        change_typer_to_defaults(self.model)
        change_typer_to_defaults(self.learner_kwargs)
        change_typer_to_defaults(self.loss_func)
        change_typer_to_defaults(self.metrics)
        change_typer_to_defaults(self.learner)
        change_typer_to_defaults(self.callbacks)
        change_typer_to_defaults(self.train)
        change_typer_to_defaults(self.show_batch)
        change_typer_to_defaults(self.tune)
        change_typer_to_defaults(self.pretrained_local_path)
        change_typer_to_defaults(self.__call__)
        change_typer_to_defaults(self.validate)
        change_typer_to_defaults(self.dataloaders)
        change_typer_to_defaults(self.pretrained_location)
        change_typer_to_defaults(self.one_batch_output_size)
        change_typer_to_defaults(self.one_batch_output)
        change_typer_to_defaults(self.one_batch_loss)
        change_typer_to_defaults(self.lr_finder)
        change_typer_to_defaults(self.inference_dataloader)
        change_typer_to_defaults(self.output_results)

        # Store a bool to let the app know later on (in self.assert_initialized)
        # that __init__ has been called on this parent class
        self.torchapp_initialized = True
        self.learner_obj = None
        # self.console = console

    def __str__(self):
        return self.__class__.__name__

    def get_bibtex_files(self):
        return [
            bibtex_dir / "fastai.bib", 
            bibtex_dir / "torchapp.bib",
        ]

    def copy_method(self, method):
        return MethodType(copy_func(method.__func__), self)

    def pretrained_location(self) -> Union[str, Path]:
        """
        The location of a pretrained model.

        It can be a URL, in which case it will need to be downloaded.
        Or it can be part of the package bundle in which case,
        it needs to be a relative path from directory which contains the code which defines the app.

        This function by default returns an empty string.
        Inherited classes need to override this method to use pretrained models.

        Returns:
            Union[str, Path]: The location of the pretrained model.
        """
        return ""

    def pretrained_local_path(
        self,
        pretrained: str = Param(default=None, help="The location (URL or filepath) of a pretrained model."),
        reload: bool = Param(
            default=False,
            help="Should the pretrained model be downloaded again if it is online and already present locally.",
        ),
        **kwargs,
    ) -> Path:
        """
        The local path of the pretrained model.

        If it is a URL, then it is downloaded.
        If it is a relative path, then this method returns the absolute path to it.

        Args:
            pretrained (str, optional): The location (URL or filepath) of a pretrained model. If it is a relative path, then it is relative to the current working directory. Defaults to using the result of the `pretrained_location` method.
            reload (bool, optional): Should the pretrained model be downloaded again if it is online and already present locally. Defaults to False.

        Raises:
            FileNotFoundError: If the file cannot be located in the local environment.

        Returns:
            Path: The absolute path to the model on the local filesystem.
        """
        if pretrained:
            location = pretrained
            base_dir = Path.cwd()
        else:
            location = str(call_func(self.pretrained_location, **kwargs))
            module = inspect.getmodule(self)
            base_dir = Path(module.__file__).parent.resolve()

        if not location:
            raise FileNotFoundError(f"Please pass in a pretrained model.")

        # Check if needs to be downloaded
        location = str(location)
        if location.startswith("http"):
            name = location.split("/")[-1]
            extension_location = name.rfind(".")
            if extension_location:
                name_stem = name[:extension_location]
                extension = name[extension_location:]
            else:
                name_stem = name
                extension = ".dat"
            url_hash = hashlib.md5(location.encode()).hexdigest()
            path = self.cache_dir()/f"{name_stem}-{url_hash}{extension}"
            cached_download(location, path, force=reload)
        else:
            path = Path(location)
            if not path.is_absolute():
                path = base_dir / path

        if not path or not path.is_file():
            raise FileNotFoundError(f"Cannot find pretrained model at '{path}'")

        return path

    def prepare_source(self, data):
        return data

    def inference_dataloader(self, learner, **kwargs):
        dataloader = learner.dls.test_dl(**kwargs)
        return dataloader

    def validate(
        self,
        gpu: bool = Param(True, help="Whether or not to use a GPU for processing if available."),
        **kwargs,
    ):
        path = call_func(self.pretrained_local_path, **kwargs)

        # Check if CUDA is available
        gpu = gpu and torch.cuda.is_available()

        try:
            learner = load_learner(path, cpu=not gpu)
        except Exception:
            import dill
            learner = load_learner(path, cpu=not gpu, pickle_module=dill)

        # Create a dataloader for inference
        dataloaders = call_func(self.dataloaders, **kwargs)

        table = Table(title="Validation", box=SIMPLE)

        values = learner.validate(dl=dataloaders.valid)
        names = [learner.recorder.loss.name] + [metric.name for metric in learner.metrics]
        result = {name: value for name, value in zip(names, values)}

        table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        for name, value in result.items():
            table.add_row(name, str(value))

        console.print(table)

        return result

    def __call__(
        self, 
        gpu: bool = Param(True, help="Whether or not to use a GPU for processing if available."), 
        **kwargs
    ):
        # Check if CUDA is available
        gpu = gpu and torch.cuda.is_available()

        # Open the exported learner from a pickle file
        path = call_func(self.pretrained_local_path, **kwargs)
        learner = self.learner_obj = load_learner(path, cpu=not gpu)

        # Create a dataloader for inference
        dataloader = call_func(self.inference_dataloader, learner, **kwargs)

        results = learner.get_preds(dl=dataloader, reorder=False, with_decoded=False, act=self.activation(), cbs=self.inference_callbacks())

        # Output results
        output_results = call_func(self.output_results, results, **kwargs)
        return output_results if output_results is not None else results

    def inference_callbacks(self):
        return None

    @classmethod
    def main(cls, inference_only:bool=False):
        """
        Creates an instance of this class and runs the command-line interface.
        """
        cli = cls.click(inference_only=inference_only)
        return cli()

    @classmethod
    def inference_only_main(cls):
        """
        Creates an instance of this class and runs the command-line interface for only the inference command.
        """
        return cls.main(inference_only=True)

    @classmethod
    def click(cls, inference_only:bool=False):
        """
        Creates an instance of this class and returns the click object for the command-line interface.
        """
        self = cls()
        cli = self.cli(inference_only=inference_only)
        return cli

    @classmethod
    def inference_only_click(cls):
        """
        Creates an instance of this class and returns the click object for the command-line interface.
        """
        return cls.click(inference_only=True)

    def assert_initialized(self):
        """
        Asserts that this app has been initialized.

        All sub-classes of TorchApp need to call super().__init__() if overriding the __init__() function.

        Raises:
            TorchAppInitializationError: If the app has not been properly initialized.
        """
        if not self.torchapp_initialized:
            raise TorchAppInitializationError(
                """The initialization function for this TorchApp has not been called.
                Please ensure sub-classes of TorchApp call 'super().__init__()'"""
            )

    def version(self, verbose: bool = False):
        """
        Prints the version of the package that defines this app.

        Used in the command-line interface.

        Args:
            verbose (bool, optional): Whether or not to print to stdout. Defaults to False.

        Raises:
            Exception: If it cannot find the package.

        """
        if verbose:
            from importlib import metadata

            module = inspect.getmodule(self)
            package = ""
            if module.__package__:
                package = module.__package__.split('.')[0]
            else:
                path = Path(module.__file__).parent
                while path.name:
                    try:
                        if metadata.distribution(path.name):
                            package = path.name
                            break
                    except Exception:
                        pass
                    path = path.parent

            if package:
                version = metadata.version(package)
                print(version)
            else:
                raise Exception("Cannot find package.")

            raise typer.Exit()

    def cli(self, inference_only:bool=False):
        """
        Returns a 'Click' object which defines the command-line interface of the app.
        """
        self.assert_initialized()

        cli = typer.Typer()

        @cli.callback()
        def base_callback(
            version: Optional[bool] = typer.Option(
                None,
                "--version",
                "-v",
                callback=self.version,
                is_eager=True,
                help="Prints the current version.",
            ),
        ):
            pass

        typer_click_object = typer.main.get_command(cli)

        params, _, _ = get_params_convertors_ctx_param_name_from_function(self.infer_cli)
        command = click.Command(
            name="infer",
            callback=self.infer_cli,
            params=params,
        )
        if inference_only:
            return command
        typer_click_object.add_command(command)


        train_params, _, _ = get_params_convertors_ctx_param_name_from_function(self.train_cli)
        train_command = click.Command(
            name="train",
            callback=self.train_cli,
            params=train_params,
        )
        typer_click_object.add_command(train_command)

        show_batch_params, _, _ = get_params_convertors_ctx_param_name_from_function(self.show_batch_cli)
        command = click.Command(
            name="show-batch",
            callback=self.show_batch_cli,
            params=show_batch_params,
        )
        typer_click_object.add_command(command)

        params, _, _ = get_params_convertors_ctx_param_name_from_function(self.tune_cli)
        tuning_params = self.tuning_params()
        for param in params:
            if param.name in tuning_params:
                param.default = None
        command = click.Command(
            name="tune",
            callback=self.tune_cli,
            params=params,
        )
        typer_click_object.add_command(command)

        params, _, _ = get_params_convertors_ctx_param_name_from_function(self.validate_cli)
        command = click.Command(
            name="validate",
            callback=self.validate_cli,
            params=params,
        )
        typer_click_object.add_command(command)

        params, _, _ = get_params_convertors_ctx_param_name_from_function(self.lr_finder_cli)
        command = click.Command(
            name="lr-finder",
            callback=self.lr_finder_cli,
            params=params,
        )
        typer_click_object.add_command(command)

        command = click.Command(
            name="bibliography",
            callback=self.print_bibliography,
        )
        typer_click_object.add_command(command)

        command = click.Command(
            name="bibtex",
            callback=self.print_bibtex,
        )
        typer_click_object.add_command(command)

        return typer_click_object

    def tuning_params(self):
        tuning_params = {}
        signature = inspect.signature(self.tune_cli)

        for key, value in signature.parameters.items():
            default_value = value.default
            if isinstance(default_value, Param) and default_value.tune == True:

                # Override annotation if given in typing hints
                if value.annotation:
                    default_value.annotation = value.annotation

                default_value.check_choices()

                tuning_params[key] = default_value
                
        return tuning_params

    def dataloaders(self):
        raise NotImplementedError(
            f"Please ensure that the 'dataloaders' method is implemented in {self.__class__.__name__}."
        )

    def model(self) -> nn.Module:
        raise NotImplementedError(f"Please ensure that the 'model' method is implemented in {self.__class__.__name__}.")

    def build_learner_func(self):
        return Learner

    def learner(
        self,
        fp16: bool = Param(
            default=True,
            help="Whether or not the floating-point precision of learner should be set to 16 bit.",
        ),
        **kwargs,
    ) -> Learner:
        """
        Creates a fastai learner object.
        """
        console.print("Building dataloaders", style="bold")
        dataloaders = call_func(self.dataloaders, **kwargs)

        # Allow the dataloaders to go to GPU so long as it hasn't explicitly been set as a different device
        if dataloaders.device is None:
            dataloaders.cuda()  # This will revert to CPU if cuda is not available

        console.print("Building model", style="bold")
        model = call_func(self.model, **kwargs)

        console.print("Building learner", style="bold")
        learner_kwargs = call_func(self.learner_kwargs, **kwargs)
        build_learner_func = self.build_learner_func()
        learner = build_learner_func(
            dataloaders,
            model,
            **learner_kwargs,
        )

        learner.training_kwargs = kwargs

        if fp16:
            console.print("Setting floating-point precision of learner to 16 bit", style="red")
            learner = learner.to_fp16()

        # Save a pointer to the learner
        self.learner_obj = learner

        return learner

    def learner_kwargs(
        self,
        output_dir: Path = Param("./outputs", help="The location of the output directory."),
        weight_decay: float = Param(
            None, help="The amount of weight decay. If None then it uses the default amount of weight decay in fastai."
        ),
        # l2_regularization: bool = Param(False, help="Whether to add decay to the gradients (L2 regularization) instead of to the weights directly (weight decay)."),
        **kwargs,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        return dict(
            loss_func=call_func(self.loss_func, **kwargs),
            metrics=call_func(self.metrics, **kwargs),
            path=self.output_dir,
            wd=weight_decay,
        )

    def loss_func(self, **kwargs):
        """The loss function. If None, then fastai will use the default loss function if it exists for this model."""
        return None

    def activation(self):
        """The activation for the last layer. If None, then fastai will use the default activiation of the loss if it exists."""
        return None

    def metrics(self) -> List:
        """
        The list of metrics to use with this app.

        By default this list is empty. This method should be subclassed to add metrics in child classes of TorchApp.

        Returns:
            List: The list of metrics.
        """
        return []

    def monitor(self) -> str:
        """
        The metric to optimize for when performing hyperparameter tuning.

        By default it returns 'valid_loss'.
        """
        return "valid_loss"

    def goal(self) -> str:
        """
        Sets the optimality direction when evaluating the metric from `monitor`.

        By default it produces the same behaviour as fastai callbacks (fastai.callback.tracker)
        ie. it is set to "minimize" if the monitor metric has the string 'loss' or 'err' otherwise it is "maximize".

        If the monitor is empty then this function returns None.
        """
        monitor = self.monitor()
        if not monitor or not isinstance(monitor, str):
            return None

        return "minimize" if ("loss" in monitor) or ("err" in monitor) else "maximize"

    def callbacks(
        self,
        project_name: str = Param(default=None, help="The name for this project for logging purposes."),
        run_name: str = Param(default=None, help="The name for this particular run for logging purposes."),
        run_id: str = Param(default=None, help="A unique ID for this particular run for logging purposes."),
        notes: str = Param(None, help="A longer description of the run for logging purposes."),
        tag: List[str] = Param(
            None, help="A tag for logging purposes. Multiple tags can be added each introduced with --tag."
        ),
        wandb: bool = Param(default=False, help="Whether or not to use 'Weights and Biases' for logging."),
        wandb_mode: str = Param(default="online", help="The mode for 'Weights and Biases'."),
        wandb_dir: Path = Param(None, help="The location for 'Weights and Biases' output."),
        wandb_entity: str = Param(None, help="An entity is a username or team name where you're sending runs."),
        wandb_group: str = Param(None, help="Specify a group to organize individual runs into a larger experiment."),
        wandb_job_type: str = Param(
            None,
            help="Specify the type of run, which is useful when you're grouping runs together into larger experiments using group.",
        ),
        mlflow: bool = Param(default=False, help="Whether or not to use MLflow for logging."),
    ) -> List:
        """
        The list of callbacks to use with this app in the fastai training loop.

        Args:
            project_name (str): The name for this project for logging purposes. If no name is given then the name of the app is used.

        Returns:
            list: The list of callbacks.
        """
        callbacks = [CSVLogger()]
        monitor = self.monitor()
        if monitor:
            callbacks.append(SaveModelCallback(monitor=monitor))

        if wandb:
            callback = TorchAppWandbCallback(
                app=self,
                project_name=project_name,
                id=run_id,
                name=run_name,
                mode=wandb_mode,
                dir=wandb_dir,
                entity=wandb_entity,
                group=wandb_group,
                job_type=wandb_job_type,
                notes=notes,
                tags=tag,
            )
            callbacks.append(callback)
            self.add_bibtex_file(bibtex_dir / "wandb.bib")  # this should be in the callback

        if mlflow:
            callbacks.append(TorchAppMlflowCallback(app=self, experiment_name=project_name))
            self.add_bibtex_file(bibtex_dir / "mlflow.bib")  # this should be in the callback

        extra_callbacks = self.extra_callbacks()
        if extra_callbacks:
            callbacks += extra_callbacks

        return callbacks

    def extra_callbacks(self):
        return None

    def show_batch(
        self, output_path: Path = Param(None, help="A location to save the HTML which summarizes the batch."), **kwargs
    ):
        dataloaders = call_func(self.dataloaders, **kwargs)

        # patch the display function of ipython so we can capture the HTML
        def mock_display(html_object):
            self.batch_html = html_object

        import IPython.display

        ipython_display = IPython.display.display
        IPython.display.display = mock_display

        dataloaders.show_batch()
        batch_html = getattr(self, "batch_html", None)

        if not batch_html:
            console.print(f"Cannot display batch as HTML")
            return

        html = batch_html.data

        # Write output
        if output_path:
            console.print(f"Writing batch HTML to '{output_path}'")
            with open(output_path, 'w') as f:
                f.write(html)
        else:
            console.print(html)
            console.print(f"To write this HTML output to a file, give an output path.")

        # restore the ipython display function
        IPython.display.display = ipython_display
        return self.batch_html

    def train(
        self,
        distributed: bool = Param(default=False, help="If the learner is distributed."),
        **kwargs,
    ) -> Learner:
        """
        Trains a model for this app.

        Args:
            distributed (bool, optional): If the learner is distributed. Defaults to Param(default=False, help="If the learner is distributed.").

        Returns:
            Learner: The fastai Learner object created for training.
        """
        self.training_kwargs = kwargs
        self.training_kwargs['distributed'] = distributed

        callbacks = call_func(self.callbacks, **kwargs)
        learner = call_func(self.learner, **kwargs)

        self.print_bibliography(verbose=True)

        # with learner.distrib_ctx() if distributed == True else nullcontext():
        call_func(self.fit, learner, callbacks, **kwargs)

        learner.export()

        return learner

    def fit(
        self,
        learner,
        callbacks,
        epochs: int = Param(default=20, help="The number of epochs."),
        freeze_epochs: int = Param(
            default=3,
            help="The number of epochs to train when the learner is frozen and the last layer is trained by itself. Only if `fine_tune` is set on the app.",
        ),
        learning_rate: float = Param(
            default=1e-4, help="The base learning rate (when fine tuning) or the max learning rate otherwise."
        ),
        **kwargs,
    ):
        if self.fine_tune:
            return learner.fine_tune(
                epochs, freeze_epochs=freeze_epochs, base_lr=learning_rate, cbs=callbacks, **kwargs
            )  # hack

        return learner.fit_one_cycle(epochs, lr_max=learning_rate, cbs=callbacks, **kwargs)

    def project_name(self) -> str:
        """
        The name to use for a project for logging purposes.

        The default is to use the class name.
        """
        return self.__class__.__name__

    def tune(
        self,
        runs: int = Param(default=1, help="The number of runs to attempt to train the model."),
        engine: str = Param(
            default="skopt",
            help="The optimizer to use to perform the hyperparameter tuning. Options: wandb, optuna, skopt.",
        ),  # should be enum
        id: str = Param(
            default="",
            help="The ID of this hyperparameter tuning job. "
            "If using wandb, then this is the sweep id. "
            "If using optuna, then this is the storage. "
            "If using skopt, then this is the file to store the results. ",
        ),
        name: str = Param(
            default="",
            help="An informative name for this hyperparameter tuning job. If empty, then it creates a name from the project name.",
        ),
        method: str = Param(
            default="", help="The sampling method to use to perform the hyperparameter tuning. By default it chooses the default method of the engine."
        ),  # should be enum
        min_iter: int = Param(
            default=None,
            help="The minimum number of iterations if using early termination. If left empty, then early termination is not used.",
        ),
        seed: int = Param(
            default=None,
            help="A seed for the random number generator.",
        ),
        **kwargs,
    ):
        if not name:
            name = f"{self.project_name()}-tuning"

        if engine == "wandb":
            from .tuning.wandb import wandb_tune

            self.add_bibtex_file(bibtex_dir / "wandb.bib")

            return wandb_tune(
                self,
                runs=runs,
                sweep_id=id,
                name=name,
                method=method,
                min_iter=min_iter,
                **kwargs,
            )
        elif engine == "optuna":
            from .tuning.optuna import optuna_tune

            self.add_bibtex_file(bibtex_dir / "optuna.bib")

            return optuna_tune(
                self,
                runs=runs,
                storage=id,
                name=name,
                method=method,
                seed=seed,
                **kwargs,
            )
        elif engine in ["skopt", "scikit-optimize"]:
            from .tuning.skopt import skopt_tune

            self.add_bibtex_file(bibtex_dir / "skopt.bib")

            return skopt_tune(
                self,
                runs=runs,
                file=id,
                name=name,
                method=method,
                seed=seed,
                **kwargs,
            )
        else:
            raise NotImplementedError(f"Optimizer engine {engine} not implemented.")

    def get_best_metric(self, learner):
        # The slice is there because 'epoch' is prepended to the list but it isn't included in the values
        metric_index = learner.recorder.metric_names[1:].index(self.monitor())
        metric_values = list(map(lambda row: row[metric_index], learner.recorder.values))
        metric_function = min if self.goal()[:3] == "min" else max
        metric_value = metric_function(metric_values)
        return metric_value

    def one_batch_output(self, **kwargs):
        learner = call_func(self.learner, **kwargs)
        batch = learner.dls.train.one_batch()
        n_inputs = getattr(learner.dls, 'n_inp', 1 if len(batch) == 1 else len(batch) - 1)
        batch_x = batch[:n_inputs]
        with torch.no_grad():
            output = learner.model(*batch_x)
        return output

    def one_batch_output_size(self, **kwargs):
        output = self.one_batch_output(**kwargs)
        return output.size()

    def one_batch_loss(self, **kwargs):
        learner = call_func(self.learner, **kwargs)
        batch = learner.dls.train.one_batch()
        n_inputs = getattr(learner.dls, 'n_inp', 1 if len(batch) == 1 else len(batch) - 1)
        batch_x = batch[:n_inputs]
        batch_y = batch[n_inputs:]
        with torch.no_grad():
            output = learner.model(*batch_x)
            loss = learner.loss_func(output, *batch_y)

        return loss

    def lr_finder(
        self, plot_filename: Path = None, start_lr: float = 1e-07, end_lr: float = 10, iterations: int = 100, **kwargs
    ):
        learner = call_func(self.learner, **kwargs)

        from matplotlib import pyplot as plt
        from fastai.callback.schedule import SuggestionMethod

        suggest_funcs = (
            SuggestionMethod.Valley,
            SuggestionMethod.Minimum,
            SuggestionMethod.Slide,
            SuggestionMethod.Steep,
        )

        result = learner.lr_find(
            stop_div=False,
            num_it=iterations,
            start_lr=start_lr,
            end_lr=end_lr,
            show_plot=plot_filename is not None,
            suggest_funcs=suggest_funcs,
        )

        if plot_filename is not None:
            plt.savefig(str(plot_filename))

        print("\n")
        table = Table(title="Suggested Learning Rates", box=SIMPLE)

        table.add_column("Method", style="cyan", no_wrap=True)
        table.add_column("Learning Rate", style="magenta")
        table.add_column("Explanation")

        for method, value in zip(suggest_funcs, result):
            table.add_row(method.__name__, str(value), method.__doc__)

        console.print(table)

        return result

    def cache_dir(self) -> Path:
        """ Returns a path to a directory where data files for this app can be cached. """
        cache_dir = Path(user_cache_dir("torchapps"))/self.__class__.__name__
        cache_dir.mkdir(exist_ok=True, parents=True)
        return cache_dir

    def output_results(
        self, 
        results, 
        **kwargs
    ):
        print(results)
        return results