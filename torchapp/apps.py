from typing import Type
from pathlib import Path
import os
from collections.abc import Iterable
import inspect
import typer
import torch
from torch import nn
import lightning as L
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torchmetrics import Metric
from pytorch_lightning.profilers import Profiler, PyTorchProfiler

from .modules import GeneralLightningModule
from .callbacks import TimeLoggingCallback, LogOptimizerCallback
from .cli import CLIApp, method, main, tool
from .citations import Citable
from .params import Param


BIBTEX_DIR = Path(__file__).parent / "bibtex"


class TorchApp(Citable,CLIApp):
    @method
    def setup(self) -> None:
        pass

    def get_bibtex_files(self):
        return [
            BIBTEX_DIR / "torchapp.bib",
        ]

    @method
    def model(self) -> nn.Module:
        raise NotImplementedError(f"Please ensure that the 'model' method is implemented in {self.__class__.__name__}.")
    
    @method
    def loss_function(self):
        raise NotImplementedError(f"Please ensure that the 'loss_function' method is implemented in {self.__class__.__name__}.")
    
    @method()
    def data(self) -> Iterable|L.LightningDataModule:
        raise NotImplementedError(f"Please ensure that the 'data' method is implemented in {self.__class__.__name__}.")
    
    @method
    def validation_dataloader(self) -> Iterable|None:
        return None

    @method
    def prediction_dataloader(self, module) -> Iterable:
        raise NotImplementedError(f"Please ensure that the 'data' method is implemented in {self.__class__.__name__}.")

    @method("extra_callbacks")
    def callbacks(self, **kwargs):
        callbacks = [
            RichProgressBar(leave=True),
            TimeLoggingCallback(),
            LogOptimizerCallback(),
        ]
        callbacks += self.extra_callbacks(**kwargs) or []
        return callbacks
    
    @method
    def extra_callbacks(self, **kwargs) -> list[L.Callback]:
        return []

    @method
    def profiler(
        self,
        profiler_path:Path=None,
        profile_memory:bool=False,
        **kwargs,
    ) -> Profiler|None:
        if profiler_path:
            return PyTorchProfiler(
                filename=str(profiler_path),
                profile_memory=profile_memory,
            )
        return None
    
    @method
    def project_name(self, project_name:str=Param(default="", help="The name of this project (for logging purposes). Defaults to the name of the app."), **kwargs) -> str:
        return project_name or self.__class__.__name__

    @method("callbacks", "profiler", "project_name")
    def trainer(
        self,
        output_dir:Path=Param("./outputs", help="The location of the output directory."),
        max_epochs:int=20,
        run_name:str="",
        wandb:bool=False,
        wandb_project:str="",
        wandb_entity:str="",
        max_gpus:int=0,
        **kwargs,
    ) -> L.Trainer:
        run_name = run_name or output_dir.name

        loggers = [
            CSVLogger("logs", name=run_name)
        ]
        if wandb:
            if wandb_project:
                os.environ["WANDB_PROJECT"] = wandb_project
            if wandb_entity:
                os.environ["WANDB_ENTITY"] = wandb_entity

            wandb_logger = WandbLogger(name=run_name, project=self.project_name(**kwargs))
            loggers.append(wandb_logger)
        
        # If GPUs are available, use all of them; otherwise, use CPUs
        gpus = torch.cuda.device_count()
        if max_gpus:
            gpus = min(max_gpus, gpus)
        
        if gpus > 1:
            devices = gpus
            strategy = 'ddp'  # Distributed Data Parallel
        else:
            devices = "auto"  # Will use CPU if no GPU is available
            strategy = "auto"

        return L.Trainer(
            default_root_dir=output_dir,
            accelerator="auto",
            devices=devices, 
            strategy=strategy, 
            logger=loggers, 
            max_epochs=max_epochs,
            callbacks=self.callbacks(**kwargs),
            profiler=self.profiler(**kwargs),
        )
    
    @method
    def metrics(self) -> list[tuple[str,Metric]]:
        return []
    
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

    @method
    def input_count(self) -> int:
        return 1
    
    @method
    def module_class(self) -> Type[GeneralLightningModule]:
        return GeneralLightningModule
        
    @method("model", "loss_function")
    def lightning_module(
        self,
        max_learning_rate:float = 1e-4,
        **kwargs,
    ) -> L.LightningModule:
        model = self.model(**kwargs)
        loss_function = self.loss_function(**kwargs)
        metrics = self.metrics(**kwargs)
        input_count = self.input_count(**kwargs)

        module_class = self.module_class(**kwargs)

        return module_class(
            model=model,
            loss_function=loss_function,
            max_learning_rate=max_learning_rate,
            input_count=input_count,
            metrics=metrics,
        )
    
    @tool("setup", "data", "lightning_module", "trainer")
    def train(
        self,
        **kwargs,
    ):
        """Train the model."""
        self.setup(**kwargs)
        data = self.data(**kwargs)
        data.setup("fit")
        validation_dataloader = self.validation_dataloader(**kwargs)

        lightning_module = self.lightning_module(**kwargs)
        trainer = self.trainer(**kwargs)

        # Dummy data to set the number of weights in the model
        dummy_batch = next(iter(data.train_dataloader()))
        dummy_x = dummy_batch[:lightning_module.input_count]
        with torch.no_grad():
            lightning_module.model(*dummy_x)

        trainer.fit( lightning_module, data, validation_dataloader )

    @method
    def checkpoint(self, checkpoint:Path=None, **kwargs) -> Path:
        """ Returns a path to a checkpoint to use for prediction. """
        if not checkpoint:
            raise ValueError("Please provide a checkpoint path or implement the 'checkpoint' method in your app.")
        return checkpoint
        
    @method("checkpoint")
    def load_checkpoint(self, **kwargs) -> L.LightningModule:
        module_class = self.module_class(**kwargs)
        return module_class.load_from_checkpoint(self.checkpoint(**kwargs))
    
    @method
    def prediction_trainer(self, module) -> L.Trainer:
        # TODO multigpu
        return L.Trainer()
    
    @method
    def output_results(self, results, **kwargs):
        raise NotImplementedError(f"Please ensure that the 'output_results' method is implemented in {self.__class__.__name__}.")

    @main("load_checkpoint", "prediction_trainer", "prediction_dataloader", "output_results")
    def predict(self, **kwargs):
        """ Make predictions with the model. """
        module = self.load_checkpoint(**kwargs)
        trainer = self.prediction_trainer(module, **kwargs)
        prediction_dataloader = self.prediction_dataloader(module, **kwargs)

        results = trainer.predict(module, dataloaders=prediction_dataloader)

        return self.output_results(results, **kwargs)

    @tool("train", "project_name")
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
        """ Perform hyperparameter tuning. """
        if not name:
            name = f"{self.project_name(**kwargs)}-tuning"

        if engine == "wandb":
            from .tuning.wandb import wandb_tune

            self.add_bibtex_file(BIBTEX_DIR / "wandb.bib")

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

            self.add_bibtex_file(BIBTEX_DIR / "optuna.bib")

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

            self.add_bibtex_file(BIBTEX_DIR / "skopt.bib")

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



