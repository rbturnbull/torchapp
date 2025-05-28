from typing import Type
from pathlib import Path
import os
import sys
from collections.abc import Iterable
import inspect
import typer
import torch
import hashlib
from torch import nn
from appdirs import user_cache_dir
import lightning as L
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torchmetrics import Metric
from pytorch_lightning.profilers import Profiler, PyTorchProfiler
from rich.console import Console

from .modules import GeneralLightningModule
from .callbacks import TimeLoggingCallback, LogOptimizerCallback
from .cli import CLIApp, method, main, tool, flag
from .citations import Citable
from .params import Param
from .download import cached_download

console = Console()

BIBTEX_DIR = Path(__file__).parent / "bibtex"


def version_callback():
    print("version callback func - needs to be implemented in the app")


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
    
    @method("data")
    def one_batch(self, **kwargs) -> torch.Tensor:
        """ Returns a single batch of data. """
        data = self.data(**kwargs)
        if isinstance(data, L.LightningDataModule):
            train_dataloader = data.train_dataloader()
        else:
            train_dataloader = data

        first_batch = next(iter(train_dataloader))
        if isinstance(first_batch, tuple):
            return first_batch[0]
        else:
            return first_batch
        
    @tool("one_batch")
    def one_batch_size(self, **kwargs) -> torch.Size:
        """ Returns the size of a single batch. """
        batch = self.one_batch(**kwargs)
        size = [item.size() for item in batch]
        print(size)
        return size
    
    @tool("one_batch", "lightning_module")
    def one_batch_loss(self, **kwargs) -> torch.Tensor:
        """ Returns the loss of a single batch. """
        module = self.lightning_module(**kwargs)
        batch = self.one_batch(**kwargs)                
        loss = module.training_step(batch, batch_idx=0)
        print(loss)
        return loss
    
    @tool("data", "lightning_module")
    def one_batch_output_size(
        self, 
        **kwargs
    ) -> torch.Size:
        """ Returns the size of the output of a single batch. """
        data = self.data(**kwargs)
        module = self.lightning_module(**kwargs)
        if isinstance(data, L.LightningDataModule):
            train_dataloader = data.train_dataloader()
        else:
            train_dataloader = data

        first_batch = next(iter(train_dataloader))
        if isinstance(first_batch, (tuple, list)):
            inputs = first_batch[:module.input_count]
        else:
            inputs = first_batch
        
        results = module.model(*inputs)
        return results.shape if isinstance(results, torch.Tensor) else tuple(result.shape for result in results)

    @method
    def validation_dataloader(self) -> Iterable|None:
        return None

    @method
    def prediction_dataloader(self, module) -> Iterable:
        raise NotImplementedError(f"Please ensure that the 'prediction_dataloader' method is implemented in {self.__class__.__name__}.")

    @method
    def monitor(self) -> str:
        return "valid_loss"
    
    @method
    def goal(self) -> str:
        monitor = self.monitor() or "valid_loss"
        return "minimize" if "loss" in monitor else "maximize"

    @method
    def checkpoint(self, checkpoint:Path=None, **kwargs) -> Path:
        """ Returns a path to a checkpoint to use for prediction. """
        if not checkpoint:
            raise ValueError("Please provide a checkpoint path or implement the 'checkpoint' method in your app.")
        return checkpoint

    @method("monitor")
    def checkpoint_callback(self, save_top_k:int=1) -> ModelCheckpoint|list[ModelCheckpoint]:
        monitor = self.monitor()

        goal = self.goal()
        goal = goal.lower()[:3]
        assert goal in ["min", "max"], f"Goal '{goal}' not recognized."

        class WeightsOnlyCheckpoint(ModelCheckpoint):
            @property
            def state_key(self) -> str:
                return "weights_only_checkpoint"

        checkpoints = []
        weights_checkpoint = WeightsOnlyCheckpoint(
            save_top_k=save_top_k,
            monitor=monitor,
            mode=goal,
            save_weights_only=True,
            filename="weights-{epoch:02d}-{"+monitor+":.2g}",
            verbose=True,
        )
        # weights_checkpoint.state_key = "weights"
        checkpoints.append(weights_checkpoint)

        checkpoint = ModelCheckpoint(
            save_top_k=save_top_k,
            monitor=monitor,
            mode=goal,
            save_weights_only=False,
            filename="checkpoint-{epoch:02d}-{"+monitor+":.2g}",
            verbose=True,
        )
        # checkpoint.state_key = "checkpoint"

        checkpoints.append(checkpoint)

        return checkpoints

    @method("extra_callbacks")
    def callbacks(self, **kwargs):
        callbacks = [
            TimeLoggingCallback(),
            LogOptimizerCallback(),
        ]
        if sys.stdout.isatty():
            callbacks.append(RichProgressBar(leave=True))

        if checkpoint_callbacks := self.checkpoint_callback(**kwargs):
            if isinstance(checkpoint_callbacks, ModelCheckpoint):
                checkpoint_callbacks = [checkpoint_callbacks]
            callbacks.extend(checkpoint_callbacks)
            
        callbacks += self.extra_callbacks(**kwargs) or []
        return callbacks
    
    # Hack - this should be done through calling the 'super' of 'callbacks'
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
        wandb_entity:str="",
        wandb_offline:bool=False,
        wandb_dir:Path=None,
        max_gpus:int=0,
        log_every_n_steps:int=50,
        gradient_clip_val:float=Param(None, help="The value to clip the gradients to. If None, then no clipping is done."),
        **kwargs,
    ) -> L.Trainer:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        run_name = run_name or output_dir.name

        loggers = [
            CSVLogger(save_dir=output_dir),
        ]
        if wandb:
            if wandb_entity:
                os.environ["WANDB_ENTITY"] = wandb_entity

            project_name = self.project_name(**kwargs)
            if wandb_dir:
                wandb_dir = Path(wandb_dir)
                wandb_dir.mkdir(exist_ok=True, parents=True)
            wandb_logger = WandbLogger(name=run_name, project=project_name, offline=wandb_offline, save_dir=wandb_dir)
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
            log_every_n_steps=log_every_n_steps,
            gradient_clip_val=gradient_clip_val,
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
    def extra_hyperparameters(self) -> dict:
        """ Extra hyperparameters to save with the module. """
        return {}
    
    @method
    def module_class(self) -> Type[GeneralLightningModule]:
        return GeneralLightningModule
        
    @method("model", "loss_function", "extra_hyperparameters", "input_count", "metrics", "module_class")
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
        extras = self.extra_hyperparameters(**kwargs)

        return module_class(
            model=model,
            loss_function=loss_function,
            max_learning_rate=max_learning_rate,
            input_count=input_count,
            metrics=metrics,
            **extras,
        )
    
    @tool("setup", "data", "lightning_module", "trainer")
    def train(
        self,
        **kwargs,
    ) -> L.LightningModule:
        """Train the model."""
        style = 'bold red'
        console.rule("Setting up training", style=style)
        self.setup(**kwargs)
        
        console.print("Setting up dataloaders")
        data = self.data(**kwargs)
        data.setup("fit")
        validation_dataloader = self.validation_dataloader(**kwargs)

        console.print("Setting up Module")
        lightning_module = self.lightning_module(**kwargs)

        console.print("Setting up Trainer")
        trainer = self.trainer(**kwargs)

        # Dummy data to set the number of weights in the model
        console.print("Training Dummy Batch")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dummy_batch = next(iter(data.train_dataloader()))
        dummy_x = dummy_batch[:lightning_module.input_count]
        with torch.no_grad():
            dummy_x = [x.to(device) for x in dummy_x]
            model = lightning_module.model.to(device)
            model(*dummy_x)

        console.rule("Training", style=style)
        trainer.fit( lightning_module, data, validation_dataloader )

        return lightning_module, trainer

    @tool("setup", "data", "load_checkpoint", "trainer")
    def validate(
        self,
        **kwargs,
    ) -> L.LightningModule:
        """ Validate the model. """
        style = 'bold red'
        console.rule("Setting up", style=style)
        self.setup(**kwargs)
        
        console.print("Setting up dataloaders")
        data = self.data(**kwargs)
        data.setup("fit")
        validation_dataloader = self.validation_dataloader(**kwargs)

        console.print("Setting up Module")
        lightning_module = self.load_checkpoint(**kwargs)

        console.print("Setting up Trainer")
        trainer = self.trainer(**kwargs)

        console.rule("Validating", style=style)
        result = trainer.validate( lightning_module, data, validation_dataloader )

        return result

    def process_location(self, location: str, reload:bool=False) -> Path:
        """
        Process a location string into a Path
        
        Can be a URL or a local path.
        """
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
            location = path
        else:
            path = Path(location)
        
        return path

    @method("checkpoint")
    def load_checkpoint(
        self, 
        reload: bool = Param(
            default=False,
            help="Should the checkpoint be downloaded again if it is online and already present locally.",
        ),
        **kwargs,
    ) -> L.LightningModule:
        module_class = self.module_class(**kwargs)

        location = self.checkpoint(**kwargs)
        path = self.process_location(location, reload=reload)

        if not path or not path.is_file():
            raise FileNotFoundError(f"Cannot find pretrained model at '{path}'")

        return module_class.load_from_checkpoint(location)
        
    def cache_dir(self) -> Path:
        """ Returns a path to a directory where data files for this app can be cached. """
        cache_dir = Path(user_cache_dir("torchapps"))/self.__class__.__name__
        cache_dir.mkdir(exist_ok=True, parents=True)
        return cache_dir
    
    @method
    def prediction_trainer(self, module) -> L.Trainer:
        # TODO multigpu
        return L.Trainer()
    
    @method
    def output_results(self, results, **kwargs):
        raise NotImplementedError(f"Please ensure that the 'output_results' method is implemented in {self.__class__.__name__}.")

    @method("predict")
    def __call__(self, **kwargs):
        return self.predict(**kwargs)

    @main("load_checkpoint", "prediction_trainer", "prediction_dataloader", "output_results")
    def predict(self, **kwargs):
        """ Make predictions with the model. """
        module = self.load_checkpoint(**kwargs)
        trainer = self.prediction_trainer(module, **kwargs)
        prediction_dataloader = self.prediction_dataloader(module, **kwargs)

        results = trainer.predict(module, dataloaders=prediction_dataloader)

        # Concatenate results of prediction batches
        if len(results) == 0:
            return None
        if isinstance(results[0], tuple):
            # zip and cat elements of tuples
            results = tuple(torch.cat(elements, dim=0) for elements in zip(*results))
        else:
            results = torch.cat(results, dim=0)

        return self.output_results(results, **kwargs)        

    @tool("train", "project_name")
    def tune(
        self,
        output_dir:Path=Param("./outputs", help="The location of the output directory."),
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
                output_dir=output_dir,
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

    def tuning_params(self):
        tuning_params = {}
        signature = inspect.signature(self.tune)

        for key, value in signature.parameters.items():
            default_value = value.default
            if isinstance(default_value, Param) and default_value.tune == True:

                # Override annotation if given in typing hints
                if value.annotation:
                    default_value.annotation = value.annotation

                default_value.check_choices()

                tuning_params[key] = default_value
                
        return tuning_params

    @flag(shortcut="-v")
    def version(
        self,
    ) -> str:
        """
        The version of this package.
        """
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
        else:
            raise Exception("Cannot find package.")
        
        return version

    @flag
    def bibtex(self) -> str:
        """
        The BibTeX entry for this app.
        """
        return super().bibtex()
    
    @flag
    def bibliography(self) -> str:
        """
        The bibliography for this app.
        """
        return super().bibliography()    