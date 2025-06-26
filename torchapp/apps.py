from typing import Type, TYPE_CHECKING
from pathlib import Path
import os

from rich.console import Console

from .cli import CLIApp, method, main, tool, flag
from .citations import Citable
from .params import Param


if TYPE_CHECKING:
    # These are only needed for type-checking (mypy, pylance, etc.)
    import torch
    from torch import nn
    from torchmetrics import Metric
    from lightning import LightningModule, Trainer, LightningDataModule
    from lightning.pytorch.callbacks import ModelCheckpoint, Callback
    from pytorch_lightning.profilers import Profiler
    from collections.abc import Iterable
    from .modules import GeneralLightningModule


console = Console()
BIBTEX_DIR = Path(__file__).parent / "bibtex"


class TorchApp(Citable, CLIApp):
    @method
    def setup(self) -> None:
        pass

    def get_bibtex_files(self):
        return [
            BIBTEX_DIR / "torchapp.bib",
        ]

    @method
    def model(self) -> 'nn.Module':
        raise NotImplementedError(
            f"Please ensure that the 'model' method is implemented in {self.__class__.__name__}."
        )

    @method
    def loss_function(self):
        raise NotImplementedError(
            f"Please ensure that the 'loss_function' method is implemented in {self.__class__.__name__}."
        )

    @method()
    def data(self) -> 'Iterable|LightningDataModule':
        raise NotImplementedError(
            f"Please ensure that the 'data' method is implemented in {self.__class__.__name__}."
        )

    @method("setup", "data", "validation_dataloader")
    def setup_and_data(self, **kwargs) -> 'LightningDataModule|Iterable':
        """
        Sets up the app and returns either a LightningDataModule or an Iterable (e.g. a DataLoader).
        """
        from lightning import LightningDataModule

        style = 'bold red'
        console.rule("Setting up app", style=style)
        self.setup(**kwargs)

        console.print("Setting up data")
        data = self.data(**kwargs)

        if isinstance(data, LightningDataModule):
            data.setup("fit")
        return data

    @method("setup_and_data")
    def one_batch(self, **kwargs) -> 'torch.Tensor':
        """
        Returns a single batch of data, without labels.
        """
        from lightning import LightningDataModule  # to check instance

        data = self.setup_and_data(**kwargs)
        if isinstance(data, LightningDataModule):
            train_dataloader = data.train_dataloader()
        else:
            train_dataloader = data

        first_batch = next(iter(train_dataloader))
        if isinstance(first_batch, tuple):
            return first_batch[0]
        else:
            return first_batch

    @tool("one_batch")
    def one_batch_size(self, **kwargs) -> 'torch.Size':
        """ Returns the size of a single batch. """
        batch = self.one_batch(**kwargs)
        size = [item.size() for item in batch]
        print(size)
        return size
    
    @tool("one_batch", "lightning_module")
    def one_batch_loss(self, **kwargs) -> 'torch.Tensor':
        """
        Returns the loss (a tensor) of that single batch using the module's training_step.
        """
        batch = self.one_batch(**kwargs)
        module = self.lightning_module(**kwargs)

        # We assume LightningModule.training_step(batch, batch_idx) returns a loss-tensor
        loss = module.training_step(batch, batch_idx=0)
        print(loss)
        return loss

    @tool("setup_and_data", "lightning_module")
    def one_batch_output_size(self, **kwargs) -> 'torch.Size':
        """
        Returns the shape of the output from model(*inputs) for that single batch.
        """
        import torch
        from lightning import LightningDataModule

        data = self.setup_and_data(**kwargs)
        module = self.lightning_module(**kwargs)

        if isinstance(data, LightningDataModule):
            train_dataloader = data.train_dataloader()
        else:
            train_dataloader = data

        first_batch = next(iter(train_dataloader))

        # If the batch is a tuple/list, split off inputs up to input_count
        if isinstance(first_batch, (tuple, list)):
            inputs = first_batch[: module.input_count]
        else:
            inputs = first_batch

        # Forward-pass through the model
        results = module.model(*inputs)

        if isinstance(results, torch.Tensor):
            return results.shape
        else:
            # If model returns a tuple of tensors, return a tuple of shapes
            return tuple(r.shape for r in results)

    @method
    def validation_dataloader(self) -> 'Iterable|None':
        return None

    @method
    def prediction_dataloader(self, module) -> 'Iterable':
        raise NotImplementedError(
            f"Please ensure that the 'prediction_dataloader' method is implemented in {self.__class__.__name__}."
        )

    @method
    def monitor(self) -> str:
        return "valid_loss"

    @method
    def goal(self) -> str:
        monitor = self.monitor() or "valid_loss"
        return "minimize" if "loss" in monitor else "maximize"

    @method
    def checkpoint(self, checkpoint: Path = None, **kwargs) -> Path:
        """
        Returns a path (local or remote) to a checkpoint.  
        By default, you must override or pass `--checkpoint`.
        """
        if not checkpoint:
            raise ValueError(
                "Please provide a checkpoint path or implement the 'checkpoint' method in your app."
            )
        return checkpoint

    @method("monitor")
    def checkpoint_callback(self, save_top_k: int = 1) -> 'ModelCheckpoint|list[ModelCheckpoint]':
        """
        Build both a Weights-only checkpoint and a full-state checkpoint,
        saving the top K models based on `monitor`.
        """
        from lightning.pytorch.callbacks import ModelCheckpoint

        monitor = self.monitor()
        goal = self.goal().lower()[:3]
        assert goal in ["min", "max"], f"Goal '{goal}' not recognized."

        class WeightsOnlyCheckpoint(ModelCheckpoint):
            @property
            def state_key(self) -> str:
                return "weights_only_checkpoint"

        checkpoints = []

        # 1) weights-only checkpoint
        weights_ckpt = WeightsOnlyCheckpoint(
            save_top_k=save_top_k,
            monitor=monitor,
            mode=goal,
            save_weights_only=True,
            filename="weights-{epoch:02d}-{"+monitor+":.2g}",
            verbose=True,
        )
        checkpoints.append(weights_ckpt)

        # 2) full checkpoint
        full_ckpt = ModelCheckpoint(
            save_top_k=save_top_k,
            monitor=monitor,
            mode=goal,
            save_weights_only=False,
            filename="checkpoint-{epoch:02d}-{"+monitor+":.2g}",
            verbose=True,
        )
        checkpoints.append(full_ckpt)

        return checkpoints

    @method("extra_callbacks")
    def callbacks(self, **kwargs):
        """
        Combine all callbacks: 
        - Always run TimeLoggingCallback and LogOptimizerCallback.
        - Optionally show a RichProgressBar if running interactively.
        - Always include the checkpoint callbacks.
        - Also include any extra_callbacks() provided by the user.
        """
        from lightning.pytorch.callbacks import ModelCheckpoint  # needed for isinstance
        from .callbacks import TimeLoggingCallback, LogOptimizerCallback
        import sys
        
        callbacks = [
            TimeLoggingCallback(),
            LogOptimizerCallback(),
        ]

        # If stdout is a TTY, add a nice progress bar
        if sys.stdout.isatty():
            from lightning.pytorch.callbacks import RichProgressBar
            callbacks.append(RichProgressBar(leave=True))

        # Insert both weight-only and full checkpoints
        if checkpoint_cbs := self.checkpoint_callback(**kwargs):
            if isinstance(checkpoint_cbs, ModelCheckpoint):
                checkpoint_cbs = [checkpoint_cbs]
            callbacks.extend(checkpoint_cbs)

        # Add any user-defined extra_callbacks
        callbacks += self.extra_callbacks(**kwargs) or []
        return callbacks

    @method
    def extra_callbacks(self, **kwargs) -> 'list[Callback]':
        """
        If you want to add your own callbacks on top of the defaults,
        override this. Must return a list of `lightning.pytorch.callbacks.Callback` objects.
        """
        return []

    @method
    def profiler(
        self,
        profiler_path: Path = None,
        profile_memory: bool = False,
        **kwargs,
    ) -> 'Profiler|None':
        """
        If `profiler_path` is given, return a PyTorchProfiler that writes to that file.
        """
        from pytorch_lightning.profilers import PyTorchProfiler

        if profiler_path:
            return PyTorchProfiler(
                filename=str(profiler_path),
                profile_memory=profile_memory,
            )
        return None

    @method
    def project_name(self, project_name: str = Param(default="", help="The name of this project."), **kwargs) -> str:
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
    ) -> 'Trainer':
        """
        Instantiate and return a Lightning `Trainer`.  
        - Sets up CSVLogger by default under `output_dir`.  
        - If `wandb=True`, also sets up a WandbLogger.  
        - Automatically chooses GPUs if available (up to `max_gpus`).  
        """
        import torch
        from pytorch_lightning.loggers import CSVLogger, WandbLogger
        from lightning import Trainer

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        run_name = run_name or output_dir.name

        # 1) Always have a CSV logger writing to `output_dir`
        loggers = [CSVLogger(save_dir=output_dir)]

        # 2) If wandb is requested, configure a WandbLogger
        if wandb:
            if wandb_entity:
                os.environ["WANDB_ENTITY"] = wandb_entity

            project = self.project_name(**kwargs)
            if wandb_dir:
                wandb_dir = Path(wandb_dir)
                wandb_dir.mkdir(exist_ok=True, parents=True)

            wandb_logger = WandbLogger(
                name=run_name,
                project=project,
                offline=wandb_offline,
                save_dir=wandb_dir,
            )
            loggers.append(wandb_logger)

        # 3) GPU logic: use all available up to max_gpus; otherwise, use CPUs
        available_gpus = torch.cuda.device_count()
        if max_gpus:
            available_gpus = min(max_gpus, available_gpus)

        if available_gpus > 1:
            devices = available_gpus
            strategy = 'ddp'  # Distributed Data Parallel
        else:
            devices = "auto"  # "auto" will pick CPU if no GPU is found
            strategy = "auto"

        return Trainer(
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
    def metrics(self) -> 'list[tuple[str,Metric]]':
        return []

    @method
    def input_count(self) -> int:
        return 1

    @method
    def extra_hyperparameters(self) -> dict:
        """
        Any extra hyperparameters to pass to GeneralLightningModule.__init__.
        """
        return {}

    @method
    def module_class(self) -> 'Type[GeneralLightningModule]':
        from .modules import GeneralLightningModule
        return GeneralLightningModule

    @method("model", "loss_function", "extra_hyperparameters", "input_count", "metrics", "module_class")
    def lightning_module(
        self,
        max_learning_rate: float = 1e-4,
        **kwargs,
    ) -> 'LightningModule':
        """
        Build and return an instance of your LightningModule subclass,
        passing in the model, loss, metrics, etc.
        """
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

    @tool("setup_and_data", "validation_dataloader", "lightning_module", "trainer")
    def train(self, **kwargs) -> 'LightningModule':
        """
        Train for `max_epochs`.  
        Steps:  
          1. call setup_and_data → get DataModule or DataLoader  
          2. call validation_dataloader()  
          3. instantiate LightningModule + Trainer  
          4. do a dummy forward-pass (so Lightning can count params)  
          5. `trainer.fit(...)`  
        Returns `(lightning_module, trainer)` after training completes.
        """
        import torch
        from lightning import LightningDataModule

        style = 'bold red'
        data = self.setup_and_data(**kwargs)
        validation_dataloader = self.validation_dataloader(**kwargs)

        console.print("Setting up Module")
        lightning_module = self.lightning_module(**kwargs)

        console.print("Setting up Trainer")
        trainer = self.trainer(**kwargs)

        # Do one dummy forward to let Lightning “see” the model structure
        console.print("Training Dummy Batch")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # get a single train batch
        if isinstance(data, LightningDataModule):
            dummy_loader = data.train_dataloader()
        else:
            dummy_loader = data
        dummy_batch = next(iter(dummy_loader))

        # split off inputs
        dummy_inputs = dummy_batch[: lightning_module.input_count]
        with torch.no_grad():
            dummy_inputs = [x.to(device) for x in dummy_inputs]
            model = lightning_module.model.to(device)
            model(*dummy_inputs)

        console.rule("Training", style=style)
        trainer.fit(lightning_module, data, validation_dataloader)

        return lightning_module, trainer

    @tool("setup_and_data", "load_checkpoint", "trainer")
    def validate(self, **kwargs) -> 'LightningModule':
        """
        Validate the model on the validation set.
        """
        from lightning import LightningDataModule

        style = 'bold red'
        data = self.setup_and_data(**kwargs)
        data.setup("fit")
        validation_dataloader = self.validation_dataloader(**kwargs)

        console.print("Setting up Module")
        lightning_module = self.load_checkpoint(**kwargs)

        console.print("Setting up Trainer")
        trainer = self.trainer(**kwargs)

        console.rule("Validating", style=style)
        result = trainer.validate(lightning_module, data, validation_dataloader)

        return result

    def process_location(self, location: str, reload: bool = False) -> Path:
        """
        Turn a URL or filesystem path into a Path that points to a local file.
        If it's a URL, download (and cache) via cached_download().  
        """
        from pathlib import Path
        import hashlib
        from .download import cached_download

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
            cache_path = self.cache_dir() / f"{name_stem}-{url_hash}{extension}"
            cached_download(location, cache_path, force=reload)
            return cache_path
        else:
            return Path(location)

    @method("checkpoint")
    def load_checkpoint(
        self,
        reload: bool = Param(
            default=False,
            help="Should the checkpoint be re-downloaded if it's remote and already exists locally?",
        ),
        **kwargs,
    ) -> 'LightningModule':
        """
        Locate a checkpoint (via `self.checkpoint(...)`), download it if needed,
        then call `.load_from_checkpoint(...)` on the module class.
        """
        module_class = self.module_class(**kwargs)

        location = self.checkpoint(**kwargs)
        path = self.process_location(location, reload=reload)

        if not path or not path.is_file():
            raise FileNotFoundError(f"Cannot find pretrained model at '{path}'")

        # The class method `load_from_checkpoint` expects the local file path
        return module_class.load_from_checkpoint(str(path))

    def cache_dir(self) -> Path:
        """
        Returns a directory under the user's cache folder for storing downloads,
        e.g. `~/.cache/torchapps/<AppName>/`.
        """
        from appdirs import user_cache_dir
        cache_dir = Path(user_cache_dir("torchapps")) / self.__class__.__name__
        cache_dir.mkdir(exist_ok=True, parents=True)
        return cache_dir

    @method
    def prediction_trainer(self, module) -> 'Trainer':
        """
        Return a Trainer for making predictions. By default, just a brand-new Trainer().
        """
        from lightning import Trainer

        # TODO multigpu
        return Trainer()

    @method
    def output_results(self, results, **kwargs):
        raise NotImplementedError(f"Please ensure that the 'output_results' method is implemented in {self.__class__.__name__}.")

    @method("predict")
    def __call__(self, **kwargs):
        return self.predict(**kwargs)

    @main("load_checkpoint", "prediction_trainer", "prediction_dataloader", "output_results")
    def predict(self, **kwargs):
        """ Runs predictions with model and outputs the results. """
        import torch

        module = self.load_checkpoint(**kwargs)
        trainer = self.prediction_trainer(module, **kwargs)
        prediction_dataloader = self.prediction_dataloader(module, **kwargs)

        results_list = trainer.predict(module, dataloaders=prediction_dataloader)

        if not results_list:
            return None

        # If each batch returns a tuple, zip-and-concatenate along dim=0
        first_item = results_list[0]
        if isinstance(first_item, tuple):
            results = tuple(
                torch.cat(elements, dim=0) for elements in zip(*results_list)
            )
        else:
            results = torch.cat(results_list, dim=0)

        return self.output_results(results, **kwargs)

    @tool("train", "project_name")
    def tune(
        self,
        output_dir: Path = Param("./outputs", help="Where to write tuning results."),
        runs: int = Param(default=1, help="Number of parallel runs for tuning."),
        engine: str = Param(
            default="skopt",
            help="Which optimizer to use: 'wandb', 'optuna', or 'skopt'.",
        ),
        id: str = Param(
            default="",
            help="If using wandb: sweep ID. If optuna: storage URL. If skopt: file name.",
        ),
        name: str = Param(
            default="",
            help="An identifier for this tuning job. Defaults to `<project_name>-tuning`.",
        ),
        method: str = Param(
            default="",
            help="Sampling method for the tuner (engine-specific).",
        ),
        min_iter: int = Param(
            default=None,
            help="Minimum number of iterations for early stopping (if supported).",
        ),
        seed: int = Param(
            default=None,
            help="Random-seed for reproducibility (if supported).",
        ),
        **kwargs,
    ):
        """
        Perform hyperparameter tuning with one of three engines:
        - wandb (requires a wandb sweep ID),
        - optuna (requires a storage string),
        - skopt (requires a file path).
        """
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
            raise NotImplementedError(f"Optimizer engine '{engine}' not implemented.")

    def tuning_params(self):
        """
        Inspect the signature of self.tune(...) and return a dict of only
        those Param() arguments where `Param.tune == True`.  
        (So that an external tuner can see which hyperparameters are tunable.)
        """
        import inspect

        tuning_params = {}
        signature = inspect.signature(self.tune)

        for key, value in signature.parameters.items():
            default_value = value.default
            if isinstance(default_value, Param) and default_value.tune is True:
                # Override annotation if given in typing hints
                if value.annotation:
                    default_value.annotation = value.annotation

                default_value.check_choices()

                tuning_params[key] = default_value
        return tuning_params

    def package_name(self) -> str:
        """
        Attempt to discover this app's package name by:
        - Checking `__package__` on the module  
        - Otherwise walking up the filesystem until we find a folder that holds a distribution
        """
        import inspect
        from importlib import metadata

        module_obj = inspect.getmodule(self)
        package = ""
        if module_obj and module_obj.__package__:
            package = module_obj.__package__.split('.')[0]
        else:
            path = Path(module_obj.__file__).parent
            while path.name:
                try:
                    if metadata.distribution(path.name):
                        package = path.name
                        break
                except Exception:
                    pass
                path = path.parent
        return package
    @flag(shortcut="-v")
    def version(self) -> str:
        """
        Return the installed version of this package (from `importlib.metadata`).
        """
        import importlib.metadata as metadata

        package = self.package_name()
        if package:
            try:
                return metadata.version(package)
            except metadata.PackageNotFoundError as e:
                print(e)
                return ""

        else:
            print("Package name could not be determined.")
            return ""

    @flag
    def bibtex(self) -> str:
        """
        Return a combined BibTeX string of all citations in this app.
        """
        return super().bibtex()

    @flag
    def bibliography(self) -> str:
        """
        Return human-readable bibliography entries for this app.
        """
        return super().bibliography()
