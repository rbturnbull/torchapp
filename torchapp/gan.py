from torch import nn
from fastai.learner import Learner
from pathlib import Path
from fastai.vision.gan import GANLearner

from .citations import torchapp_bibtex_dir
from .util import copy_func, call_func, change_typer_to_defaults, add_kwargs
from .apps import TorchApp, console
from .params import Param


class GANApp(TorchApp):
    def __init__(self):
        super().__init__()

        # Make deep copies of methods so that we can change the function signatures dynamically
        self.generator = self.copy_method(self.generator)
        self.critic = self.copy_method(self.critic)

        add_kwargs(to_func=self.learner, from_funcs=[self.generator, self.critic])

        # Remove params from defaults in methods not used for the cli
        change_typer_to_defaults(self.generator)
        change_typer_to_defaults(self.critic)


    def build_learner_func(self):
        """
        Returns GANLearner.wgan

        For more information see: https://docs.fast.ai/vision.gan.html
        """
        return GANLearner.wgan

    def generator(self) -> nn.Module:
        raise NotImplementedError(f"Please ensure that the 'generator' method is implemented in {self.__class__.__name__}.")

    def critic(self) -> nn.Module:
        raise NotImplementedError(f"Please ensure that the 'critic' method is implemented in {self.__class__.__name__}.")

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

        console.print("Building generator", style="bold")
        generator = call_func(self.generator, **kwargs)

        console.print("Building critic", style="bold")
        critic = call_func(self.critic, **kwargs)

        console.print("Building learner", style="bold")
        learner_kwargs = call_func(self.learner_kwargs, **kwargs)
        build_learner_func = self.build_learner_func()
        learner = build_learner_func(
            dataloaders,
            generator,
            critic,
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
        **kwargs,
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        return dict(
            path=output_dir,
            wd=weight_decay,
        )

    def get_bibtex_files(self):
        files = super().get_bibtex_files()
        files.append(torchapp_bibtex_dir() / "gan.bib")
        files.append(torchapp_bibtex_dir() / "wgan.bib")
        return files
