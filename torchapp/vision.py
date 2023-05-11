import enum
import types
from pathlib import Path
from typing import get_type_hints, List
import torchvision.models as models
from torch import nn
from fastai.vision.learner import cnn_learner, unet_learner

from .apps import TorchApp
from .params import Param


def torchvision_model_choices() -> List[str]:
    """
    Returns a list of function names in torchvision.models which can produce torch modules.

    For more information see: https://pytorch.org/vision/stable/models.html
    """
    model_choices = [""]  # Allow for blank option
    for item in dir(models):
        obj = getattr(models, item)

        # Only accept functions
        if isinstance(obj, types.FunctionType):

            # Only accept if the return value is a pytorch module
            hints = get_type_hints(obj)
            return_value = hints.get("return", "")
            try:
                mro = return_value.mro()
                if nn.Module in mro:
                    model_choices.append(item)
            except TypeError:
                pass

    return model_choices

TorchvisionModelEnum = enum.Enum(
    "TorchvisionModelName",
    {model_name if model_name else "default": model_name for model_name in torchvision_model_choices()},
)


class VisionApp(TorchApp):
    """
    A TorchApp which uses a model from torchvision.

    The default base torchvision model is resnet18.
    """

    def default_model_name(self):
        return "resnet18"

    def model(
        self,
        model_name: TorchvisionModelEnum = Param(
            default="",
            help="The name of a model architecture in torchvision.models (https://pytorch.org/vision/stable/models.html). If not given, then it is given by `default_model_name`",
        ),
    ):
        if not model_name:
            model_name = self.default_model_name()

        if not hasattr(models, model_name):
            raise ValueError(f"Model '{model_name}' not recognized.")

        return getattr(models, model_name)

    def build_learner_func(self):
        return cnn_learner

    def learner_kwargs(
        self,
        output_dir: Path = Param("./outputs", help="The location of the output directory."),
        pretrained: bool = Param(default=True, help="Whether or not to use the pretrained weights."),
        weight_decay: float = Param(
            None, help="The amount of weight decay. If None then it uses the default amount of weight decay in fastai."
        ),
        **kwargs,
    ):
        kwargs = super().learner_kwargs(output_dir=output_dir, weight_decay=weight_decay, **kwargs)
        kwargs['pretrained'] = pretrained
        self.fine_tune = pretrained
        return kwargs


class UNetApp(VisionApp):
    """
    A TorchApp which uses a base model from torchvision which is modified.

    Useful for image segmentation, super-resolution or colorization.
    The default base torchvision model is resnet18.

    For more information see:
    Olaf Ronneberger, Philipp Fischer, Thomas Brox,
    U-Net: Convolutional Networks for Biomedical Image Segmentation,
    https://arxiv.org/abs/1505.04597
    https://github.com/fastai/fastbook/blob/master/15_arch_details.ipynb
    """

    def build_learner_func(self):
        """
        Returns unet_learner

        For more information see: https://docs.fast.ai/vision.learner.html#unet_learner
        """
        return unet_learner
