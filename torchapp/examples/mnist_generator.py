#!/usr/bin/env python3
from functools import partial
from pathlib import Path
import torch
from torch import nn
from fastai.data.transforms import get_image_files, Normalize
from fastai.data.block import DataBlock, TransformBlock
from fastai.vision.data import ImageBlock
from fastai.vision.core import PILImageBW
from fastai.vision.augment import Resize, ResizeMethod
from fastai.vision.gan import basic_generator, basic_critic, generate_noise
from fastai.layers import NormType
from fastai.data.external import URLs, untar_data
import torchapp as ta
from torchapp.gan import GANApp
from enum import Enum


class MNISTDataset(Enum):
    MNIST_TINY = "MNIST_TINY"
    MNIST_SAMPLE = "MNIST_SAMPLE"
    MNIST = "MNIST"


class MNISTGenerator(GANApp):
    """
    An app to generate images from the MNIST dataset.

    Designed to demonstrate how to build a GAN App.

    Based on https://github.com/fastai/fastai/blob/master/dev_nbs/course/lesson7-wgan.ipynb
    """
    def dataloaders(
        self,
        mnist_dataset:MNISTDataset = ta.Param(MNISTDataset.MNIST_TINY, case_sensitive=False),
        image_size: int = ta.Param(default=64, help="The size of the images to generate"),
        noise_size:int = ta.Param(100, help="Size of the input noise vector for the generator"),
        batch_size: int = ta.Param(default=32, tune_min=8, tune_max=128, tune_log=True, tune=True),
    ):
        # Download dataset
        if isinstance(mnist_dataset, MNISTDataset):
            mnist_dataset = str(mnist_dataset.value)

        assert hasattr(URLs, mnist_dataset)
        dataset_url = getattr(URLs, mnist_dataset)
        path = untar_data(dataset_url)

        self.image_size = image_size
        self.noise_size = noise_size
        dblock = DataBlock(
            blocks = (TransformBlock, ImageBlock(cls=PILImageBW)),
            get_x = partial(generate_noise, size=noise_size),
            get_items = get_image_files,
            item_tfms=Resize(image_size, method=ResizeMethod.Crop),
            batch_tfms = Normalize.from_stats(torch.tensor([0.5]), torch.tensor([0.5])),
        )

        return dblock.dataloaders(path, path=path, bs=batch_size)

    def generator(
        self,
        generator_extra_layers:int = ta.Param(default=1, help="Number of extra hidden layers in the generator"),
        generator_features:int = ta.Param(default=64, help="Number of features used in the generator"),
    ) -> nn.Module:
        return basic_generator(
            out_size=self.image_size, 
            n_channels=1, 
            in_sz=self.noise_size,
            n_extra_layers=generator_extra_layers,
            n_features=generator_features,
        )

    def critic(
        self,
        critic_extra_layers:int = ta.Param(default=1, help="Number of extra hidden layers in the critic"),
        critic_features:int = ta.Param(default=64, help="Number of features used in the critic"),
        norm_type:NormType = ta.Param(NormType.Batch, help="Type of normalization to use in the critic"),
    ) -> nn.Module:
        return basic_critic(
            in_size=self.image_size, 
            n_channels=1, 
            n_extra_layers=critic_extra_layers, 
            n_features=critic_features,
            norm_type=norm_type,
            act_cls=partial(nn.LeakyReLU, negative_slope=0.2)
        )

    def get_bibtex_files(self):
        files = super().get_bibtex_files()
        files.append(Path(__file__).parent / "mnist.bib")
        return files

    def monitor(self):
        return False


if __name__ == "__main__":
    MNISTGenerator.main()
