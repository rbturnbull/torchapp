from pathlib import Path
from fastai.data.block import DataBlock, CategoryBlock
from fastai.data.transforms import ColReader, RandomSplitter, DisplayedTransform, ColSplitter
from fastai.metrics import accuracy
from fastai.vision.data import ImageBlock
from fastai.vision.augment import Resize, ResizeMethod
import pandas as pd
import torchapp as ta

from torchapp.vision import VisionApp


class PathColReader(DisplayedTransform):
    def __init__(self, column_name: str, base_dir: Path):
        self.column_name = column_name
        self.base_dir = base_dir

    def __call__(self, row, **kwargs):
        path = Path(row[self.column_name])
        if not path.is_absolute():
            path = self.base_dir / path
        return path


class ImageClassifier(VisionApp):
    """
    A TorchApp for classifying images.

    For training, it expects a CSV with image paths and categories.
    """

    def dataloaders(
        self,
        csv: Path = ta.Param(default=None, help="A CSV with image paths and categories."),
        image_column: str = ta.Param(default="image", help="The name of the column with the image paths."),
        category_column: str = ta.Param(
            default="category", help="The name of the column with the category of the image."
        ),
        base_dir: Path = ta.Param(default="./", help="The base directory for images with relative paths."),
        validation_column: str = ta.Param(
            default="validation",
            help="The column in the dataset to use for validation. "
            "If the column is not in the dataset, then a validation set will be chosen randomly according to `validation_proportion`.",
        ),
        validation_proportion: float = ta.Param(
            default=0.2,
            help="The proportion of the dataset to keep for validation. Used if `validation_column` is not in the dataset.",
        ),
        batch_size: int = ta.Param(default=16, help="The number of items to use in each batch."),
        width: int = ta.Param(default=224, help="The width to resize all the images to."),
        height: int = ta.Param(default=224, help="The height to resize all the images to."),
        resize_method: str = ta.Param(default="squish", help="The method to resize images."),
    ):
        df = pd.read_csv(csv)

        # Create splitter for training/validation images
        if validation_column and validation_column in df:
            splitter = ColSplitter(validation_column)
        else:
            splitter = RandomSplitter(validation_proportion)

        datablock = DataBlock(
            blocks=[ImageBlock, CategoryBlock],
            get_x=PathColReader(column_name=image_column, base_dir=base_dir),
            get_y=ColReader(category_column),
            splitter=splitter,
            item_tfms=Resize((height, width), method=resize_method),
        )

        return datablock.dataloaders(df, bs=batch_size)

    def metrics(self):
        return [accuracy]

    def monitor(self):
        return "accuracy"


if __name__ == "__main__":
    ImageClassifier.main()
