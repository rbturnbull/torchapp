import enum
from pathlib import Path
import types
import pandas as pd
import torch
from typing import get_type_hints
from torch import nn
import torchvision.models as models
import lightning as L
from torchmetrics import Metric, Accuracy
import torchapp as ta

from torchapp.examples.vision import VisionApp
from rich.console import Console
console = Console()


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the accuracy of predictions against targets.
    
    Args:
        predictions (torch.Tensor): The predicted values.
        targets (torch.Tensor): The ground truth values.
    
    Returns:
        torch.Tensor: The accuracy as a tensor.
    """
    return (predictions == targets).float().mean()

def torchvision_model_choices() -> list[str]:
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

class PathColReader():
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
    def default_model_name(self):
        return "resnet18"

    @ta.method
    def model(
        self,
        model_name: TorchvisionModelEnum = ta.Param(
            default="",
            help="The name of a model architecture in torchvision.models (https://pytorch.org/vision/stable/models.html). If not given, then it is given by `default_model_name`",
        ),
    ):
        if not model_name:
            model_name = self.default_model_name()

        if not hasattr(models, model_name):
            raise ValueError(f"Model '{model_name}' not recognized.")

        return getattr(models, model_name)
    
    @ta.method
    def data(
        self,
        csv: Path = ta.Param(default=None, help="A CSV with image paths and categories."),
        image_column: str = ta.Param(default="image", help="The name of the column with the image paths."),
        category_column: str = ta.Param(
            default="category", help="The name of the column with the category of the image."
        ),
        base_dir: Path = ta.Param(default=None, help="The base directory for images with relative paths. If not given, then it is relative to the csv directory."),
        validation_column: str = ta.Param(
            default="validation",
            help="The column in the dataset to use for validation. "
            "If the column is not in the dataset, then a validation set will be chosen randomly according to `validation_proportion`.",
        ),
        validation_value: str = ta.Param(
            default=None,
            help="If set, then the value in the `validation_column` must equal this string for the item to be in the validation set. "
        ),
        validation_proportion: float = ta.Param(
            default=0.2,
            help="The proportion of the dataset to keep for validation. Used if `validation_column` is not in the dataset.",
        ),
        batch_size: int = ta.Param(default=16, help="The number of items to use in each batch."),
        width: int = ta.Param(default=224, help="The width to resize all the images to."),
        height: int = ta.Param(default=224, help="The height to resize all the images to."),
        resize_method: str = ta.Param(default="squish", help="The method to resize images."),
        max_lighting:float=0.0,
        max_rotate:float=0.0,
        max_warp:float=0.0,
        max_zoom:float=1.0,
        do_flip:bool=False,
        p_affine:float=0.75,
        p_lighting:float=0.75,
    ) -> L.LightningDataModule:
        df = pd.read_csv(csv)

        base_dir = base_dir or Path(csv).parent
        
        # Create splitter for training/validation images
        if validation_value is not None:
            validation_column_new = f"{validation_column} is {validation_value}"
            df[validation_column_new] = df[validation_column].astype(str) == validation_value
            validation_column = validation_column_new
            
        if validation_column and validation_column in df:
            splitter = ColSplitter(validation_column)
        else:
            splitter = RandomSplitter(validation_proportion)

        batch_transforms = aug_transforms(
            p_lighting=p_lighting,
            p_affine=p_affine,
            max_rotate=max_rotate, 
            do_flip=do_flip, 
            max_lighting=max_lighting, 
            max_zoom=max_zoom,
            max_warp=max_warp,
            pad_mode='zeros',
        )

        datablock = DataBlock(
            blocks=[ImageBlock, CategoryBlock],
            get_x=PathColReader(column_name=image_column, base_dir=base_dir),
            get_y=ColReader(category_column),
            splitter=splitter,
            item_tfms=Resize((height, width), method=resize_method),
            batch_tfms=batch_transforms,
        )

        return datablock.dataloaders(df, bs=batch_size)

    @ta.method
    def metrics(self):
        return [accuracy]

    @ta.method
    def monitor(self):
        return "accuracy"

    @ta.method
    def prediction_dataloader(
        self, 
        learner, 
        items:list[Path] = None, 
        csv: Path = ta.Param(default=None, help="A CSV with image paths."),
        image_column: str = ta.Param(default="image", help="The name of the column with the image paths."),
        base_dir: Path = ta.Param(default="./", help="The base directory for images with relative paths."),
        **kwargs
    ):
        self.items = []
        if isinstance(items, (Path, str)):
            self.items.append(Path(items))
        else:
            try:
                for item in items:
                    item = Path(item)
                    # If the item is a directory then get all images in that directory
                    if item.is_dir():
                        self.items.extend( get_image_files(item) )
                    else:
                        self.items.append(item)
            except:
                raise ValueError(f"Cannot interpret list of items.")

        # Read CSV if available
        if csv is not None:
            df = pd.read_csv(csv)
            for _, row in df.iterrows():
                self.items.append(Path(row[image_column]))

        if not self.items:
            raise ValueError(f"No items found.")

        # Set relative to base dir
        if base_dir:
            base_dir = Path(base_dir)
        
            self.items = [base_dir / item if not item.is_absolute() else item for item in self.items]

        return learner.dls.test_dl(self.items, **kwargs)

    @ta.method
    def output_results(
        self, 
        results, 
        output_csv:Path = ta.Param(None, help="Path to write predictions in CSV format"), 
        verbose:bool = True, 
        **kwargs
    ):
        data = []
        vocab = self.learner_obj.dls.vocab
        for item, scores in zip(self.items, results[0]): 
            probabilities = torch.softmax(torch.as_tensor(scores), dim=-1)           
            prediction = vocab[torch.argmax(probabilities)]
            if verbose:
                console.print(f"'{item}': '{prediction}'")
            data.append( [item,prediction] + probabilities.tolist() )

        df = pd.DataFrame(data, columns=["path","prediction"]+list(vocab))
        if output_csv:
            df.to_csv(output_csv)

        if verbose:
            console.print(df)

        return df


if __name__ == "__main__":
    ImageClassifier.tools()
