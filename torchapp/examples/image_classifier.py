from PIL import Image
import enum
from pathlib import Path
import types
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from typing import get_type_hints
from torch import nn
import torchvision.models as models
from torchvision import transforms
import lightning as L
from dataclasses import dataclass
import torchapp as ta
from torchapp.metrics import accuracy

from rich.console import Console
console = Console()

def replace_imagenet_classification_layer(model, out_features) -> bool:
    """
    Recursively replaces the last classification layer in a model if it outputs 1000 classes.
    Supports nn.Linear and nn.Conv2d used in torchvision models like ResNet and SqueezeNet.
    """
    for name, module in reversed(list(model.named_children())):
        # Handle Linear classifier (e.g., ResNet, VGG)
        if isinstance(module, nn.Linear) and module.out_features == 1000:
            in_features = module.in_features
            setattr(model, name, nn.Linear(in_features, out_features))
            return True

        # Handle 1x1 Conv2d classifier (e.g., SqueezeNet, MobileNet)
        elif isinstance(module, nn.Conv2d) and module.out_channels == 1000 and module.kernel_size == (1, 1):
            in_channels = module.in_channels
            setattr(model, name, nn.Conv2d(in_channels, out_features, kernel_size=1))
            return True

        # Recurse into submodules
        elif replace_imagenet_classification_layer(module, out_features):
            return True

    return False  # no classification layer found


def get_image_paths(directory:Path|str) -> list[Path]:
    directory = Path(directory)
    extensions = ["jpg", "jpeg", "png", "tif", "tiff"]
    paths = []
    for extension in extensions:
        paths += directory.glob(f"*.{extension.lower()}")
        paths += directory.glob(f"*.{extension.upper()}")
        
    return paths


@dataclass
class ImageItem():
    path:Path
    height:int = 224
    width:int = 224

    def image_as_tensor(self):
        # Standard ImageNet preprocessing for pretrained torchvision models
        transform = transforms.Compose([
            transforms.Resize((self.height, self.width)),  # Resize directly to desired size
            transforms.ToTensor(),       # converts to [0,1] and puts channel first
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]    # ImageNet std
            )
        ])

        image = Image.open(self.path).convert("RGB")
        tensor = transform(image)  # shape: (3, height, width)
        return tensor


@dataclass(kw_only=True)
class ImageTrainingItem(ImageItem):
    category_id:int


@dataclass(kw_only=True)
class ImageClassifierDataset(Dataset):
    items:list[ImageTrainingItem]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image_tensor = item.image_as_tensor()
        if isinstance(item, ImageTrainingItem):
            return image_tensor, item.category_id
        return image_tensor


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


class ImageClassifier(ta.TorchApp):
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

        model = getattr(models, model_name)()

        # configure last layer
        n_categories = len(self.target_names)
        result = replace_imagenet_classification_layer(model, n_categories)
        assert result, f"Model '{model_name}' does not have a classification layer to replace. Please choose another model."
        return model
    
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
        # max_lighting:float=0.0,
        # max_rotate:float=0.0,
        # max_warp:float=0.0,
        # max_zoom:float=1.0,
        # do_flip:bool=False,
        # p_affine:float=0.75,
        # p_lighting:float=0.75,
    ) -> L.LightningDataModule:
        df = pd.read_csv(csv)

        base_dir = base_dir or Path(csv).parent
        
        # Create splitter for training/validation images
        if validation_value is not None:
            validation_column_new = f"{validation_column} is {validation_value}"
            df[validation_column_new] = df[validation_column].astype(str) == validation_value
            validation_column = validation_column_new

        if validation_column not in df:
            # randomly assign validation set based on validation_proportion
            df[validation_column] = df.sample(frac=validation_proportion, random_state=42).index.isin(df.index)

        assert image_column in df, f"Image column '{image_column}' not found in the CSV. Columns available {df.columns.tolist()}"
        assert category_column in df, f"Category column '{category_column}' not found in the CSV. Columns available {df.columns.tolist()}"

        training_data = []
        validation_data = []

        df['category_id'], self.target_names = pd.factorize(df[category_column])

        self.width = width
        self.height = height

        for _, row in df.iterrows():
            image_path = Path(row[image_column])
            if not image_path.is_absolute():
                image_path = base_dir/image_path
        
            item = ImageTrainingItem(path=image_path, category_id=row['category_id'], width=width, height=height)
            dataset = validation_data if row[validation_column] else training_data
            dataset.append(item)

        training_dataset = ImageClassifierDataset(items=training_data)
        validation_dataset = ImageClassifierDataset(items=validation_data)

        data_module = L.LightningDataModule()
        data_module.train_dataloader = lambda: DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
        data_module.val_dataloader = lambda: DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
        return data_module

    @ta.method
    def metrics(self):
        return [accuracy]

    @ta.method
    def monitor(self):
        return "accuracy"
    
    @ta.method
    def extra_hyperparameters(self):
        return dict(
            target_names=self.target_names,
            width=self.width,
            height=self.height,
        )

    @ta.method
    def prediction_dataloader(
        self, 
        module, 
        items:list[Path] = None, 
        batch_size:int = 16,
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
                        self.items.extend( get_image_paths(item) )
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

        width = module.hparams.width
        height = module.hparams.height

        dataset = ImageClassifierDataset(items=[ImageItem(path=path, width=width, height=height) for path in self.items])
        self.target_names = module.hparams.target_names
        return DataLoader(dataset, batch_size=batch_size)

    @ta.method
    def output_results(
        self, 
        results, 
        output_csv:Path = ta.Param(None, help="Path to write predictions in CSV format"), 
        verbose:bool = True, 
        **kwargs
    ):
        data = []
        for item, scores in zip(self.items, results[0]): 
            probabilities = torch.softmax(torch.as_tensor(scores), dim=-1)           
            prediction = self.target_names[torch.argmax(probabilities)]
            if verbose:
                console.print(f"'{item}': '{prediction}'")
            data.append( [item,prediction] + probabilities.tolist() )

        df = pd.DataFrame(data, columns=["path","prediction"]+list(self.target_names))
        if output_csv:
            df.to_csv(output_csv)

        if verbose:
            console.print(df)

        return df

    @ta.method
    def loss_function(self):
        return nn.CrossEntropyLoss()


if __name__ == "__main__":
    ImageClassifier.tools()
