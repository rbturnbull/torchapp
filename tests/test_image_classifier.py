from torchapp.testing import TorchAppTestCase
from torchapp.examples.image_classifier import ImageClassifier


class TestImageClassifier(TorchAppTestCase):
    app_class = ImageClassifier
