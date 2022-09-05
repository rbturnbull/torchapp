from torchapp.testing import TorchAppTestCase
from torchapp.examples.iris import IrisApp


class TestIris(TorchAppTestCase):
    app_class = IrisApp
