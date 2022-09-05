import pytest
from torchapp.testing import TorchAppTestCase
from torchapp.vision import VisionApp


class TestVisionApp(TorchAppTestCase):
    app_class = VisionApp

    def test_model_incorrect(self):
        app = self.get_app()
        with pytest.raises(ValueError):
            app.model(model_name="resnet1000")
