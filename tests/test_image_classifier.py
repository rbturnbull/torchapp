import pytest
from torchapp.testing import TorchAppTestCase
from torchapp.examples.image_classifier import ImageClassifier, get_image_paths


class TestImageClassifier(TorchAppTestCase):
    app_class = ImageClassifier

    def test_model_incorrect(self):
        app = self.get_app()
        with pytest.raises(ValueError):
            app.model(model_name="resnet1000")


    def test_get_image_paths(self):
        expected_dir = self.get_expected_dir()
        paths = get_image_paths(expected_dir)
        assert len(paths) == 3
        names = [p.name for p in paths]
        assert 'vanjari-banner.jpg' in names
        assert 'bloodhound.png' in names