from unittest.mock import patch
from torchapp.examples.iris import IrisApp
from torchapp.testing import TorchAppTestCase


def get_test_case():
    class DummyApp(IrisApp):
        pass

    class TestDummyApp(TorchAppTestCase):
        app_class = DummyApp

    return TestDummyApp()


@patch('builtins.input', lambda _: 'y')
def test_model_interactive():
    test_case = get_test_case()
    test_case.test_model(interactive=True)
