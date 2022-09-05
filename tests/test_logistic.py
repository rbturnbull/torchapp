from torchapp.testing import TorchAppTestCase
from torchapp.examples.logistic_regression import LogisticRegressionApp


class TestLogisticRegressionApp(TorchAppTestCase):
    app_class = LogisticRegressionApp
