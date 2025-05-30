import torch
import torchapp as ta
import numpy as np

MOCK_METRIC = "metric"

categorical_choices = ["abcdefghij", "baby", "c"]


class MockCheckpoint:
    def __init__(self, value):
        self.best_model_score = torch.tensor(value)


class MockTrainer:
    def __init__(self, value):
        self.checkpoint_callback = MockCheckpoint(value=value)


class MockLightningModule:
    pass


class TuningTestApp(ta.TorchApp):
    @ta.method
    def data(self, **kwargs):
        return []

    @ta.method
    def monitor(self):
        return MOCK_METRIC

    @ta.tool
    def train(
        self,
        x: float = ta.Param(default=0.0, tune=True, min=-10.0, max=10.0, help="A real parameter in [-10.0,10.0]."),
        a: int = ta.Param(default=2, tune=True, min=1, max=12, help="An integer parameter in [1,12]."),
        string: str = ta.Param(
            default="baby",
            tune=True,
            tune_choices=categorical_choices,
            help="An string parameter which is either 'abcdefghij', 'baby', or 'c'.",
        ),
        switch: bool = ta.Param(default=False, tune=True, help="A bool that can be true or false."),
        activation: ta.Activation = ta.Param(default=ta.Activation.ReLU, tune=True, help="An activation function to use."),
        **kwargs,
    ):
        assert isinstance(x, float)
        assert -10.0 <= x <= 10.0
        assert isinstance(a, int) or isinstance(a, np.integer)
        assert 1 <= a <= 12
        assert isinstance(string, str)
        assert string in categorical_choices
        assert switch in [True, False]
        assert activation in ta.Activation.default_tune_choices()

        c = len(string)

        value = c * switch - a * (x - 2.0) ** 2 - a + 1
        
        return MockLightningModule(), MockTrainer(value=value)
