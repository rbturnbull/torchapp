from torch import nn
from torchapp.enums import Activation


def test_activations():
    for activation in Activation:
        assert isinstance(activation.module(), nn.Module)
