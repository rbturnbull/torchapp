
from torchapp.examples.iris import IrisApp
from torchapp import TorchApp
from torchapp.cli import collect_arguments

def test_multiple_apps():
    basic_app = TorchApp()
    basic_tune_arguments = collect_arguments(basic_app.tune)
    assert 'hidden_size' not in basic_tune_arguments

    iris = IrisApp()
    iris_tune_arguments = collect_arguments(iris.tune)
    assert 'hidden_size' in iris_tune_arguments
    
    basic_tune_arguments2 = collect_arguments(basic_app.tune)
    assert 'hidden_size' not in basic_tune_arguments2

