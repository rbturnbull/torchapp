from torch import nn
import torchapp as ta
from torchapp.examples.logistic_regression import LogisticRegressionApp

class MultilayerPerceptronApp(LogisticRegressionApp):
    def model(
        self,
        hidden_layers:int = ta.Param(default=1, tune_min=0, tune_max=4, tune=True, help="The number of hidden layers"),
        hidden_size:int = ta.Param(default=64, tune_min=8, tune_max=256, tune=True, tune_log=True, help="The size of the hidden layers"),
        hidden_bias:bool = ta.Param(default=True, tune=True, help="Whether or not the hidden layers have bias"),
    ) -> nn.Module:
        in_features = 1

        layer_list = []
        for _ in range(hidden_layers):
            layer_list.append(nn.Linear(in_features=in_features, out_features=hidden_size, bias=hidden_bias))
            in_features = hidden_size

        # final layer
        layer_list.append(nn.Linear(in_features=in_features, out_features=1, bias=True))
        
        return nn.Sequential(*layer_list)


if __name__ == "__main__":
    MultilayerPerceptronApp.main()
