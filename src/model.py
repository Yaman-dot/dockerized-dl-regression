import torch
import torch.nn as nn

#device = "cuda" if torch.cuda.is_available() else "cpu"


class RegressionModel(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(in_features=in_features, out_features=64),
                                    nn.ReLU(),
                                    nn.Linear(64, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 32),
                                    nn.ReLU(),
                                    nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.layers(x)
