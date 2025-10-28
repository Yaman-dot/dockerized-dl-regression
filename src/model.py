import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"


class RegressionModel(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(in_features=in_features, out_features=32),
                                    nn.BatchNorm1d(32),
                                    nn.ReLU(),
                                    nn.Linear(in_features=32, out_features=16),
                                    nn.BatchNorm1d(16),
                                    nn.ReLU(),
                                    nn.Linear(in_features=16, out_features=1))
    def forward(self, x):
        return self.layers(x)
