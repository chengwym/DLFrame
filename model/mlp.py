import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(
        self,
        layers_dim,
        hidden_layers_nums,
    ):
        super(MLP, self).__init__()
        layers = []
        for i in range(hidden_layers_nums-1):
            layers.append(nn.Linear(layers_dim[i], layers_dim[i+1]))
            layers.append(nn.GELU())
        layers.append(nn.Linear(layers_dim[-2], layers_dim[-1]))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    