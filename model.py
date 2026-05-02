import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.red = nn.Sequential(
            nn.Linear(5, 64),    # 5 entradas
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)     # 1 salida: DALYs
        )

    def forward(self, x):
        return self.red(x)