import torch.nn as nn

class MLP_Simple(nn.Module):
    """Arquitectura original (5→16→8→1)"""
    def __init__(self):
        super(MLP_Simple, self).__init__()
        self.capa1 = nn.Linear(5, 16)
        self.capa2 = nn.Linear(16, 8)
        self.salida = nn.Linear(8, 1) # codificacion de enfermedades 
        self.relu = nn.ReLU()#softmax para clasificacion, relu para regresion

    def forward(self, x):
        x = self.relu(self.capa1(x))
        x = self.relu(self.capa2(x))
        return self.salida(x)


class MLP_Grande(nn.Module):
    """Arquitectura alternativa (5→64→32→1)"""
    def __init__(self):
        super(MLP_Grande, self).__init__()
        self.red = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.red(x)