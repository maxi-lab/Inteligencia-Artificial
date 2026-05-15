import torch.nn as nn

# CAMBIO: ambos modelos ahora tienen 5 neuronas de salida en lugar de 1.
# Antes: nn.Linear(8, 1)  →  Ahora: nn.Linear(8, 5) / nn.Linear(32, 5)
# Esto permite que la red prediga simultáneamente los DALYs de las
# 5 enfermedades: depresión, esquizofrenia, bipolaridad, alimentarios, ansiedad.

class MLP_Simple(nn.Module):
    """
    Arquitectura original adaptada: 5 entradas → 16 → 8 → 5 salidas.
    Una salida por cada trastorno mental (DALYs).
    """
    def __init__(self):
        super(MLP_Simple, self).__init__()
        self.capa1  = nn.Linear(5, 16)
        self.capa2  = nn.Linear(16, 8)
        self.salida = nn.Linear(8, 5)   # ← era Linear(8, 1)
        self.relu   = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.capa1(x))
        x = self.relu(self.capa2(x))
        return self.salida(x)


class MLP_Grande(nn.Module):
    """
    Arquitectura alternativa adaptada: 5 entradas → 64 → 32 → 5 salidas.
    Una salida por cada trastorno mental (DALYs).
    """
    def __init__(self):
        super(MLP_Grande, self).__init__()
        self.red = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)   # ← era Linear(32, 1)
        )

    def forward(self, x):
        return self.red(x)
