import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def entrenar(modelo, datos, epocas=500, lr=0.01, graficar=True):
    X_train = torch.tensor(datos['X_train'], dtype=torch.float32)
    y_train = torch.tensor(datos['y_train'], dtype=torch.float32)

    criterio = nn.MSELoss()
    optimizador = optim.Adam(modelo.parameters(), lr=lr)

    historial = []

    print(f"Entrenando {modelo.__class__.__name__}...")
    for epoca in range(epocas):
        predicciones = modelo(X_train)
        perdida = criterio(predicciones, y_train)

        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()

        historial.append(perdida.item())

        if (epoca + 1) % 50 == 0:
            print(f'  Época [{epoca+1}/{epocas}] - Pérdida: {perdida.item():.4f}')

    if graficar:
        plt.plot(historial, label=modelo.__class__.__name__)
        plt.title('Curva de aprendizaje')
        plt.xlabel('Época')
        plt.ylabel('Pérdida (MSE)')
        plt.legend()
        plt.savefig(f'results/curva_{modelo.__class__.__name__}.png')
        plt.clf()

    return modelo, historial