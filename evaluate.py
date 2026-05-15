import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluar(modelo, datos, nombre_modelo='modelo'):
    X_test = torch.tensor(datos['X_test'], dtype=torch.float32)
    y_test = torch.tensor(datos['y_test'], dtype=torch.float32)
    scaler_y = datos['scaler_y']

    modelo.eval()
    with torch.no_grad():
        predicciones_scaled = modelo(X_test)
        perdida_test = nn.MSELoss()(predicciones_scaled, y_test)

    y_real     = scaler_y.inverse_transform(y_test.numpy())
    y_predicho = scaler_y.inverse_transform(predicciones_scaled.numpy())

    # Métricas
    mae  = mean_absolute_error(y_real, y_predicho)
    rmse = np.sqrt(mean_squared_error(y_real, y_predicho))
    r2   = r2_score(y_real, y_predicho)

    print(f'\n[{nombre_modelo}]')
    print(f'  MSE  : {perdida_test.item():.4f}')
    print(f'  MAE  : {mae:.4f}')
    print(f'  RMSE : {rmse:.4f}')
    print(f'  R²   : {r2:.4f}')

    # Tabla para el informe
    tabla = pd.DataFrame({
        'Salida esperada (DALY)':   y_real.flatten(),
        'Salida del modelo (DALY)': y_predicho.flatten(),
    })
    tabla['Diferencia']       = (tabla['Salida esperada (DALY)'] - tabla['Salida del modelo (DALY)']).abs()
    tabla['Error relativo %'] = (tabla['Diferencia'] / tabla['Salida esperada (DALY)'].abs() * 100).round(2)
    tabla['Correcto (<10%)']  = tabla['Error relativo %'] < 10

    tabla.round(2).to_csv(f'results/tabla_{nombre_modelo}.csv', index=False)
    print(f'  Correctos (<10% error): {tabla["Correcto (<10%)"].sum()} / {len(tabla)}')
    print(f'  CSV guardado en results/tabla_{nombre_modelo}.csv')

    # Gráfico real vs predicho
    plt.figure(figsize=(8, 6))
    plt.scatter(y_real, y_predicho, alpha=0.5, label='Predicciones')
    plt.plot([y_real.min(), y_real.max()],
             [y_real.min(), y_real.max()], 'r--', label='Predicción perfecta')
    plt.xlabel('DALY real')
    plt.ylabel('DALY predicho')
    plt.title(f'Real vs Predicho — {nombre_modelo}')
    plt.legend()
    plt.savefig(f'results/scatter_{nombre_modelo}.png')
    plt.clf()
    print(f'  Gráfico guardado en results/scatter_{nombre_modelo}.png')

    return tabla, {'mae': mae, 'rmse': rmse, 'r2': r2, 'mse': perdida_test.item()}