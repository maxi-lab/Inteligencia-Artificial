import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_loader import ENFERMEDADES_SALIDA


def evaluar(modelo, datos, nombre_modelo='modelo'):
    X_test = torch.tensor(datos['X_test'], dtype=torch.float32)
    y_test = torch.tensor(datos['y_test'], dtype=torch.float32)
    scaler_y = datos['scaler_y']

    modelo.eval()
    with torch.no_grad():
        predicciones_scaled = modelo(X_test)

        # MSELoss sobre la matriz completa (N×5): un único valor de pérdida global
        perdida_test = nn.MSELoss()(predicciones_scaled, y_test)

    # Desnormalizar: volvemos a la escala original de DALYs por cada enfermedad
    # scaler_y recuerda la media y desvío de cada columna, así que
    # inverse_transform reconstruye correctamente cada una de las 5 salidas.
    y_real     = scaler_y.inverse_transform(y_test.numpy())           # (N, 5)
    y_predicho = scaler_y.inverse_transform(predicciones_scaled.numpy())  # (N, 5)

    print(f'\n{"="*60}')
    print(f'  EVALUACIÓN: {nombre_modelo}')
    print(f'{"="*60}')
    print(f'  MSE global (escala normalizada): {perdida_test.item():.4f}')
    print()

    metricas_globales = {'mse': perdida_test.item()}
    metricas_por_enfermedad = {}

    # ── Métricas individuales por enfermedad ────────────────────────────────
    # CAMBIO: antes había una sola métrica porque había una sola salida.
    # Ahora iteramos sobre las 5 columnas y calculamos MAE, RMSE y R² para cada una.
    for i, nombre_col in enumerate(ENFERMEDADES_SALIDA):
        yr = y_real[:, i]
        yp = y_predicho[:, i]

        mae  = mean_absolute_error(yr, yp)
        rmse = np.sqrt(mean_squared_error(yr, yp))
        r2   = r2_score(yr, yp)

        metricas_por_enfermedad[nombre_col] = {'mae': mae, 'rmse': rmse, 'r2': r2}

        print(f'  [{nombre_col}]')
        print(f'    MAE : {mae:.4f}  |  RMSE : {rmse:.4f}  |  R² : {r2:.4f}')

    # ── Tabla para el informe (con las 5 enfermedades) ──────────────────────
    # Construimos la tabla de comparación esperado vs predicho para cada trastorno.
    tabla_dict = {}
    for i, nombre_col in enumerate(ENFERMEDADES_SALIDA):
        label = nombre_col.replace('daly_', '').capitalize()
        tabla_dict[f'Esperado_{label}']  = y_real[:, i].round(2)
        tabla_dict[f'Predicho_{label}']  = y_predicho[:, i].round(2)
        tabla_dict[f'ErrRel%_{label}']   = (
            np.abs(y_real[:, i] - y_predicho[:, i])
            / np.abs(y_real[:, i]) * 100
        ).round(2)

    tabla = pd.DataFrame(tabla_dict)

    # Columna "Correcto": todas las enfermedades con error relativo < 10%
    col_errores = [c for c in tabla.columns if c.startswith('ErrRel%')]
    tabla['Correcto_todas'] = tabla[col_errores].lt(10).all(axis=1)

    tabla.to_csv(f'results/tabla_{nombre_modelo}.csv', index=False)
    correctos = tabla['Correcto_todas'].sum()
    print(f'\n  Correctos (<10% error en TODAS las enfermedades): {correctos} / {len(tabla)}')
    print(f'  CSV guardado en results/tabla_{nombre_modelo}.csv')

    # ── Gráfico: Real vs Predicho por enfermedad ────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, nombre_col in enumerate(ENFERMEDADES_SALIDA):
        ax = axes[i]
        yr = y_real[:, i]
        yp = y_predicho[:, i]
        ax.scatter(yr, yp, alpha=0.4, s=10)
        ax.plot([yr.min(), yr.max()], [yr.min(), yr.max()], 'r--', lw=1.5)
        ax.set_xlabel('DALY real')
        ax.set_ylabel('DALY predicho')
        label = nombre_col.replace('daly_', '').capitalize()
        r2 = metricas_por_enfermedad[nombre_col]['r2']
        ax.set_title(f'{label}  (R²={r2:.3f})')

    axes[5].axis('off')  # el 6.º panel queda vacío (tenemos 5 enfermedades)
    fig.suptitle(f'Real vs Predicho — {nombre_modelo}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'results/scatter_{nombre_modelo}.png', dpi=100)
    plt.clf()
    print(f'  Gráfico guardado en results/scatter_{nombre_modelo}.png')

    return tabla, metricas_por_enfermedad, metricas_globales
