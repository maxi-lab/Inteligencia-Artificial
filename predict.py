import torch
import pandas as pd
from data_loader import ENFERMEDADES_SALIDA

def predecir(modelo, scaler_X, scaler_y):
    """
    Realiza una predicción con un dato de ejemplo y muestra los DALYs
    estimados para las 5 enfermedades.

    CAMBIO respecto a la versión original:
    - Antes: la salida era un único valor (daly_depresion).
    - Ahora: la salida es un vector de 5 valores, uno por cada trastorno mental.
      inverse_transform desnormaliza cada uno usando la escala correcta.
    """
    modelo.eval()

    # Dato de ejemplo: prevalencias (%) de las 5 enfermedades de entrada
    dato_nuevo = pd.DataFrame(
        [[0.28, 4.80, 6.30, 0.70, 0.55]],
        columns=['esquizofrenia', 'depresion', 'ansiedad', 'bipolaridad', 'alimentarios']
    )

    dato_scaled = scaler_X.transform(dato_nuevo)
    dato_tensor = torch.tensor(dato_scaled, dtype=torch.float32)

    with torch.no_grad():
        prediccion_scaled = modelo(dato_tensor)   # shape: (1, 5)

    # Desnormalizamos el vector de 5 salidas
    dalys_predichos = scaler_y.inverse_transform(prediccion_scaled.numpy())  # (1, 5)

    print("\n--- Predicción con dato nuevo ---")
    print(f"Tasas de prevalencia ingresadas (%):")
    print(dato_nuevo.iloc[0].to_string())
    print(f"\nDALYs predichos por trastorno (por 100k habitantes):")

    # Mostramos un valor por cada enfermedad
    for nombre_col, valor in zip(ENFERMEDADES_SALIDA, dalys_predichos[0]):
        label = nombre_col.replace('daly_', '').capitalize()
        print(f"  {label:<20}: {valor:.2f}")

    return dalys_predichos[0]
