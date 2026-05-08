import torch
import pandas as pd

def predecir(modelo, scaler_X, scaler_y):
    modelo.eval()

    dato_nuevo = pd.DataFrame(
        [[0.28, 4.80, 6.30, 0.70, 0.55]],
        columns=['esquizofrenia', 'depresion', 'ansiedad', 'bipolaridad', 'alimentarios']
    )

    dato_scaled = scaler_X.transform(dato_nuevo)
    dato_tensor = torch.tensor(dato_scaled, dtype=torch.float32)

    with torch.no_grad():
        prediccion_scaled = modelo(dato_tensor)

    daly_predicho = scaler_y.inverse_transform(prediccion_scaled.numpy())

    print("\n--- Predicción con dato nuevo ---")
    print(f"Tasas ingresadas (%):\n{dato_nuevo.iloc[0].to_string()}")
    print(f"\n-> DALY predicho: {daly_predicho[0][0]:.2f} por cada 100k habitantes")

    return daly_predicho[0][0]