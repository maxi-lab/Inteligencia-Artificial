import os
from data_loader import cargar_datos
from model import MLP_Simple, MLP_Grande
from train import entrenar
from evaluate import evaluar
from predict import predecir

os.makedirs('results', exist_ok=True) # Asegura que la carpeta de resultados exista

print("Cargando datos...")
datos = cargar_datos()

# Entrenar ambos modelos
modelo_simple, _ = entrenar(MLP_Simple(), datos)
modelo_grande, _ = entrenar(MLP_Grande(), datos)

# Evaluar y generar tablas
tabla_simple, metricas_simple = evaluar(modelo_simple, datos, 'MLP_Simple')
tabla_grande, metricas_grande = evaluar(modelo_grande, datos, 'MLP_Grande')

# Comparación final para el informe
print("\n=== COMPARACIÓN FINAL ===")
print(f"{'Métrica':<10} {'MLP_Simple':>12} {'MLP_Grande':>12}")
print("-" * 36)
for m in ['mse', 'mae', 'rmse', 'r2']:
    print(f"{m.upper():<10} {metricas_simple[m]:>12.4f} {metricas_grande[m]:>12.4f}")

# Predicción con dato nuevo
print("\n--- Predicción MLP_Simple ---")
predecir(modelo_simple, datos['scaler_X'], datos['scaler_y'])

print("\n--- Predicción MLP_Grande ---")
predecir(modelo_grande, datos['scaler_X'], datos['scaler_y'])