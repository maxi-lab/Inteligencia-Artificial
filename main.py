import os

from data_loader import cargar_datos, ENFERMEDADES_SALIDA


from model import MLP_Simple, MLP_Grande
from train import entrenar
from evaluate import evaluar
from predict import predecir

os.makedirs('results', exist_ok=True)

print("Cargando datos...")
datos = cargar_datos()
print(f"Patrones de entrenamiento: {len(datos['X_train'])}")
print(f"Patrones de test:          {len(datos['X_test'])}")
print(f"Salidas del modelo:        {len(ENFERMEDADES_SALIDA)} enfermedades\n")

# ── Entrenar ambos modelos ──────────────────────────────────────────────────
modelo_simple, _ = entrenar(MLP_Simple(), datos)
modelo_grande, _ = entrenar(MLP_Grande(), datos)

# ── Evaluar y generar tablas/gráficos ───────────────────────────────────────
tabla_simple, metricas_simple, globales_simple = evaluar(modelo_simple, datos, 'MLP_Simple')
tabla_grande, metricas_grande, globales_grande = evaluar(modelo_grande, datos, 'MLP_Grande')

# ── Comparación final por enfermedad ───────────────────────────────────────
print("\n" + "="*70)
print("COMPARACIÓN FINAL: MLP_Simple vs MLP_Grande")
print("="*70)

enf_labels = [e.replace('daly_', '').capitalize() for e in ENFERMEDADES_SALIDA]
header = f"{'Enfermedad':<20} {'MAE_S':>8} {'MAE_G':>8} {'R²_S':>8} {'R²_G':>8}"
print(header)
print("-" * 70)

for nombre_col, label in zip(ENFERMEDADES_SALIDA, enf_labels):
    ms = metricas_simple[nombre_col]
    mg = metricas_grande[nombre_col]
    print(
        f"{label:<20} "
        f"{ms['mae']:>8.4f} {mg['mae']:>8.4f} "
        f"{ms['r2']:>8.4f} {mg['r2']:>8.4f}"
    )

# ── Predicción con dato nuevo ───────────────────────────────────────────────
print("\n\n--- Predicción MLP_Simple ---")
predecir(modelo_simple, datos['scaler_X'], datos['scaler_y'])

print("\n--- Predicción MLP_Grande ---")
predecir(modelo_grande, datos['scaler_X'], datos['scaler_y'])
