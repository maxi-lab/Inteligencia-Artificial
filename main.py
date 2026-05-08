import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

print("Cargando y fusionando datasets...")


# Cargar los CSV

"""
- Primer archivo: Prevalencias de enfermedades (porcentaje de enfermos de cada enfermedad)

1-mental-illnesses-prevalence.csv:
- Entity: País o región.
- Code: Código del país.
- Year: Año del registro.
- Schizophrenia disorders: Participación de la población con esquizofrenia.
- Depressive disorders: Participación de la población con trastornos depresivos.
- Anxiety disorders: Participación de la población con trastornos de ansiedad.
- Bipolar disorders: Participación de la población con trastornos bipolares.
- Eating disorders: Participación de la población con trastornos alimenticios.

- Segundo archivo: valor DALY de cada una de esas mismas enfermedades

2-burden-disease-from-each-mental-illness.csv
- Entity: País o región.
- Code: Código del país.
- Year: Año del registro.
- DALYs - Depressive disorders: Tasa de DALY por trastornos depresivos.
- DALYs - Schizophrenia: Tasa de DALY por esquizofrenia.
- DALYs - Bipolar disorder: Tasa de DALY por trastorno bipolar.
- DALYs - Eating disorders: Tasa de DALY por trastornos alimenticios.
- DALYs - Anxiety disorders: Tasa de DALY por trastornos de ansiedad.
"""

df_prev = pd.read_csv('datasets/1-mental-illnesses-prevalence.csv')
df_daly = pd.read_csv('datasets/2-burden-disease-from-each-mental-illness.csv') 


df_prev.rename(columns={
    'Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized': 'esquizofrenia',
    'Depressive disorders (share of population) - Sex: Both - Age: Age-standardized': 'depresion',
    'Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized': 'ansiedad',
    'Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized': 'bipolaridad',
    'Eating disorders (share of population) - Sex: Both - Age: Age-standardized': 'alimentarios'
}, inplace=True)

# 3. Renombrar el DALY específico que pide el TP
df_daly.rename(columns={
    'DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Depressive disorders': 'daly_objetivo'
}, inplace=True)

df_daly = df_daly[['Entity', 'Code', 'Year', 'daly_objetivo']]

# 4. Fusionar (Merge) ambos datasets por País y Año
df = pd.merge(df_prev, df_daly, on=['Entity', 'Code', 'Year'], how='inner')

# Eliminar filas con valores nulos (NaN) para que la red neuronal no falle
df = df.dropna()

# 5. Separar Entradas (X) y Salida (y)
X_pandas = df[['esquizofrenia', 'depresion', 'ansiedad', 'bipolaridad', 'alimentarios']]
y_pandas = df['daly_objetivo']

# 6. Normalizar los datos
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X_pandas)
y_scaled = scaler_y.fit_transform(y_pandas.values.reshape(-1, 1))

# 7. Separar en conjunto de Entrenamiento (80%) y Prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# ==========================================
# PASO 2: CONVERSIÓN A TENSORES DE PYTORCH
# ==========================================
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# ==========================================
# PASO 3: DEFINICIÓN DE LA RED NEURONAL (MLP)
# ==========================================
class PrediccionDALY_MLP(nn.Module):
    def __init__(self):
        super(PrediccionDALY_MLP, self).__init__()
        self.capa1 = nn.Linear(5, 16)
        self.capa2 = nn.Linear(16, 8)
        self.salida = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.capa1(x))
        x = self.relu(self.capa2(x))
        x = self.salida(x) 
        return x

modelo = PrediccionDALY_MLP()

# ==========================================
# PASO 4: CONFIGURACIÓN DEL ENTRENAMIENTO
# ==========================================
criterio = nn.MSELoss()
optimizador = optim.Adam(modelo.parameters(), lr=0.01)

epocas = 500
historial_perdida = []

print("Iniciando entrenamiento...")
for epoca in range(epocas):
    predicciones = modelo(X_train_tensor)
    perdida = criterio(predicciones, y_train_tensor)
    
    optimizador.zero_grad() 
    perdida.backward()      
    optimizador.step()      
    
    historial_perdida.append(perdida.item())
    
    if (epoca+1) % 50 == 0:
        print(f'Época [{epoca+1}/{epocas}], Pérdida (MSE): {perdida.item():.4f}')

plt.plot(historial_perdida)
plt.title('Curva de Aprendizaje del Modelo')
plt.xlabel('Época')
plt.ylabel('Pérdida (MSE)')
plt.show()

# ==========================================
# PASO 5: EVALUACIÓN Y PREDICCIÓN (INFERENCIA)
# ==========================================
modelo.eval() # Ponemos el modelo en modo evaluación
with torch.no_grad(): # Desactivamos el cálculo de gradientes
    predicciones_test = modelo(X_test_tensor)
    perdida_test = criterio(predicciones_test, y_test_tensor)
    print(f'\nPérdida en conjunto de prueba (MSE): {perdida_test.item():.4f}')

# --- EJEMPLO DE PREDICCIÓN CON UN PAÍS NUEVO ---
print("\n--- Simulando Predicción ---")

# Creamos el dato nuevo como DataFrame para evitar el Warning de sklearn
dato_nuevo = pd.DataFrame(
    [[0.28, 4.80, 6.30, 0.70, 0.55]], 
    columns=['esquizofrenia', 'depresion', 'ansiedad', 'bipolaridad', 'alimentarios']
)

# 1. Normalizamos
dato_nuevo_scaled = scaler_X.transform(dato_nuevo)
dato_tensor = torch.tensor(dato_nuevo_scaled, dtype=torch.float32)

# 2. Predecimos (y usamos with torch.no_grad() por buenas prácticas)
with torch.no_grad():
    prediccion_scaled = modelo(dato_tensor)

# 3. Revertimos la normalización usando .detach().numpy() para evitar el error de PyTorch
daly_predicho = scaler_y.inverse_transform(prediccion_scaled.detach().numpy())

print(f"Tasas ingresadas (%):\n{dato_nuevo.iloc[0].to_string()}")
print(f"\n-> El modelo predice un DALY por Trastornos Depresivos de: {daly_predicho[0][0]:.2f} por cada 100k habitantes")