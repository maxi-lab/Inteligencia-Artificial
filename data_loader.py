import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Nombres cortos para las 5 enfermedades de salida
ENFERMEDADES_SALIDA = [
    'daly_depresion',
    'daly_esquizofrenia',
    'daly_bipolaridad',
    'daly_alimentarios',
    'daly_ansiedad'
]

def cargar_datos():
    df_prev = pd.read_csv('datasets/1-mental-illnesses-prevalence.csv')
    df_daly = pd.read_csv('datasets/2-burden-disease-from-each-mental-illness.csv')

    # ── Renombrar entradas (prevalencia de cada trastorno) ──────────────────
    df_prev.rename(columns={
        'Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized': 'esquizofrenia',
        'Depressive disorders (share of population) - Sex: Both - Age: Age-standardized':    'depresion',
        'Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized':       'ansiedad',
        'Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized':       'bipolaridad',
        'Eating disorders (share of population) - Sex: Both - Age: Age-standardized':        'alimentarios'
    }, inplace=True)

    # ── Renombrar salidas (DALYs de cada trastorno) ─────────────────────────
    # CAMBIO CLAVE: antes solo se usaba la columna de depresión.
    # Ahora renombramos las 5 columnas de DALYs para usarlas todas como salida.
    df_daly.rename(columns={
        'DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Depressive disorders': 'daly_depresion',
        'DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Schizophrenia':        'daly_esquizofrenia',
        'DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Bipolar disorder':     'daly_bipolaridad',
        'DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Eating disorders':     'daly_alimentarios',
        'DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Anxiety disorders':    'daly_ansiedad'
    }, inplace=True)

    # ── Merge y limpieza ────────────────────────────────────────────────────
    df = pd.merge(df_prev, df_daly, on=['Entity', 'Code', 'Year'], how='inner')
    df = df.dropna()

    # ── Definir X e y ───────────────────────────────────────────────────────
    X = df[['esquizofrenia', 'depresion', 'ansiedad', 'bipolaridad', 'alimentarios']]

    # CAMBIO: y ahora es una matriz de 5 columnas (una por enfermedad)
    # Antes: y = df['daly_objetivo']  → vector 1D
    # Ahora: y = df[ENFERMEDADES_SALIDA]  → matriz (N, 5)
    y = df[ENFERMEDADES_SALIDA]

    # Split primero
    X_train, X_test, y_train, y_test = train_test_split(
        X, y.values,
        test_size=0.25,
        random_state=42
    )

    # Scalers SOLO con train
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_test  = scaler_X.transform(X_test)

    y_train = scaler_y.fit_transform(y_train)
    y_test  = scaler_y.transform(y_test)

    return {
        'X_train': X_train,
        'X_test':  X_test,
        'y_train': y_train,
        'y_test':  y_test,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'df': df
    }
