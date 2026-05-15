import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def cargar_datos():
    df_prev = pd.read_csv('datasets/1-mental-illnesses-prevalence.csv')
    df_daly = pd.read_csv('datasets/2-burden-disease-from-each-mental-illness.csv')

    df_prev.rename(columns={
        'Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized': 'esquizofrenia',
        'Depressive disorders (share of population) - Sex: Both - Age: Age-standardized': 'depresion',
        'Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized': 'ansiedad',
        'Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized': 'bipolaridad',
        'Eating disorders (share of population) - Sex: Both - Age: Age-standardized': 'alimentarios'
    }, inplace=True)

    df_daly.rename(columns={
        'DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Depressive disorders': 'daly_objetivo'
    }, inplace=True)

    df_daly = df_daly[['Entity', 'Code', 'Year', 'daly_objetivo']]
    df = pd.merge(df_prev, df_daly, on=['Entity', 'Code', 'Year'], how='inner')
    df = df.dropna()

    X = df[['esquizofrenia', 'depresion', 'ansiedad', 'bipolaridad', 'alimentarios']]
    y = df['daly_objetivo']

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.25, random_state=42
    )

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'df': df  # útil para evaluate.py después
    }