# Proyecto de Inteligencia Artificial - UTN FRLP

## Integrantes
- Baccarini, Camila
- Blanco Cavallero, Federico
- Di Grappa, Emiliano
- Diez, Lucas
- Machado Taier, Ivan Ignacio
- Montanari, Santiago
- Preneste, Máximo
- Vidmar, Ian

## Descripción
Este proyecto es una implementación de una Red Neuronal Feedforward Multicapa (Multilayer Perceptron, MLP) con aprendizaje supervisado mediante backpropagation.

El objetivo es predecir la carga total de enfermedad mental, medida en DALYs por trastornos depresivos, para un país en un año dado. La predicción se realiza a partir de las tasas de prevalencia de los distintos trastornos mentales presentes en la población en ese momento.

## Características principales
- Implementación de un MLP multicapa
- Entrenamiento supervisado con backpropagation
- Predicción de carga de enfermedad mental (DALYs) basada en datos de prevalencia

## Uso
1. Preparar los datos de entrada con las tasas de prevalencia de los trastornos mentales.
2. Entrenar la red neuronal con el conjunto de datos disponible.
3. Utilizar el modelo entrenado para predecir DALYs para un país y un año.

## Instalación de dependencias
1. Crear y activar un entorno virtual Python:
   - Windows: `python -m venv env` y luego `env\Scripts\activate`
   - Linux/macOS: `python3 -m venv env` y luego `source env/bin/activate`
2. Actualizar pip y herramientas básicas:
   - `pip install --upgrade pip setuptools wheel`
3. Instalar las dependencias del proyecto:
   - `pip install -r requirements.txt`

## Tecnologías
- Python 3: lenguaje estándar en el ámbito del aprendizaje automático y redes neuronales, con un ecosistema completo y activo de librerías.
- PyTorch: implementación de la red neuronal con soporte para:
  - control total sobre la arquitectura, la función de pérdida y el ciclo de entrenamiento,
  - comprensión didáctica del flujo de datos (forward pass) y del ajuste de pesos (backpropagation),
  - soporte nativo para GPU mediante CUDA (aunque no es requerido para este dataset),
  - comunidad activa, documentación completa y amplia adopción en investigación y producción.

## Nota
Este trabajo fue desarrollado para la materia de Inteligencia Artificial de la UTN FRLP.
