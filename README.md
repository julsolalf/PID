# PID

# 🛑 Sistema de Reconocimiento Automático de Señales de Tráfico

Este proyecto implementa un sistema de reconocimiento de señales de tráfico utilizando redes neuronales convolucionales (CNN), entrenado sobre el conjunto de datos GTSRB. El sistema incluye un modelo de clasificación y una interfaz en tiempo real que utiliza la cámara del ordenador para detectar y clasificar señales.

## 📁 Estructura del proyecto

📦 Proyecto  
├── creacion_modelo_43.ipynb      → Entrenamiento del modelo CNN  
├── camera.py                     → Detección en tiempo real con webcam  
├── models/                       → Carpeta opcional para guardar el modelo entrenado  
└── README.md                     → Este archivo  

## 🧠 Descripción

- `creacion_modelo_43.ipynb`: cuaderno Jupyter donde se entrena un modelo CNN sobre imágenes en escala de grises (30×30) del dataset GTSRB. Incluye técnicas como:
  - Batch Normalization
  - Global Average Pooling
  - Data Augmentation
  - Ponderación de clases

  El modelo final alcanza una precisión del 97–98% sobre el conjunto de validación.

- `camera.py`: script que accede a la cámara del ordenador, detecta regiones con señales de tráfico y las clasifica utilizando el modelo previamente entrenado. El resultado se muestra en tiempo real sobre la imagen de la webcam.

## 🔧 Requisitos

- Python 3.8+
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- scikit-learn

Puedes instalar los requisitos con: 
```{python}
pip install -r requirements.txt
```
## 🚀 Cómo usar

1. Entrenar el modelo ejecutando el notebook:

2. Al final del entrenamiento se guarda el modelo.

3. Ejecuta la detección en tiempo real con `camera.py`

## 📊 Dataset

- Se utiliza el conjunto **GTSRB (German Traffic Sign Recognition Benchmark)**.
- Contiene más de 50.000 imágenes distribuidas en 43 clases de señales reales.

Más información: https://benchmark.ini.rub.de/gtsrb_dataset.html
