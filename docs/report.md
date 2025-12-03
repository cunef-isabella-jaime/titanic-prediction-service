Informe Final — Titanic Prediction Service

1. Objetivo del proyecto

El objetivo principal fue simular un entorno de trabajo colaborativo real utilizando Git y GitHub, incluyendo:

Trabajo por ramas (feature branches)

Issues

Pull Requests con revisión cruzada

Políticas de protección

Documentación final

Como ejemplo práctico, se implementó un modelo simple de Machine Learning para predecir la supervivencia de pasajeros del Titanic utilizando un dataset público.


2. Dataset utilizado

Se usó el dataset público Titanic, disponible en Kaggle y repositorios educativos.

Para ejecutar el proyecto localmente, el archivo debe estar en:

data/raw/titanic.csv

3. Exploratory Data Analysis (EDA)
Tareas realizadas:

Inspección inicial del dataset

Análisis de nulos

Distribución de variables relevantes (Pclass, Sex, Age, Fare, Survived)

Visualizaciones

Conclusiones preliminares

Cada integrante realizó su propio notebook:

notebooks/eda_isabella.ipynb
notebooks/eda_jaime.ipynb


4. Preprocesamiento aplicado
El preprocesado incluye:

Selección de columnas:

["Pclass", "Sex", "Age", "Fare"]


Codificación del sexo:

male → 0
female → 1


Relleno de valores faltantes:

Edad → mediana

Fare → mediana

Separación en variables:

X: features

y: Survived


5. Entrenamiento del modelo

Se entrenó un modelo de Regresión Logística con:

LogisticRegression(max_iter=1000)


División de datos:

train_test_split(X, y, test_size=0.2, random_state=42)

 Resultado:
Accuracy en test: 0.804


El modelo entrenado se guarda en:

models/titanic_model.pkl

6. Predicción

El script:

src/prediction.py


Carga el modelo y permite predecir manualmente:

Ejemplo:

Pclass = 3
Sex = "male"
Age = 28
Fare = 7.25


Salida:

Predicción (0=no sobrevive, 1=sobrevive): 0


7. Estructura del proyecto

titanic-prediction-service/
│
├── data/raw/titanic.csv
├── docs/report.md
├── models/titanic_model.pkl
├── notebooks/
├── src/
│   ├── training.py
│   └── prediction.py
├── tests/
└── README.md


8. Cómo ejecutar el proyecto

1) Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

2) Instalar dependencias
pip install -r requirements.txt

3) Verificar dataset
data/raw/titanic.csv

4) Entrenar modelo
python src/training.py

5) Realizar predicción
python src/prediction.py

9. Equipo

Isabella Fabani

Jaime Martínez Martínez
