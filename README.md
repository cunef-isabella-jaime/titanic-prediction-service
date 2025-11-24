# Titanic Prediction Service

Repositorio del proyecto de la Actividad de Evaluación Continua - Tema 1
(Asignatura: Herramientas de Trabajo Colaborativo).

El objetivo principal es simular un entorno de trabajo colaborativo real
utilizando Git y GitHub: organización, equipos, ramas, commits, Pull Requests,
Issues, Projects, CODEOWNERS y políticas de protección de ramas.

## Objetivo técnico

Como ejemplo, se entrena un modelo sencillo de Machine Learning para predecir
la supervivencia de pasajeros del Titanic a partir de un dataset público.

## Estructura principal

- src/: scripts de entrenamiento (training.py) y predicción (prediction.py).
- notebooks/: notebooks de experimentación individuales (uno por integrante).
- tests/: tests mínimos.
- docs/: documentación del proyecto.
- data/: datasets brutos y procesados.
- .github/: configuración de GitHub (CODEOWNERS y plantillas de Issues).

## Cómo empezar

1. Clonar el repo desde la organización.
2. (Opcional) Crear y activar un entorno virtual.
3. Instalar dependencias: pip install -r requirements.txt
4. Colocar el dataset en data/raw/titanic.csv.
5. Ejecutar el entrenamiento: python src/training.py
6. Ejecutar un ejemplo de predicción: python src/prediction.py

## Equipo

- Isabella Fabani
- Jaime Martinez Martinez
