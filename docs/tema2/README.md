# Titanic Prediction Service — Tema 2
### Documentación del Proyecto (Actividad de Evaluación Continua — Tema 2, CUNEF)

Este documento forma parte del repositorio general que contiene todas las prácticas del curso de MLOps.
El **Tema 2** se centra en la construcción de un **servicio reproducible de Machine Learning** utilizando buenas prácticas de ingeniería, control de versiones y documentación técnica.

El objetivo principal es implementar un flujo completo que permita:

1. Preparar y limpiar un dataset real.
2. Entrenar un modelo reproducible.
3. Guardar dicho modelo siguiendo buenas prácticas.
4. Realizar predicciones mediante un script independiente.
5. Añadir pruebas unitarias.
6. Documentar todo el workflow.

---

## 🧩 Problema a Resolver

Dado un conjunto de características de los pasajeros del Titanic, se busca predecir si un pasajero **sobrevive (1)** o **no sobrevive (0)**.

Más formalmente, entrenamos un modelo de clasificación binaria que aprende:

\[
\hat{y} = f(X)
\]

donde:

- **X** = características del pasajero
- **\hat{y}** = predicción de supervivencia (0 o 1)

---

## 📊 Descripción del Dataset

Se utiliza el dataset clásico del Titanic. Las principales columnas utilizadas en el modelo son:

| Columna   | Tipo / Codificación           | Descripción |
|-----------|-------------------------------|-------------|
| Survived  | 0 / 1                         | Variable objetivo |
| Pclass    | 1, 2, 3                       | Clase del pasajero |
| Sex       | male/female → 0/1             | Sexo |
| Age       | Numérico (imputado)           | Edad |
| Fare      | Numérico                      | Tarifa |
| Otras     | Variables auxiliares          | SibSp, Parch, Embarked |

### 🛠 Preprocesado en `training.py`

1. Selección de columnas relevantes.
2. Imputación de nulos (mediana de Age y Fare).
3. Codificación binaria del sexo.
4. División train/test.

---

##  Descripción del Pipeline del Modelo

El flujo completo del proyecto consiste en:

1. Carga del dataset (`data/raw/titanic.csv`).
2. Preprocesado: limpieza, imputación y codificación.
3. División en train/test.
4. Entrenamiento de un modelo de regresión logística.
5. Evaluación del rendimiento.
6. Guardado del modelo entrenado en `models/titanic_model.pkl`.
7. Generación de predicciones mediante el script `src/prediction.py`.

### Diagrama del pipeline (Mermaid)

```mermaid
flowchart TD
    A[Carga de datos<br>data/raw/titanic.csv] --> B[Preprocesado<br>limpieza e imputación]
    B --> C[Codificación de variables<br>Sex -> 0/1]
    C --> D[División Train/Test]
    D --> E[Entrenamiento<br>Regresión logística]
    E --> F[Evaluación en Test]
    F --> G[Guardado del modelo<br>models/titanic_model.pkl]
    G --> H[Script de predicción<br>src/prediction.py]


---

## Pruebas Unitarias

Como parte de las buenas prácticas de MLOps, se implementaron pruebas unitarias utilizando **pytest**.

### ✔ `tests/test_training.py`
Comprueba:

- Que el preprocesado no devuelve valores nulos.
- Que las columnas esperadas existen tras el procesamiento.
- Que el modelo puede entrenarse sin errores.

### ✔ `tests/test_prediction.py`
Verifica:

- La función `validate_input` rechaza entradas incorrectas.
- El formato de entrada es el requerido (Pclass, Sex, Age, Fare).
- La función `make_prediction` devuelve un resultado válido (0 o 1).
- El manejo de excepciones funciona correctamente.

Para ejecutar los tests:
```markdown
```bash
pytest
