# Titanic Prediction Service — Documentación del Proyecto (Tema 2)

Este proyecto forma parte de la Actividad Continua del Tema 2 del curso de MLOps de CUNEF.  
El objetivo es construir un servicio reproducible de *Machine Learning* que prediga la supervivencia de pasajeros del Titanic aplicando buenas prácticas de:

- Control de versiones con Git y GitHub.
- Organización del código en módulos (`src/`, `notebooks/`, `models/`, `docs/`…).
- Calidad de código y notebooks mediante *pre-commit hooks*.
- Documentación en Markdown y Jupyter Book.

---

## 🧩 Problema a resolver

Dado un conjunto de características de los pasajeros (edad, clase, sexo, tarifa pagada, etc.), queremos predecir si un pasajero **sobrevive (1)** o **no sobrevive (0)** al hundimiento del Titanic.

De forma más formal, entrenamos un modelo de clasificación binaria que aprende una función:

$$
\hat{y} = f(X) \quad\text{con}\quad \hat{y} \in \{0,1\}
$$

donde \(X\) son las características de cada pasajero y \(\hat{y}\) es la predicción de supervivencia.

---

## 📊 Descripción del Dataset

El dataset utilizado es el clásico dataset del Titanic. A continuación se resumen las columnas principales utilizadas por el modelo:

| Columna   | Tipo / Codificación                                           | Descripción                                           |
|----------|----------------------------------------------------------------|-------------------------------------------------------|
| `Survived` | 0 / 1                                                        | Variable objetivo: 1 si el pasajero sobrevivió, 0 si no |
| `Pclass`   | 1, 2, 3                                                     | Clase del pasajero (1 = 1ª clase, 3 = 3ª clase)      |
| `Sex`      | `male` / `female` (codificado como 0 / 1)                   | Sexo del pasajero                                     |
| `Age`      | Numérico (años, con imputación de nulos)                    | Edad del pasajero                                     |
| `Fare`     | Numérico (tarifa pagada)                                    | Importe del billete                                   |
| Otras      | (no siempre usadas en el modelo base)                       | `SibSp`, `Parch`, `Embarked`, etc.                    |

Preprocesado aplicado en `training.py`:

1. Selección de columnas relevantes.
2. Imputación de valores nulos (por ejemplo, mediana de `Age` y `Fare`).
3. Codificación del sexo (`Sex`) como variable binaria 0/1.
4. División en conjuntos de *train* y *test*.

---

## ⚙️ Descripción del Pipeline del modelo

El flujo completo de entrenamiento puede resumirse en los siguientes pasos:

1. **Carga del dataset** desde `data/raw/titanic.csv`.
2. **Preprocesado** de las variables (limpieza, imputación, codificación).
3. **División train/test** para evaluar el rendimiento del modelo.
4. **Entrenamiento** de un modelo de regresión logística.
5. **Evaluación** sobre el conjunto de test.
6. **Guardado del modelo** entrenado en `models/titanic_model.pkl`.
7. **Script de predicción** que carga el modelo y genera predicciones a partir de nuevas instancias.

El siguiente diagrama **Mermaid** representa este pipeline:

```mermaid
flowchart TD
    A[Carga de datos<br/>data/raw/titanic.csv] --> B[Preprocesado<br/>limpieza e imputación]
    B --> C[Codificación de variables<br/>Sex -> 0/1]
    C --> D[Split train/test]
    D --> E[Entrenamiento<br/>Regresión logística]
    E --> F[Evaluación en test]
    F --> G[Exportar modelo<br/>models/titanic_model.pkl]
    G --> H[Script de predicción<br/>src/prediction.py]

