import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path


# ============================
#     VALIDACIÓN DE DATOS
# ============================
def validate_input(df: pd.DataFrame) -> None:
    """
    Valida la integridad del dataset antes del preprocesado.
    Si encuentra errores, lanza una excepción clara para detener el proceso.
    """
    required_columns = ["Pclass", "Sex", "Age", "Fare", "Survived"]

    # 1) Validar columnas requeridas
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas obligatorias: {missing_cols}")

    # 2) Validar tipos de datos básicos
    if not pd.api.types.is_numeric_dtype(df["Pclass"]):
        raise TypeError("La columna 'Pclass' debe ser numérica.")

    if not pd.api.types.is_numeric_dtype(df["Age"]):
        raise TypeError("La columna 'Age' debe ser numérica.")

    if not pd.api.types.is_numeric_dtype(df["Fare"]):
        raise TypeError("La columna 'Fare' debe ser numérica.")

    # 3) Validar rangos
    if (df["Pclass"] < 1).any() or (df["Pclass"] > 3).any():
        raise ValueError("Valores inválidos en 'Pclass'. Solo se permiten 1, 2, 3.")

    if (df["Age"] < 0).any():
        raise ValueError("La columna 'Age' no puede ser negativa.")

    if (df["Fare"] < 0).any():
        raise ValueError("La columna 'Fare' no puede ser negativa.")

    print("✔ Datos validados correctamente.")


# ============================
#     CARGA DE DATOS
# ============================
def load_data(path: str) -> pd.DataFrame:
    """Carga el dataset desde un CSV y aplica validación."""
    df = pd.read_csv(path)
    validate_input(df)
    return df


# ============================
#     PREPROCESADO
# ============================
def preprocess(df: pd.DataFrame):
    """
    Preprocesado básico:
    - Selección de columnas
    - Imputación de nulos
    - Codificación de 'Sex'
    """
    target_col = "Survived"
    feature_cols = ["Pclass", "Sex", "Age", "Fare"]

    df = df[feature_cols + [target_col]].copy()

    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    X = df[feature_cols]
    y = df[target_col]

    return X, y


# ============================
#      ENTRENAMIENTO
# ============================
def train_model(X, y):
    """Entrena un modelo de regresión logística sencillo."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy en test: {acc:.3f}")
    return model


# ============================
#        GUARDADO
# ============================
def save_model(model, path: str):
    """Guarda el modelo entrenado en disco."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Modelo guardado en: {path}")


# ============================
#        SCRIPT MAIN
# ============================
def main():
    data_path = "data/raw/titanic.csv"
    model_path = "models/titanic_model.pkl"

    df = load_data(data_path)
    X, y = preprocess(df)
    model = train_model(X, y)
    save_model(model, model_path)


if __name__ == "__main__":
    main()
