import sys
from pathlib import Path

import pandas as pd
import joblib


def validate_input(df: pd.DataFrame) -> None:
    """
    Valida la integridad del dataset de entrada para predicción.
    Si encuentra errores, lanza una excepción clara para detener el proceso.

    Validaciones:
    - Columnas obligatorias presentes.
    - Tipos correctos (numéricos / categóricos).
    - No existen valores fuera de rango.
    """

    # Para predicción NO esperamos la columna "Survived"
    required_columns = ["Pclass", "Sex", "Age", "Fare"]

    # 1) Columnas requeridas
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas obligatorias: {missing_cols}")

    # 2) Tipos de datos
    if not pd.api.types.is_numeric_dtype(df["Pclass"]):
        raise TypeError("La columna 'Pclass' debe ser numérica.")

    if not pd.api.types.is_numeric_dtype(df["Age"]):
        raise TypeError("La columna 'Age' debe ser numérica (int o float).")

    if not pd.api.types.is_numeric_dtype(df["Fare"]):
        raise TypeError("La columna 'Fare' debe ser numérica.")

    # 3) Rangos de valores
    if (df["Pclass"] < 1).any() or (df["Pclass"] > 3).any():
        raise ValueError("Valores inválidos en 'Pclass'. Solo se permiten 1, 2, 3.")

    if (df["Age"] < 0).any():
        raise ValueError("La columna 'Age' no puede contener valores negativos.")

    if (df["Fare"] < 0).any():
        raise ValueError("La columna 'Fare' no puede tener tarifas negativas.")

    print("✔ Datos de entrada validados correctamente.")


def load_input(path: str) -> pd.DataFrame:
    """Carga el CSV de entrada y aplica validaciones básicas."""
    df = pd.read_csv(path)
    validate_input(df)
    return df


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica el mismo preprocesado que en el entrenamiento,
    pero solo sobre las variables de entrada.
    """
    feature_cols = ["Pclass", "Sex", "Age", "Fare"]

    # Nos quedamos solo con las columnas que interesan
    df = df[feature_cols].copy()

    # Codificar sexo: male=0, female=1
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    # Rellenar nulos con la mediana
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    return df


def load_model(path: str):
    """Carga el modelo entrenado desde disco."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"No se encontró el modelo en: {path_obj}")
    model = joblib.load(path_obj)
    return model


def main():
    """
    Script de línea de comandos para hacer predicciones.

    Uso:
        python src/prediction.py models/titanic_model.pkl data/raw/titanic.csv
    """
    if len(sys.argv) != 3:
        print(
            "Uso: python src/prediction.py <ruta_modelo.pkl> <ruta_csv_entrada>\n"
            "Ejemplo:\n"
            "  python src/prediction.py models/titanic_model.pkl data/raw/titanic.csv"
        )
        sys.exit(1)

    model_path = sys.argv[1]
    input_path = sys.argv[2]

    try:
        # Cargar modelo y datos
        model = load_model(model_path)
        df_raw = load_input(input_path)

        # Preprocesar y predecir
        X = preprocess_features(df_raw)
        y_pred = model.predict(X)

        print("Predicción (0 = no sobrevive, 1 = sobrevive):")
        for i, pred in enumerate(y_pred, start=1):
            print(f"Pasajero {i}: {pred}")

    except (ValueError, TypeError) as e:
        # Errores derivados de datos inválidos
        print(f"[ERROR] Datos de entrada no válidos: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"[ERROR] Archivo no encontrado: {e}")
        sys.exit(1)
    except Exception as e:
        # Cualquier otro error inesperado
        print(f"[ERROR] Error inesperado al ejecutar la predicción: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
