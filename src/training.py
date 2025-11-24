import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

import joblib

from pathlib import Path
def load_data(path: str) -> pd.DataFrame:"""Carga el dataset desde un CSV."""
    df = pd.read_csv(path)
    return df
def preprocess(df: pd.DataFrame):
    """
    Preprocesado muy básico:
    - Seleccionar algunas columnas.
    - Rellenar nulos.
    - Codificar 'Sex' como 0/1.
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
def train_model(X, y):
    """Entrena un modelo de regresión logística sencillo."""
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy en test: {acc:.3f}")

    return model


def save_model(model, path: str):
    """Guarda el modelo entrenado en disco."""
    from pathlib import Path
    import joblib

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Modelo guardado en: {path}")


def main():
    data_path = "data/raw/titanic.csv"  # Ajusta el nombre si es distinto
    model_path = "models/titanic_model.pkl"

    df = load_data(data_path)
    X, y = preprocess(df)
    model = train_model(X, y)
    save_model(model, model_path)


if __name__ == "__main__":
    main()

