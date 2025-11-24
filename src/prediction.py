"""
prediction.py
Script de ejemplo para cargar el modelo entrenado y hacer predicciones
para un pasajero nuevo del Titanic.
"""

import joblib
import numpy as np


def load_model(path: str):
    model = joblib.load(path)
    return model


def predict_single_passenger(model, pclass: int, sex: str, age: float, fare: float) -> int:
    """
    sex: 'male' o 'female'
    Devuelve 0 = no sobrevive, 1 = sobrevive.
    """
    sex_encoded = 0 if sex == "male" else 1
    features = np.array([[pclass, sex_encoded, age, fare]])
    prediction = model.predict(features)[0]
    return int(prediction)


def main():
    model_path = "models/titanic_model.pkl"
    model = load_model(model_path)

    # Ejemplo de pasajero
    pred = predict_single_passenger(
        model,
        pclass=3,
        sex="male",
        age=22,
        fare=7.25,
    )
    print(f"Predicción (0 = no sobrevive, 1 = sobrevive): {pred}")


if __name__ == "__main__":
    main()
