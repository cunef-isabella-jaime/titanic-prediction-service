import pandas as pd
import pytest

from src.prediction import validate_input


def test_validate_input_ok():
    """
    Caso feliz: el DataFrame tiene todas las columnas necesarias,
    tipos correctos y valores en rango. No debe lanzar excepción.
    """
    df = pd.DataFrame(
        {
            "Pclass": [1, 2, 3],
            "Sex": ["male", "female", "male"],
            "Age": [20, 35, 50],
            "Fare": [10.0, 50.5, 80.0],
            "Survived": [0, 1, 0],
        }
    )

    # Si hay algún error, el test fallará
    validate_input(df)


def test_validate_input_missing_column():
    """
    Si falta alguna columna obligatoria, debe lanzar ValueError.
    Aquí quitamos 'Fare'.
    """
    df = pd.DataFrame(
        {
            "Pclass": [1, 2],
            "Sex": ["male", "female"],
            "Age": [30, 40],
            # Falta 'Fare'
            "Survived": [0, 1],
        }
    )

    with pytest.raises(ValueError) as excinfo:
        validate_input(df)

    assert "Faltan columnas obligatorias" in str(excinfo.value)


def test_validate_input_wrong_type_pclass():
    """
    Si 'Pclass' no es numérica, debe lanzar TypeError.
    """
    df = pd.DataFrame(
        {
            "Pclass": ["1", "2"],  # mal tipo: string
            "Sex": ["male", "female"],
            "Age": [25, 30],
            "Fare": [15.0, 20.0],
            "Survived": [0, 1],
        }
    )

    with pytest.raises(TypeError) as excinfo:
        validate_input(df)

    assert "Pclass" in str(excinfo.value)


def test_validate_input_negative_age():
    """
    Si 'Age' tiene valores negativos, debe lanzar ValueError.
    """
    df = pd.DataFrame(
        {
            "Pclass": [1, 2],
            "Sex": ["male", "female"],
            "Age": [-5, 30],  # edad negativa
            "Fare": [10.0, 20.0],
            "Survived": [0, 1],
        }
    )

    with pytest.raises(ValueError) as excinfo:
        validate_input(df)

    assert "Age" in str(excinfo.value)


def test_validate_input_negative_fare():
    """
    Si 'Fare' tiene valores negativos, debe lanzar ValueError.
    """
    df = pd.DataFrame(
        {
            "Pclass": [1, 2],
            "Sex": ["male", "female"],
            "Age": [20, 30],
            "Fare": [-1.0, 20.0],  # tarifa negativa
            "Survived": [0, 1],
        }
    )

    with pytest.raises(ValueError) as excinfo:
        validate_input(df)

    assert "Fare" in str(excinfo.value)
