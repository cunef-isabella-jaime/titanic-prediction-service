import pandas as pd
import pytest

from src.training import validate_input, preprocess


def make_valid_df():
    """DataFrame mínimo válido para los tests."""
    return pd.DataFrame(
        {
            "Pclass": [1, 3],
            "Sex": ["male", "female"],
            "Age": [25.0, 30.0],
            "Fare": [50.0, 80.0],
            "Survived": [0, 1],
        }
    )


# --------- tests de validate_input ---------


def test_validate_input_ok_no_lanza_excepcion():
    df = make_valid_df()
    # Si algo va mal, pytest marcará el test como FAILED
    validate_input(df)


def test_validate_input_falta_columna_obligatoria():
    df = make_valid_df().drop(columns=["Fare"])
    with pytest.raises(ValueError):
        validate_input(df)


def test_validate_input_pclass_no_numerica():
    df = make_valid_df()
    df["Pclass"] = ["uno", "tres"]  # tipo incorrecto
    with pytest.raises(TypeError):
        validate_input(df)


def test_validate_input_age_fuera_de_rango():
    df = make_valid_df()
    df.loc[0, "Age"] = -5  # edad negativa
    with pytest.raises(ValueError):
        validate_input(df)


def test_validate_input_fare_negativa():
    df = make_valid_df()
    df.loc[0, "Fare"] = -10.0
    with pytest.raises(ValueError):
        validate_input(df)


# --------- tests de preprocess ---------


def test_preprocess_devuelve_X_y_con_formato_correcto():
    df = make_valid_df()
    X, y = preprocess(df)

    # columnas de features correctas
    assert list(X.columns) == ["Pclass", "Sex", "Age", "Fare"]

    # target correcto
    assert y.name == "Survived"

    # Sex debe estar codificada a 0/1
    assert set(X["Sex"].unique()) <= {0, 1}


def test_preprocess_rellena_nulos():
    df = make_valid_df()
    df.loc[0, "Age"] = None
    df.loc[1, "Fare"] = None

    X, y = preprocess(df)

    # después del preprocesado no debería haber nulos en Age y Fare
    assert X["Age"].isna().sum() == 0
    assert X["Fare"].isna().sum() == 0
