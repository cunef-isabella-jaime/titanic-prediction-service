import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# --------------------------
# LOAD MODEL
# --------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# --------------------------
# TITLE
# --------------------------
st.title("🚢 Titanic Survival Prediction App")

# --------------------------
# USER INPUTS
# --------------------------
st.header("Introduce los datos del pasajero")

sex = st.selectbox("Sexo", ["male", "female"])
pclass = st.selectbox("Clase del billete (Pclass)", [1, 2, 3])
age = st.number_input("Edad", min_value=0, max_value=100, value=30)
fare = st.number_input("Precio del billete", min_value=0.0, value=50.0)

# Validación
if age < 0:
    st.error("La edad no puede ser negativa.")
if fare < 0:
    st.error("El precio no puede ser negativo.")

# --------------------------
# PREDICCIÓN
# --------------------------
if st.button("Predecir supervivencia"):
    df = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": 1 if sex == "female" else 0,
        "Age": age,
        "Fare": fare
    }])

    prediction = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    st.subheader("Resultado:")
    st.write("Probabilidad de supervivencia:", round(prob, 3))

    if prediction == 1:
        st.success("El pasajero **sobrevive** ✔")
    else:
        st.error("El pasajero **no sobrevive** ✘")

# --------------------------
# GRÁFICAS
# --------------------------
st.header("📊 Métricas históricas del modelo")

fig, ax = plt.subplots()
ax.bar(["Exactitud", "Precision", "Recall"], [0.82, 0.78, 0.75])
plt.title("Métricas del modelo (simuladas)")

st.pyplot(fig)
