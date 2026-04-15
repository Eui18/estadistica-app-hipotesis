import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Analisis Estadistico", layout="wide")

st.title("Analisis Estadistico Interactivo con IA")
st.write("Carga tus datos para comenzar el analisis.")

st.header("1. Carga de Datos")

fuente = st.radio("Como quieres cargar los datos?", ["Subir CSV", "Generar datos sinteticos"])

df = None

if fuente == "Subir CSV":
    archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])
    if archivo:
        df = pd.read_csv(archivo)
        st.success("Archivo cargado correctamente.")

else:
    n = st.number_input("Numero de datos (n)", min_value=30, max_value=5000, value=100)
    media = st.number_input("Media", value=50.0)
    std = st.number_input("Desviacion estandar", value=10.0)

    if st.button("Generar datos"):
        datos = np.random.normal(media, std, int(n))
        df = pd.DataFrame({"variable": datos})
        st.success("Datos generados correctamente.")