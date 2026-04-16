import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Analisis Estadistico", layout="wide")

st.title("Analisis Estadistico Interactivo con IA")
st.write("Carga tus datos para comenzar el analisis.")

st.header("1. Carga de Datos")

fuente = st.radio("Como quieres cargar los datos?", ["Subir CSV", "Generar datos sinteticos"])

df = None
variable = None

if fuente == "Subir CSV":
    archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])
    if archivo:
        df = pd.read_csv(archivo)
        st.success("Archivo cargado correctamente.")
else:
    n = st.number_input("Numero de datos (n)", min_value=1, max_value=5000, value=100)
    media = st.number_input("Media", value=50.0)
    std = st.number_input("Desviacion estandar", value=10.0)

    if st.button("Generar datos"):
        if n < 30:
            st.error("El tamaño de muestra debe ser al menos 30 para esta aplicación.")
        else:
            datos = np.random.normal(media, std, int(n))
            df = pd.DataFrame({"variable": datos})
            st.success("Datos generados correctamente.")

#vista previa
if df is not None:
    st.subheader("Vista previa de datos")
    st.dataframe(df.head())

    columnas = df.select_dtypes(include=np.number).columns.tolist()

    if len(columnas) == 0:
        st.error("No hay columnas numericas en el archivo.")
    else:
        variable = st.selectbox("Selecciona una variable", columnas)
        st.info(f"Variable seleccionada: {variable}")
        
#Graficas
if df is not None and variable is not None:
    datos = df[variable].dropna()
    st.header("2. Visualizacion de Datos")
    col1, col2 = st.columns(2)

    # HISTOGRAMA
    with col1:
        fig, grafica1 = plt.subplots(figsize=(6,4))
        grafica1.hist(datos, bins=25, color="#5427fa", edgecolor="white", alpha=0.85)
        grafica1.set_title("Histograma")
        grafica1.set_xlabel(variable)
        grafica1.set_ylabel("Frecuencia")
        grafica1.grid(axis="y", alpha=0.3)

        #Línea de la media
        grafica1.axvline(datos.mean(), color="red", linestyle="--", linewidth=2, label=f"Media: {datos.mean():.2f}")
        grafica1.legend()
        st.pyplot(fig)

    # BOXPLOT
    with col2:
        fig2, grafica2 = plt.subplots(figsize=(5,4))
        grafica2.boxplot(
            datos,
            patch_artist=True,
            boxprops=dict(facecolor="#bc137e", color="#a311b4"),
            medianprops=dict(color="red", linewidth=2),
            whiskerprops=dict(color="#9a3db3"),
            capprops=dict(color="#7f0079")
        )

        grafica2.set_title("Boxplot")
        grafica2.set_ylabel(variable)
        grafica2.grid(axis="y", alpha=0.3)

        st.pyplot(fig2)