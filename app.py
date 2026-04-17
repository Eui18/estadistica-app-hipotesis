import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="Analisis Estadistico", layout="wide")

#Inicializar almacenamiento en memoria (para que los datos no se borren)
if "df" not in st.session_state:
    st.session_state.df = None

st.title("Analisis Estadistico Interactivo con IA")
st.write("Carga tus datos para comenzar el analisis.")

#Carga de datos
st.sidebar.header("1. Carga de Datos")

fuente = st.sidebar.radio(
    "Como quieres cargar los datos?",
    ["Subir CSV", "Generar datos sinteticos"]
)

if fuente == "Subir CSV":
    archivo = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])
    if archivo:
        st.session_state.df = pd.read_csv(archivo)
        st.sidebar.success("Archivo cargado correctamente")

else:
    n = st.sidebar.number_input("Numero de datos (n)", min_value=1, max_value=5000, value=100)
    media = st.sidebar.number_input("Media", value=50.0)
    std = st.sidebar.number_input("Desviacion estandar", value=10.0)

    if st.sidebar.button("Generar datos"):
        if n < 30:
            st.sidebar.error("El tamaño de muestra debe ser al menos 30")
        else:
            datos = np.random.normal(media, std, int(n))
            st.session_state.df = pd.DataFrame({"variable": datos})
            st.sidebar.success("Datos generados correctamente")

#Obtener los datos desde session_state
df = st.session_state.df
variable = None

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

    if len(datos) < 30:
        st.error("La muestra debe tener al menos 30 datos para aplicar la Prueba Z.")
        st.stop()

    st.header("2. Visualizacion de Datos")
    col1, col2, col3 = st.columns(3)

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
        fig2, grafica2 = plt.subplots(figsize=(6,4))
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
        
    #KDE
    with col3:
        fig3, grafica3 = plt.subplots(figsize=(6,4))

        if len(datos.unique()) > 1: 
            kde = stats.gaussian_kde(datos)
            x = np.linspace(datos.min(), datos.max(), 300)

            grafica3.plot(x, kde(x), color="#16a34a", linewidth=2, label="KDE")
            grafica3.fill_between(x, kde(x), alpha=0.2, color="#16a34a")
            grafica3.legend()
        else:
            grafica3.text(0.5, 0.5, "No hay variacion suficiente para KDE", ha='center', va='center')
            
        grafica3.set_title("KDE (densidad)")
        grafica3.set_xlabel(variable)
        grafica3.set_ylabel("Densidad")
        grafica3.grid(alpha=0.3)
        st.pyplot(fig3)
    
    st.subheader("Analisis de la distribucion")

    #analisis automatico
    sesgo = datos.skew()
    curtosis = datos.kurtosis()
    q1, q3 = datos.quantile(0.25), datos.quantile(0.75)
    iqr = q3 - q1
    outliers = datos[(datos < q1 - 1.5 * iqr) | (datos > q3 + 1.5 * iqr)]

    # Mostrar valores
    st.write(f"Sesgo: {sesgo:.4f} | Curtosis: {curtosis:.4f}")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**La distribucion parece normal?**")
        if abs(sesgo) < 0.5 and abs(curtosis) < 1:
            st.success("Si, parece aproximadamente normal (sesgo y curtosis bajos).")
        else:
            st.warning("No parece completamente normal. Revisa el sesgo y la curtosis.")

    with col2:
        st.write("**Hay sesgo o valores atipicos?**")
        if sesgo > 0.5:
            st.info(f"Sesgo positivo (cola a la derecha): {sesgo:.3f}")
        elif sesgo < -0.5:
            st.info(f"Sesgo negativo (cola a la izquierda): {sesgo:.3f}")
        else:
            st.success(f"Sin sesgo significativo: {sesgo:.3f}")

        if len(outliers) > 0:
            st.warning(f"Se detectaron {len(outliers)} valores atipicos.")
        else:
            st.success("No se detectaron valores atipicos.")
            
    #PRUEBA Z
    st.header("3. Prueba de Hipotesis - Z")
    col1, col2 = st.columns(2)

    with col1:
        mu0 = st.number_input(
            "Media hipotetica (H0)",
            value=float(round(datos.mean(), 2))
        )
        sigma = st.number_input(
            "Desviacion estandar poblacional (sigma)",
            value=float(round(datos.std(), 2)),
            min_value=0.0001
        )
        alpha = st.select_slider(
            "Nivel de significancia (alpha)",
            options=[0.01, 0.05, 0.10],
            value=0.05
        )

    with col2:
        tipo_prueba = st.selectbox(
            "Tipo de prueba",
            ["Bilateral", "Cola izquierda", "Cola derecha"]
        )
        st.write("**Hipotesis:**")
        st.latex(r"H_0: \mu = " + str(mu0))

        if tipo_prueba == "Bilateral":
            st.latex(r"H_1: \mu \neq " + str(mu0))
        elif tipo_prueba == "Cola izquierda":
            st.latex(r"H_1: \mu < " + str(mu0))
        else:
            st.latex(r"H_1: \mu > " + str(mu0))

    #Ejecutar prueba
    if st.button("Ejecutar Prueba Z"):

        # Cálculo del estadístico Z
        n_muestra = len(datos)
        media_muestral = datos.mean()
        error_estandar = sigma / np.sqrt(n_muestra)
        z_calc = (media_muestral - mu0) / error_estandar

        # Calcular p-value y región crítica
        if tipo_prueba == "Bilateral":
            p_value = 2 * (1 - stats.norm.cdf(abs(z_calc)))
            z_critico = stats.norm.ppf(1 - alpha / 2)
            rechazar = abs(z_calc) > z_critico
        elif tipo_prueba == "Cola izquierda":
            p_value = stats.norm.cdf(z_calc)
            z_critico = -stats.norm.ppf(1 - alpha)
            rechazar = z_calc < z_critico
        else:
            p_value = 1 - stats.norm.cdf(z_calc)
            z_critico = stats.norm.ppf(1 - alpha)
            rechazar = z_calc > z_critico
        
        st.subheader("Resultados de la Prueba Z")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Z calculado", f"{z_calc:.4f}")

        if tipo_prueba == "Bilateral":
            c2.metric("Z critico", f"±{z_critico:.4f}")
        else:
            c2.metric("Z critico", f"{z_critico:.4f}")

        c3.metric("p-value", f"{p_value:.4f}")
        c4.metric("n", n_muestra)

        if rechazar:
            st.error(f"Se RECHAZA H0 — el p-value ({p_value:.4f}) es menor que alpha ({alpha}).")
        else:
            st.success(f"No se rechaza H0 — el p-value ({p_value:.4f}) es mayor o igual que alpha ({alpha}).")

        #Grafica de la distribución normal estándar (curva Z)
        st.subheader("Curva normal con region de rechazo")
        fig4, figura4 = plt.subplots(figsize=(13, 4))
        x = np.linspace(-4, 4, 400)
        y = stats.norm.pdf(x)

        figura4.plot(x, y, color="black", linewidth=2, label="Distribucion normal")
        figura4.fill_between(x, y, alpha=0.1, label="Zona de no rechazo")

        if tipo_prueba == "Bilateral":
            figura4.fill_between(x, y, where=(x <= -z_critico), alpha=0.45, color="#dc2626", label="Region de rechazo")
            figura4.fill_between(x, y, where=(x >= z_critico), alpha=0.45, color="#dc2626")
            figura4.axvline(-z_critico, color="#dc2626", linestyle="--", linewidth=1.5)
            figura4.axvline(z_critico, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Z critico = ±{z_critico:.2f}")
        elif tipo_prueba == "Cola izquierda":
            figura4.fill_between(x, y, where=(x <= z_critico), alpha=0.45, color="#dc2626", label="Region de rechazo")
            figura4.axvline(z_critico, color="#dc2626", linestyle="--", linewidth=1.5)
        else:
            figura4.fill_between(x, y, where=(x >= z_critico), alpha=0.45, color="#dc2626", label="Region de rechazo")
            figura4.axvline(z_critico, color="#dc2626", linestyle="--", linewidth=1.5)

        figura4.axvline(z_calc, color="#16a34a", linewidth=2.5, label=f"Z calculado = {z_calc:.2f}")
        figura4.set_title("Distribucion Normal Estandar - Prueba Z")
        figura4.set_xlabel("Z")
        figura4.set_ylabel("Densidad")
        figura4.legend()
        figura4.grid(alpha=0.3)
        st.pyplot(fig4, use_container_width=True)
    
        # Guardar resultados para usar en API o IA
        st.session_state["resultados_z"] = {
            "media_muestral": media_muestral,
            "mu0": mu0,
            "n": n_muestra,
            "sigma": sigma,
            "alpha": alpha,
            "tipo_prueba": tipo_prueba,
            "z_calc": z_calc,
            "z_critico": z_critico,
            "p_value": p_value,
            "rechazar": rechazar,
            "sesgo": sesgo,
            "curtosis": curtosis,
            "outliers": len(outliers)
        }