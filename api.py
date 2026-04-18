import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = "gemini-2.5-flash"
_cliente = None

def _get_cliente():
    global _cliente
    if _cliente is None:
        if not GEMINI_API_KEY:
            raise ValueError("No se encontró la API KEY de Gemini en el archivo .env")
        _cliente = genai.Client(api_key=GEMINI_API_KEY)
    return _cliente

def consultar_gemini(prompt: str, temperatura: float = 0.7) -> str:
    try:
        cliente = _get_cliente()
        respuesta = cliente.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "Eres un asistente experto en estadística, "
                    "explica de forma clara, didáctica y en español."
                ),
                temperature=temperatura,
                max_output_tokens=2048,
            ),
        )
        return respuesta.text if respuesta.text else "Sin respuesta."
    except Exception as e:
        return f"Error Gemini: {e}"

def construir_prompt(zr: dict, pregunta_extra: str = "") -> str:
    prompt = f"""
Actúa como un profesor experto en estadística.

Analiza los siguientes resultados de una Prueba Z de una muestra:

Datos:
- Media muestral (x̄): {zr['x_bar']:.4f}
- Media hipotética (μ₀): {zr['mu0']}
- Tamaño de muestra (n): {zr['n']}
- Desviación estándar (σ): {zr['sigma']:.4f}
- Error estándar (SE): {zr['se']:.4f}

Resultados:
- Z calculado: {zr['z_calc']:.4f}
- Z crítico: {zr['z_crit']:.4f}
- p-value: {zr['p_value']:.6f}
- Nivel de significancia (α): {zr['alpha']}
- Tipo de prueba: {zr['cola']}

Diagnóstico de datos:
- Asimetría: {zr['skew']:.3f}
- Curtosis: {zr['kurt']:.3f}
- Outliers detectados: {zr['n_out']}

Instrucciones IMPORTANTES:
- NO des teoría general.
- Enfócate SOLO en estos resultados.
- Sé claro, directo y específico.

Responde obligatoriamente en este formato:

1. DECISIÓN:
(Escribe EXACTAMENTE: "Se rechaza H0" o "No se rechaza H0")

2. JUSTIFICACIÓN:
(Explica usando el valor de Z y el p-value en relación con α)

3. INTERPRETACIÓN SIMPLE:
(Explica qué significa el resultado en palabras sencillas)

4. ANÁLISIS DE LOS DATOS:
(Comenta si la distribución parece adecuada para usar Prueba Z)

5. CONCLUSIÓN FINAL:
(Una conclusión clara como si fuera respuesta de examen)

"""

    if pregunta_extra.strip():
        prompt += f"\nPregunta adicional del usuario: {pregunta_extra}\nResponde también esta pregunta de forma clara.\n"

    return prompt