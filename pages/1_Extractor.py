import streamlit as st
import re
import zipfile
import io
from datetime import datetime
from typing import List, Dict, Tuple
from pathlib import Path
import base64
import json
import os
import time
import anthropic

st.set_page_config(page_title="Extractor SIDERA", page_icon="⚙️", layout="wide")

# =============================================================================
# FUNCIONES BASE (TUS FUNCIONES ORIGINALES)
# =============================================================================
def limpiar_nombre(nombre: str) -> str:
    if not nombre: return ""
    return nombre.replace(",", "").strip()

def limpiar_monto(monto_str: str) -> str:
    if not monto_str: return "0"
    monto = re.sub(r'[$USD€ARS\s]', '', monto_str)
    if re.search(r'[.,]\d{2}$', monto):
        if '.' in monto and ',' in monto:
            if monto.rindex('.') > monto.rindex(','):
                monto = monto.replace(',', '').replace('.', ',')
            else:
                monto = monto.replace('.', '')
        elif '.' in monto:
            monto = monto.replace('.', ',')
    else:
        monto = monto.replace('.', '').replace(',', '')
    if re.search(r',00$', monto):
        monto = re.sub(r',00$', '', monto)
    return monto.strip()

def pdf_a_imagen_png(pdf_bytes: bytes) -> bytes:
    import fitz  
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    mat = fitz.Matrix(2.0, 2.0) # Resolución optimizada
    return doc[0].get_pixmap(matrix=mat).tobytes("png")

# =============================================================================
# LÓGICA DE DOBLE PARTIDA (COLUMNA B VACÍA PARA TUS CHIPS)
# =============================================================================
def generar_asientos(emisor: str, monto: str, id_op: str, cuenta: str) -> Dict[str, str]:
    emisor = emisor if emisor else (id_op if id_op else "FALTA_NOMBRE")
    monto_limpio = "".join(monto.split()) 
    asientos = {}
    
    if cuenta in ["Giardino", "Cta Cte"]:
        asientos["Nexo"] = f',,,,,,,,,,{"".join(monto_limpio)}' # L(Monto)
        asientos[cuenta] = f',,,,,,,{"".join(monto_limpio)}' # I(Monto)
        
    elif cuenta in ["Celso", "Canella", "Vertice"]:
        asientos["Nexo"] = f',"{emisor}",,,,,,,,{monto_limpio}' # C(Emisor), K(Monto)
        asientos[cuenta] = f',"{emisor}",,,,,,,,,{monto_limpio}' # C(Emisor), L(Monto)
        
    elif cuenta == "Nexo Directo (Pega en Col C)":
        asientos["Nexo"] = f'"{emisor}",,,,,,,,{monto_limpio}' # Original pega en C
        
    return asientos

# =============================================================================
# IA VISION (TU PROMPT ORIGINAL INTACTO)
# =============================================================================
def extraer_datos_con_vision_api(archivo_contenido: bytes, nombre_archivo: str, tipo_archivo: str, api_key: str):
    client = anthropic.Anthropic(api_key=api_key)
    try:
        if tipo_archivo == 'application/pdf' or nombre_archivo.lower().endswith('.pdf'):
            archivo_contenido = pdf_a_imagen_png(archivo_contenido)
            media_type = 'image/png'
        else:
            media_type = 'image/jpeg' if tipo_archivo in ['image/jpeg', 'image/jpg'] else 'image/png'

        base64_data = base64.b64encode(archivo_contenido).decode('utf-8')

        with st.spinner(f'🤖 Procesando {nombre_archivo}...'):
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": base64_data}},
                        {"type": "text", "text": '''Analiza este comprobante bancario y extrae EXACTAMENTE estos campos:

**IMPORTANTE - Reglas especiales para el EMISOR:**
- El EMISOR es quien ENVÍA el dinero (quien hace la transferencia)
- En Personal Pay: busca "De:" o el nombre al inicio del comprobante
- En Ualá: busca "De" o "Enviaste desde" o el nombre del usuario que envía
- En Mercado Pago: busca "Enviaste dinero a" o el remitente
- Si hay un alias o CVU pero también un nombre, usa el NOMBRE, no el alias
- Si solo aparece un alias/CVU sin nombre, usa el alias
- NO confundas emisor con destinatario (quien recibe)

**Campos a extraer:**
- emisor: Nombre completo de quien ENVÍA el dinero (ver reglas arriba)
- monto: Cantidad transferida (número con formato, incluye $ si está visible)
- destinatario: Nombre completo de quien RECIBE el dinero
- id_operacion: Número o código único de la operación/transacción
- fecha: Fecha de la operación en formato YYYY-MM-DD
- horario: Hora de la operación en formato HH:MM:SS

Responde ÚNICAMENTE con un objeto JSON válido con estas claves exactas.
Si algún campo no está visible, usa una cadena vacía "".
NO agregues texto explicativo antes o después del JSON.'''
                        }
                    ]
                }]
            )
        
        texto = message.content[0].text.strip().replace('```json', '').replace('```', '').strip()
        datos = json.loads(texto)
        return {
            "archivo": nombre_archivo, "emisor": limpiar_nombre(datos.get("emisor", "")),
            "monto": limpiar_monto(datos.get("monto", "")), "id_operacion": datos.get("id_operacion", "")
        }
    except Exception as e:
        st.error(f"Error con {nombre_archivo}: {str(e)}")
        return {"archivo": nombre_archivo, "emisor": "ERROR", "monto": "0", "id_operacion": ""}

def procesar_archivos_zip(archivos_subidos):
    procesados = []
    for f in archivos_subidos:
        contenido = f.read()
        if f.name.lower().endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(contenido), 'r') as z:
                for info in z.filelist:
                    if not info.is_dir():
                        ext = Path(info.filename).suffix.lower()
                        mime = 'application/pdf' if ext == '.pdf' else ('image/png' if ext == '.png' else 'image/jpeg')
                        procesados.append((info.filename, z.read(info.filename), mime))
        else:
            procesados.append((f.name, contenido, f.type))
    return procesados

# =============================================================================
# INTERFAZ MAIN
# =============================================================================
st.title("⚙️ Extractor y Formateador")

api_key = st.secrets.get("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY"))

archivos_subidos = st.file_uploader("Sube comprobantes (ZIP, JPG, PDF)", accept_multiple_files=True)

if archivos_subidos:
    if st.button("🚀 Extraer Datos", type="primary"):
        if not api_key:
            st.error("⛔ Falta API Key.")
            st.stop()
            
        archivos_a_procesar = procesar_archivos_zip(archivos_subidos)
        resultados = []
        progreso = st.progress(0)
        
        for idx, (nombre, cont, mime) in enumerate(archivos_a_procesar):
            datos = extraer_datos_con_vision_api(cont, nombre, mime, api_key)
            resultados.append(datos)
            progreso.progress((idx + 1) / len(archivos_a_procesar))
            time.sleep(3) # Freno de mano
            
        progreso.empty()
        st.session_state.resultados_anteriores = resultados

# SI HAY RESULTADOS, MOSTRAMOS EL FORMULARIO DE ASIGNACIÓN
if 'resultados_anteriores' in st.session_state and st.session_state.resultados_anteriores:
    st.header("2. Asignar Cuentas")
    
    opciones = ["Seleccionar...", "Giardino", "Cta Cte", "Celso", "Canella", "Vertice", "Nexo Directo (Pega en Col C)", "Ignorar"]
    
    # El formulario evita que la página parpadee o se borre al seleccionar
    with st.form("form_asignacion"):
        selecciones = []
        for idx, res in enumerate(st.session_state.resultados_anteriores):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**{res['archivo']}** | Emisor: {res['emisor']} | Monto: {res['monto']}")
            with col2:
                sel = st.selectbox("Asignar:", opciones, key=f"sel_{idx}", label_visibility="collapsed")
                selecciones.append(sel)
            st.markdown("---")
            
        submit = st.form_submit_button("⚡ Generar Doble Partida", type="primary")
        
    if submit:
        bloques = {"Nexo": [], "Giardino": [], "Cta Cte": [], "Celso": [], "Canella": [], "Vertice": []}

        for idx, res in enumerate(st.session_state.resultados_anteriores):
            cuenta = selecciones[idx]
            if cuenta in ["Seleccionar...", "Ignorar"]:
                continue
            
            asientos = generar_asientos(res["emisor"], res["monto"], res["id_operacion"], cuenta)
            
            if "Nexo" in asientos:
                bloques["Nexo"].append(asientos["Nexo"])
            if cuenta in bloques and cuenta in asientos:
                bloques[cuenta].append(asientos[cuenta])

        st.header("📋 Textos Listos para Pegar (Pegar en Columna B)")
        colA, colB = st.columns(2)
        
        with colA:
            st.subheader("Hoja NEXO")
            if bloques["Nexo"]:
                st.code("\n".join(bloques["Nexo"]))
            else:
                st.info("No hay datos")

        with colB:
            st.subheader("Hoja CONTRAPARTIDAS")
            for cliente in ["Giardino", "Cta Cte", "Celso", "Canella", "Vertice"]:
                if bloques[cliente]:
                    st.markdown(f"**{cliente}**")
                    st.code("\n".join(bloques[cliente]))
