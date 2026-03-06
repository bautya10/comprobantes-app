"""
Extractor y Formateador de Comprobantes Bancarios
Aplicación Streamlit dividida en Control de Pendientes y Procesamiento
"""

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
import pandas as pd
import anthropic

# =============================================================================
# CONFIGURACIÓN Y ESTADO
# =============================================================================
def _cargar_env_local():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())

_cargar_env_local()

st.set_page_config(page_title="SIDERA App", page_icon="🏦", layout="wide")

if 'pendientes' not in st.session_state:
    st.session_state.pendientes = pd.DataFrame([{"Listo": False, "Titular": "", "Monto": ""}] * 10)
if 'resultados_crudos' not in st.session_state:
    st.session_state.resultados_crudos = None

# =============================================================================
# FUNCIONES DE LIMPIEZA
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
    mat = fitz.Matrix(2.0, 2.0)
    return doc[0].get_pixmap(matrix=mat).tobytes("png")

# =============================================================================
# LÓGICA DE DOBLE PARTIDA (Sin texto en Columna B)
# =============================================================================
def generar_asientos(emisor: str, monto: str, id_op: str, cuenta: str) -> Dict[str, str]:
    if not emisor or emisor == "SIN_EMISOR":
        emisor = id_op if id_op else "FALTA_NOMBRE"

    monto = "".join(monto.split()) 
    asientos = {}
    
    if cuenta in ["Giardino", "Cta Cte"]:
        # Nexo: L(Monto) -> 10 comas (B vacío)
        asientos["Nexo"] = f',,,,,,,,,,{"".join(monto)}'
        # Cliente: I(Monto) -> 7 comas (B vacío)
        asientos[cuenta] = f',,,,,,,{"".join(monto)}'
        
    elif cuenta in ["Celso", "Canella", "Vertice"]:
        # Nexo: C(Emisor), K(Monto) -> 1 coma, emisor, 8 comas, monto
        asientos["Nexo"] = f',"{emisor}",,,,,,,,{monto}'
        # Cliente: C(Emisor), L(Monto) -> 1 coma, emisor, 9 comas, monto
        asientos[cuenta] = f',"{emisor}",,,,,,,,,{monto}'
        
    elif cuenta == "Nexo Directo (Pega en Col C)":
        asientos["Nexo"] = f'"{emisor}",,,,,,,,{monto}'
        
    return asientos

# =============================================================================
# IA VISION (PROMPT ORIGINAL)
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
        
        response_text = message.content[0].text.strip().replace('```json', '').replace('```', '').strip()
        datos = json.loads(response_text)
        
        return {
            "archivo": nombre_archivo,
            "emisor": limpiar_nombre(datos.get("emisor", "")),
            "monto": limpiar_monto(datos.get("monto", "")),
            "id_operacion": datos.get("id_operacion", "")
        }
    except Exception as e:
        return {"archivo": nombre_archivo, "emisor": "ERROR_LECTURA", "monto": "0", "id_operacion": ""}

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
def main():
    api_key = st.secrets.get("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY"))

    # Navegación lateral
    st.sidebar.title("Navegación")
    pagina = st.sidebar.radio("Ir a:", ["📝 Control de Pendientes", "⚙️ Procesador de Comprobantes"])
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"API Key: {'✅ Configurada' if api_key else '❌ Falta'}")

    # --- PÁGINA 1: PENDIENTES ---
    if pagina == "📝 Control de Pendientes":
        st.title("📝 Control de Transferencias Pendientes")
        st.markdown("Tabla persistente. No se borrará mientras navegues.")
        
        edited_pendientes = st.data_editor(
            st.session_state.pendientes,
            key="editor_pendientes",
            column_config={
                "Listo": st.column_config.CheckboxColumn("Listo", default=False),
                "Titular": st.column_config.TextColumn("Titular / Cuenta"),
                "Monto": st.column_config.TextColumn("Monto Solicitado")
            },
            hide_index=True,
            use_container_width=True
        )
        
        if st.button("🧹 Borrar marcados como 'Listo'"):
            st.session_state.pendientes = edited_pendientes[edited_pendientes["Listo"] == False].reset_index(drop=True)
            if len(st.session_state.pendientes) < 5:
                extras = pd.DataFrame([{"Listo": False, "Titular": "", "Monto": ""}] * 5)
                st.session_state.pendientes = pd.concat([st.session_state.pendientes, extras], ignore_index=True)
            st.rerun()
        else:
            st.session_state.pendientes = edited_pendientes

    # --- PÁGINA 2: PROCESADOR ---
    elif pagina == "⚙️ Procesador de Comprobantes":
        st.title("⚙️ Procesador y Doble Partida")
        
        if not api_key:
            st.error("⛔ Configura la API Key para continuar.")
            return

        # Si no hay extracciones activas, mostrar uploader
        if st.session_state.resultados_crudos is None:
            archivos = st.file_uploader("1. Sube los comprobantes (ZIP, JPG, PDF)", accept_multiple_files=True)
            
            if archivos and st.button("🚀 Extraer Datos (Paso 1)", type="primary"):
                archivos_listos = procesar_archivos_zip(archivos)
                resultados = []
                progreso = st.progress(0)
                
                for i, (nombre, cont, mime) in enumerate(archivos_listos):
                    datos = extraer_datos_con_vision_api(cont, nombre, mime, api_key)
                    resultados.append(datos)
                    progreso.progress((i + 1) / len(archivos_listos))
                    time.sleep(3) # Freno anti-overload
                    
                progreso.empty()
                st.session_state.resultados_crudos = resultados
                st.rerun()

        # Si hay extracciones, mostrar el Formulario de Asignación
        else:
            st.success("✅ Datos extraídos correctamente.")
            st.header("2. Asignación de Cuentas")
            st.markdown("Asigna las cuentas y haz clic en Generar. (Este formulario no recarga la página al seleccionar).")
            
            opciones = ["Seleccionar...", "Giardino", "Cta Cte", "Celso", "Canella", "Vertice", "Nexo Directo (Pega en Col C)", "Ignorar"]
            selecciones = []

            # USO DE ST.FORM PARA EVITAR RECARGAS
            with st.form("form_asignacion"):
                for idx, res in enumerate(st.session_state.resultados_crudos):
                    col1, col2, col3 = st.columns([3, 2, 2])
                    with col1:
                        st.markdown(f"**Archivo:** {res['archivo']}")
                    with col2:
                        st.markdown(f"**Monto:** {res['monto']} | **Emisor:** {res['emisor']}")
                    with col3:
                        cuenta_seleccionada = st.selectbox("Asignar a:", opciones, key=f"sel_{idx}", label_visibility="collapsed")
                        selecciones.append(cuenta_seleccionada)
                    st.markdown("---")
                
                btn_generar = st.form_submit_button("⚡ Generar Textos", type="primary")

            if btn_generar:
                bloques_finales = {"Nexo": [], "Giardino": [], "Cta Cte": [], "Celso": [], "Canella": [], "Vertice": []}

                for idx, res in enumerate(st.session_state.resultados_crudos):
                    cuenta = selecciones[idx]
                    if cuenta in ["Seleccionar...", "Ignorar"]:
                        continue
                    
                    asientos = generar_asientos(res["emisor"], res["monto"], res["id_operacion"], cuenta)
                    
                    if "Nexo" in asientos:
                        bloques_finales["Nexo"].append(asientos["Nexo"])
                    if cuenta in bloques_finales and cuenta in asientos:
                        bloques_finales[cuenta].append(asientos[cuenta])

                st.header("📋 Textos Listos para Pegar")
                colA, colB = st.columns(2)
                
                with colA:
                    st.subheader("NEXO")
                    texto_nexo = "\n".join(bloques_finales["Nexo"])
                    st.code(texto_nexo if texto_nexo else "No hay movimientos para Nexo")

                with colB:
                    st.subheader("CONTRAPARTIDAS")
                    for cliente in ["Giardino", "Cta Cte", "Celso", "Canella", "Vertice"]:
                        if bloques_finales[cliente]:
                            st.markdown(f"**{cliente}**")
                            st.code("\n".join(bloques_finales[cliente]))

            if st.button("🔄 Borrar Comprobantes y Empezar de Nuevo Lote"):
                st.session_state.resultados_crudos = None
                st.rerun()

if __name__ == "__main__":
    main()
