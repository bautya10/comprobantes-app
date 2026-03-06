"""
Extractor y Formateador de Comprobantes Bancarios con Doble Partida
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

st.set_page_config(page_title="SIDERA - Controlador", page_icon="🏦", layout="wide")

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
# LÓGICA DE DOBLE PARTIDA EXACTA (Columna B Vacía)
# =============================================================================
def generar_asientos(emisor: str, monto: str, id_op: str, cuenta: str) -> Dict[str, str]:
    """
    Genera los textos asumiendo que SIEMPRE se pega en la COLUMNA B.
    Al empezar con coma (,), la Columna B queda vacía para el chip manual.
    B(0), C(1), D(2), E(3), F(4), G(5), H(6), I(7), J(8), K(9), L(10)
    """
    if not emisor or emisor == "SIN_EMISOR":
        emisor = id_op if id_op else "FALTA_NOMBRE"

    monto_limpio = "".join(monto.split()) # Asegurar que no haya espacios en el número
    asientos = {}
    
    if cuenta in ["Giardino", "Cta Cte"]:
        # Piden transferencia a Nexo
        # Nexo: B(vacío), L(Monto) -> 10 comas
        asientos["Nexo"] = f',,,,,,,,,,{"".join(monto_limpio)}'
        # Cuenta: B(vacío), I(Monto) -> 7 comas
        asientos[cuenta] = f',,,,,,,{"".join(monto_limpio)}'
        
    elif cuenta in ["Celso", "Canella", "Vertice"]:
        # Cargan plata en Nexo
        # Nexo: B(vacío), C(Emisor), K(Monto) -> 1 coma, emisor, 8 comas, monto
        asientos["Nexo"] = f',"{emisor}",,,,,,,,{monto_limpio}'
        # Cuenta: B(vacío), C(Emisor), L(Monto) -> 1 coma, emisor, 9 comas, monto
        asientos[cuenta] = f',"{emisor}",,,,,,,,,{monto_limpio}'
        
    elif cuenta == "Nexo Directo (Pega en Col C)":
        # Formato original pegando en C. C(Emisor), K(Monto).
        asientos["Nexo"] = f'"{emisor}",,,,,,,,{monto_limpio}'
        
    else:
        asientos["Sin Asignar"] = f'{monto_limpio}'
        
    return asientos

# =============================================================================
# IA VISION (Motor Restaurado)
# =============================================================================
def extraer_datos_con_vision(archivo_contenido: bytes, nombre_archivo: str, tipo_archivo: str, api_key: str):
    client = anthropic.Anthropic(api_key=api_key)
    try:
        if tipo_archivo == 'application/pdf' or nombre_archivo.lower().endswith('.pdf'):
            archivo_contenido = pdf_a_imagen_png(archivo_contenido)
            media_type = 'image/png'
        else:
            media_type = 'image/jpeg' if tipo_archivo in ['image/jpeg', 'image/jpg'] else 'image/png'

        base64_data = base64.b64encode(archivo_contenido).decode('utf-8')
        
        with st.spinner(f'🤖 Leyendo {nombre_archivo}...'):
            message = client.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": base64_data}},
                        {"type": "text", "text": '''Analiza este comprobante bancario y extrae EXACTAMENTE estos campos:

**IMPORTANTE - Reglas especiales:**
- emisor: Nombre de quien ENVÍA. (En Ualá usa EXCLUSIVAMENTE el "Id Op.").
- monto: Cantidad transferida.
- destinatario: Nombre de quien RECIBE.
- id_operacion: Código de la transacción.

Responde ÚNICAMENTE con JSON válido.'''
                        }
                    ]
                }]
            )
        
        texto = message.content[0].text.strip()
        texto = texto.replace('```json', '').replace('```', '').strip()
        datos = json.loads(texto)
        
        return {
            "archivo": nombre_archivo,
            "emisor": limpiar_nombre(datos.get("emisor", "")),
            "monto": limpiar_monto(datos.get("monto", "")),
            "id_operacion": datos.get("id_operacion", "")
        }
    except Exception as e:
        st.error(f"Error procesando {nombre_archivo}: {str(e)}")
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

    # INICIALIZAR ESTADOS (Fundamentales para que no se borre la data)
    if 'pendientes' not in st.session_state:
        st.session_state.pendientes = pd.DataFrame([{"Listo": False, "Titular": "", "Monto": ""}] * 10)
    if 'extracciones' not in st.session_state:
        st.session_state.extracciones = None
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0 # Llave maestra para borrar archivos

    st.title("🏦 SIDERA - Controlador de Operaciones")

    # TABS PARA SEPARAR EL BLOC DE NOTAS DEL PROCESADOR
    tab_pendientes, tab_procesador = st.tabs(["📝 Bloc de Pendientes", "⚙️ Procesador de Comprobantes"])

    # --- PESTAÑA 1: BLOC DE NOTAS SEGURO ---
    with tab_pendientes:
        st.markdown("### Anotador de Transferencias")
        st.markdown("Este cuadro no se borra. Escribí tranquilo.")
        
        # KEY asignada para que el widget mantenga su estado
        edited_pendientes = st.data_editor(
            st.session_state.pendientes,
            key="editor_fijo_pendientes", 
            column_config={
                "Listo": st.column_config.CheckboxColumn("Listo", default=False),
                "Titular": st.column_config.TextColumn("Titular / Cuenta"),
                "Monto": st.column_config.TextColumn("Monto Solicitado")
            },
            hide_index=True,
            num_rows="dynamic",
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

    # --- PESTAÑA 2: PROCESAMIENTO ---
    with tab_procesador:
        
        if not api_key:
            st.error("⚠️ Falta configurar la API Key.")
            return

        # PASO 1: CARGA Y EXTRACCIÓN (Usa la uploader_key para poder ser reseteado)
        archivos = st.file_uploader("1. Sube los comprobantes (ZIP, JPG, PDF)", 
                                    accept_multiple_files=True, 
                                    key=f"uploader_{st.session_state.uploader_key}")
        
        if archivos and st.button("🚀 Extraer Datos (Paso 1)", type="primary"):
            archivos_listos = procesar_archivos_zip(archivos)
            resultados = []
            
            progreso = st.progress(0)
            for i, (nombre, cont, mime) in enumerate(archivos_listos):
                datos = extraer_datos_con_vision(cont, nombre, mime, api_key)
                datos["Asignar a Cuenta"] = "Seleccionar..."
                resultados.append(datos)
                progreso.progress((i + 1) / len(archivos_listos))
                time.sleep(3) # Freno anti-overload
                
            progreso.empty()
            st.session_state.extracciones = resultados
            st.success("✅ Extracción completada.")
            st.rerun()

        # PASO 2: ASIGNACIÓN MANUAL
        if st.session_state.extracciones is not None:
            st.header("2. Asignación de Comprobantes")
            
            df_extraido = pd.DataFrame(st.session_state.extracciones)
            opciones_cuentas = [
                "Seleccionar...", 
                "Giardino", "Cta Cte", "Celso", "Canella", "Vertice", 
                "Nexo Directo (Pega en Col C)", "Ignorar"
            ]
            
            df_editado = st.data_editor(
                df_extraido,
                key="editor_extracciones",
                column_config={
                    "archivo": st.column_config.TextColumn("Archivo", disabled=True),
                    "monto": st.column_config.TextColumn("Monto Extraído", disabled=True),
                    "emisor": st.column_config.TextColumn("Emisor / ID", disabled=True),
                    "id_operacion": None, 
                    "Asignar a Cuenta": st.column_config.SelectboxColumn("📌 Asignar a:", options=opciones_cuentas, required=True)
                },
                hide_index=True,
                use_container_width=True
            )

            # PASO 3: GENERACIÓN FINAL
            if st.button("⚡ Generar Textos de Doble Partida (Paso 3)", type="primary"):
                
                bloques_finales = {"Nexo": [], "Giardino": [], "Cta Cte": [], "Celso": [], "Canella": [], "Vertice": []}

                for _, row in df_editado.iterrows():
                    cuenta = row["Asignar a Cuenta"]
                    if cuenta in ["Seleccionar...", "Ignorar"]:
                        continue
                    
                    asientos = generar_asientos(row["emisor"], row["monto"], row.get("id_operacion",""), cuenta)
                    
                    if "Nexo" in asientos:
                        bloques_finales["Nexo"].append(asientos["Nexo"])
                    
                    if cuenta in bloques_finales and cuenta in asientos:
                        bloques_finales[cuenta].append(asientos[cuenta])

                st.header("📋 Textos Listos para Pegar")
                st.warning("⚠️ Asegurate de pegar en la **COLUMNA B** de cada hoja (salvo Nexo Directo que pega en C). El texto del chip lo ponés a mano.")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Hoja: NEXO")
                    texto_nexo = "\n".join(bloques_finales["Nexo"])
                    st.code(texto_nexo if texto_nexo else "No hay movimientos para Nexo")

                with col2:
                    st.subheader("Hojas de Clientes (Contrapartidas)")
                    for cliente in ["Giardino", "Cta Cte", "Celso", "Canella", "Vertice"]:
                        if bloques_finales[cliente]:
                            st.markdown(f"**Hoja: {cliente}**")
                            st.code("\n".join(bloques_finales[cliente]))

            # BOTÓN NUKE (Limpia todo absolutamente todo)
            st.markdown("---")
            if st.button("🔄 Borrar Comprobantes y Empezar de Nuevo"):
                st.session_state.extracciones = None
                st.session_state.uploader_key += 1 # Esto obliga al cuadro de archivos a resetearse
                st.rerun()

if __name__ == "__main__":
    main()
