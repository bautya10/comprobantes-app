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

st.set_page_config(page_title="Extractor Financiero SIDERA", page_icon="🏦", layout="wide")

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
# LÓGICA DE DOBLE PARTIDA EXACTA
# =============================================================================
def generar_asientos(emisor: str, monto: str, id_op: str, cuenta: str) -> Dict[str, str]:
    """
    Genera los textos para pegar asumiendo que SIEMPRE se pega en la COLUMNA B.
    B(0), C(1), D(2), E(3), F(4), G(5), H(6), I(7), J(8), K(9), L(10)
    """
    if not emisor or emisor == "SIN_EMISOR":
        emisor = id_op if id_op else "FALTA_NOMBRE"

    asientos = {}
    
    if cuenta in ["Giardino", "Cta Cte"]:
        # Piden transferencia a Nexo
        # Nexo (Pega en B): Col B "Envio de transferencia", Col L monto (Crédito) -> 10 comas
        asientos["Nexo"] = f'"Envio de transferencia",,,,,,,,,,{monto}'
        # Cuenta (Pega en B): Col B "Transferencia", Col I monto (Débito) -> 7 comas
        asientos[cuenta] = f'"Transferencia",,,,,,,{monto}'
        
    elif cuenta in ["Celso", "Canella", "Vertice"]:
        # Cargan plata en Nexo
        # Nexo (Pega en B): Col B "Transferencia", Col C Nombre, Col K monto (Débito) -> 8 comas post-nombre
        asientos["Nexo"] = f'"Transferencia","{emisor}",,,,,,,,{monto}'
        # Cuenta (Pega en B): Col B "Transferencia", Col C Nombre, Col L monto (Crédito) -> 9 comas post-nombre
        asientos[cuenta] = f'"Transferencia","{emisor}",,,,,,,,,{monto}'
        
    elif cuenta == "Nexo Directo (Pega en Col C)":
        # Formato original (pega en C). C(0)... K(8)
        asientos["Nexo"] = f'"{emisor}",,,,,,,,{monto}'
        
    else:
        asientos["Sin Asignar"] = f'{monto}'
        
    return asientos

# =============================================================================
# IA VISION
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
                        {"type": "text", "text": '''Extrae EXACTAMENTE:
- emisor: Nombre de quien ENVÍA (En Ualá usa EXCLUSIVAMENTE el "Id Op.").
- monto: Cantidad transferida.
- destinatario: Nombre de quien RECIBE.
- id_operacion: Código de la transacción.
Responde ÚNICAMENTE con JSON válido.'''
                        }
                    ]
                }]
            )
        
        texto = message.content[0].text.strip().replace('```json', '').replace('```', '').strip()
        datos = json.loads(texto)
        return {
            "archivo": nombre_archivo,
            "emisor": limpiar_nombre(datos.get("emisor", "")),
            "monto": limpiar_monto(datos.get("monto", "")),
            "id_operacion": datos.get("id_operacion", "")
        }
    except Exception as e:
        return {"archivo": nombre_archivo, "emisor": "ERROR", "monto": "0", "id_operacion": "ERROR"}

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
    st.title("🏦 SIDERA - Controlador de Operaciones")
    
    api_key = st.secrets.get("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY"))

    # INICIALIZAR ESTADOS
    if 'pendientes' not in st.session_state:
        st.session_state.pendientes = pd.DataFrame([{"Listo": False, "Titular": "", "Monto": ""}] * 5)
    if 'extracciones' not in st.session_state:
        st.session_state.extracciones = None

    # --- SIDEBAR: CONTROL DE PENDIENTES ---
    with st.sidebar:
        st.header("📝 Transferencias Pendientes")
        st.markdown("Anotá acá las que piden por WhatsApp hasta que llegue el comprobante.")
        
        # Tabla editable interactiva
        edited_pendientes = st.data_editor(
            st.session_state.pendientes,
            column_config={
                "Listo": st.column_config.CheckboxColumn("Listo", default=False),
                "Titular": st.column_config.TextColumn("Titular / Cuenta"),
                "Monto": st.column_config.TextColumn("Monto Solicitado")
            },
            hide_index=True,
            num_rows="dynamic"
        )
        # Botón para limpiar los terminados
        if st.button("🧹 Borrar marcados como 'Listo'"):
            st.session_state.pendientes = edited_pendientes[edited_pendientes["Listo"] == False].reset_index(drop=True)
            # Asegurar que siempre haya filas vacías
            if len(st.session_state.pendientes) < 3:
                extras = pd.DataFrame([{"Listo": False, "Titular": "", "Monto": ""}] * 3)
                st.session_state.pendientes = pd.concat([st.session_state.pendientes, extras], ignore_index=True)
            st.rerun()
        else:
            st.session_state.pendientes = edited_pendientes

        st.markdown("---")
        st.info(f"API Key: {'✅ Configurada' if api_key else '❌ Falta'}")

    # --- ÁREA PRINCIPAL ---
    # PASO 1: CARGA Y EXTRACCIÓN
    archivos = st.file_uploader("1. Sube los comprobantes (ZIP, JPG, PDF)", accept_multiple_files=True)
    
    if archivos and st.button("🚀 Extraer Datos (Paso 1)", type="primary"):
        if not api_key: st.error("Falta API Key"); return
        
        archivos_listos = procesar_archivos_zip(archivos)
        resultados = []
        
        progreso = st.progress(0)
        for i, (nombre, cont, mime) in enumerate(archivos_listos):
            datos = extraer_datos_con_vision(cont, nombre, mime, api_key)
            datos["Asignar a Cuenta"] = "Seleccionar..." # Campo para el Paso 2
            resultados.append(datos)
            progreso.progress((i + 1) / len(archivos_listos))
            time.sleep(3) # Freno de mano anti-overload
            
        progreso.empty()
        st.session_state.extracciones = resultados
        st.success("Extracción completada. Pasa al Paso 2.")

    # PASO 2: ASIGNACIÓN MANUAL (Data Editor)
    if st.session_state.extracciones is not None:
        st.header("2. Asignación de Comprobantes")
        st.markdown("Mirá el monto y asignale la cuenta a cada comprobante. Luego generá los asientos.")
        
        df_extraido = pd.DataFrame(st.session_state.extracciones)
        
        opciones_cuentas = [
            "Seleccionar...", 
            "Giardino", "Cta Cte", "Celso", "Canella", "Vertice", 
            "Nexo Directo (Pega en Col C)", "Ignorar"
        ]
        
        # Editor visual donde el usuario elige la cuenta
        df_editado = st.data_editor(
            df_extraido,
            column_config={
                "archivo": st.column_config.TextColumn("Archivo", disabled=True),
                "monto": st.column_config.TextColumn("Monto Extraído", disabled=True),
                "emisor": st.column_config.TextColumn("Emisor / ID", disabled=True),
                "id_operacion": None, # Ocultar para limpiar pantalla
                "Asignar a Cuenta": st.column_config.SelectboxColumn(
                    "📌 Asignar a:", options=opciones_cuentas, required=True
                )
            },
            hide_index=True,
            use_container_width=True
        )

        # PASO 3: GENERACIÓN FINAL
        if st.button("⚡ Generar Textos de Doble Partida (Paso 3)", type="primary"):
            
            # Agrupar los resultados por hoja de destino
            bloques_finales = {
                "Nexo": [],
                "Giardino": [],
                "Cta Cte": [],
                "Celso": [],
                "Canella": [],
                "Vertice": []
            }

            for _, row in df_editado.iterrows():
                cuenta = row["Asignar a Cuenta"]
                if cuenta in ["Seleccionar...", "Ignorar"]:
                    continue
                
                asientos = generar_asientos(row["emisor"], row["monto"], row.get("id_operacion",""), cuenta)
                
                if "Nexo" in asientos:
                    bloques_finales["Nexo"].append(asientos["Nexo"])
                
                if cuenta in bloques_finales and cuenta in asientos:
                    bloques_finales[cuenta].append(asientos[cuenta])

            # MOSTRAR RESULTADOS ORDENADOS PARA COPIAR Y PEGAR
            st.header("📋 Textos Listos para Pegar")
            st.warning("⚠️ Asegurate de pegar en la **COLUMNA B** de cada hoja (salvo Nexo Directo que pega en C).")
            
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

            if st.button("🔄 Limpiar todo e Iniciar nuevo lote"):
                st.session_state.extracciones = None
                st.rerun()

if __name__ == "__main__":
    main()
