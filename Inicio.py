"""
Extractor Sidera PRO - Versión Full con Auditoría de Duplicados y Flujo Lineal
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
import anthropic

# Cargar .env
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

st.set_page_config(page_title="Sidera PRO | Extractor", page_icon="🏦", layout="wide")

# =============================================================================
# FUNCIONES DE LIMPIEZA
# =============================================================================

def limpiar_nombre(nombre: str) -> str:
    return nombre.replace(",", "").strip() if nombre else ""

def limpiar_monto(monto_str: str) -> str:
    if not monto_str: return "0"
    monto = re.sub(r'[$USD€ARS\s]', '', monto_str)
    if re.search(r'[.,]\d{2}$', monto):
        if '.' in monto and ',' in monto:
            if monto.rindex('.') > monto.rindex(','):
                monto = monto.replace(',', '').replace('.', ',')
            else:
                monto = monto.replace('.', '')
        elif '.' in monto: monto = monto.replace('.', ',')
    else:
        monto = monto.replace('.', '').replace(',', '')
    if re.search(r',00$', monto): monto = re.sub(r',00$', '', monto)
    return monto.strip()

def pdf_a_imagen_png(pdf_bytes: bytes) -> bytes:
    import fitz
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
    return pix.tobytes("png")

def extraer_datos_con_vision_api(archivo_contenido: bytes, nombre_archivo: str, tipo_archivo: str) -> Dict[str, str]:
    api_key = st.secrets.get("ANTHROPIC_API_KEY", None) if hasattr(st, "secrets") else os.getenv("ANTHROPIC_API_KEY")
    if not api_key: return {"emisor": "ERROR_API", "monto": "0", "id_operacion": ""}
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        if tipo_archivo == 'application/pdf' or nombre_archivo.lower().endswith('.pdf'):
            archivo_contenido = pdf_a_imagen_png(archivo_contenido)
            media_type = 'image/png'
        else: media_type = 'image/jpeg'

        base64_data = base64.b64encode(archivo_contenido).decode('utf-8')

        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": base64_data}},
                    {"type": "text", "text": 'Analiza y extrae en JSON: emisor, monto, id_operacion (solo nros/letras sin "ID"), destinatario.'}
                ]
            }]
        )
        res = message.content[0].text.strip().replace('```json', '').replace('```', '').strip()
        return json.loads(res)
    except: return {"emisor": "ERROR_LECTURA", "monto": "0", "id_operacion": ""}

# =============================================================================
# DETECTOR DE DUPLICADOS
# =============================================================================
def encontrar_duplicados(datos_lote: List[Dict]) -> Dict[str, List[str]]:
    ids_map = {}
    duplicados = {}
    for item in datos_lote:
        id_op = str(item.get("id_operacion", "")).strip()
        if id_op and id_op != "" and id_op != "None":
            if id_op in ids_map:
                if id_op not in duplicados:
                    duplicados[id_op] = [ids_map[id_op]]
                duplicados[id_op].append(item["archivo"])
            else:
                ids_map[id_op] = item["archivo"]
    return duplicados

# =============================================================================
# LÓGICA DE DOBLE PARTIDA
# =============================================================================
def generar_bloques_texto(datos: Dict, cliente: str) -> Dict[str, str]:
    emisor = limpiar_nombre(datos.get("emisor", ""))
    id_op = str(datos.get("id_operacion", "")).strip()
    monto = limpiar_monto(datos.get("monto", ""))
    
    if not emisor: emisor = id_op if id_op else "SIN_NOMBRE"

    res = {"Nexo": "", "Cliente": ""}
    
    if cliente == "Celso":
        # Nexo: Nombre en B, Monto en K (8 comas)
        res["Nexo"] = f'"{emisor}",,,,,,,,{monto}'
        # Celso: Nombre en B, ID en D, Monto en L (9 comas totales)
        res["Cliente"] = f'"{emisor}",,"{id_op}",,,,,,,{monto}'
    elif cliente in ["Canella", "Vertice"]:
        res["Nexo"] = f'"{emisor}",,,,,,,,{monto}'
        res["Cliente"] = f'"{emisor}",,,,,,,,,{monto}'
    elif cliente in ["Giardino", "Cta Cte"]:
        res["Nexo"] = f',,,,,,,,,,{"".join(monto.split())}'
        res["Cliente"] = f',,,,,,,{"".join(monto.split())}'
    else:
        res["Nexo"] = f'"{emisor}",,,,,,,,{monto}'
        res["Cliente"] = monto
        
    return res

# =============================================================================
# INTERFAZ PRINCIPAL
# =============================================================================
def main():
    st.title("🏦 Sidera PRO - Procesador de Lotes")
    
    # Check de API Key en la barra lateral para no molestar
    api_key = st.secrets.get("ANTHROPIC_API_KEY", None) if hasattr(st, "secrets") else os.getenv("ANTHROPIC_API_KEY")
    with st.sidebar:
        st.write("Estado del sistema:")
        if api_key:
            st.success("✅ API Conectada")
        else:
            st.error("⚠️ Faltan credenciales API")

    # --- FLUJO LINEAL ---
    
    st.header("1️⃣ Paso 1: ¿De quién es el lote?")
    cliente_global = st.selectbox(
        "Seleccioná el cliente ANTES de subir los archivos:", 
        ["Celso", "Canella", "Vertice", "Giardino", "Cta Cte", "Otro"]
    )
    
    if cliente_global == "Celso":
        st.info("💡 Modo Celso activado: Se inyectará el ID de Operación en la Columna D para su hoja.")

    st.markdown("---")

    st.header("2️⃣ Paso 2: Subir Comprobantes")
    archivos_subidos = st.file_uploader(
        f"Arrastrá aquí el ZIP o las capturas enviadas por {cliente_global}", 
        type=['jpg', 'jpeg', 'png', 'pdf', 'zip'], 
        accept_multiple_files=True
    )
    
    st.markdown("---")

    st.header("3️⃣ Paso 3: Procesar y Auditar")
    if archivos_subidos and st.button(f"🚀 PROCESAR LOTE DE {cliente_global.upper()}", type="primary", use_container_width=True):
        if not api_key:
            st.error("⛔ Detenido: No hay API Key cargada.")
            return

        archivos_finales = []
        for a in archivos_subidos:
            if a.name.lower().endswith('.zip'):
                with zipfile.ZipFile(a) as z:
                    for n in z.namelist():
                        if not n.endswith('/') and '__MACOSX' not in n:
                            archivos_finales.append((n, z.read(n), 'image/jpeg'))
            else: archivos_finales.append((a.name, a.read(), a.type))

        datos_procesados = []
        progress = st.progress(0)
        status = st.empty()
        
        for idx, (nombre, contenido, tipo) in enumerate(archivos_finales):
            status.text(f"Leyendo comprobante {idx+1}/{len(archivos_finales)}: {nombre}")
            info = extraer_datos_con_vision_api(contenido, nombre, tipo)
            info["archivo"] = nombre
            
            lineas = generar_bloques_texto(info, cliente_global)
            info["linea_nexo"] = lineas["Nexo"]
            info["linea_cliente"] = lineas["Cliente"]
            
            datos_procesados.append(info)
            progress.progress((idx + 1) / len(archivos_finales))
            time.sleep(1.5) # Freno mínimo para estabilidad

        status.empty()
        
        # Detectar Duplicados
        dups = encontrar_duplicados(datos_procesados)

        st.success(f"✅ {len(datos_procesados)} comprobantes extraídos con éxito.")

        # Armar Pestañas
        tab_list = ["📋 Textos para Excel", "🔍 Auditoría y Detalle"]
        if dups:
            tab_list.insert(0, "🚨 ALERTA: DUPLICADOS")
        
        tabs = st.tabs(tab_list)
        
        # Pestaña de Duplicados (Dinámica)
        if dups:
            with tabs[0]:
                st.error(f"### ⚠️ Se detectaron {len(dups)} IDs repetidos en este envío.")
                st.write("Por favor, chequeá estos archivos antes de copiar los montos al Excel:")
                for id_op, lista_archivos in dups.items():
                    with st.expander(f"🆔 ID: {id_op} - Aparece {len(lista_archivos)} veces"):
                        for f in lista_archivos:
                            st.write(f"• `{f}`")
            text_tab = tabs[1]
            audit_tab = tabs[2]
        else:
            text_tab = tabs[0]
            audit_tab = tabs[1]

        # Pestaña de Textos
        with text_tab:
            st.info("💡 Hacé clic adentro del cuadro para seleccionar todo el texto rápido.")
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Para Hoja NEXO")
                st.code("\n".join([d["linea_nexo"] for d in datos_procesados]), language=None)
            with c2:
                st.subheader(f"Para Hoja {cliente_global.upper()}")
                st.code("\n".join([d["linea_cliente"] for d in datos_procesados]), language=None)

        # Pestaña de Auditoría
        with audit_tab:
            st.subheader("Desglose individual por archivo")
            for d in datos_procesados:
                with st.expander(f"📄 {d['archivo']} - ${d.get('monto','0')}"):
                    col_a, col_b = st.columns(2)
                    col_a.write(f"**Emisor:** {d.get('emisor')}")
                    col_a.write(f"**Destinatario:** {d.get('destinatario')}")
                    col_b.write(f"**Monto Limpio:** {limpiar_monto(d.get('monto'))}")
                    col_b.write(f"**🆔 ID Operación:** `{d.get('id_operacion')}`")

if __name__ == "__main__":
    main()
