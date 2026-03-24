"""
Extractor y Formateador de Comprobantes Bancarios
Aplicación Streamlit para procesar comprobantes y generar formato compatible con Google Sheets
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
import time  # <-- Agregado para el freno de mano
import anthropic

# Cargar .env manualmente (funciona en local; en Streamlit Cloud usa st.secrets)
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

# Configuración de la página
st.set_page_config(
    page_title="Extractor de Comprobantes",
    page_icon="🏦",
    layout="wide"
)

# =============================================================================
# FUNCIONES DE LIMPIEZA Y PROCESAMIENTO
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
                monto = monto.replace(',', '')
                monto = monto.replace('.', ',')
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
    import fitz  # pymupdf
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pagina = doc[0]
    mat = fitz.Matrix(2.0, 2.0)
    pix = pagina.get_pixmap(matrix=mat)
    return pix.tobytes("png")

def extraer_datos_con_vision_api(archivo_contenido: bytes, nombre_archivo: str, tipo_archivo: str) -> Dict[str, str]:
    api_key = st.secrets.get("ANTHROPIC_API_KEY", None) if hasattr(st, "secrets") else None
    if not api_key: api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        st.error("⚠️ API Key no configurada.")
        return {"emisor": "", "monto": "0", "destinatario": "", "id_operacion": "", "fecha": "", "horario": ""}
    
    try:
        client = anthropic.Anthropic(api_key=api_key)

        if tipo_archivo == 'application/pdf' or nombre_archivo.lower().endswith('.pdf'):
            try:
                archivo_contenido = pdf_a_imagen_png(archivo_contenido)
                media_type = 'image/png'
            except Exception as e:
                st.error(f"❌ No se pudo convertir el PDF '{nombre_archivo}': {e}")
                return {"emisor": "", "monto": "0", "destinatario": "", "id_operacion": "", "fecha": "", "horario": ""}
        elif tipo_archivo in ['image/jpeg', 'image/jpg']: media_type = 'image/jpeg'
        elif tipo_archivo == 'image/png': media_type = 'image/png'
        else: media_type = 'image/jpeg'

        base64_data = base64.b64encode(archivo_contenido).decode('utf-8')

        with st.spinner(f'🤖 Procesando {nombre_archivo} con Claude...'):
            message = client.messages.create(
                model="claude-sonnet-4-6",
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
- NO confundas emisor con destinatario (quien recibe)

**Campos a extraer:**
- emisor: Nombre completo de quien ENVÍA el dinero (ver reglas arriba)
- monto: Cantidad transferida (número con formato, incluye $ si está visible)
- destinatario: Nombre completo de quien RECIBE el dinero
- id_operacion: Número o código único de la operación/transacción (puede estar como "Nro de operación", "ID", "Código", etc.). Extrae solo los números/letras, sin la palabra "ID".
- fecha: Fecha de la operación en formato YYYY-MM-DD
- horario: Hora de la operación en formato HH:MM:SS (si solo hay HH:MM, agrega :00 al final)

**Formato de respuesta:**
Responde ÚNICAMENTE con un objeto JSON válido con estas claves exactas.
Si algún campo no está visible, usa una cadena vacía "".
NO agregues texto explicativo antes o después del JSON.

Ejemplo de respuesta correcta:
{
    "emisor": "Juan Carlos Pérez",
    "monto": "$1.500,00",
    "destinatario": "María González",
    "id_operacion": "123456789",
    "fecha": "2024-02-11",
    "horario": "14:30:00"
}'''}
                    ]
                }]
            )
        
        response_text = message.content[0].text.strip().replace('```json', '').replace('```', '').strip()
        datos = json.loads(response_text)
        
        claves_requeridas = ["emisor", "monto", "destinatario", "id_operacion", "fecha", "horario"]
        for clave in claves_requeridas:
            if clave not in datos: datos[clave] = ""
        
        return datos
        
    except Exception as e:
        st.error(f"❌ Error inesperado al procesar {nombre_archivo}: {str(e)}")
        return {"emisor": "", "monto": "0", "destinatario": "", "id_operacion": "", "fecha": "", "horario": ""}

# =============================================================================
# FUNCIONES DE MANEJO DE ARCHIVOS
# =============================================================================
def extraer_archivos_zip(archivo_zip: bytes) -> List[Tuple[str, bytes, str]]:
    archivos = []
    try:
        with zipfile.ZipFile(io.BytesIO(archivo_zip), 'r') as zip_ref:
            for file_info in zip_ref.filelist:
                if not file_info.is_dir():
                    nombre = file_info.filename
                    contenido = zip_ref.read(nombre)
                    extension = Path(nombre).suffix.lower()
                    if extension in ['.jpg', '.jpeg']: tipo_mime = 'image/jpeg'
                    elif extension == '.png': tipo_mime = 'image/png'
                    elif extension == '.pdf': tipo_mime = 'application/pdf'
                    else: tipo_mime = 'application/octet-stream'
                    archivos.append((nombre, contenido, tipo_mime))
    except Exception as e: st.error(f"❌ Error al extraer ZIP: {str(e)}")
    return archivos

def procesar_archivos_cargados(archivos_subidos) -> List[Tuple[str, bytes, str]]:
    archivos_procesados = []
    for archivo_subido in archivos_subidos:
        contenido = archivo_subido.read()
        nombre = archivo_subido.name
        tipo = archivo_subido.type
        if nombre.lower().endswith('.zip'): archivos_procesados.extend(extraer_archivos_zip(contenido))
        else: archivos_procesados.append((nombre, contenido, tipo))
    return archivos_procesados

# =============================================================================
# LÓGICA DE DOBLE PARTIDA (NUEVO FLUJO)
# =============================================================================
def generar_asientos_por_cliente(datos: Dict[str, str], cliente_seleccionado: str) -> Dict[str, str]:
    """Genera las líneas de texto según el cliente preseleccionado."""
    
    emisor_raw = datos.get("emisor", "")
    id_op = datos.get("id_operacion", "")
    monto_raw = datos.get("monto", "")
    
    emisor = limpiar_nombre(emisor_raw)
    if not emisor: emisor = id_op if id_op else "FALTA_NOMBRE"
    
    monto_limpio = "".join(limpiar_monto(monto_raw).split())
    asientos = {"Nexo": "", "Cliente": ""}
    
    if cliente_seleccionado in ["Giardino", "Cta Cte"]:
        asientos["Nexo"] = f',,,,,,,,,,{"".join(monto_limpio)}'
        asientos["Cliente"] = f',,,,,,,{"".join(monto_limpio)}'
        
    elif cliente_seleccionado in ["Canella", "Vertice"]:
        asientos["Nexo"] = f'"{emisor}",,,,,,,,{monto_limpio}'
        asientos["Cliente"] = f'"{emisor}",,,,,,,,,{monto_limpio}'
        
    elif cliente_seleccionado == "Celso":
        asientos["Nexo"] = f'"{emisor}",,,,,,,,{monto_limpio}'
        # LÓGICA ESPECIAL CELSO: ID en Columna D (2 comas antes, 6 después para llegar a L)
        asientos["Cliente"] = f'"{emisor}",,"{id_op}",,,,,,,{monto_limpio}'
        
    elif cliente_seleccionado == "Nexo Directo (Pega en Col C)":
        asientos["Nexo"] = f'"{emisor}",,,,,,,,{monto_limpio}'
        
    return asientos

# =============================================================================
# INTERFAZ DE STREAMLIT
# =============================================================================
def main():
    st.title("🏦 Generador de Doble Partida")
    
    api_key = st.secrets.get("ANTHROPIC_API_KEY", None) if hasattr(st, "secrets") else None
    if not api_key: api_key = os.getenv("ANTHROPIC_API_KEY")

    with st.sidebar:
        st.header("⚙️ Configuración del Lote")
        if not api_key: st.error("⚠️ API Key NO configurada")
            
        st.markdown("---")
        # 1. SELECTOR GLOBAL
        st.subheader("1. ¿De quién son estos comprobantes?")
        opciones_clientes = ["Celso", "Canella", "Vertice", "Giardino", "Cta Cte", "Nexo Directo (Pega en Col C)"]
        cliente_global = st.selectbox("Cliente:", opciones_clientes)
        
        if cliente_global == "Celso":
            st.info("💡 Lógica Celso activada: Se incluirá el ID de Operación para la Columna D.")
    
    st.header("📤 Cargar y Procesar")
    
    # 2. CARGA DE ARCHIVOS
    archivos_subidos = st.file_uploader(
        f"Sube los archivos de {cliente_global} (Imágenes, PDF, ZIP)",
        type=['jpg', 'jpeg', 'png', 'pdf', 'zip'],
        accept_multiple_files=True
    )
    
    if archivos_subidos:
        # 3. BOTÓN PROCESAR
        if st.button("🚀 Extraer y Generar Textos", type="primary", use_container_width=True):
            if not api_key:
                st.error("⛔ No se puede procesar sin API Key.")
                return
            
            with st.spinner(f"Procesando lote para {cliente_global}..."):
                archivos_a_procesar = procesar_archivos_cargados(archivos_subidos)
            
            if not archivos_a_procesar:
                st.error("❌ No se encontraron archivos válidos.")
                return
            
            resultados_nexo = []
            resultados_cliente = []
            datos_auditoria = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, (nombre, contenido, tipo_mime) in enumerate(archivos_a_procesar):
                progress = (idx + 1) / len(archivos_a_procesar)
                progress_bar.progress(progress)
                status_text.text(f"Analizando {idx + 1}/{len(archivos_a_procesar)}: {nombre}")
                
                datos_extraidos = extraer_datos_con_vision_api(contenido, nombre, tipo_mime)
                
                # Generar la doble partida en el momento
                asientos = generar_asientos_por_cliente(datos_extraidos, cliente_global)
                
                if asientos.get("Nexo"): resultados_nexo.append(asientos["Nexo"])
                if asientos.get("Cliente"): resultados_cliente.append(asientos["Cliente"])
                
                datos_extraidos["archivo"] = nombre
                datos_auditoria.append(datos_extraidos)
                
                time.sleep(3) # Freno de mano API
            
            progress_bar.empty()
            status_text.empty()
            
            # 4. MOSTRAR RESULTADOS DIRECTAMENTE
            st.success("✅ ¡Procesamiento completado!")
            
            tab_textos, tab_auditoria = st.tabs(["📋 Textos para Excel", "🔍 Auditoría de IDs y Datos"])
            
            with tab_textos:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Para Hoja: NEXO")
                    if resultados_nexo:
                        st.code("\n".join(resultados_nexo), language=None)
                    else:
                        st.write("Sin datos para Nexo.")
                
                with col2:
                    st.subheader(f"Para Hoja: {cliente_global.upper()}")
                    if resultados_cliente:
                        st.code("\n".join(resultados_cliente), language=None)
                    else:
                        st.write(f"Sin datos para {cliente_global}.")
            
            with tab_auditoria:
                for d in datos_auditoria:
                    st.markdown(f"**Archivo:** {d['archivo']} | **Emisor:** {d['emisor']} | **Monto:** {d['monto']} | 🆔 **ID:** `{d['id_operacion']}`")

if __name__ == "__main__":
    main()
