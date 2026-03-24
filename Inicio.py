"""
Extractor y Formateador de Comprobantes Bancarios
Aplicación Streamlit para procesar comprobantes y generar formato compatible con Google Sheets
"""

import streamlit as st
import re
import zipfile
import io
from datetime import datetime
from typing import List, Dict, Tuple, Optional
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
    """Limpia el nombre del emisor removiendo comas internas."""
    if not nombre:
        return ""
    return nombre.replace(",", "").strip()

def limpiar_monto(monto_str: str) -> str:
    """Limpia el monto según las reglas estrictas."""
    if not monto_str:
        return "0"
    
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
    """Convierte la primera página de un PDF a PNG en memoria."""
    import fitz  # pymupdf
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pagina = doc[0]
    # BAJADO A 2.0 PARA QUE NO EXPLOTE LA MAC NI LA API
    mat = fitz.Matrix(2.0, 2.0)
    pix = pagina.get_pixmap(matrix=mat)
    return pix.tobytes("png")

def extraer_datos_con_vision_api(archivo_contenido: bytes, nombre_archivo: str, 
                                 tipo_archivo: str) -> Dict[str, str]:
    """Extrae datos del comprobante usando Anthropic Claude Vision API."""
    
    api_key = st.secrets.get("ANTHROPIC_API_KEY", None) if hasattr(st, "secrets") else None
    if not api_key:
        api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        st.error("⚠️ API Key no configurada. Configura ANTHROPIC_API_KEY en Secrets (Streamlit Cloud) o en el archivo .env (local)")
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
        elif tipo_archivo in ['image/jpeg', 'image/jpg']:
            media_type = 'image/jpeg'
        elif tipo_archivo == 'image/png':
            media_type = 'image/png'
        else:
            media_type = 'image/jpeg'

        base64_data = base64.b64encode(archivo_contenido).decode('utf-8')

        with st.spinner(f'🤖 Procesando {nombre_archivo} con Claude...'):
            # MODELO OFICIAL ESTABLE DE ANTHROPIC
            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": '''Analiza este comprobante bancario y extrae EXACTAMENTE estos campos:

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
- id_operacion: Número o código único de la operación/transacción (puede estar como "Nro de operación", "ID", "Código", etc.)
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
}'''
                        }
                    ]
                }]
            )
        
        response_text = message.content[0].text.strip()
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        datos = json.loads(response_text)
        
        claves_requeridas = ["emisor", "monto", "destinatario", "id_operacion", "fecha", "horario"]
        for clave in claves_requeridas:
            if clave not in datos:
                datos[clave] = ""
        
        return datos
        
    except anthropic.APIError as e:
        st.error(f"❌ Error de API de Anthropic: {str(e)}")
        return {"emisor": "", "monto": "0", "destinatario": "", "id_operacion": "", "fecha": "", "horario": ""}
    except json.JSONDecodeError as e:
        st.error(f"❌ Error al parsear JSON: {str(e)}")
        return {"emisor": "", "monto": "0", "destinatario": "", "id_operacion": "", "fecha": "", "horario": ""}
    except Exception as e:
        st.error(f"❌ Error inesperado al procesar {nombre_archivo}: {str(e)}")
        return {"emisor": "", "monto": "0", "destinatario": "", "id_operacion": "", "fecha": "", "horario": ""}

def aplicar_logica_formateo(datos: Dict[str, str]) -> Tuple[str, str, str]:
    emisor_raw = datos.get("emisor", "")
    monto_raw = datos.get("monto", "")
    destinatario = datos.get("destinatario", "").strip()
    id_operacion = datos.get("id_operacion", "")
    fecha = datos.get("fecha", "")
    horario = datos.get("horario", "")
    
    emisor = limpiar_nombre(emisor_raw)
    
    if not emisor:
        if id_operacion:
            emisor = id_operacion
        elif fecha and horario:
            emisor = f"{fecha} {horario}"
        else:
            emisor = "SIN_EMISOR"
    
    monto = limpiar_monto(monto_raw)
    destinatario_lower = destinatario.lower()
    
    # REGLAS DE DESTINATARIOS IMPORTANTES
    es_jessica = "jessica" in destinatario_lower and "giuliani" in destinatario_lower
    es_credibank = "credibank" in destinatario_lower
    es_ganadera = "estancia la ganadera" in destinatario_lower or "ganadera srl" in destinatario_lower

    if es_jessica or es_credibank or es_ganadera:
        linea = f'"{emisor}",,,,,,,,{monto}'
    else:
        linea = monto

    return linea, emisor, id_operacion

def detectar_duplicados(procesados: List[Dict[str, str]]) -> List[str]:
    ids_vistos = {}
    duplicados = []
    for item in procesados:
        id_op = item.get("id_operacion", "")
        if id_op:
            if id_op in ids_vistos:
                if id_op not in duplicados:
                    duplicados.append(id_op)
            else:
                ids_vistos[id_op] = True
    return duplicados

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
                    if extension in ['.jpg', '.jpeg']:
                        tipo_mime = 'image/jpeg'
                    elif extension == '.png':
                        tipo_mime = 'image/png'
                    elif extension == '.pdf':
                        tipo_mime = 'application/pdf'
                    else:
                        tipo_mime = 'application/octet-stream'
                    archivos.append((nombre, contenido, tipo_mime))
    except Exception as e:
        st.error(f"❌ Error al extraer ZIP: {str(e)}")
    return archivos

def procesar_archivos_cargados(archivos_subidos) -> List[Tuple[str, bytes, str]]:
    archivos_procesados = []
    for archivo_subido in archivos_subidos:
        contenido = archivo_subido.read()
        nombre = archivo_subido.name
        tipo = archivo_subido.type
        if nombre.lower().endswith('.zip'):
            archivos_zip = extraer_archivos_zip(contenido)
            archivos_procesados.extend(archivos_zip)
        else:
            archivos_procesados.append((nombre, contenido, tipo))
    return archivos_procesados

# =============================================================================
# LÓGICA DE DOBLE PARTIDA (SECCIÓN NUEVA CON CELSO)
# =============================================================================
def generar_asientos_doble_partida(emisor: str, monto: str, id_op: str, cuenta: str) -> Dict[str, str]:
    emisor = emisor if emisor else (id_op if id_op else "FALTA_NOMBRE")
    monto_limpio = "".join(monto.split()) 
    asientos = {}
    
    if cuenta in ["Giardino", "Cta Cte"]:
        asientos["Nexo"] = f',,,,,,,,,,{"".join(monto_limpio)}' 
        asientos[cuenta] = f',,,,,,,{"".join(monto_limpio)}' 
        
    elif cuenta in ["Canella", "Vertice"]:
        asientos["Nexo"] = f'"{emisor}",,,,,,,,{monto_limpio}' 
        asientos[cuenta] = f'"{emisor}",,,,,,,,,{monto_limpio}' 
        
    elif cuenta == "Celso":
        asientos["Nexo"] = f'"{emisor}",,,,,,,,{monto_limpio}' 
        # ID inyectado en la Columna D (2 comas, luego el ID, luego 7 comas para llegar a L)
        asientos[cuenta] = f'"{emisor}",,"{id_op}",,,,,,,{monto_limpio}' 
        
    elif cuenta == "Nexo Directo (Pega en Col C)":
        asientos["Nexo"] = f'"{emisor}",,,,,,,,{monto_limpio}' 
        
    return asientos

# =============================================================================
# INTERFAZ DE STREAMLIT
# =============================================================================
def main():
    st.title("🏦 Extractor y Formateador de Comprobantes")
    
    api_key = st.secrets.get("ANTHROPIC_API_KEY", None) if hasattr(st, "secrets") else None
    if not api_key:
        api_key = os.getenv("ANTHROPIC_API_KEY")

    with st.sidebar:
        st.header("ℹ️ Información")
        if api_key:
            st.success("✅ API Key configurada")
        else:
            st.error("⚠️ API Key NO configurada")
            
        st.markdown("---")
        st.markdown("""
        **Reglas de formateo:**
        - Si destinatario = "Jessica", "Credibank" o **"Estancia la Ganadera SRL"**
          → `"EMISOR",,,,,,,,MONTO`
        - Si destinatario = Otro → `MONTO`
        """)
    
    st.header("📤 1. Cargar Comprobantes")
    archivos_subidos = st.file_uploader(
        "Selecciona uno o más archivos (imágenes, PDFs o ZIPs)",
        type=['jpg', 'jpeg', 'png', 'pdf', 'zip'],
        accept_multiple_files=True
    )
    
    if archivos_subidos:
        if st.button("🚀 Procesar Comprobantes", type="primary", use_container_width=True):
            if not api_key:
                st.error("⛔ No se puede procesar sin API Key.")
                return
            
            if 'resultados_anteriores' in st.session_state:
                del st.session_state.resultados_anteriores
            
            with st.spinner("Extrayendo archivos..."):
                archivos_a_procesar = procesar_archivos_cargados(archivos_subidos)
            
            if not archivos_a_procesar:
                st.error("❌ No se encontraron archivos válidos para procesar")
                return
            
            resultados = []
            datos_completos = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, (nombre, contenido, tipo_mime) in enumerate(archivos_a_procesar):
                progress = (idx + 1) / len(archivos_a_procesar)
                progress_bar.progress(progress)
                status_text.text(f"Procesando {idx + 1}/{len(archivos_a_procesar)}: {nombre}")
                
                # Extracción base original
                datos_extraidos = extraer_datos_con_vision_api(contenido, nombre, tipo_mime)
                linea_formateada, emisor, id_op = aplicar_logica_formateo(datos_extraidos)
                
                resultado = {
                    "archivo": nombre,
                    "linea": linea_formateada,
                    "emisor": emisor,
                    "id_operacion": id_op,
                    "monto": limpiar_monto(datos_extraidos.get("monto", "")),
                    "datos_raw": datos_extraidos
                }
                resultados.append(resultado)
                datos_completos.append(datos_extraidos)
                
                # FRENO ANTI OVERLOAD
                time.sleep(3)
            
            progress_bar.empty()
            status_text.empty()
            
            st.session_state.resultados_anteriores = resultados

    # MOSTRAR RESULTADOS (Mantiene tu visualización original + Paso 2)
    if 'resultados_anteriores' in st.session_state:
        resultados = st.session_state.resultados_anteriores
        
        st.header("📊 Resultados de Extracción")
        tab1, tab2, tab3 = st.tabs(["📋 Formato Original", "🔍 Detalle", "📝 Datos Crudos"])
        with tab1:
            lineas_salida = [r["linea"] for r in resultados]
            st.code("\n".join(lineas_salida), language=None)
        with tab2:
            for r in resultados:
                st.markdown(f"**Archivo:** {r['archivo']} | **Resultado:** {r['linea']}")
        with tab3:
            st.json(resultados)

        st.markdown("---")
        
        # EL SELECTOR PARA DOBLE PARTIDA
        st.header("🎯 2. Asignar Cuentas para Doble Partida")
        opciones = ["Seleccionar...", "Giardino", "Cta Cte", "Celso", "Canella", "Vertice", "Nexo Directo (Pega en Col C)", "Ignorar"]
        
        with st.form("form_doble_partida"):
            st.markdown("Seleccioná de quién es cada comprobante para armar las contrapartidas automáticas.")
            selecciones = []
            for idx, res in enumerate(resultados):
                colA, colB = st.columns([2, 1])
                with colA:
                    # AGREGADO: Mostrar ID en la vista para control visual
                    st.write(f"📄 **{res['archivo']}** | 👤 {res['emisor']} | 🆔 **{res.get('id_operacion', '')}** | 💰 ${res['monto']}")
                with colB:
                    sel = st.selectbox("Asignar a:", opciones, key=f"sel_dp_{idx}", label_visibility="collapsed")
                    selecciones.append(sel)
                st.markdown("---")
            
            btn_generar = st.form_submit_button("⚡ Generar Textos de Contrapartidas", type="primary")
            
        if btn_generar:
            bloques = {"Nexo": [], "Giardino": [], "Cta Cte": [], "Celso": [], "Canella": [], "Vertice": []}

            for idx, res in enumerate(resultados):
                cuenta = selecciones[idx]
                if cuenta in ["Seleccionar...", "Ignorar"]: continue
                
                asientos = generar_asientos_doble_partida(res["emisor"], res["monto"], res.get("id_operacion", ""), cuenta)
                if "Nexo" in asientos: bloques["Nexo"].append(asientos["Nexo"])
                if cuenta in bloques and cuenta in asientos: bloques[cuenta].append(asientos[cuenta])

            st.success("✅ Asientos generados (Listos para pegar en Columna B)")
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Hoja: NEXO")
                st.code("\n".join(bloques["Nexo"]) if bloques["Nexo"] else "Sin datos")
            with c2:
                st.subheader("Hojas: CLIENTES")
                for cliente in ["Giardino", "Cta Cte", "Celso", "Canella", "Vertice"]:
                    if bloques[cliente]:
                        st.markdown(f"**{cliente}**")
                        st.code("\n".join(bloques[cliente]))

if __name__ == "__main__":
    main()
