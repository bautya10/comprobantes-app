"""
Extractor y Formateador de Comprobantes Bancarios
Aplicación Streamlit para procesar comprobantes y generar formato compatible con Google Sheets
Con sistema de doble partida automática
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
import time
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
        elif tipo_archivo in ['image/jpeg', 'image/jpg']:
            media_type = 'image/jpeg'
        elif tipo_archivo == 'image/png':
            media_type = 'image/png'
        else:
            media_type = 'image/jpeg'

        base64_data = base64.b64encode(archivo_contenido).decode('utf-8')

        with st.spinner(f'🤖 Procesando {nombre_archivo}...'):
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
        st.error(f"❌ Error de API: {str(e)}")
        return {"emisor": "", "monto": "0", "destinatario": "", "id_operacion": "", "fecha": "", "horario": ""}
    except json.JSONDecodeError as e:
        st.error(f"❌ Error al parsear JSON: {str(e)}")
        return {"emisor": "", "monto": "0", "destinatario": "", "id_operacion": "", "fecha": "", "horario": ""}
    except Exception as e:
        st.error(f"❌ Error inesperado: {str(e)}")
        return {"emisor": "", "monto": "0", "destinatario": "", "id_operacion": "", "fecha": "", "horario": ""}


def detectar_duplicados(datos_completos: List[Dict[str, str]]) -> List[str]:
    """
    Detecta IDs de operación duplicados en el lote.
    Retorna lista de IDs que están duplicados.
    """
    ids_vistos = {}
    duplicados = []
    
    for item in datos_completos:
        id_op = item.get("id_operacion", "")
        if id_op and id_op.strip():  # Solo si el ID existe y no está vacío
            if id_op in ids_vistos:
                if id_op not in duplicados:
                    duplicados.append(id_op)
            else:
                ids_vistos[id_op] = True
    
    return duplicados


def generar_doble_partida(emisor: str, monto: str, id_op: str, cliente: str) -> Dict[str, str]:
    """
    Genera las líneas de doble partida según el cliente.
    
    Reglas:
    - NEXO (todos): "Titular",,,,,,,,MONTO (8 comas)
    - CLIENTE (excepto Celso): "Titular",,,,,MONTO (5 comas)
    - CELSO hoja cliente: "Titular",ID,,,,,,,,,MONTO (1 coma, ID, 9 comas)
    
    Args:
        emisor: Nombre del emisor limpio
        monto: Monto limpio
        id_op: ID de operación
        cliente: Cliente seleccionado
        
    Returns:
        Diccionario con las líneas para NEXO y CLIENTE
    """
    resultado = {}
    
    # Línea para NEXO (siempre igual para todos los clientes)
    resultado["nexo"] = f'"{emisor}",,,,,,,,{monto}'
    
    # Línea para hoja del CLIENTE (varía según quién sea)
    if cliente == "Celso":
        # Celso: "Titular",ID,,,,,,,,,MONTO (1 coma + ID + 9 comas)
        resultado["cliente"] = f'"{emisor}",{id_op},,,,,,,,,{monto}'
    elif cliente in ["Canella", "Vertice", "3D Land", "Moreira", "Giampaoli"]:
        # Otros clientes: "Titular",,,,,MONTO (5 comas)
        resultado["cliente"] = f'"{emisor}",,,,,{monto}'
    else:
        # Caso "Otro" (manual) - usar formato estándar de 5 comas
        resultado["cliente"] = f'"{emisor}",,,,,{monto}'
    
    return resultado


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
# INTERFAZ DE STREAMLIT
# =============================================================================

def main():
    st.title("🏦 Extractor de Comprobantes - Doble Partida")
    st.markdown("**Sistema automático para NEXO con doble partida por cliente**")
    
    # Verificar API Key
    api_key = st.secrets.get("ANTHROPIC_API_KEY", None) if hasattr(st, "secrets") else None
    if not api_key:
        api_key = os.getenv("ANTHROPIC_API_KEY")

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Información")
        if api_key:
            st.success("✅ API Key configurada")
        else:
            st.error("⚠️ API Key NO configurada")
        
        st.markdown("---")
        st.markdown("""
        **Lógica de Doble Partida:**
        
        **NEXO (todos):**
        `"Titular",,,,,,,,MONTO`
        
        **Cliente (excepto Celso):**
        `"Titular",,,,,MONTO`
        
        **Celso:**
        `"Titular",ID,,,,,,,,,MONTO`
        
        **Duplicados:**
        Se detectan por ID y se eliminan automáticamente del output.
        """)
    
    # PASO 1: Seleccionar cliente
    st.header("1️⃣ Seleccionar Cliente")
    cliente_opciones = ["Seleccionar...", "Celso", "Canella", "Vertice", "3D Land", "Moreira", "Giampaoli", "Otro (manual)"]
    cliente_seleccionado = st.selectbox(
        "¿De quién son estos comprobantes?",
        cliente_opciones,
        help="Seleccioná el cliente emisor de los comprobantes para generar la doble partida correcta"
    )
    
    # PASO 2: Cargar archivos
    st.header("2️⃣ Cargar Comprobantes")
    archivos_subidos = st.file_uploader(
        "Selecciona uno o más archivos (imágenes, PDFs o ZIPs)",
        type=['jpg', 'jpeg', 'png', 'pdf', 'zip'],
        accept_multiple_files=True,
        help="Los archivos ZIP serán extraídos automáticamente"
    )
    
    # PASO 3: Procesar
    if archivos_subidos and cliente_seleccionado != "Seleccionar...":
        if st.button("🚀 Procesar Comprobantes", type="primary", use_container_width=True):
            if not api_key:
                st.error("⛔ No se puede procesar sin API Key.")
                return
            
            # Limpiar resultados anteriores
            if 'resultados_procesados' in st.session_state:
                del st.session_state.resultados_procesados
            
            # Extraer archivos
            with st.spinner("Extrayendo archivos..."):
                archivos_a_procesar = procesar_archivos_cargados(archivos_subidos)
            
            if not archivos_a_procesar:
                st.error("❌ No se encontraron archivos válidos para procesar")
                return
            
            st.success(f"✅ {len(archivos_a_procesar)} archivo(s) detectado(s)")
            
            # Procesar cada archivo
            resultados = []
            datos_completos = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, (nombre, contenido, tipo_mime) in enumerate(archivos_a_procesar):
                progress = (idx + 1) / len(archivos_a_procesar)
                progress_bar.progress(progress)
                status_text.text(f"Procesando {idx + 1}/{len(archivos_a_procesar)}: {nombre}")
                
                # Extraer datos con API
                datos_extraidos = extraer_datos_con_vision_api(contenido, nombre, tipo_mime)
                
                # Limpiar datos
                emisor = limpiar_nombre(datos_extraidos.get("emisor", ""))
                if not emisor:
                    id_op = datos_extraidos.get("id_operacion", "")
                    emisor = id_op if id_op else "SIN_EMISOR"
                
                monto = limpiar_monto(datos_extraidos.get("monto", ""))
                id_operacion = datos_extraidos.get("id_operacion", "")
                
                resultado = {
                    "archivo": nombre,
                    "emisor": emisor,
                    "monto": monto,
                    "id_operacion": id_operacion,
                    "datos_raw": datos_extraidos
                }
                
                resultados.append(resultado)
                datos_completos.append(datos_extraidos)
                
                # Pausa entre llamadas para no sobrecargar la API
                if idx < len(archivos_a_procesar) - 1:
                    time.sleep(1)
            
            progress_bar.empty()
            status_text.empty()
            
            # Detectar duplicados
            duplicados = detectar_duplicados(datos_completos)
            
            # Filtrar duplicados de los resultados
            resultados_sin_duplicados = []
            ids_ya_vistos = set()
            
            for resultado in resultados:
                id_op = resultado["id_operacion"]
                if id_op and id_op in duplicados:
                    # Si el ID está duplicado, solo incluir la primera aparición
                    if id_op not in ids_ya_vistos:
                        ids_ya_vistos.add(id_op)
                        resultados_sin_duplicados.append(resultado)
                    # Si ya lo vimos, lo saltamos (es el duplicado)
                else:
                    # Si no está en la lista de duplicados, incluir siempre
                    resultados_sin_duplicados.append(resultado)
            
            # Mostrar advertencia si hubo duplicados
            if duplicados:
                eliminados = len(resultados) - len(resultados_sin_duplicados)
                st.warning(f"⚠️ Se detectaron y eliminaron {eliminados} comprobante(s) duplicado(s) (IDs: {', '.join(duplicados)})")
            
            # Guardar en session state
            st.session_state.resultados_procesados = {
                "resultados": resultados_sin_duplicados,
                "cliente": cliente_seleccionado
            }
    
    elif archivos_subidos and cliente_seleccionado == "Seleccionar...":
        st.warning("⚠️ Por favor, seleccioná un cliente antes de procesar")
    
    # MOSTRAR RESULTADOS
    if 'resultados_procesados' in st.session_state:
        datos = st.session_state.resultados_procesados
        resultados = datos["resultados"]
        cliente = datos["cliente"]
        
        st.success(f"✅ Procesamiento completado - {len(resultados)} comprobantes válidos")
        
        # Generar líneas de doble partida
        lineas_nexo = []
        lineas_cliente = []
        
        for resultado in resultados:
            doble_partida = generar_doble_partida(
                resultado["emisor"],
                resultado["monto"],
                resultado["id_operacion"],
                cliente
            )
            lineas_nexo.append(doble_partida["nexo"])
            lineas_cliente.append(doble_partida["cliente"])
        
        # Mostrar outputs
        st.header("3️⃣ Resultados - Copiar y Pegar")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Hoja NEXO")
            output_nexo = "\n".join(lineas_nexo)
            st.code(output_nexo, language=None)
            st.download_button(
                label="💾 Descargar NEXO",
                data=output_nexo,
                file_name=f"nexo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key="btn_nexo"
            )
        
        with col2:
            st.subheader(f"📊 Hoja {cliente.upper()}")
            output_cliente = "\n".join(lineas_cliente)
            st.code(output_cliente, language=None)
            st.download_button(
                label=f"💾 Descargar {cliente.upper()}",
                data=output_cliente,
                file_name=f"{cliente.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key="btn_cliente"
            )
        
        # Tab con detalles
        with st.expander("🔍 Ver Detalle de Comprobantes"):
            for idx, resultado in enumerate(resultados, 1):
                st.markdown(f"**#{idx} - {resultado['archivo']}**")
                st.json(resultado['datos_raw'])
                st.markdown("---")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
        💡 Desarrollado por y para SIDERA | Sistema de Doble Partida Automática
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
