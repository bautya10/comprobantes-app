
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

    """

    Limpia el nombre del emisor removiendo comas internas.

    

    Args:

        nombre: Nombre a limpiar

        

    Returns:

        Nombre sin comas

    """

    if not nombre:

        return ""

    return nombre.replace(",", "").strip()





def limpiar_monto(monto_str: str) -> str:

    """

    Limpia el monto según las reglas estrictas:

    - Sin símbolos de moneda ($, USD, etc.)

    - Sin puntos de miles

    - Solo coma (,) para decimales si son > ,00

    

    Args:

        monto_str: String con el monto a limpiar

        

    Returns:

        Monto limpio en formato correcto

    """

    if not monto_str:

        return "0"

    

    # Remover símbolos de moneda comunes

    monto = re.sub(r'[$USD€ARS\s]', '', monto_str)

    

    # Detectar si usa punto o coma como separador decimal

    # Patrón: número con punto/coma seguido de exactamente 2 dígitos al final

    if re.search(r'[.,]\d{2}$', monto):

        # Tiene decimales

        # Remover puntos que sean separadores de miles

        if '.' in monto and ',' in monto:

            # Ambos presentes: el último es decimal

            if monto.rindex('.') > monto.rindex(','):

                # Punto es decimal

                monto = monto.replace(',', '')  # Remover comas de miles

                monto = monto.replace('.', ',')  # Cambiar punto decimal a coma

            else:

                # Coma es decimal

                monto = monto.replace('.', '')  # Remover puntos de miles

        elif '.' in monto:

            # Solo punto: es decimal

            monto = monto.replace('.', ',')

        # Si solo tiene coma, ya está bien

    else:

        # No tiene decimales o son ceros

        monto = monto.replace('.', '').replace(',', '')

        # Si los decimales son ,00, no los incluimos

    

    # Verificar si los decimales son ,00 y removerlos

    if re.search(r',00$', monto):

        monto = re.sub(r',00$', '', monto)

    

    return monto.strip()





def pdf_a_imagen_png(pdf_bytes: bytes) -> bytes:

    """

    Convierte la primera página de un PDF a PNG en memoria.

    La API de Anthropic solo acepta imágenes (jpeg/png/gif/webp), no PDFs directamente.

    Usa alta resolución (3x) para mejor OCR de textos pequeños.

    """

    import fitz  # pymupdf

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    pagina = doc[0]

    # 3x resolución para mejor calidad OCR (especialmente en PDFs de bancos con texto pequeño)

    mat = fitz.Matrix(3.0, 3.0)

    pix = pagina.get_pixmap(matrix=mat)

    return pix.tobytes("png")





def extraer_datos_con_vision_api(archivo_contenido: bytes, nombre_archivo: str, 

                                  tipo_archivo: str) -> Dict[str, str]:

    """

    Extrae datos del comprobante usando Anthropic Claude Vision API.

    

    Args:

        archivo_contenido: Bytes del archivo

        nombre_archivo: Nombre del archivo original

        tipo_archivo: Tipo MIME del archivo

        

    Returns:

        Diccionario con los datos extraídos

    """

    

    # Obtener API key: primero intenta st.secrets (Streamlit Cloud), luego variable de entorno (local)

    api_key = st.secrets.get("ANTHROPIC_API_KEY", None) if hasattr(st, "secrets") else None

    if not api_key:

        api_key = os.getenv("ANTHROPIC_API_KEY")



    if not api_key:

        st.error("⚠️ API Key no configurada. Configura ANTHROPIC_API_KEY en Secrets (Streamlit Cloud) o en el archivo .env (local)")

        return {

            "emisor": "",

            "monto": "0",

            "destinatario": "",

            "id_operacion": "",

            "fecha": "",

            "horario": ""

        }

    

    try:

        # Inicializar cliente de Anthropic

        client = anthropic.Anthropic(api_key=api_key)



        # Si es PDF, convertir primera página a PNG

        # (Anthropic API solo acepta: image/jpeg, image/png, image/gif, image/webp)

        if tipo_archivo == 'application/pdf' or nombre_archivo.lower().endswith('.pdf'):

            try:

                archivo_contenido = pdf_a_imagen_png(archivo_contenido)

                media_type = 'image/png'

            except Exception as e:

                st.error(f"❌ No se pudo convertir el PDF '{nombre_archivo}': {e}")

                return {"emisor": "", "monto": "0", "destinatario": "",

                        "id_operacion": "", "fecha": "", "horario": ""}

        elif tipo_archivo in ['image/jpeg', 'image/jpg']:

            media_type = 'image/jpeg'

        elif tipo_archivo == 'image/png':

            media_type = 'image/png'

        else:

            media_type = 'image/jpeg'



        # Codificar en base64 (después de la posible conversión)

        base64_data = base64.b64encode(archivo_contenido).decode('utf-8')



        # Mostrar indicador de progreso

        with st.spinner(f'🤖 Procesando {nombre_archivo} con Claude 4.6...'):

            # Llamar a la API con Sonnet 4.6 (más rápido, mejor y más barato que Sonnet 4)

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

        

        # Extraer texto de la respuesta

        response_text = message.content[0].text.strip()

        

        # Limpiar posibles backticks de markdown

        response_text = response_text.replace('```json', '').replace('```', '').strip()

        

        # Parsear JSON

        datos = json.loads(response_text)

        

        # Validar que tenga todas las claves necesarias

        claves_requeridas = ["emisor", "monto", "destinatario", "id_operacion", "fecha", "horario"]

        for clave in claves_requeridas:

            if clave not in datos:

                datos[clave] = ""

        

        return datos

        

    except anthropic.APIError as e:

        st.error(f"❌ Error de API de Anthropic: {str(e)}")

        return {

            "emisor": "",

            "monto": "0",

            "destinatario": "",

            "id_operacion": "",

            "fecha": "",

            "horario": ""

        }

    except json.JSONDecodeError as e:

        st.error(f"❌ Error al parsear JSON: {str(e)}")

        st.text(f"Respuesta recibida: {response_text if 'response_text' in locals() else 'N/A'}")

        return {

            "emisor": "",

            "monto": "0",

            "destinatario": "",

            "id_operacion": "",

            "fecha": "",

            "horario": ""

        }

    except Exception as e:

        st.error(f"❌ Error inesperado al procesar {nombre_archivo}: {str(e)}")

        return {

            "emisor": "",

            "monto": "0",

            "destinatario": "",

            "id_operacion": "",

            "fecha": "",

            "horario": ""

        }





def aplicar_logica_formateo(datos: Dict[str, str]) -> Tuple[str, str, str]:

    """

    Aplica la lógica de formateo según las reglas de negocio.

    

    Reglas:

    - Si destinatario es "Jessica Andrea Giuliani" o "Credibank":

      Formato: "NOMBRE_EMISOR",,,,,,,,MONTO (8 comas para columna K)

    - Si destinatario es otro o no figura:

      Formato: MONTO

    

    Args:

        datos: Diccionario con los datos extraídos

        

    Returns:

        Tupla (línea_formateada, nombre_emisor, id_operacion)

    """

    

    # Extraer y limpiar campos

    emisor_raw = datos.get("emisor", "")

    monto_raw = datos.get("monto", "")

    destinatario = datos.get("destinatario", "").strip()

    id_operacion = datos.get("id_operacion", "")

    fecha = datos.get("fecha", "")

    horario = datos.get("horario", "")

    

    # Limpiar nombre del emisor

    emisor = limpiar_nombre(emisor_raw)

    

    # Aplicar fallbacks para el emisor

    if not emisor:

        if id_operacion:

            emisor = id_operacion

        elif fecha and horario:

            emisor = f"{fecha} {horario}"

        else:

            emisor = "SIN_EMISOR"

    

    # Limpiar monto

    monto = limpiar_monto(monto_raw)

    

    # Aplicar lógica de formateo según destinatario

    destinatario_lower = destinatario.lower()

    es_jessica = "jessica" in destinatario_lower and "giuliani" in destinatario_lower

    es_credibank = "credibank" in destinatario_lower



    if es_jessica or es_credibank:

        # Formato: "NOMBRE",,,,,,,,MONTO (8 comas)

        linea = f'"{emisor}",,,,,,,,{monto}'

    else:

        # Formato: solo MONTO

        linea = monto



    return linea, emisor, id_operacion





def detectar_duplicados(procesados: List[Dict[str, str]]) -> List[str]:

    """

    Detecta IDs de operación duplicados en el lote actual.

    

    Args:

        procesados: Lista de diccionarios con datos procesados

        

    Returns:

        Lista de IDs duplicados

    """

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

    """

    Extrae archivos de un ZIP.

    

    Args:

        archivo_zip: Contenido del archivo ZIP

        

    Returns:

        Lista de tuplas (nombre_archivo, contenido, tipo_mime)

    """

    archivos = []

    

    try:

        with zipfile.ZipFile(io.BytesIO(archivo_zip), 'r') as zip_ref:

            for file_info in zip_ref.filelist:

                if not file_info.is_dir():

                    nombre = file_info.filename

                    contenido = zip_ref.read(nombre)

                    

                    # Determinar tipo MIME básico

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

    """

    Procesa los archivos cargados, expandiendo ZIPs si es necesario.

    

    Args:

        archivos_subidos: Lista de archivos de st.file_uploader

        

    Returns:

        Lista de tuplas (nombre_archivo, contenido, tipo_mime) en orden de carga

    """

    archivos_procesados = []

    

    for archivo_subido in archivos_subidos:

        contenido = archivo_subido.read()

        nombre = archivo_subido.name

        tipo = archivo_subido.type

        

        if nombre.lower().endswith('.zip'):

            # Es un ZIP, extraer contenido

            archivos_zip = extraer_archivos_zip(contenido)

            archivos_procesados.extend(archivos_zip)

        else:

            archivos_procesados.append((nombre, contenido, tipo))

    

    return archivos_procesados





# =============================================================================

# INTERFAZ DE STREAMLIT

# =============================================================================



def main():

    """Función principal de la aplicación"""

    

    # Título y descripción

    st.title("🏦 Extractor y Formateador de Comprobantes")

    st.markdown("""

    Esta aplicación procesa comprobantes bancarios (imágenes, PDFs, ZIPs) y genera 

    un formato compatible con Google Sheets según reglas de negocio específicas.

    **Solo aplica para NEXO!**

    """)

    

    # Verificar API Key (Streamlit Cloud usa st.secrets, local usa .env)

    api_key = st.secrets.get("ANTHROPIC_API_KEY", None) if hasattr(st, "secrets") else None

    if not api_key:

        api_key = os.getenv("ANTHROPIC_API_KEY")



    # Sidebar con información

    with st.sidebar:

        st.header("ℹ️ Información")

        

        # Estado de la API

        if api_key:

            st.success("✅ API Key configurada")

        else:

            st.error("⚠️ API Key NO configurada")

            st.markdown("""

            **Para configurar:**

            1. Crea un archivo `.env`

            2. Agrega: `ANTHROPIC_API_KEY=tu-clave-aqui`

            3. Reinicia la aplicación

            """)

        

        st.markdown("---")

        

        st.markdown("""

        **Formatos soportados:**

        - Imágenes: JPG, PNG

        - Documentos: PDF

        - Archivos: ZIP

        

        **Reglas de formateo:**

        - Si destinatario = "Jessica Andrea Giuliani" o "Credibank"

          → `"EMISOR",,,,,,,,MONTO`

        - Si destinatario = Otro

          → `MONTO`

        

        **Modelo:**

        - Claude Sonnet 4.6 (más rápido y preciso)

        

        **Bancos soportados:**

        - Personal Pay ✅

        - Ualá ✅

        - Mercado Pago ✅

        - Todos los demás ✅

        """)

    

    # Área principal

    st.header("📤 Cargar Comprobantes")

    

    # File uploader con múltiples archivos

    archivos_subidos = st.file_uploader(

        "Selecciona uno o más archivos (imágenes, PDFs o ZIPs)",

        type=['jpg', 'jpeg', 'png', 'pdf', 'zip'],

        accept_multiple_files=True,

        help="Puedes seleccionar múltiples archivos. Los ZIPs serán extraídos automáticamente."

    )

    

    # Botón de procesamiento

    if archivos_subidos:

        st.info(f"📁 {len(archivos_subidos)} archivo(s) cargado(s)")

        

        if st.button("🚀 Procesar Comprobantes", type="primary", use_container_width=True):

            

            if not api_key:

                st.error("⛔ No se puede procesar sin API Key. Por favor, configura tu ANTHROPIC_API_KEY en el archivo .env")

                return

            

            # LIMPIEZA DE ESTADO: Olvidar datos anteriores

            if 'resultados_anteriores' in st.session_state:

                del st.session_state.resultados_anteriores

            

            # Procesar archivos (expandir ZIPs)

            with st.spinner("Extrayendo archivos..."):

                archivos_a_procesar = procesar_archivos_cargados(archivos_subidos)

            

            if not archivos_a_procesar:

                st.error("❌ No se encontraron archivos válidos para procesar")

                return

            

            st.success(f"✅ {len(archivos_a_procesar)} comprobante(s) detectado(s)")

            

            # Procesar cada archivo

            resultados = []

            datos_completos = []

            

            progress_bar = st.progress(0)

            status_text = st.empty()

            

            for idx, (nombre, contenido, tipo_mime) in enumerate(archivos_a_procesar):

                progress = (idx + 1) / len(archivos_a_procesar)

                progress_bar.progress(progress)

                status_text.text(f"Procesando {idx + 1}/{len(archivos_a_procesar)}: {nombre}")

                

                # Extraer datos con API de visión

                datos_extraidos = extraer_datos_con_vision_api(contenido, nombre, tipo_mime)

                

                # Aplicar lógica de formateo

                linea_formateada, emisor, id_op = aplicar_logica_formateo(datos_extraidos)

                

                # Guardar resultados

                resultado = {

                    "archivo": nombre,

                    "linea": linea_formateada,

                    "emisor": emisor,

                    "id_operacion": id_op,

                    "datos_raw": datos_extraidos

                }

                resultados.append(resultado)

                datos_completos.append(datos_extraidos)

            

            progress_bar.empty()

            status_text.empty()

            

            # Detectar duplicados

            duplicados = detectar_duplicados(datos_completos)

            

            if duplicados:

                st.warning(f"⚠️ IDs de operación duplicados detectados: {', '.join(duplicados)}")

            

            # Mostrar resultados

            st.header("📊 Resultados")

            

            # Tabs para diferentes vistas

            tab1, tab2, tab3 = st.tabs(["📋 Formato Sheets", "🔍 Detalle", "📝 Datos Crudos"])

            

            with tab1:

                st.subheader("Formato para Google Sheets")

                st.markdown("**Copia el siguiente texto y pégalo en tu hoja de cálculo:**")

                

                # Generar output final

                lineas_salida = [r["linea"] for r in resultados]

                output_final = "\n".join(lineas_salida)

                

                # Área de texto con el resultado

                st.code(output_final, language=None)

                

                # Botón de copiado

                st.download_button(

                    label="💾 Descargar como TXT",

                    data=output_final,

                    file_name=f"comprobantes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",

                    mime="text/plain"

                )

            

            with tab2:

                st.subheader("Detalle por Comprobante")

                

                for idx, resultado in enumerate(resultados, 1):

                    with st.expander(f"#{idx} - {resultado['archivo']}", expanded=False):

                        col1, col2 = st.columns(2)

                        

                        with col1:

                            st.markdown("**Datos extraídos:**")

                            st.json(resultado['datos_raw'])

                        

                        with col2:

                            st.markdown("**Resultado:**")

                            st.code(resultado['linea'])

                            

                            if resultado['id_operacion'] in duplicados:

                                st.error("⚠️ ID duplicado")

            

            with tab3:

                st.subheader("Datos Crudos (JSON)")

                st.json(resultados)

            

            # Guardar en session state

            st.session_state.resultados_anteriores = resultados

    

    else:

        st.info("👆 Carga uno o más archivos para comenzar")

    

    # Footer

    st.markdown("---")

    st.markdown("""

    <div style='text-align: center; color: gray; font-size: 0.9em;'>

        💡 Desarrollado por y para SIDERA | 2026 Priorizando la eficiencia

    </div>

    """, unsafe_allow_html=True)





if __name__ == "__main__":

    main()



