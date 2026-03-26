"""
Extractor Sidera - Versión Aislada (Solo Carga Manual)
"""

import streamlit as st
import re
import zipfile
import base64
import json
import os
import time
import anthropic
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Extractor de Comprobantes", page_icon="🏦", layout="wide")

# =============================================================================
# FUNCIONES DE LIMPIEZA Y LÓGICA
# =============================================================================

def limpiar_nombre(nombre: str) -> str:
    return nombre.replace(",", "").strip() if nombre else ""

def limpiar_monto(monto_str) -> str:
    if not monto_str: return "0"
    monto = str(monto_str)
    monto = re.sub(r'[$USD€ARS\s]', '', monto)
    if re.search(r'[.,]\d{2}$', monto):
        if '.' in monto and ',' in monto:
            monto = monto.replace(',', '').replace('.', ',') if monto.rindex('.') > monto.rindex(',') else monto.replace('.', '')
        elif '.' in monto: monto = monto.replace('.', ',')
    else: monto = monto.replace('.', '').replace(',', '')
    if re.search(r',00$', monto): monto = re.sub(r',00$', '', monto)
    return monto.strip()

def pdf_a_imagen_png(pdf_bytes: bytes) -> bytes:
    import fitz
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
    return pix.tobytes("png")

def extraer_datos_con_vision_api(archivo_contenido: bytes, nombre_archivo: str, tipo_archivo: str) -> dict:
    # Busca la API Key de manera segura
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["ANTHROPIC_API_KEY"]
        except:
            pass

    if not api_key: return {"emisor": "ERROR", "monto": "0", "id_operacion": ""}
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        if tipo_archivo == 'application/pdf' or nombre_archivo.lower().endswith('.pdf'):
            archivo_contenido = pdf_a_imagen_png(archivo_contenido)
            media_type = 'image/png'
        else: media_type = 'image/jpeg'
        base64_data = base64.b64encode(archivo_contenido).decode('utf-8')
        message = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=1024,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": base64_data}},
                {"type": "text", "text": 'Extrae en JSON: emisor (quien envía), monto, destinatario (quien recibe), id_operacion (solo nros/letras sin "ID"). ATENCIÓN: En Argentina usamos puntos para los miles. Lee con cuidado los ceros.'}
            ]}]
        )
        res = message.content[0].text.strip().replace('```json', '').replace('```', '').strip()
        return json.loads(res)
    except: return {"emisor": "ERROR_LECTURA", "monto": "0", "id_operacion": ""}

def generar_asientos_doble_partida(emisor: str, monto: str, id_op: str, cuenta: str) -> dict:
    emisor_str = str(emisor).replace(",", "").strip() if emisor and str(emisor) != "None" else (str(id_op).strip() if id_op else "FALTA_NOMBRE")
    id_op_str = str(id_op).strip() if id_op and str(id_op) != "None" else ""
    monto_limpio = "".join(limpiar_monto(monto).split()) 
    asientos = {"Nexo": "", "Cliente": ""}
    
    if cuenta in ["Giardino", "Cta Cte"]:
        asientos["Nexo"] = f',,,,,,,,,,{"".join(monto_limpio)}' 
        asientos["Cliente"] = f',,,,,,,{"".join(monto_limpio)}' 
    elif cuenta in ["Canella", "Vertice"]:
        asientos["Nexo"] = f'"{emisor_str}",,,,,,,,{monto_limpio}' 
        asientos["Cliente"] = f'"{emisor_str}",,,,,,,,,{monto_limpio}' 
    elif cuenta == "Celso":
        asientos["Nexo"] = f'"{emisor_str}",,,,,,,,{monto_limpio}' 
        # COMAS CORREGIDAS PARA CELSO
        asientos["Cliente"] = f'"{emisor_str}",{id_op_str},,,,,,,,,{monto_limpio}' 
    elif cuenta == "Nexo Directo (Pega en Col C)":
        asientos["Nexo"] = f'"{emisor_str}",,,,,,,,{monto_limpio}' 
    return asientos

# =============================================================================
# INTERFAZ PRINCIPAL
# =============================================================================
def main():
    st.title("⚙️ Extractor Manual (ZIP/Fotos)")
    st.write("Subí tus archivos para generar los textos del Excel.")
    
    opciones_cliente = ["Celso", "Canella", "Vertice", "Giardino", "Cta Cte", "Nexo Directo (Pega en Col C)"]
    cliente_seleccionado = st.selectbox("Elige el cliente para todo este lote:", opciones_cliente, key="sel_manual")
    
    archivos_subidos = st.file_uploader("Selecciona archivos", type=['jpg', 'jpeg', 'png', 'pdf', 'zip'], accept_multiple_files=True)
    
    if archivos_subidos:
        if st.button("🚀 Procesar Lote Manual", type="primary"):
            
            archivos_a_procesar = []
            for archivo_subido in archivos_subidos:
                if archivo_subido.name.lower().endswith('.zip'):
                    with zipfile.ZipFile(archivo_subido) as z:
                        for n in z.namelist():
                            if not n.endswith('/') and '__MACOSX' not in n:
                                archivos_a_procesar.append((n, z.read(n), 'image/jpeg'))
                else: archivos_a_procesar.append((archivo_subido.name, archivo_subido.read(), archivo_subido.type))

            resultados_limpios = []
            resultados_duplicados = []
            ids_vistos_en_lote = set()
            
            progreso = st.progress(0)
            status_text = st.empty()
            
            for idx, (nombre, contenido, tipo_mime) in enumerate(archivos_a_procesar):
                status_text.text(f"Procesando {idx + 1}/{len(archivos_a_procesar)}: {nombre}")
                
                datos_ext = extraer_datos_con_vision_api(contenido, nombre, tipo_mime)
                id_operacion = str(datos_ext.get("id_operacion", ""))
                
                res = {
                    "archivo": nombre,
                    "emisor": str(datos_ext.get("emisor", "")),
                    "id_operacion": id_operacion,
                    "monto": str(datos_ext.get("monto", "0"))
                }
                
                # ESCUDO ANTI-DUPLICADOS (DENTRO DEL MISMO LOTE)
                if id_operacion and id_operacion != "ERROR" and id_operacion in ids_vistos_en_lote:
                    res["motivo"] = "Repetido adentro de este mismo ZIP"
                    resultados_duplicados.append(res)
                else:
                    if id_operacion and id_operacion != "ERROR":
                        ids_vistos_en_lote.add(id_operacion)
                    resultados_limpios.append(res)
                    
                progreso.progress((idx + 1) / len(archivos_a_procesar))
                time.sleep(2) # Freno para no saturar la API
            
            progreso.empty()
            status_text.empty()
            st.success("✅ Extracción manual completada.")
            
            if resultados_duplicados:
                st.error(f"🚨 ATENCIÓN: Se omitieron {len(resultados_duplicados)} comprobantes duplicados en este lote.")
                for dup in resultados_duplicados:
                    st.markdown(f"- 📄 `{dup['archivo']}` | 🆔 **{dup['id_operacion']}**")
            
            if resultados_limpios:
                st.info(f"Armando textos para {len(resultados_limpios)} comprobantes válidos...")
                bloques = {"Nexo": [], "Cliente": []}
                for r in resultados_limpios:
                    asientos = generar_asientos_doble_partida(r["emisor"], r["monto"], r["id_operacion"], cliente_seleccionado)
                    if asientos["Nexo"]: bloques["Nexo"].append(asientos["Nexo"])
                    if asientos["Cliente"]: bloques["Cliente"].append(asientos["Cliente"])
                
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Hoja: NEXO")
                    st.code("\n".join(bloques["Nexo"]), language=None)
                with c2:
                    st.subheader(f"Hoja: {cliente_seleccionado}")
                    st.code("\n".join(bloques["Cliente"]), language=None)

if __name__ == "__main__":
    main()
