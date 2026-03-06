import streamlit as st
import pandas as pd

st.set_page_config(page_title="SIDERA - Control", page_icon="📝", layout="wide")
st.title("📝 Control de Transferencias Pendientes")
st.markdown("Anotá acá las transferencias solicitadas. Este cuadro mantiene su estado.")

if 'pendientes' not in st.session_state:
    st.session_state.pendientes = pd.DataFrame([{"Listo": False, "Titular": "", "Monto": ""}] * 10)

edited_pendientes = st.data_editor(
    st.session_state.pendientes,
    key="editor_maestro",
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
