#!/bin/sh
# Abre la interfaz en el navegador. Con el entorno del proyecto activado.
cd "$(dirname "$0")" && python -m streamlit run interfaz/app.py
