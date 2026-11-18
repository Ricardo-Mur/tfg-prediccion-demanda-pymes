@echo off
rem Abre la interfaz en el navegador. Hay que ejecutarlo desde la carpeta del proyecto, con su entorno activado.
cd /d "%~dp0"
python -m streamlit run interfaz\app.py
