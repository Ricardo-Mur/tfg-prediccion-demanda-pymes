# Sistema web de evaluación y predicción de demanda para pymes

Aplicación local para cargar históricos de ventas en CSV o Excel, confirmar datos ambiguos mediante revisión agrupada, calcular previsiones mensuales y consultar sus limitaciones. No requiere publicar ni enviar los datos a un servidor externo.

## Requisitos

- Windows 11 o Linux y **Python 3.13**.
- Dependencias indicadas en `requisitos.txt`. Para reproducir exactamente una ejecución deben coincidir el sistema, las versiones y el número de procesos utilizado.

## Instalación

Desde la carpeta del proyecto, en una terminal:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requisitos.txt
.venv\Scripts\python.exe -m streamlit run interfaz\app.py
```

Para Linux, después de crear y activar el entorno virtual: `python -m pip install -r requisitos.txt` y `python -m streamlit run interfaz/app.py`.

También puede ejecutarse `iniciar_interfaz.bat` desde Windows cuando el entorno esté activado y `python` corresponda a ese entorno.

## Primer uso

La aplicación se abre en el navegador del propio equipo. Se puede pulsar **Datos de ejemplo** para generar entradas completamente ficticias, o cargar CSV/Excel propios. Revisa las propuestas de columnas, aclara las operaciones que conozcas y consulta los avisos antes de calcular. La aplicación presenta resultados por productos, familias y total, cuando los datos lo permiten.

## Integridad del motor

`predecir_v2.py` y los módulos incluidos en `src/` mantienen su contenido y huellas SHA-256 originales de la congelación del 15/11/2026. `congelacion_v2.json` de esta distribución contiene **solo las entradas de código incluidas**; las pruebas históricas y los conjuntos reservados no se distribuyen aquí. No representa por sí solo la auditoría histórica completa de 73 archivos.

`SHA256SUMS_PUBLICACION.txt` permite comprobar la integridad de los archivos publicados. Esta copia es independiente de la carpeta de trabajo original.

## Estado

La interfaz corresponde a una **versión preliminar funcional**. Existe una limitación de navegación detectada tras realizar un segundo cálculo y regresar a algunas secciones. El motor predictivo no se ha cambiado para esta distribución.
