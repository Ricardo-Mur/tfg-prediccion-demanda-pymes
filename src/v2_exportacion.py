"""Versión 2: cómo se escriben los archivos de salida. No calcula nada.

Textos que no se abren como fórmulas
------------------------------------
Los códigos de producto, los nombres de familia, los números de factura y cualquier otro texto vienen de archivos
ajenos al programa. Si uno empieza por `=`, `+`, `-` o `@` (o por un tabulador o un retorno de carro), Excel o
LibreOffice lo tratarían como una fórmula al abrir el CSV. Para evitarlo, al escribir un CSV:

    a todo texto que empieza por uno de esos caracteres, o por un apóstrofo, se le antepone un apóstrofo (').

El apóstrofo también se antepone a los textos que ya empezaban por apóstrofo para que la regla se pueda deshacer
sin dudas: para recuperar el texto original basta con quitar el primer apóstrofo de los textos que empiezan por
apóstrofo. Dos textos distintos siguen siendo distintos después de protegerlos, así que dos productos nunca se
confunden. `leer_csv` deshace la regla.

- Solo se protege lo que se escribe en los CSV. Dentro del programa, y en los archivos JSON (que no se abren como
  hojas de cálculo), los códigos son los originales.
- Los números no se tocan: una cantidad negativa es un número, no un texto.

Redondeo
--------
Los importes y los precios se redondean solo al escribirlos (`decimales`). Los cálculos usan todos los decimales.

Entorno
-------
`guardar_entorno` anota la versión de Python y de las librerías y la huella SHA-256 de cada archivo de código que
el programa ha cargado, para saber con qué se calculó cada resultado.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime
from importlib import metadata
from pathlib import Path

import pandas as pd

from . import motor_v2 as M

MARCA = "'"
INICIOS_DE_FORMULA = ('=', '+', '-', '@', '\t', '\r')
# Columnas que son textos aunque parezcan números (códigos): al leer un CSV exportado se leen como texto.
COLUMNAS_DE_TEXTO = ('producto', 'grupo', 'columna', 'id_factura', 'id_linea', 'serie', 'grupos_sin_total', 'filas')
LIBRERIAS = ('numpy', 'pandas', 'statsmodels', 'xgboost', 'scipy', 'scikit-learn', 'openpyxl')
RAIZ = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ textos seguros

def texto_seguro(valor):
    """El valor tal como se escribe en un CSV: con un apóstrofo delante si es un texto que una hoja de cálculo
    podría tomar por una fórmula (o que ya empieza por apóstrofo). Lo que no es texto no cambia."""
    if isinstance(valor, str) and valor and (valor[0] in INICIOS_DE_FORMULA or valor[0] == MARCA):
        return MARCA + valor
    return valor


def texto_original(valor):
    """Deshace `texto_seguro`: quita el primer apóstrofo de un texto que empieza por apóstrofo."""
    if isinstance(valor, str) and valor.startswith(MARCA):
        return valor[1:]
    return valor


def proteger(tabla: pd.DataFrame) -> pd.DataFrame:
    """Copia de la tabla con sus textos (y sus cabeceras) protegidos. Si no hay nada que proteger, la misma tabla."""
    columnas = {c: texto_seguro(c) for c in tabla.columns}
    cambian = [c for c in tabla.columns if tabla[c].dtype.kind in 'OTU' or str(tabla[c].dtype) in ('str', 'string')]
    cambian = [c for c in cambian if any(isinstance(v, str) and texto_seguro(v) != v for v in tabla[c])]
    if not cambian and all(k == v for k, v in columnas.items()):
        return tabla
    t = tabla.copy()
    for c in cambian:
        t[c] = t[c].map(texto_seguro)
    return t.rename(columns=columnas)


def escribir_csv(tabla: pd.DataFrame, ruta: Path, *, decimales: dict[str, int] | None = None) -> Path:
    """Escribe un CSV del programa: punto y coma, coma decimal, UTF-8 con marca, textos protegidos.

    `decimales`: columna -> número de decimales con los que se escribe. La tabla que se recibe no cambia.
    """
    if decimales:
        redondear = {c: n for c, n in decimales.items() if c in tabla.columns and tabla[c].dtype.kind == 'f'}
        if redondear:
            tabla = tabla.assign(**{c: tabla[c].round(n) for c, n in redondear.items()})
    proteger(tabla).to_csv(ruta, sep=';', decimal=',', encoding='utf-8-sig', index=False)
    return Path(ruta)


def leer_csv(ruta: Path, *, texto: tuple[str, ...] = ()) -> pd.DataFrame:
    """Lee un CSV escrito por el programa y devuelve los textos originales (sin el apóstrofo de protección).

    Las columnas de códigos (`COLUMNAS_DE_TEXTO`, las que empiezan por `valor_en_` y las indicadas en `texto`) se
    leen como texto, para que «007» no se convierta en 7. Solo una celda vacía se toma como dato ausente.
    """
    cabeceras = list(pd.read_csv(ruta, sep=';', encoding='utf-8-sig', nrows=0).columns)
    como_texto = {c: str for c in cabeceras
                  if texto_original(c) in COLUMNAS_DE_TEXTO + tuple(texto) or texto_original(c).startswith('valor_en_')}
    t = pd.read_csv(ruta, sep=';', decimal=',', encoding='utf-8-sig', dtype=como_texto, keep_default_na=False, na_values=[''])
    for c in t.columns:
        if t[c].dtype.kind in 'OTU' or str(t[c].dtype) in ('str', 'string'):
            t[c] = t[c].map(texto_original)
    return t.rename(columns={c: texto_original(c) for c in t.columns})


# ------------------------------------------------------------------ resultados del motor

def exportar_motor(informe: M.InformeV2, carpeta: Path) -> dict[str, Path]:
    """Los mismos archivos que `motor_v2.exportar_v2`, con los textos protegidos y, en `candidatos_v2.csv`, el mes
    como fecha (como en los demás archivos) en vez de como número de mes."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    candidatos = informe.candidatos
    if len(candidatos):
        candidatos = candidatos.assign(mes=[M.fecha(int(m)) for m in candidatos.mes])
    rutas = {}
    for nombre, tabla in (('prevision_v2', informe.resumen), ('historial_v2', informe.historial),
                          ('validacion_v2', informe.validacion), ('candidatos_v2', candidatos),
                          ('candidatos_excluidos_v2', informe.excluidos)):
        rutas[nombre] = escribir_csv(tabla, carpeta / f'{nombre}.csv')
    rutas['configuracion'] = carpeta / 'configuracion_v2.json'
    rutas['configuracion'].write_text(json.dumps(
        asdict(informe.configuracion) | {'mes_previsto': str(informe.mes_previsto)}, ensure_ascii=False, indent=2),
        encoding='utf-8')
    return rutas


# ------------------------------------------------------------------ entorno y carpeta de salidas

def huella(ruta: Path) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def entorno(raiz: Path = RAIZ) -> dict:
    """Versión de Python y de las librerías, y huella de cada archivo de código del proyecto que está cargado."""
    versiones = {}
    for nombre in LIBRERIAS:
        try:
            versiones[nombre] = metadata.version(nombre)
        except metadata.PackageNotFoundError:
            versiones[nombre] = None
    raiz = Path(raiz).resolve()
    codigo = {}
    for modulo in list(sys.modules.values()):
        archivo = getattr(modulo, '__file__', None)
        if not archivo:
            continue
        ruta = Path(archivo).resolve()
        if ruta.suffix == '.py' and raiz in ruta.parents and ruta.exists():
            codigo[ruta.relative_to(raiz).as_posix()] = huella(ruta)
    return {'python': platform.python_version(), 'implementacion': platform.python_implementation(),
            'sistema': platform.system(), 'librerias': versiones, 'codigo_sha256': dict(sorted(codigo.items()))}


def guardar_entorno(carpeta: Path, raiz: Path = RAIZ) -> dict[str, Path]:
    ruta = Path(carpeta) / 'entorno_v2.json'
    ruta.write_text(json.dumps(entorno(raiz), ensure_ascii=False, indent=2), encoding='utf-8')
    return {'entorno': ruta}


def carpeta_nueva(raiz: Path, ahora: datetime | None = None) -> Path:
    """Crea una carpeta de salidas nueva con la fecha y la hora. Nunca reutiliza una que ya exista: si dos
    ejecuciones coinciden en el mismo segundo, la segunda recibe un sufijo."""
    base = (ahora or datetime.now()).strftime('%Y%m%d_%H%M%S')
    raiz = Path(raiz)
    raiz.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        carpeta = raiz / (base if n == 1 else f'{base}_{n}')
        try:
            carpeta.mkdir()
            return carpeta
        except FileExistsError:
            n += 1
