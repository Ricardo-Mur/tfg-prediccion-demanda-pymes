"""Espacio de ejecución aislado para el programa congelado.

`predecir_v2.py` lee siempre de la carpeta `datos_prueba` que tiene al lado y escribe en `salidas_v2`, también
al lado. No admite otras carpetas, y no se puede modificar. Por eso cada sesión trabaja en un espacio propio,
fuera del proyecto:

    <carpeta temporal>/tfg_prediccion_espacios/espacio_<id>/
        .espacio_tfg                 marca: solo se borra una carpeta que la tenga
        programa/predecir_v2.py      copia idéntica del código congelado (se comprueba con sus huellas)
        programa/src/*.py
        programa/datos_prueba/       los archivos de ventas del usuario
        programa/salidas_v2/         lo que escribe el programa
        familias/                    el archivo de familias, si lo hay

En el espacio solo entra el código que el programa necesita para ejecutarse. No entran los datos reservados,
ni los datos de prueba del proyecto, ni las pruebas, ni la evaluación. Al terminar se borra entero.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

PROYECTO = Path(__file__).resolve().parent.parent
RAIZ_ESPACIOS = Path(tempfile.gettempdir()) / 'tfg_prediccion_espacios'
MARCA = '.espacio_tfg'
CERROJO = 'en_curso.lock'
EXTENSIONES = ('.csv', '.xlsx')


class EspacioNoValido(RuntimeError):
    """El espacio no es lo que debe ser: no se ejecuta nada en él, o no se borra."""


def huella(ruta: Path) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def codigo_congelado(proyecto: Path = PROYECTO) -> dict[str, str]:
    """Los archivos que el programa necesita para ejecutarse, con su huella congelada: predecir_v2.py y src/*.py."""
    congelados = json.loads((proyecto / 'congelacion_v2.json').read_text(encoding='utf-8'))['archivos']
    return {r: h for r, h in congelados.items() if r == 'predecir_v2.py' or (r.startswith('src/') and r.count('/') == 1)}


def _nombre_seguro(nombre: str, extensiones: tuple[str, ...]) -> str:
    limpio = Path(str(nombre).replace('\\', '/')).name.strip()
    if not limpio or limpio.startswith('.') or Path(limpio).suffix.lower() not in extensiones:
        raise ValueError(f'No puedo usar el archivo «{nombre}»: tiene que ser un ' + ' o un '.join(extensiones) + '.')
    return limpio


class Espacio:
    def __init__(self, ruta: Path):
        self.ruta = Path(ruta)
        self.programa = self.ruta / 'programa'
        self.entradas = self.programa / 'datos_prueba'
        self.salidas = self.programa / 'salidas_v2'
        self.familias = self.ruta / 'familias'

    # ------------------------------------------------------------------ creación

    @classmethod
    def crear(cls, raiz: Path | None = None, proyecto: Path = PROYECTO) -> 'Espacio':
        raiz = Path(raiz) if raiz is not None else RAIZ_ESPACIOS
        raiz.mkdir(parents=True, exist_ok=True)
        identificador = uuid.uuid4().hex
        espacio = cls(raiz / f'espacio_{identificador}')
        espacio.ruta.mkdir()                                      # nunca se reutiliza una carpeta que ya exista
        (espacio.ruta / MARCA).write_text(json.dumps({'id': identificador, 'creado': time.time()}), encoding='utf-8')
        for relativo in codigo_congelado(proyecto):
            destino = espacio.programa / relativo
            destino.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(proyecto / relativo, destino)
        espacio.entradas.mkdir()
        espacio.comprobar_codigo(proyecto)
        return espacio

    def comprobar_codigo(self, proyecto: Path = PROYECTO) -> None:
        """El código del espacio es exactamente el congelado: ni un archivo distinto, ni uno de más, ni uno de menos."""
        esperado = codigo_congelado(proyecto)
        presentes = {p.relative_to(self.programa).as_posix() for p in self.programa.rglob('*.py')}
        if presentes != set(esperado):
            raise EspacioNoValido('El código del espacio de ejecución no es el congelado: sobran o faltan archivos '
                                  f'({sorted(presentes ^ set(esperado))[:5]}).')
        distintos = [r for r, h in esperado.items() if huella(self.programa / r) != h]
        if distintos:
            raise EspacioNoValido(f'El código del espacio de ejecución no coincide con el congelado: {distintos[:5]}.')

    # ------------------------------------------------------------------ archivos del usuario

    def guardar_ventas(self, nombre: str, contenido: bytes) -> Path:
        destino = self.entradas / _nombre_seguro(nombre, EXTENSIONES)
        if destino.exists():
            raise ValueError(f'Ya hay un archivo que se llama «{destino.name}».')
        destino.write_bytes(contenido)
        return destino

    def quitar_ventas(self, nombre: str) -> None:
        destino = self.entradas / _nombre_seguro(nombre, EXTENSIONES)
        if destino.is_file() and not destino.is_symlink():
            destino.unlink()

    def archivos_de_ventas(self) -> list[Path]:
        """En el mismo orden en que los enumera el programa."""
        return sorted(p for p in self.entradas.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONES)

    def guardar_familias(self, nombre: str, contenido: bytes) -> Path:
        self.familias.mkdir(exist_ok=True)
        destino = self.familias / _nombre_seguro(nombre, EXTENSIONES)
        destino.write_bytes(contenido)
        return destino

    # ------------------------------------------------------------------ resultados

    def carpeta_de_resultados(self) -> Path | None:
        carpetas = [p for p in self.salidas.iterdir() if p.is_dir()] if self.salidas.is_dir() else []
        return carpetas[0] if len(carpetas) == 1 else None

    def zip_de_resultados(self) -> bytes:
        """Los archivos que ha escrito el programa, tal cual, en un ZIP en memoria: se puede descargar y después
        borrar el espacio."""
        carpeta = self.carpeta_de_resultados()
        if carpeta is None:
            raise EspacioNoValido('No hay resultados que descargar.')
        memoria = io.BytesIO()
        with zipfile.ZipFile(memoria, 'w', zipfile.ZIP_DEFLATED) as z:
            for archivo in sorted(p for p in carpeta.iterdir() if p.is_file()):
                z.write(archivo, f'{carpeta.name}/{archivo.name}')
        return memoria.getvalue()

    def tocar(self) -> None:
        """Anota que la sesión sigue en uso: un espacio con actividad reciente no se considera abandonado."""
        try:
            os.utime(self.ruta / MARCA)
        except OSError:
            pass

    def contenido(self) -> list[str]:
        return sorted(p.relative_to(self.ruta).as_posix() for p in self.ruta.rglob('*') if p.is_file())

    # ------------------------------------------------------------------ borrado

    def es_borrable(self) -> None:
        """Solo se borra una carpeta creada por esta clase: con su marca, con su nombre, que no sea un enlace y que
        no tenga una ejecución en curso."""
        if self.ruta.is_symlink() or not self.ruta.is_dir():
            raise EspacioNoValido('El espacio no es una carpeta: no se borra.')
        try:
            marca = json.loads((self.ruta / MARCA).read_text(encoding='utf-8'))
        except (OSError, ValueError):
            raise EspacioNoValido('La carpeta no tiene la marca de un espacio de ejecución: no se borra.') from None
        if self.ruta.name != f'espacio_{marca.get("id")}':
            raise EspacioNoValido('La marca no corresponde a esta carpeta: no se borra.')
        if (self.ruta / CERROJO).exists():
            raise EspacioNoValido('Hay una ejecución en curso en este espacio: primero hay que cancelarla.')

    def eliminar(self) -> None:
        """Borra el espacio entero: código copiado, archivos del usuario, resultados y registros."""
        self.es_borrable()
        for intento in range(5):          # en Windows, un archivo recién cerrado puede tardar un instante en soltarse
            try:
                shutil.rmtree(self.ruta)  # no sigue enlaces: borra el enlace, no aquello a lo que apunta
                break
            except OSError:
                if intento == 4:
                    raise
                time.sleep(0.3)
        if self.ruta.exists():
            raise EspacioNoValido('El espacio no se ha podido borrar del todo.')


def espacios_abandonados(raiz: Path | None = None, horas: float = 24.0) -> list[Espacio]:
    """Espacios de sesiones que nadie cerró: con marca, sin ejecución en curso y sin actividad desde hace más de
    `horas`. La actividad es la última vez que la sesión usó el espacio (`tocar`), no cuándo se creó."""
    raiz = Path(raiz) if raiz is not None else RAIZ_ESPACIOS
    viejos = []
    for carpeta in (sorted(raiz.glob('espacio_*')) if raiz.is_dir() else []):
        espacio = Espacio(carpeta)
        try:
            espacio.es_borrable()
            creado = json.loads((carpeta / MARCA).read_text(encoding='utf-8')).get('creado', 0)
            actividad = max(float(creado), (carpeta / MARCA).stat().st_mtime)
        except (EspacioNoValido, OSError, ValueError):
            continue
        if time.time() - actividad > horas * 3600:
            viejos.append(espacio)
    return viejos
