"""Separación de resultados por archivos y asignaciones confirmadas.

La identidad incluye huellas SHA-256 de los bytes de los archivos de origen,
hojas, fila de cabecera y asignaciones de columnas confirmadas. Nunca modifica
los archivos fuente ni utiliza la fecha de ejecución para identificar el conjunto.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import unicodedata


def _huella_archivo(archivo: Path) -> str:
    h = hashlib.sha256()
    with archivo.open('rb') as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b''):
            h.update(bloque)
    return h.hexdigest()


def _nombre_seguro(nombre: str) -> str:
    normalizado = unicodedata.normalize('NFKD', nombre)
    basico = normalizado.encode('ascii', 'ignore').decode('ascii')
    return re.sub(r'[^A-Za-z0-9_-]+', '_', basico).strip('_')[:48] or 'datos'


def identificar_conjunto(resultado: dict) -> tuple[str, dict]:
    """Identificador estable y manifiesto de las fuentes confirmadas."""
    fuentes = resultado.get('fuentes')
    if not fuentes:
        raise ValueError('No están identificados los archivos de origen confirmados.')
    elementos = []
    for fuente in fuentes:
        archivo = Path(fuente.archivo)
        if not archivo.is_file():
            raise FileNotFoundError(f'No se encuentra la fuente: {archivo}')
        columnas = getattr(fuente, 'columnas', None)
        if not isinstance(columnas, dict) or not {'fecha', 'producto', 'cantidad'}.issubset(columnas):
            raise ValueError(f'Falta confirmar la correspondencia de columnas de {archivo.name}.')
        elemento = {
            'archivo': archivo.name,
            'sha256_archivo': _huella_archivo(archivo),
            'hoja': fuente.hoja,
            'fila_cabecera': int(fuente.fila_cabecera),
            'columnas': {str(c): int(i) for c, i in sorted(columnas.items())},
            'identificador_indice': getattr(fuente, 'identificador_indice', None),
            'estado_indice': getattr(fuente, 'estado_indice', None),
        }
        elementos.append(elemento)
    # La selección múltiple en diferente orden no debe producir dos carpetas.
    elementos.sort(key=lambda e: json.dumps(e, ensure_ascii=False, sort_keys=True))
    manifest = {'version_identidad': 1, 'fuentes': elementos}
    datos = json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                       separators=(',', ':')).encode('utf-8')
    digest = hashlib.sha256(datos).hexdigest()
    manifest['huella_conjunto'] = digest
    archivos_unicos = {e['archivo'] for e in elementos}
    nombre = next(iter(archivos_unicos)) if len(archivos_unicos) == 1 else 'archivos_combinados'
    nombre_base = _nombre_seguro(Path(nombre).stem)
    return f'{nombre_base}__{digest[:16]}', manifest


def crear_carpeta_salidas(resultado: dict, raiz_salidas: str | Path) -> Path:
    """Crea o reutiliza la carpeta del conjunto; no escribe predicciones ni CSV."""
    identificador, manifest = identificar_conjunto(resultado)
    carpeta = Path(raiz_salidas) / identificador
    manifiesto = carpeta / 'identidad_conjunto.json'
    if manifiesto.exists():
        previo = json.loads(manifiesto.read_text(encoding='utf-8'))
        if previo != manifest:
            raise ValueError('La carpeta existente tiene una identidad de datos diferente.')
        return carpeta
    if carpeta.exists() and any(carpeta.iterdir()):
        raise ValueError(f'Carpeta de resultados ocupada sin identidad válida: {carpeta}')
    carpeta.mkdir(parents=True, exist_ok=True)
    temporal = carpeta / 'identidad_conjunto.json.tmp'
    temporal.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
                        encoding='utf-8')
    temporal.replace(manifiesto)
    return carpeta
