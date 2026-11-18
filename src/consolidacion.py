"""Fase 3.3: unión conservadora de históricos de ventas de varias fuentes.

Los archivos originales se conservan. Cada fila puede rastrearse hasta su
archivo, hoja y posición en la tabla cargada. Se normalizan SOLO las tres
columnas esenciales (descripción opcional) sin asumir identidad comercial
entre referencias ni eliminar coincidencias entre ficheros.
"""
from __future__ import annotations

from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from itertools import combinations
from pathlib import Path

import pandas as pd

from .carga import seleccionar_campos
from .preparacion_historico import _cantidad_finita


@dataclass
class FuenteVentas:
    archivo: Path
    hoja: str | None
    fila_cabecera: int
    datos_originales: pd.DataFrame
    columnas: dict[str, int]
    identificador_indice: int | None = None
    estado_indice: int | None = None


@dataclass
class DatosConsolidados:
    tabla: pd.DataFrame
    originales: pd.DataFrame
    trazabilidad: pd.DataFrame
    identificadores: pd.Series | None
    estados_movimiento: pd.Series | None
    nombres: list[str]
    archivos_unicos: int


@dataclass
class InformeConsolidacion:
    archivos: int
    tablas: int
    filas_por_archivo: dict[str, int]
    fechas_por_archivo: dict[str, tuple[date | None, date | None]]
    periodos_solapados: list[tuple[str, str]]
    meses_sin_registros: int
    coincidencias_entre_archivos: int
    grupos_coincidentes: int
    ejemplos_coincidentes: list[dict]
    filas_con_origen: int


def consolidar_fuentes(fuentes: list[FuenteVentas]) -> DatosConsolidados:
    """Unifica columnas confirmadas conservando la relación con cada origen.

    No presupone que columnas extra estén presentes en todos los archivos:
    los registros de origen siguen disponibles individualmente en `fuentes`.
    Para auditar filas idénticas dentro del mismo archivo, se añade la fuente
    pero nunca el número de fila (que las haría artificialmente únicas).
    """
    if not fuentes:
        raise ValueError('Selecciona al menos un archivo de ventas.')
    # Un Excel puede contener varias hojas de ventas. Cada una es una fuente
    # lógica distinta, aunque todas procedan del mismo archivo físico.
    rutas = [str(f.archivo.resolve()) for f in fuentes]
    repeticiones = Counter(rutas)
    nombres = [
        (f'{f.archivo.name} [{f.hoja}]' if repeticiones[rutas[i]] > 1
         else f.archivo.name)
        for i, f in enumerate(fuentes)
    ]
    claves = [(rutas[i], f.hoja) for i, f in enumerate(fuentes)]
    if len(claves) != len(set(claves)):
        raise ValueError('Se ha seleccionado dos veces la misma hoja.')
    if len(nombres) != len(set(nombres)):
        raise ValueError('Hay dos fuentes distintas con el mismo nombre.')
    normalizados: list[pd.DataFrame] = []
    originales: list[pd.DataFrame] = []
    trazas: list[pd.DataFrame] = []
    ids: list[pd.Series] = []
    estados: list[pd.Series] = []
    hay_ids = hay_estados = False
    for f, nombre_fuente in zip(fuentes, nombres):
        raw = f.datos_originales.reset_index(drop=True)
        if not len(raw):
            raise ValueError(f'{f.archivo.name} no contiene transacciones.')
        normal = seleccionar_campos(raw, f.columnas)
        if 'descripcion' not in normal:
            normal['descripcion'] = pd.NA
        normalizados.append(normal[['fecha', 'producto', 'cantidad', 'descripcion']])
        original = raw.copy()
        col_archivo = '__tfg_origen_archivo__'
        if col_archivo in original.columns:
            raise ValueError(f'El archivo contiene una columna reservada: {col_archivo}')
        original[col_archivo] = f.archivo.name
        col_hoja = '__tfg_origen_hoja__'
        if col_hoja in original.columns:
            raise ValueError(f'El archivo contiene una columna reservada: {col_hoja}')
        original[col_hoja] = f.hoja or 'CSV'
        originales.append(original)
        trazas.append(pd.DataFrame({
            'archivo': [f.archivo.name] * len(raw),
            'fuente': [nombre_fuente] * len(raw),
            'hoja': [f.hoja or 'CSV'] * len(raw),
            'fila_en_tabla': range(1, len(raw) + 1),
            'fila_cabecera': [f.fila_cabecera + 1] * len(raw),
        }))
        if f.identificador_indice is not None:
            hay_ids = True
            ids.append(raw.iloc[:, f.identificador_indice].copy())
        else:
            ids.append(pd.Series([pd.NA]*len(raw), dtype=object))
        if f.estado_indice is not None:
            hay_estados = True
            estados.append(raw.iloc[:, f.estado_indice].copy())
        else:
            estados.append(pd.Series([pd.NA]*len(raw), dtype=object))
    return DatosConsolidados(
        tabla=pd.concat(normalizados, ignore_index=True),
        originales=pd.concat(originales, ignore_index=True, sort=False),
        trazabilidad=pd.concat(trazas, ignore_index=True),
        identificadores=(pd.concat(ids, ignore_index=True) if hay_ids else None),
        estados_movimiento=(pd.concat(estados, ignore_index=True) if hay_estados else None),
        nombres=nombres,
        archivos_unicos=len(set(rutas)),
    )


def _id_texto(valor: object) -> str | None:
    if valor is None:
        return None
    try:
        if bool(pd.isna(valor)):
            return None
    except (TypeError, ValueError):
        return None
    texto = str(valor).strip()
    return texto if texto else None


def analizar_consolidacion(datos: DatosConsolidados, limpieza: object,
                          max_ejemplos: int = 5) -> InformeConsolidacion:
    """Detecta solapamientos y posibles duplicados entre ficheros.

    - Solapamiento: intersección de rangos entre primera/última fecha registrada
      (no prueba que existan ventas todos los días intermedios).
    - Coincidencia: fecha-hora, código y cantidad iguales en archivos distintos.
      Si ambos tienen identificador de operación, exige que coincidan; sin
      identificadores, queda como mera coincidencia a revisar.
    """
    copia = limpieza.datos
    if len(copia) != len(datos.trazabilidad):
        raise ValueError('No coincide el tamaño de la limpieza y la trazabilidad.')
    filas_por_archivo = datos.trazabilidad['fuente'].value_counts().reindex(datos.nombres).fillna(0).astype(int).to_dict()
    rangos: dict[str, tuple[date | None, date | None]] = {}
    fechas_por_fuente: dict[str, set[date]] = defaultdict(set)
    agrupadas: dict[tuple[pd.Timestamp, str, Decimal], list[tuple[int, str, str | None]]] = defaultdict(list)
    meses: set[tuple[int, int]] = set()
    ids = (datos.identificadores.tolist() if datos.identificadores is not None
           else [None] * len(copia))
    for i, (fecha, producto, cantidad, archivo, id_op) in enumerate(zip(
        copia['fecha'].tolist(), copia['producto'].tolist(), copia['cantidad'].tolist(),
        datos.trazabilidad['fuente'].tolist(), ids,
    )):
        if isinstance(fecha, pd.Timestamp) and not pd.isna(fecha):
            dia = fecha.date()
            fechas_por_fuente[archivo].add(dia)
            meses.add((dia.year, dia.month))
            valor = _cantidad_finita(cantidad)
            try:
                codigo_ausente = bool(pd.isna(producto))
            except (TypeError, ValueError):
                codigo_ausente = False
            if not codigo_ausente and str(producto).strip() and valor is not None:
                clave = (fecha, str(producto).strip(), valor)
                agrupadas[clave].append((i, archivo, _id_texto(id_op)))
    for nombre in datos.nombres:
        observadas = fechas_por_fuente.get(nombre, set())
        rangos[nombre] = ((min(observadas), max(observadas)) if observadas else (None, None))
    solapamientos: list[tuple[str, str]] = []
    for a, b in combinations(datos.nombres, 2):
        a0, a1 = rangos[a]
        b0, b1 = rangos[b]
        if a0 is not None and b0 is not None and max(a0,b0) <= min(a1,b1):
            solapamientos.append((a, b))

    implicadas: set[int] = set()
    grupos = 0
    muestras: list[dict] = []
    for clave, registros in agrupadas.items():
        por_fuente: dict[str, list[tuple[int,str|None]]] = defaultdict(list)
        for pos, archivo, identificador in registros:
            por_fuente[archivo].append((pos, identificador))
        if len(por_fuente) <= 1:
            continue
        ids_por_fuente = {fuente: {idv for _,idv in filas if idv is not None}
                          for fuente,filas in por_fuente.items()}
        tiene_sin_id = {fuente: any(idv is None for _,idv in filas)
                        for fuente,filas in por_fuente.items()}
        validas: list[tuple[int,str]] = []
        for fuente, filas in por_fuente.items():
            for pos, identificador in filas:
                for otra in por_fuente:
                    if otra == fuente:
                        continue
                    # Si ambas operaciones tienen id distinto, no hay
                    # evidencia suficiente de duplicación de la transacción.
                    if (identificador is None or tiene_sin_id[otra]
                            or identificador in ids_por_fuente[otra]):
                        validas.append((pos, fuente))
                        break
        if validas:
            grupos += 1
            implicadas.update(pos for pos,_ in validas)
            if len(muestras) < max_ejemplos:
                fuentes_involucradas = sorted({f for _,f in validas})
                muestras.append({
                    'archivos': fuentes_involucradas,
                    'posiciones': [
                        {'archivo': fuente, 'fila_en_tabla': int(datos.trazabilidad.iloc[pos]['fila_en_tabla'])}
                        for pos, fuente in validas[:5]
                    ],
                })
    if meses:
        min_mes, max_mes = min(meses), max(meses)
        meses_intervalo = (max_mes[0] - min_mes[0]) * 12 + max_mes[1] - min_mes[1] + 1
        meses_sin = meses_intervalo - len(meses)
    else:
        meses_sin = 0
    return InformeConsolidacion(
        archivos=datos.archivos_unicos, tablas=len(datos.nombres), filas_por_archivo=filas_por_archivo,
        fechas_por_archivo=rangos, periodos_solapados=solapamientos,
        meses_sin_registros=meses_sin,
        coincidencias_entre_archivos=len(implicadas), grupos_coincidentes=grupos,
        ejemplos_coincidentes=muestras, filas_con_origen=len(datos.trazabilidad),
    )


def mostrar_consolidacion_corta(informe: InformeConsolidacion) -> None:
    print('\nHISTÓRICO COMBINADO')
    print(f'Archivos reunidos: {informe.archivos}.')
    if informe.tablas != informe.archivos:
        print(f'Hojas o tablas de ventas incorporadas: {informe.tablas}.')
    for nombre, cantidad in informe.filas_por_archivo.items():
        inicio, fin = informe.fechas_por_archivo[nombre]
        rango = (f'{inicio:%d/%m/%Y} a {fin:%d/%m/%Y}'
                 if inicio is not None and fin is not None else 'sin fechas interpretables')
        print(f'  {nombre}: {cantidad:,} registros; {rango}'.replace(',', '.'))
    if informe.periodos_solapados:
        print(f'AVISO: hay periodos que se solapan entre {len(informe.periodos_solapados)} par(es) de archivos.')
    if informe.coincidencias_entre_archivos:
        print(f'Coincidencias entre archivos para revisar: {informe.coincidencias_entre_archivos} registros'
              f' ({informe.grupos_coincidentes} grupos). No se ha eliminado ninguno.')
    else:
        print('No se han detectado coincidencias de fecha, producto y cantidad entre archivos.')
    if informe.meses_sin_registros:
        print(f'Meses sin registros en el intervalo combinado: {informe.meses_sin_registros}. No se suponen ventas cero.')
    print('Se conserva el origen de cada registro. Se han mantenido los códigos de producto tal como aparecen.')
    print('Un solapamiento o una coincidencia no demuestra por sí solo que haya un duplicado.')
