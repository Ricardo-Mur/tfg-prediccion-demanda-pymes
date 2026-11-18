"""Pruebas no destructivas adicionales de calidad del histórico de ventas.

Las coincidencias y los valores extremos son indicios, NO errores probados.
No se borran filas, no se sustituyen cantidades y no se predice demanda.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from math import isfinite

import pandas as pd

from .calidad import _sin_valor, _cantidad_estado


@dataclass(frozen=True)
class AuditoriaAvanzada:
    filas_completamente_vacias: int
    muestra_filas_vacias: list[int]
    filas_identicas: int
    grupos_filas_identicas: int
    muestra_grupos_identicos: list[list[int]]
    filas_misma_fecha_producto_cantidad: int
    grupos_misma_fecha_producto_cantidad: int
    muestra_grupos_misma_clave: list[list[int]]
    productos_descripciones_inconsistentes: int
    muestra_descripciones: list[dict]
    cantidades_extremas: int
    muestra_extremos: list[dict]


def _grupos_repetidos(datos: pd.DataFrame, max_muestra: int = 5) -> tuple[int, int, list[list[int]]]:
    """Duplicidad por fila completa, incluso con columnas de igual nombre.

    La huella agrupa candidatos. Cada grupo mostrado se verifica mediante
    comparación de valores, sin publicar el contenido original de la fila.
    """
    if datos.empty:
        return 0, 0, []
    datos = datos.reset_index(drop=True)
    mascara = datos.duplicated(keep=False)
    indices = [int(i) for i in mascara[mascara].index]
    if not indices:
        return 0, 0, []
    huellas = pd.util.hash_pandas_object(datos.loc[mascara], index=False)
    grupos: list[list[int]] = []
    for _, serie in huellas.groupby(huellas, sort=False):
        candidatos = [int(x) for x in serie.index]
        # Posibles colisiones de huella: verificamos igualdad de fila.
        grupos_verificados: list[list[int]] = []
        for pos in candidatos:
            fila = datos.iloc[pos]
            encontrado = False
            for grupo in grupos_verificados:
                if fila.equals(datos.iloc[grupo[0]]):
                    grupo.append(pos)
                    encontrado = True
                    break
            if not encontrado:
                grupos_verificados.append([pos])
        grupos.extend(g for g in grupos_verificados if len(g) >= 2)
    filas_en_grupos = sum(map(len, grupos))
    ejemplos = [sorted(i+1 for i in g) for g in grupos[:max_muestra]]
    return filas_en_grupos, len(grupos), ejemplos


def auditar_calidad_avanzada(
    seleccion: pd.DataFrame, originales: pd.DataFrame, *,
    max_muestras: int = 5, min_observaciones_atipicos: int = 12,
) -> AuditoriaAvanzada:
    """Audita filas, patrones y extremos sin cambiar ninguna tabla de entrada.

    La posición de fila es 1-based DENTRO de la tabla cargada (no Excel físico).
    Los atípicos se estiman sobre cantidades positivas por producto con
    suficiente repetición: umbral conservador max(Q3+3*IQR, 3*Q3).
    """
    if len(seleccion) != len(originales):
        raise ValueError('Los originales y los datos seleccionados deben tener el mismo número de filas')
    if not all(c in seleccion.columns for c in ('fecha','producto','cantidad')):
        raise ValueError('Faltan columnas obligatorias confirmadas')
    if max_muestras < 0 or min_observaciones_atipicos < 4:
        raise ValueError('Parámetros de auditoría inválidos')

    seleccion = seleccion.reset_index(drop=True)
    originales = originales.reset_index(drop=True)
    # La columna de procedencia no convierte una fila de ventas vacía en válida.
    columnas_de_datos = originales.drop(columns=['__tfg_origen_archivo__', '__tfg_origen_hoja__'], errors='ignore')
    mask_vacios = columnas_de_datos.apply(lambda col: col.map(_sin_valor)).all(axis=1)
    filas_vacias = [i+1 for i in range(len(originales)) if bool(mask_vacios.iat[i])]
    n_filas_identicas, n_grupos_identicos, muestras_identicos = _grupos_repetidos(originales, max_muestras)
    n_misma_clave, n_grupos_clave, muestras_clave = _grupos_repetidos(
        seleccion.loc[:, ['fecha','producto','cantidad']], max_muestras
    )

    desc_por_producto: dict[str, dict[str, str]] = defaultdict(dict)
    if 'descripcion' in seleccion:
        for prod, desc in seleccion.loc[:, ['producto','descripcion']].itertuples(index=False, name=None):
            if not _sin_valor(prod) and not _sin_valor(desc):
                codigo = str(prod).strip()
                texto = ' '.join(str(desc).split())
                desc_por_producto[codigo].setdefault(texto.casefold(), texto)
    discrepantes = {
        prod: vals for prod, vals in desc_por_producto.items() if len(vals) >= 2
    }
    muestras_desc = [
        {'producto':p, 'descripciones':list(v.values())[:4]}
        for p,v in list(discrepantes.items())[:max_muestras]
    ]

    muestras_por_producto: dict[str,list[tuple[int,float]]] = defaultdict(list)
    for i, (prod, qty) in enumerate(seleccion.loc[:, ['producto','cantidad']].itertuples(index=False,name=None)):
        if _sin_valor(prod):
            continue
        estado, numero = _cantidad_estado(qty)
        if estado not in ('valida','normalizable') or numero is None or numero <= 0:
            continue
        try:
            valor = float(numero)
        except (ValueError, OverflowError):
            continue
        if isfinite(valor):
            muestras_por_producto[str(prod).strip()].append((i+1,valor))

    extremos = []
    # Estadístico de cribado, no tratamiento automático. No presupone
    # gaussianidad y evita usar ceros/devoluciones como demanda positiva.
    for codigo, registros in muestras_por_producto.items():
        if len(registros) < min_observaciones_atipicos:
            continue
        valores = pd.Series([v for _,v in registros], dtype=float)
        q1, q3 = (float(x) for x in valores.quantile([.25,.75]).tolist())
        umbral = max(q3 + 3*(q3-q1), 3*q3)
        for fila, valor in registros:
            if valor > umbral:
                extremos.append({'fila_tabla':fila, 'producto':codigo, 'cantidad':valor, 'umbral_orientativo':round(umbral,4)})

    return AuditoriaAvanzada(
        filas_completamente_vacias=len(filas_vacias),
        muestra_filas_vacias=filas_vacias[:max_muestras],
        filas_identicas=n_filas_identicas,
        grupos_filas_identicas=n_grupos_identicos,
        muestra_grupos_identicos=muestras_identicos,
        filas_misma_fecha_producto_cantidad=n_misma_clave,
        grupos_misma_fecha_producto_cantidad=n_grupos_clave,
        muestra_grupos_misma_clave=muestras_clave,
        productos_descripciones_inconsistentes=len(discrepantes),
        muestra_descripciones=muestras_desc,
        cantidades_extremas=len(extremos),
        muestra_extremos=extremos[:max_muestras],
    )


def mostrar_auditoria_avanzada(informe: AuditoriaAvanzada) -> None:
    print('\n=== FASE 2.4: AUDITORÍA COMPLEMENTARIA NO DESTRUCTIVA ===')
    print(f'Filas completamente vacías conservadas en la importación: {informe.filas_completamente_vacias}')
    if informe.muestra_filas_vacias:
        print(f'  Primeras posiciones vacías: {informe.muestra_filas_vacias}')
    print(f'Filas idénticas en origen: {informe.filas_identicas} ({informe.grupos_filas_identicas} grupos)')
    if informe.muestra_grupos_identicos:
        print(f'  Grupos de posiciones: {informe.muestra_grupos_identicos}')
    print(f'Filas con misma fecha-producto-cantidad: {informe.filas_misma_fecha_producto_cantidad} ({informe.grupos_misma_fecha_producto_cantidad} grupos)')
    if informe.muestra_grupos_misma_clave:
        print(f'  Grupos de posiciones: {informe.muestra_grupos_misma_clave}')
    print(f'Productos con distintas descripciones: {informe.productos_descripciones_inconsistentes}')
    for x in informe.muestra_descripciones:
        print(f"  Código {x['producto']!r}: {x['descripciones']}")
    print(f'Cantidades extraordinariamente altas (criterio conservador por producto): {informe.cantidades_extremas}')
    for x in informe.muestra_extremos:
        print(f"  Fila {x['fila_tabla']}, producto {x['producto']!r}: {x['cantidad']:g} > {x['umbral_orientativo']:g}")
    print('IMPORTANTE: ninguna coincidencia, devolución o cifra extrema se borra o corrige automáticamente.')
    print('Una repetición exacta también puede ser una venta legítima; un atípico puede ser un pedido real.')
