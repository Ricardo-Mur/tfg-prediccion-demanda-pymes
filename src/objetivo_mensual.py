"""Fase 3.10: estados de la serie mensual antes de ajustar modelos.

La falta de una operación NO es un cero. Los registros con cero explícito
se diferencian de los meses sin registros. Las cantidades negativas o
pendientes bloquean la interpretación de ese mes como un objetivo conocido.

Los totales positivos registrados son una medida observada, NO ventas netas
verificadas ni demanda real. Este módulo NO entrena ni selecciona modelos.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

import pandas as pd

from .cobertura_automatica import _fecha_mes, _indice_mes, _meses_pendientes
from .limpieza import ResultadoLimpieza
from .series_mensuales import InformeSeriesMensuales
from .validacion_referencias import _decimal_positivo, _meses_pendientes_sin_producto


@dataclass
class InformeObjetivoMensual:
    detalle: pd.DataFrame
    resumen_por_producto: pd.DataFrame
    productos: int
    meses_con_positivo: int
    meses_con_cero_explicito: int
    meses_sin_registro: int
    meses_con_negativos: int
    meses_con_datos_pendientes: int
    meses_con_producto_desconocido: int
    registros_pendientes_sin_fecha: int
    ceros_inferidos: int = 0
    modelos_entrenados: int = 0


def preparar_objetivo_mensual(
    series: InformeSeriesMensuales,
    *,
    limpieza: ResultadoLimpieza | None = None,
    max_meses_por_producto: int = 1200,
) -> InformeObjetivoMensual:
    """Construye únicamente meses entre primera y última aparición por producto.

    La variable 'unidades_observadas' solamente se asigna a sumas de unidades
    positivas registradas o a meses con registros explícitos solo de cero.
    Nunca se deriva de una laguna. Debe considerarse PRELIMINAR (no demanda).
    """
    if type(max_meses_por_producto) is not int or max_meses_por_producto < 1:
        raise ValueError('El límite de meses por producto debe ser un entero positivo.')
    columnas = [
        'producto', 'mes', 'estado', 'unidades_observadas',
        'registros_positivos', 'registros_cero', 'registros_negativos',
    ]
    resumen_columnas = [
        'producto', 'meses_en_intervalo', 'meses_con_positivo',
        'meses_con_cero_explicito', 'meses_sin_registro',
        'meses_con_negativos', 'meses_con_datos_pendientes',
        'meses_con_producto_desconocido', 'meses_con_valor_observado',
    ]
    tabla = series.series_observadas
    exigidas = {
        'producto', 'mes', 'registros_positivos', 'registros_cero',
        'registros_negativos', 'unidades_positivas', 'unidades_negativas',
    }
    if not exigidas.issubset(tabla.columns):
        raise ValueError('Faltan columnas esenciales de las series mensuales.')
    if tabla[['producto', 'mes']].isna().any().any():
        raise ValueError('Existen meses o productos sin identificar.')
    if tabla.duplicated(['producto', 'mes']).any():
        raise ValueError('Se repite un producto y mes en la serie observada.')

    _, pendientes_producto, pendientes_no_localizables = _meses_pendientes(limpieza)
    bloqueados_globales, pendientes_sin_fecha = _meses_pendientes_sin_producto(limpieza)

    filas = []
    res = []
    contadores = dict.fromkeys([
        'positivo_registrado', 'cero_explicito', 'sin_registro',
        'pendiente_negativos', 'pendiente_dato', 'pendiente_producto_desconocido',
    ], 0)
    for producto, grupo in tabla.groupby('producto', sort=False):
        codigo = str(producto)
        observados = {}
        for fila in grupo.itertuples(index=False):
            if type(fila.mes) is not date or fila.mes.day != 1:
                raise ValueError('Se requieren fechas mensuales normalizadas.')
            indice = _indice_mes(fila.mes)
            counts = []
            for nombre in ('registros_positivos', 'registros_cero', 'registros_negativos'):
                valor = getattr(fila, nombre)
                if isinstance(valor, bool) or not isinstance(valor, (int,)) or valor < 0:
                    raise ValueError('Recuento de operaciones incorrecto.')
                counts.append(valor)
            pos, ceros, neg = counts
            if pos + ceros + neg == 0:
                raise ValueError('Un mes observado debe tener alguna operación.')
            unidades = _decimal_positivo(fila.unidades_positivas)
            if (pos and unidades <= 0) or (not pos and unidades != 0):
                raise ValueError('Inconsistencia entre positivos y unidades observadas.')
            valor_neg = fila.unidades_negativas
            try:
                valor_neg = valor_neg if isinstance(valor_neg, Decimal) else Decimal(str(valor_neg))
            except Exception as exc:
                raise ValueError('Unidades negativas no interpretables.') from exc
            if not valor_neg.is_finite() or (neg and valor_neg >= 0) or (not neg and valor_neg != 0):
                raise ValueError('Inconsistencia entre negativos y unidades observadas.')
            observados[indice] = (pos, ceros, neg, unidades)
        inicio, fin = min(observados), max(observados)
        if fin - inicio + 1 > max_meses_por_producto:
            raise ValueError('Histórico demasiado extenso: revise las fechas de la serie.')
        recuento = dict.fromkeys(contadores, 0)
        for indice in range(inicio, fin + 1):
            pos, ceros, neg, unidades = observados.get(indice, (0, 0, 0, Decimal(0)))
            # Priorizamos incidencias frente a valores parciales conocidos.
            if indice in bloqueados_globales:
                estado = 'pendiente_producto_desconocido'
                objetivo = None
            elif (codigo, indice) in pendientes_producto:
                estado = 'pendiente_dato'
                objetivo = None
            elif neg:
                estado = 'pendiente_negativos'
                objetivo = None
            elif pos:
                estado = 'positivo_registrado'
                objetivo = unidades
            elif ceros:
                estado = 'cero_explicito'
                objetivo = Decimal(0)
            else:
                estado = 'sin_registro'
                objetivo = None
            filas.append({
                'producto': codigo, 'mes': _fecha_mes(indice), 'estado': estado,
                'unidades_observadas': objetivo,
                'registros_positivos': pos, 'registros_cero': ceros,
                'registros_negativos': neg,
            })
            contadores[estado] += 1
            recuento[estado] += 1
        res.append({
            'producto': codigo, 'meses_en_intervalo': fin - inicio + 1,
            'meses_con_positivo': recuento['positivo_registrado'],
            'meses_con_cero_explicito': recuento['cero_explicito'],
            'meses_sin_registro': recuento['sin_registro'],
            'meses_con_negativos': recuento['pendiente_negativos'],
            'meses_con_datos_pendientes': recuento['pendiente_dato'],
            'meses_con_producto_desconocido': recuento['pendiente_producto_desconocido'],
            'meses_con_valor_observado': recuento['positivo_registrado'] + recuento['cero_explicito'],
        })
    detalle = pd.DataFrame(filas, columns=columnas).sort_values(['producto', 'mes']).reset_index(drop=True)
    resumen = pd.DataFrame(res, columns=resumen_columnas).sort_values('producto').reset_index(drop=True)
    return InformeObjetivoMensual(
        detalle=detalle, resumen_por_producto=resumen,
        productos=len(resumen),
        meses_con_positivo=contadores['positivo_registrado'],
        meses_con_cero_explicito=contadores['cero_explicito'],
        meses_sin_registro=contadores['sin_registro'],
        meses_con_negativos=contadores['pendiente_negativos'],
        meses_con_datos_pendientes=contadores['pendiente_dato'],
        meses_con_producto_desconocido=contadores['pendiente_producto_desconocido'],
        registros_pendientes_sin_fecha=pendientes_sin_fecha,
    )


def mostrar_objetivo_mensual_corto(r: InformeObjetivoMensual) -> None:
    def f(x: int) -> str:
        return f'{x:,}'.replace(',', '.')
    print('\n11. ¿QUÉ SIGNIFICAN LOS VALORES Y HUECOS DE CADA MES?')
    print(f'Productos revisados: {f(r.productos)}.')
    print(f'Meses de producto con cantidades positivas registradas: {f(r.meses_con_positivo)}.')
    print(f'Meses de producto con únicamente cantidades cero explícitas: {f(r.meses_con_cero_explicito)}.')
    print(f'Meses intermedios sin registros propios (valor desconocido): {f(r.meses_sin_registro)}.')
    print(f'Meses de producto con movimientos negativos pendientes: {f(r.meses_con_negativos)}.')
    print(f'Meses de producto con cantidades o datos pendientes: {f(r.meses_con_datos_pendientes)}.')
    if r.meses_con_producto_desconocido:
        print(f'Meses de producto afectados por registros sin producto conocido: '
              f'{f(r.meses_con_producto_desconocido)}.')
    if r.registros_pendientes_sin_fecha:
        print('Existen registros sin fecha interpretable; no se puede asignar su mes.')
    print('Los ceros explícitos se conservan, pero no prueban que el mes completo tuviera cero ventas.')
    print('Los meses sin registros conservan un valor desconocido: no se han rellenado con ceros.')
    print('Las cantidades positivas son provisionales: no equivalen a ventas netas ni demanda real.')
    print('Esta fase prepara los estados para futuras evaluaciones, sin entrenar modelos.')
