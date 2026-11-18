"""Fase 3.8: validacion temporal de dos referencias, sin completar lagunas.

PRUEBA DEL PROCEDIMIENTO, no validacion de demanda ni seleccion de modelos.
La variable observada es la suma mensual de cantidades POSITIVAS registradas,
con negativos y cantidades ilegibles excluyendo el producto/mes entero.
Los meses sin registros y los de solo ceros NO se rellenan ni se validan.

Referencia ingenua: predice con el mes anterior, y requiere otros dos meses
consecutivos anteriores para permitir tres meses iniciales de contexto.
Referencia anual: utiliza el mismo mes del ano anterior, aunque haya lagunas
entre ambos; estos periodos NO se interpretan como ventas cero.

La comparacion entre dos referencias SOLO es valida sobre las mismas fechas
para el mismo producto. Esta fase no entrena algoritmos complejos.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, localcontext

import pandas as pd

from .calidad import _sin_valor
from .cobertura_automatica import _meses_pendientes
from .limpieza import ResultadoLimpieza
from .series_mensuales import InformeSeriesMensuales


@dataclass
class InformeValidacionReferencias:
    detalle: pd.DataFrame
    evaluaciones: pd.DataFrame
    productos: int
    con_tres_cortes_ingenuos: int
    con_tres_cortes_anuales: int
    con_tres_cortes_comunes: int
    total_cortes_ingenuos: int
    total_cortes_anuales: int
    total_cortes_comunes: int
    meses_excluidos: int
    pendientes_sin_fecha: int
    ceros_inferidos: int = 0
    modelos_complejos_entrenados: int = 0


def _indice(m: date) -> int:
    if not isinstance(m, date) or isinstance(m, pd.Timestamp):
        raise ValueError('Los meses deben ser fechas normalizadas de tipo date.')
    return m.year * 12 + m.month - 1


def _decimal_positivo(v: object) -> Decimal:
    if type(v) is bool:
        raise ValueError('No se admite una cantidad booleana.')
    try:
        x = v if isinstance(v, Decimal) else Decimal(str(v))
    except Exception as exc:
        raise ValueError('La cantidad mensual debe ser numérica.') from exc
    if not x.is_finite() or x < 0:
        raise ValueError('La cantidad mensual no puede ser negativa o no finita.')
    return x


def _errores(vals: list[tuple[Decimal, Decimal]]) -> tuple[Decimal | None, Decimal | None]:
    if not vals:
        return None, None
    with localcontext() as ctx:
        ctx.prec = 36
        errores = [real - pred for real, pred in vals]
        n = Decimal(len(errores))
        mae = sum((abs(e) for e in errores), Decimal(0)) / n
        rmse = (sum((e * e for e in errores), Decimal(0)) / n).sqrt()
    return mae, rmse


def _meses_pendientes_sin_producto(limpieza: ResultadoLimpieza | None) -> tuple[set[int], int]:
    """Evita evaluar un mes si hay una fila sin producto identificable."""
    if limpieza is None or limpieza.pendientes.empty:
        return set(), 0
    meses: set[int] = set()
    sin_fecha = 0
    for numero in set(int(n) for n in limpieza.pendientes['fila_tabla']):
        if numero < 1 or numero > len(limpieza.datos):
            raise ValueError('Fila pendiente fuera de la tabla.')
        fila = limpieza.datos.iloc[numero - 1]
        fecha = fila['fecha']
        if not isinstance(fecha, pd.Timestamp) or pd.isna(fecha):
            sin_fecha += 1
        elif _sin_valor(fila['producto']):
            meses.add(_indice(fecha.date().replace(day=1)))
    return meses, sin_fecha


def evaluar_referencias_temporales(
    series: InformeSeriesMensuales,
    *,
    limpieza: ResultadoLimpieza | None = None,
    meses_contexto: int = 3,
    cortes_minimos: int = 3,
    periodicidad_anual: int = 12,
) -> InformeValidacionReferencias:
    """Calcula predicciones de referencia de UN MES y métricas por producto.

    Cada predicción se construye solo con valores cronológicamente anteriores.
    Evita comparar métricas calculadas en ventanas de evaluación diferentes.
    """
    for nom, valor in [('contexto',meses_contexto),('cortes',cortes_minimos),
                       ('periodicidad anual',periodicidad_anual)]:
        if type(valor) is not int or valor < 1:
            raise ValueError(f'Parámetro incorrecto: {nom}.')
    tabla = series.series_observadas
    columnas_detalle = [
        'producto', 'meses_utilizables', 'cortes_ingenua', 'cortes_anual',
        'cortes_comunes', 'mae_ingenua', 'rmse_ingenua', 'mae_anual', 'rmse_anual',
        'mae_ingenua_comunes', 'mae_anual_comunes', 'tres_cortes_ingenua',
        'tres_cortes_anual', 'tres_cortes_comunes',
    ]
    columnas_eval = ['producto','mes','observado_unidades_positivas',
                     'prediccion_ingenua','prediccion_anual',
                     'error_absoluto_ingenua','error_absoluto_anual']
    _, pendientes_producto, _ = _meses_pendientes(limpieza)
    bloqueados_globales, sin_fecha = _meses_pendientes_sin_producto(limpieza)
    if tabla.empty:
        return InformeValidacionReferencias(
            pd.DataFrame(columns=columnas_detalle), pd.DataFrame(columns=columnas_eval),
            0,0,0,0,0,0,0,0,sin_fecha)
    necesarios = {'producto','mes','registros_positivos','registros_negativos',
                  'registros_cero','unidades_positivas'}
    if not necesarios.issubset(tabla.columns):
        raise ValueError('Faltan columnas esenciales de las series mensuales.')
    if tabla[['producto','mes']].isna().any().any() or tabla.duplicated(['producto','mes']).any():
        raise ValueError('Cada producto y mes debe aparecer una única vez.')
    for nom in ('registros_positivos','registros_negativos','registros_cero'):
        valores = pd.to_numeric(tabla[nom], errors='coerce')
        if valores.isna().any() or (valores < 0).any() or (valores % 1 != 0).any():
            raise ValueError('Recuento de operaciones incorrecto.')
    detalles, evaluaciones = [], []
    excluidos = 0
    for producto, grupo in tabla.groupby('producto', sort=False):
        codigo = str(producto)
        valores: dict[int, Decimal] = {}
        for registro in grupo.itertuples(index=False):
            # Solo datos observados, no ceros imputados ni meses solo con cero.
            indice = _indice(registro.mes)
            unidades = _decimal_positivo(registro.unidades_positivas)
            if ((codigo, indice) in pendientes_producto or indice in bloqueados_globales
                or registro.registros_negativos > 0 or registro.registros_positivos == 0):
                excluidos += 1
                continue
            if unidades <= 0:
                raise ValueError('El recuento positivo no concuerda con las unidades.')
            valores[indice] = unidades
        naive, anual, comunes_naive, comunes_anual = [], [], [], []
        for indice in sorted(valores):
            verdad = valores[indice]
            previa = valores.get(indice - 1) if all(
                (indice - d) in valores for d in range(1, meses_contexto+1)
            ) else None
            estacional = valores.get(indice - periodicidad_anual)
            if previa is None and estacional is None:
                continue
            evaluaciones.append({
                'producto': codigo,
                'mes': date(indice // 12, indice % 12 + 1, 1),
                'observado_unidades_positivas': verdad,
                'prediccion_ingenua': previa,
                'prediccion_anual': estacional,
                'error_absoluto_ingenua': abs(verdad-previa) if previa is not None else None,
                'error_absoluto_anual': abs(verdad-estacional) if estacional is not None else None,
            })
            if previa is not None:
                naive.append((verdad, previa))
            if estacional is not None:
                anual.append((verdad, estacional))
            if previa is not None and estacional is not None:
                comunes_naive.append((verdad, previa))
                comunes_anual.append((verdad, estacional))
        mae_n, rmse_n = _errores(naive)
        mae_a, rmse_a = _errores(anual)
        mae_c_n, _ = _errores(comunes_naive)
        mae_c_a, _ = _errores(comunes_anual)
        detalles.append({
            'producto': codigo, 'meses_utilizables':len(valores),
            'cortes_ingenua':len(naive), 'cortes_anual':len(anual),
            'cortes_comunes':len(comunes_naive), 'mae_ingenua':mae_n,
            'rmse_ingenua':rmse_n, 'mae_anual':mae_a, 'rmse_anual':rmse_a,
            'mae_ingenua_comunes':mae_c_n, 'mae_anual_comunes':mae_c_a,
            'tres_cortes_ingenua':len(naive) >= cortes_minimos,
            'tres_cortes_anual':len(anual) >= cortes_minimos,
            'tres_cortes_comunes':len(comunes_naive) >= cortes_minimos,
        })
    d = pd.DataFrame(detalles,columns=columnas_detalle).sort_values('producto').reset_index(drop=True)
    e = pd.DataFrame(evaluaciones,columns=columnas_eval).sort_values(['producto','mes']).reset_index(drop=True)
    return InformeValidacionReferencias(
        detalle=d, evaluaciones=e, productos=len(d),
        con_tres_cortes_ingenuos=int(d['tres_cortes_ingenua'].sum()),
        con_tres_cortes_anuales=int(d['tres_cortes_anual'].sum()),
        con_tres_cortes_comunes=int(d['tres_cortes_comunes'].sum()),
        total_cortes_ingenuos=int(d['cortes_ingenua'].sum()),
        total_cortes_anuales=int(d['cortes_anual'].sum()),
        total_cortes_comunes=int(d['cortes_comunes'].sum()),
        meses_excluidos=excluidos, pendientes_sin_fecha=sin_fecha,
    )


def mostrar_validacion_referencias_corta(r: InformeValidacionReferencias) -> None:
    def f(x: int) -> str:
        return f'{x:,}'.replace(',','.')
    print('\n9. ¿QUÉ RESULTADOS DAN LAS PREDICCIONES MÁS SENCILLAS?')
    print(f'Productos revisados: {f(r.productos)}.')
    print(f'Productos con al menos tres meses evaluados frente al mes anterior: '
          f'{f(r.con_tres_cortes_ingenuos)}.')
    print(f'Productos con al menos tres meses evaluados frente al mismo mes del año anterior: '
          f'{f(r.con_tres_cortes_anuales)}.')
    print(f'Productos con al menos tres fechas comunes a ambas referencias: '
          f'{f(r.con_tres_cortes_comunes)}.')
    print(f'Comparaciones mensuales calculadas: mes anterior {f(r.total_cortes_ingenuos)}; '
          f'mismo mes del año previo {f(r.total_cortes_anuales)}.')
    print(f'Meses de productos no utilizados por falta de unidades positivas claras, '
          f'movimientos negativos o datos pendientes: {f(r.meses_excluidos)}.')
    if r.pendientes_sin_fecha:
        print('Existen filas sin fecha interpretable; su efecto temporal sigue siendo incierto.')
    print('Los errores MAE y RMSE de cada referencia quedan disponibles por producto en el informe interno.')
    print('Solo se compararán referencias directamente cuando tengan las mismas fechas de evaluación.')
    print('Estas pruebas usan cantidades positivas registradas: NO verifican demanda real ni ventas netas.')
    print('No se han completado meses ausentes ni se han inventado ventas cero.')
    print('Todavía no se han entrenado SARIMA, Prophet, Random Forest ni XGBoost.')
