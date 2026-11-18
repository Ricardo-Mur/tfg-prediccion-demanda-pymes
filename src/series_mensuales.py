"""Fase 3.2: agrupación mensual conservadora para demanda por producto.

Los periodos sin registros quedan desconocidos; jamás se les asigna cero.
Las cantidades negativas se conservan por separado de las positivas hasta
que la empresa aclare si corresponden a devoluciones, ajustes u otros tipos.
No se quitan posibles duplicados ni se decide suficiencia de modelos aquí.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

import pandas as pd

from .limpieza import ResultadoLimpieza
from .preparacion_historico import InformePreparacion, _cantidad_finita


@dataclass
class InformeSeriesMensuales:
    # Solo periodos con operaciones interpretables; NO hay periodos completados.
    series_observadas: pd.DataFrame
    productos_con_serie: int
    meses_con_registros: int
    meses_en_intervalo_global: int
    productos_con_un_solo_mes: int
    productos_con_menos_de_cuatro_meses: int
    dias_intervalo: int
    filas_utilizadas: int
    filas_pendientes: int


def construir_series_mensuales(
    limpieza: ResultadoLimpieza,
    preparacion: InformePreparacion,
) -> InformeSeriesMensuales:
    """Agrupa SOLO filas con fecha, producto y cantidad interpretables.

    Preserva la cantidad positiva y negativa en columnas separadas, mantiene
    ceros declarados explícitamente y no decide qué es demanda o venta neta.
    Trabaja con Decimal para no perder precisión por usar coma decimal.
    """
    datos = limpieza.datos
    if len(datos) != len(preparacion.mascara_interpretables):
        raise ValueError('La preparación y la copia normalizada no tienen el mismo tamaño.')

    aceptadas = preparacion.mascara_interpretables.to_numpy(dtype=bool)
    agregados: dict[tuple[str, date], dict] = {}
    for permitido, fila in zip(aceptadas, datos.itertuples(index=False, name=None)):
        if not permitido:
            continue
        fecha = fila[datos.columns.get_loc('fecha')]
        producto = str(fila[datos.columns.get_loc('producto')]).strip()
        cantidad = _cantidad_finita(fila[datos.columns.get_loc('cantidad')])
        if not isinstance(fecha, pd.Timestamp) or cantidad is None:
            # Protección adicional si se modificaron los datos después de preparar.
            raise ValueError('Hay un registro marcado como interpretable que no lo es.')
        mes = date(fecha.year, fecha.month, 1)
        clave = (producto, mes)
        if clave not in agregados:
            agregados[clave] = {
                'producto': producto,
                'mes': mes,
                'registros': 0,
                'registros_positivos': 0,
                'registros_negativos': 0,
                'registros_cero': 0,
                'unidades_positivas': Decimal('0'),
                'unidades_negativas': Decimal('0'),
            }
        r = agregados[clave]
        r['registros'] += 1
        if cantidad > 0:
            r['registros_positivos'] += 1
            r['unidades_positivas'] += cantidad
        elif cantidad < 0:
            r['registros_negativos'] += 1
            r['unidades_negativas'] += cantidad
        else:
            r['registros_cero'] += 1

    columnas = [
        'producto', 'mes', 'registros', 'registros_positivos',
        'registros_negativos', 'registros_cero',
        'unidades_positivas', 'unidades_negativas',
    ]
    tabla = pd.DataFrame(agregados.values(), columns=columnas)
    if not tabla.empty:
        tabla = tabla.sort_values(['producto', 'mes']).reset_index(drop=True)
        cobertura = tabla.groupby('producto', sort=False)['mes'].nunique()
        un_mes = int((cobertura == 1).sum())
        menos_cuatro = int((cobertura < 4).sum())
        productos = len(cobertura)
    else:
        un_mes = menos_cuatro = productos = 0

    dias = ((preparacion.ultimo_dia - preparacion.primer_dia).days + 1
            if preparacion.primer_dia is not None
            and preparacion.ultimo_dia is not None else 0)
    return InformeSeriesMensuales(
        series_observadas=tabla,
        productos_con_serie=productos,
        meses_con_registros=preparacion.meses_con_registros,
        meses_en_intervalo_global=preparacion.meses_en_intervalo,
        productos_con_un_solo_mes=un_mes,
        productos_con_menos_de_cuatro_meses=menos_cuatro,
        dias_intervalo=dias,
        filas_utilizadas=int(aceptadas.sum()),
        filas_pendientes=len(datos) - int(aceptadas.sum()),
    )


def mostrar_diagnostico_temporal_corto(informe: InformeSeriesMensuales) -> None:
    """Mensaje de negocio, sin umbrales técnicos ni preguntas nuevas al usuario."""
    print('\n4. ¿TENEMOS SUFICIENTES MESES PARA PREDECIR?')
    if informe.dias_intervalo == 0:
        print('No se ha podido establecer el periodo: faltan fechas utilizables.')
        print('No podemos evaluar predicciones con este archivo tal como está.')
        return
    print(f'Los registros del archivo abarcan {informe.dias_intervalo:,} días naturales.'.replace(',', '.'))
    print(f'Meses con operaciones registradas: {informe.meses_con_registros}.')
    print(f'Productos observados en un solo mes: {informe.productos_con_un_solo_mes:,} de {informe.productos_con_serie:,}.'.replace(',', '.'))
    if informe.meses_con_registros < 4:
        print('HISTÓRICO DEMASIADO CORTO para comprobar predicciones mensuales con fiabilidad.')
        print('Aunque haya muchas filas, necesitamos ventas de varios meses.')
    else:
        print('Ya hay varios meses con datos, pero aún falta estudiar cada producto')
        print('y revisar posibles lagunas antes de evaluar predicciones.')
        if informe.productos_con_menos_de_cuatro_meses:
            print(f'Productos con menos de cuatro meses observados: {informe.productos_con_menos_de_cuatro_meses:,}.'.replace(',', '.'))
    if informe.meses_con_registros < informe.meses_en_intervalo_global:
        print('Hay meses sin registros; su significado es desconocido, no cero ventas.')
    print('No se han rellenado huecos ni eliminado devoluciones o posibles duplicados.')
