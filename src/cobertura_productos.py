"""Fase 3.4: cobertura observada por producto, sin imputar periodos.

Este módulo describe la disponibilidad del histórico; NO decide suficiencia
predictiva, ni confunde ausencia de registros con cero demanda. No interpreta
cancelaciones ni elimina filas. Apto para uno o muchos archivos/hojas.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from .series_mensuales import InformeSeriesMensuales


def _indice_mes(mes: date) -> int:
    return mes.year * 12 + mes.month


@dataclass
class InformeCoberturaProductos:
    detalle: pd.DataFrame
    productos: int
    en_un_mes: int
    en_dos_o_tres_meses: int
    en_cuatro_o_mas_meses: int
    en_doce_o_mas_meses: int
    con_huecos_interiores: int
    sin_cantidades_positivas: int
    sin_registros_en_ultimo_mes: int
    codigos_solo_simbolos: int
    mes_global_inicial: date | None
    mes_global_final: date | None


def analizar_cobertura_productos(series: InformeSeriesMensuales) -> InformeCoberturaProductos:
    """Describe cada producto usando SOLO los meses realmente observados.

    - 'Meses con registros': meses distintos con operaciones interpretables.
    - 'Meses entre primera y última aparición': intervalo natural, NO
      significa ventas en todos los meses ni actividad real en todo él.
    - 'Huecos interiores': meses sin registros en ese intervalo; se dejan
      desconocidos (NO se escriben como ceros en una serie temporal).
    - 'Sin cantidades positivas': observamos solo negativos/cero; no presupone
      ausencia de demanda ni identifica productos que deban descartarse.
    """
    tabla = series.series_observadas
    columnas = (
        'producto', 'primer_mes_observado', 'ultimo_mes_observado',
        'meses_con_registros', 'meses_en_intervalo_observado',
        'meses_intermedios_sin_registros', 'maximo_hueco_consecutivo',
        'meses_con_cantidades_positivas', 'meses_con_cantidades_negativas',
        'meses_con_cantidades_cero', 'registros', 'meses_desde_ultimo_registro',
        'identificador_solo_simbolos',
    )
    if tabla.empty:
        return InformeCoberturaProductos(
            detalle=pd.DataFrame(columns=columnas), productos=0,
            en_un_mes=0, en_dos_o_tres_meses=0, en_cuatro_o_mas_meses=0,
            en_doce_o_mas_meses=0, con_huecos_interiores=0,
            sin_cantidades_positivas=0, sin_registros_en_ultimo_mes=0,
            mes_global_inicial=None, mes_global_final=None,
            codigos_solo_simbolos=0,
        )
    obligatorias = {
        'producto', 'mes', 'registros', 'registros_positivos',
        'registros_negativos', 'registros_cero',
    }
    if not obligatorias.issubset(tabla.columns):
        raise ValueError('Faltan columnas mensuales necesarias para medir la cobertura.')
    if tabla.duplicated(['producto', 'mes']).any():
        raise ValueError('Cada producto y mes deben aparecer una sola vez en la serie agregada.')
    if tabla['producto'].isna().any() or tabla['mes'].isna().any():
        raise ValueError('La serie mensual tiene un producto o mes sin identificar.')

    inicial = min(tabla['mes'])
    final = max(tabla['mes'])
    indice_ultimo = _indice_mes(final)
    detalle = []
    for producto, grupo in tabla.groupby('producto', sort=False):
        orden = sorted(grupo['mes'].tolist())
        indices = [_indice_mes(m) for m in orden]
        meses = len(indices)
        amplitud = indices[-1] - indices[0] + 1
        huecos = amplitud - meses
        mayor_hueco = max((b - a - 1 for a, b in zip(indices, indices[1:])), default=0)
        detalles = {
            'producto': producto,
            'primer_mes_observado': orden[0],
            'ultimo_mes_observado': orden[-1],
            'meses_con_registros': meses,
            'meses_en_intervalo_observado': amplitud,
            'meses_intermedios_sin_registros': huecos,
            'maximo_hueco_consecutivo': mayor_hueco,
            'meses_con_cantidades_positivas': int((grupo['registros_positivos'] > 0).sum()),
            'meses_con_cantidades_negativas': int((grupo['registros_negativos'] > 0).sum()),
            'meses_con_cantidades_cero': int((grupo['registros_cero'] > 0).sum()),
            'registros': int(grupo['registros'].sum()),
            'meses_desde_ultimo_registro': indice_ultimo - indices[-1],
            # Indicador exploratorio, nunca rechazo automático: códigos como
            # '.' o '--' podrían ser claves internas legítimas de una pyme.
            'identificador_solo_simbolos': not any(c.isalnum() for c in str(producto)),
        }
        detalle.append(detalles)

    df = pd.DataFrame(detalle, columns=columnas)
    df = df.sort_values(
        ['meses_con_registros', 'registros', 'producto'],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    n = df['meses_con_registros']
    r = InformeCoberturaProductos(
        detalle=df,
        productos=len(df),
        en_un_mes=int((n == 1).sum()),
        en_dos_o_tres_meses=int(((n >= 2) & (n <= 3)).sum()),
        en_cuatro_o_mas_meses=int((n >= 4).sum()),
        en_doce_o_mas_meses=int((n >= 12).sum()),
        con_huecos_interiores=int((df['meses_intermedios_sin_registros'] > 0).sum()),
        sin_cantidades_positivas=int((df['meses_con_cantidades_positivas'] == 0).sum()),
        sin_registros_en_ultimo_mes=int((df['meses_desde_ultimo_registro'] > 0).sum()),
        mes_global_inicial=inicial,
        mes_global_final=final,
        codigos_solo_simbolos=int(df["identificador_solo_simbolos"].sum()),
    )
    assert r.en_un_mes + r.en_dos_o_tres_meses + r.en_cuatro_o_mas_meses == r.productos
    return r


def mostrar_cobertura_corta(informe: InformeCoberturaProductos) -> None:
    """Explicación de negocio sin preguntas ni tecnicismos estadísticos."""
    print('\n5. ¿QUÉ HISTÓRICO TIENE CADA PRODUCTO?')
    if not informe.productos:
        print('Todavía no hay productos con registros utilizables para estudiar su histórico.')
        print('No se han completado datos ni inventado ventas.')
        return
    n = informe.productos
    print(f'Productos identificados: {n:,}.'.replace(',', '.'))
    print(f'  Registrados en un solo mes: {informe.en_un_mes:,}.'.replace(',', '.'))
    print(f'  Registrados en 2 o 3 meses: {informe.en_dos_o_tres_meses:,}.'.replace(',', '.'))
    print(f'  Registrados en 4 o más meses: {informe.en_cuatro_o_mas_meses:,}.'.replace(',', '.'))
    print(f'  De ellos, registrados en 12 o más meses: {informe.en_doce_o_mas_meses:,}.'.replace(',', '.'))
    print(f'Productos con meses sin registros entre su primera y última aparición: {informe.con_huecos_interiores:,}.'.replace(',', '.'))
    if informe.codigos_solo_simbolos:
        print(f'Códigos formados solo por signos, para revisar: {informe.codigos_solo_simbolos:,}.'.replace(',', '.'))
    if informe.sin_cantidades_positivas:
        print(f'Productos sin ninguna cantidad positiva registrada: {informe.sin_cantidades_positivas:,}.'.replace(',', '.'))
    print('Un mes con registros puede contener solo una operación; no garantiza un mes completo.')
    print('Los meses sin registros siguen siendo desconocidos, no ventas cero.')
    print('Esto describe el histórico observado; todavía NO decide qué productos pueden predecirse.')
