"""Fase 3.6: comprobar lagunas temporales sin pedir encuestas ni inventar ceros.

Distingue meses sin operaciones interpretables de meses con registros
incompletos. La actividad de otros productos NO acredita la integridad de una
exportación, ni permite concluir que un producto sin operaciones vendió cero.

Esta fase NO imputa periodos, clasifica demanda ni prepara automáticamente
objetivos de entrenamiento. Los ceros explícitos se conservan tal como vienen.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from .calidad import _sin_valor
from .limpieza import ResultadoLimpieza
from .series_mensuales import InformeSeriesMensuales


@dataclass
class InformeCoberturaAutomatica:
    productos: int
    meses_en_intervalo: int
    meses_con_actividad_general: int
    meses_sin_registros_generales: int
    productos_con_huecos_intermedios: int
    productos_con_huecos_en_meses_activos: int
    meses_producto_sin_registros_con_actividad_general: int
    meses_producto_sin_registros_sin_actividad_general: int
    registros_con_datos_esenciales_pendientes: int
    meses_sin_registros: tuple[date, ...]
    detalle_huecos: pd.DataFrame
    meses_solo_con_datos_pendientes: int = 0
    meses_producto_con_dato_pendiente: int = 0
    registros_pendientes_no_localizables: int = 0
    ceros_inferidos: int = 0


def _indice_mes(mes: date) -> int:
    return mes.year * 12 + mes.month - 1


def _fecha_mes(indice: int) -> date:
    return date(indice // 12, indice % 12 + 1, 1)


def _meses_pendientes(
    limpieza: ResultadoLimpieza | None,
) -> tuple[set[int], set[tuple[str, int]], int]:
    """Fechas conocidas de filas pendientes; no se interpretan cantidades.

    Una fila sin producto puede indicar actividad empresarial ese mes, pero
    jamás se atribuye a un artículo. Una fila sin fecha no se ubica en meses.
    """
    if limpieza is None or limpieza.pendientes.empty:
        return set(), set(), 0
    indices = set(int(f) for f in limpieza.pendientes['fila_tabla'])
    meses: set[int] = set()
    producto_mes: set[tuple[str, int]] = set()
    sin_fecha_o_producto = 0
    for fila in indices:
        if fila < 1 or fila > len(limpieza.datos):
            raise ValueError('La posición de un registro pendiente está fuera de la tabla.')
        registro = limpieza.datos.iloc[fila-1]
        fecha = registro['fecha']
        producto = registro['producto']
        if not isinstance(fecha, pd.Timestamp) or pd.isna(fecha):
            sin_fecha_o_producto += 1
            continue
        mes = _indice_mes(fecha.date())
        meses.add(mes)
        if _sin_valor(producto):
            sin_fecha_o_producto += 1
        else:
            producto_mes.add((str(producto).strip(), mes))
    return meses, producto_mes, sin_fecha_o_producto


def analizar_cobertura_automatica(
    series: InformeSeriesMensuales, *, limpieza: ResultadoLimpieza | None = None,
) -> InformeCoberturaAutomatica:
    """Diagnostica lagunas entre apariciones, sin convertirlas en ventas cero.

    Solo se revisan meses entre la primera y última aparición registrada de
    cada producto. No se infieren fechas de alta/baja, disponibilidad de stock,
    integridad del fichero ni una demanda nula. Un mes con una cantidad
    ilegible se trata como dato pendiente y no como mes sin operaciones.
    """
    t = series.series_observadas
    columnas = ['producto', 'mes', 'tipo_laguna']
    vacio = pd.DataFrame(columns=columnas)
    pendientes_general, pendientes_producto, sin_localizar = _meses_pendientes(limpieza)
    if t.empty:
        return InformeCoberturaAutomatica(
            productos=0, meses_en_intervalo=0, meses_con_actividad_general=0,
            meses_sin_registros_generales=0, productos_con_huecos_intermedios=0,
            productos_con_huecos_en_meses_activos=0,
            meses_producto_sin_registros_con_actividad_general=0,
            meses_producto_sin_registros_sin_actividad_general=0,
            registros_con_datos_esenciales_pendientes=series.filas_pendientes,
            meses_sin_registros=(), detalle_huecos=vacio,
            registros_pendientes_no_localizables=sin_localizar,
        )
    if not {'producto','mes'}.issubset(t.columns):
        raise ValueError('Faltan producto o mes en las series observadas.')
    if t[['producto','mes']].isna().any().any():
        raise ValueError('Las series observadas tienen productos o meses sin identificar.')
    if t.duplicated(['producto','mes']).any():
        raise ValueError('Debe haber una única fila por producto y mes.')
    if any(not isinstance(v, date) for v in t['mes']):
        raise ValueError('Los meses de la serie deben ser fechas verificadas.')

    activos = {_indice_mes(m) for m in t['mes']}
    primero, ultimo = min(activos), max(activos)
    pendientes_solos = {m for m in pendientes_general if primero <= m <= ultimo and m not in activos}
    vacios = tuple(_fecha_mes(i) for i in range(primero, ultimo+1)
                   if i not in activos and i not in pendientes_general)
    productos_con_huecos = productos_huecos_activos = 0
    lagunas_actividad = lagunas_generales = con_dato_pendiente = 0
    huecos: list[dict] = []
    for producto, grupo in t.groupby('producto', sort=False):
        meses = {_indice_mes(m) for m in grupo['mes']}
        primero_prod, ultimo_prod = min(meses), max(meses)
        tiene_huecos = tiene_huecos_activos = False
        for indice in range(primero_prod+1, ultimo_prod):
            if indice in meses:
                continue
            tiene_huecos = True
            if (producto,indice) in pendientes_producto:
                con_dato_pendiente += 1
                tipo = 'dato_pendiente_producto'
            elif indice in activos:
                lagunas_actividad += 1
                tiene_huecos_activos = True
                tipo = 'sin_registro_producto_con_actividad_general'
            elif indice in pendientes_general:
                tipo = 'mes_general_con_datos_pendientes'
            else:
                lagunas_generales += 1
                tipo = 'sin_registros_generales'
            huecos.append({'producto':producto, 'mes':_fecha_mes(indice), 'tipo_laguna':tipo})
        productos_con_huecos += int(tiene_huecos)
        productos_huecos_activos += int(tiene_huecos_activos)

    return InformeCoberturaAutomatica(
        productos=int(t['producto'].nunique()),
        meses_en_intervalo=ultimo-primero+1,
        meses_con_actividad_general=len(activos),
        meses_sin_registros_generales=len(vacios),
        productos_con_huecos_intermedios=productos_con_huecos,
        productos_con_huecos_en_meses_activos=productos_huecos_activos,
        meses_producto_sin_registros_con_actividad_general=lagunas_actividad,
        meses_producto_sin_registros_sin_actividad_general=lagunas_generales,
        registros_con_datos_esenciales_pendientes=series.filas_pendientes,
        meses_sin_registros=vacios,
        detalle_huecos=pd.DataFrame(huecos, columns=columnas),
        meses_solo_con_datos_pendientes=len(pendientes_solos),
        meses_producto_con_dato_pendiente=con_dato_pendiente,
        registros_pendientes_no_localizables=sin_localizar,
    )


def mostrar_cobertura_automatica_corta(informe: InformeCoberturaAutomatica) -> None:
    print('\n7. COMPROBACIÓN AUTOMÁTICA DE MESES SIN INFORMACIÓN')
    if not informe.meses_en_intervalo:
        print('No hay fechas interpretables suficientes para examinar posibles lagunas.')
        if informe.registros_con_datos_esenciales_pendientes:
            print('Existen registros sin interpretar; se conservan sin inventar valores.')
        return
    print(f'Meses con operaciones interpretables de algún producto: '
          f'{informe.meses_con_actividad_general} de {informe.meses_en_intervalo} del periodo observado.')
    if informe.meses_sin_registros_generales:
        print(f'Meses sin operaciones registradas entre esas fechas: {informe.meses_sin_registros_generales}.')
        ejemplos = ', '.join(m.strftime('%m/%Y') for m in informe.meses_sin_registros[:5])
        print('  ' + ejemplos + ('...' if len(informe.meses_sin_registros)>5 else ''))
    else:
        print('No se observan meses totalmente vacíos entre las fechas conocidas.')
    if informe.meses_solo_con_datos_pendientes:
        print(f'Otros {informe.meses_solo_con_datos_pendientes} mes(es) '
              'tienen registros que aún no podemos interpretar.')
    print(f'Productos con meses intermedios sin información interpretable: '
          f'{informe.productos_con_huecos_intermedios:,}.'.replace(',','.'))
    if informe.productos_con_huecos_en_meses_activos:
        print((f'De ellos, {informe.productos_con_huecos_en_meses_activos:,} presentan meses sin operaciones propias '
               'aunque haya ventas de otros productos.').replace(',','.'))
    if informe.meses_producto_con_dato_pendiente:
        print(f'Hay {informe.meses_producto_con_dato_pendiente} meses de productos '
              'con cantidades pendientes: no se tratarán como cero ventas.')
    if informe.registros_pendientes_no_localizables:
        print('Existen registros sin fecha o producto conocidos, que no pueden situarse en un mes concreto.')
    print('Que haya operaciones de otros artículos no garantiza que todas las ventas estén incluidas.')
    print('No es necesario responder ninguna pregunta sobre la completitud del archivo.')
    print('No se han creado ceros en meses sin registros; los ceros explícitos sí se conservan.')
    print('La posible estacionalidad se estudiará con años suficientes y validación por producto.')
