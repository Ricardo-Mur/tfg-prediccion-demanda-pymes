"""Fase 3.1: inventario preliminar de datos interpretables para predicción.

No aprende modelos, no supone cobertura del archivo, no rellena huecos con
ceros, no elimina devoluciones ni posibles duplicados. Solo separa los
campos esenciales interpretables de los no interpretables y resume las
fechas REALMENTE observadas. Es independiente del origen Excel/CSV.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
import numbers

import pandas as pd

from .calidad import _sin_valor
from .limpieza import ResultadoLimpieza


@dataclass
class InformePreparacion:
    total_filas: int
    filas_interpretables: int
    filas_con_campos_pendientes: int
    cantidades_positivas: int
    cantidades_cero: int
    cantidades_negativas: int
    productos_identificados: int
    fechas_distintas: int
    meses_con_registros: int
    primer_dia: date | None
    ultimo_dia: date | None
    meses_en_intervalo: int
    meses_sin_registros: int
    mascara_interpretables: pd.Series
    resumen_por_producto: pd.DataFrame


def _cantidad_finita(valor: object) -> Decimal | None:
    if _sin_valor(valor) or isinstance(valor, bool):
        return None
    if not isinstance(valor, (numbers.Number, Decimal)):
        # En fase 3 no interpretamos nuevamente cadenas no normalizadas.
        return None
    try:
        numero = Decimal(str(valor))
    except (ValueError, InvalidOperation, TypeError):
        return None
    return numero if numero.is_finite() else None


def analizar_preparacion(limpieza: ResultadoLimpieza) -> InformePreparacion:
    """Resume la *copia normalizada* sin editarla ni decidir ventas netas.

    Una fila es interpretable cuando tiene fecha normalizada, producto y
    cantidad numérica, y esos campos no figuran entre los pendientes.
    'Interpretable' no significa 'venta válida para entrenar': la naturaleza
    de devoluciones, cobertura, duplicados y objetivos aún debe evaluarse.
    """
    datos = limpieza.datos
    faltan = set(("fecha", "producto", "cantidad")) - set(datos.columns)
    if faltan:
        raise ValueError("Faltan las columnas esenciales: " + ", ".join(sorted(faltan)))

    pendientes = set()
    if not limpieza.pendientes.empty:
        pendientes = {
            int(fila) for fila, campo in zip(
                limpieza.pendientes['fila_tabla'], limpieza.pendientes['campo']
            ) if campo in ("fecha", "producto", "cantidad")
        }

    mascara = []
    positivos = ceros = negativos = 0
    fechas = set()
    meses = set()
    por_producto: dict[str, dict] = {}

    for posicion, fila in enumerate(datos.itertuples(index=False, name=None), 1):
        # Leemos por posición para que los índices arbitrarios no desplacen filas.
        # itertuples proporciona tuplas, no puede confundirse la fila con su etiqueta.
        fecha = fila[datos.columns.get_loc('fecha')]
        producto = fila[datos.columns.get_loc('producto')]
        cantidad = fila[datos.columns.get_loc('cantidad')]
        numero = _cantidad_finita(cantidad)
        valida = (
            posicion not in pendientes
            and isinstance(fecha, pd.Timestamp)
            and not pd.isna(fecha)
            and not _sin_valor(producto)
            and numero is not None
        )
        mascara.append(bool(valida))
        if not valida:
            continue

        dia = fecha.date()
        mes = (dia.year, dia.month)
        fechas.add(dia)
        meses.add(mes)
        codigo = str(producto).strip()  # No se recortan ceros iniciales de códigos.
        p = por_producto.setdefault(codigo, {
            'producto': codigo, 'registros': 0,
            'primer_dia': dia, 'ultimo_dia': dia,
            'fechas': set(), 'meses': set(),
            'cantidades_positivas': 0, 'cantidades_cero': 0,
            'cantidades_negativas': 0,
        })
        p['registros'] += 1
        p['primer_dia'] = min(p['primer_dia'], dia)
        p['ultimo_dia'] = max(p['ultimo_dia'], dia)
        p['fechas'].add(dia)
        p['meses'].add(mes)
        if numero > 0:
            positivos += 1
            p['cantidades_positivas'] += 1
        elif numero < 0:
            negativos += 1
            p['cantidades_negativas'] += 1
        else:
            ceros += 1
            p['cantidades_cero'] += 1

    columnas = (
        'producto', 'registros', 'primer_dia', 'ultimo_dia',
        'dias_con_registros', 'meses_con_registros',
        'cantidades_positivas', 'cantidades_cero', 'cantidades_negativas',
    )
    filas_producto = []
    for p in por_producto.values():
        filas_producto.append({
            **{k: p[k] for k in columnas if k in p},
            'dias_con_registros': len(p['fechas']),
            'meses_con_registros': len(p['meses']),
        })
    productos = pd.DataFrame(filas_producto, columns=columnas)
    if not productos.empty:
        productos = productos.sort_values(
            ['registros', 'producto'], ascending=[False, True]
        ).reset_index(drop=True)

    primera = min(fechas) if fechas else None
    ultima = max(fechas) if fechas else None
    intervalo = (
        (ultima.year - primera.year)*12 + ultima.month-primera.month+1
        if primera and ultima else 0
    )
    interpretables = sum(mascara)
    return InformePreparacion(
        total_filas=len(datos), filas_interpretables=interpretables,
        filas_con_campos_pendientes=len(datos)-interpretables,
        cantidades_positivas=positivos, cantidades_cero=ceros,
        cantidades_negativas=negativos,
        productos_identificados=len(productos), fechas_distintas=len(fechas),
        meses_con_registros=len(meses), primer_dia=primera, ultimo_dia=ultima,
        meses_en_intervalo=intervalo,
        meses_sin_registros=intervalo-len(meses),
        mascara_interpretables=pd.Series(mascara, index=datos.index, dtype=bool),
        resumen_por_producto=productos,
    )


def mostrar_preparacion_corta(informe: InformePreparacion) -> None:
    """Presenta un resumen de fase 3.1; no hace nuevas preguntas al usuario."""
    print("\n3. ¿QUÉ HISTÓRICO TENEMOS PARA PREDECIR?")
    print(f"Registros con fecha, producto y cantidad interpretables: {informe.filas_interpretables:,}".replace(',', '.'))
    if informe.filas_con_campos_pendientes:
        print(f"Registros con datos esenciales sin resolver: {informe.filas_con_campos_pendientes:,}".replace(',', '.'))
    print(f"Productos identificados: {informe.productos_identificados:,}".replace(',', '.'))
    if informe.primer_dia is None:
        print("No hay fechas interpretables suficientes para determinar el periodo.")
    else:
        print(f"Periodo observado: {informe.primer_dia:%d/%m/%Y} a {informe.ultimo_dia:%d/%m/%Y}")
        print(f"Meses con algún registro: {informe.meses_con_registros} de {informe.meses_en_intervalo} del intervalo")
        if informe.meses_sin_registros:
            print("Hay meses sin registros: no se han rellenado con ceros ni asumido ventas nulas.")
    if informe.cantidades_negativas:
        print(f"Cantidades negativas aún sin clasificar comercialmente: {informe.cantidades_negativas}")
    if informe.cantidades_cero:
        print(f"Registros con cantidad cero conservados: {informe.cantidades_cero}")
    print("La presencia de operaciones no garantiza que estén todas las ventas del periodo.")
    print("La capacidad predictiva se comprobará por producto con validación temporal.")
