"""Fase 3.5: descripción inicial de la actividad y clasificación condicionada.

No se interpretan periodos sin registros como demanda cero. No se calcula ADI
ni una tipología de demanda a partir de un historial cuya completitud se desconoce.
El cálculo ADI/CV² está aislado y requiere una serie mensual VERIFICADA, de
periodos explícitos, con significado comercial claro y sin valores desconocidos.

ADI/CV² describe un patrón, no demuestra capacidad predictiva ni compara modelos.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import numbers
from typing import Sequence

import pandas as pd

from .series_mensuales import InformeSeriesMensuales
from .cobertura_productos import InformeCoberturaProductos


@dataclass
class InformePatronesPreliminares:
    detalle: pd.DataFrame
    productos: int
    con_positivos_en_dos_o_mas_meses: int
    con_positivos_en_un_mes: int
    sin_positivos_observados: int
    con_movimientos_negativos: int
    con_cantidades_cero: int
    con_huecos_intermedios: int
    con_huecos_y_positivos: int
    con_cv2_descriptivo: int
    clasificaciones_adi_cv2: int = 0


@dataclass(frozen=True)
class ClasificacionVerificada:
    patron: str | None
    adi: float | None
    cv2: float | None
    periodos: int
    periodos_con_demanda: int
    motivo: str


def _cv2_positivos(valores: Sequence[Decimal]) -> float | None:
    """CV² poblacional en periodos con valor positivo; al menos 2 positivos.

    El valor con un solo periodo positivo queda sin estimar para evitar una
    aparente variabilidad nula basada en una única observación.
    """
    if len(valores) < 2:
        return None
    media = sum(valores, Decimal(0)) / Decimal(len(valores))
    if media <= 0:
        return None
    varianza = sum(((v - media) ** 2 for v in valores), Decimal(0)) / Decimal(len(valores))
    return float(varianza / (media ** 2))


def analizar_patrones_preliminares(
    series: InformeSeriesMensuales, cobertura: InformeCoberturaProductos,
) -> InformePatronesPreliminares:
    """Resume actividad positiva, ceros y negativos SIN imputar periodos.

    CV² aquí es solo una descripción de la variación del volumen positivo
    registrado en los meses con ventas positivas. NO es una clasificación de
    demanda porque no se conoce la integridad de los meses sin registros.
    """
    t = series.series_observadas
    cols = ['producto', 'meses_con_positivos', 'meses_con_negativos',
            'meses_con_ceros', 'meses_solo_con_no_positivos',
            'meses_interiores_sin_registros', 'cv2_ventas_positivas_observadas']
    if t.empty:
        if cobertura.productos:
            raise ValueError('La cobertura indica productos pero no hay series mensuales.')
        return InformePatronesPreliminares(pd.DataFrame(columns=cols), 0, 0, 0, 0, 0, 0, 0, 0, 0)
    necesarios = {'producto', 'mes', 'registros_positivos', 'registros_negativos',
                  'registros_cero', 'unidades_positivas'}
    if not necesarios.issubset(t.columns):
        raise ValueError('La serie mensual no contiene las columnas de actividad necesarias.')
    if t.duplicated(['producto', 'mes']).any():
        raise ValueError('La serie mensual contiene más de una fila por producto y mes.')
    if not {'producto','meses_intermedios_sin_registros'}.issubset(cobertura.detalle.columns):
        raise ValueError('Falta información de cobertura por producto.')
    huecos = dict(zip(cobertura.detalle['producto'],
                      cobertura.detalle['meses_intermedios_sin_registros']))
    filas: list[dict] = []
    for producto, grupo in t.groupby('producto', sort=False):
        if producto not in huecos:
            raise ValueError('La cobertura y las series no coinciden en productos.')
        positivos = grupo['registros_positivos'] > 0
        negativos = grupo['registros_negativos'] > 0
        ceros = grupo['registros_cero'] > 0
        cantidades = [v for v in grupo.loc[positivos,'unidades_positivas'].tolist()]
        if any(not isinstance(v, Decimal) or v <= 0 or not v.is_finite() for v in cantidades):
            raise ValueError('Las cantidades positivas mensuales deben ser decimales finitas mayores que cero.')
        filas.append({
            'producto': producto,
            'meses_con_positivos': int(positivos.sum()),
            'meses_con_negativos': int(negativos.sum()),
            'meses_con_ceros': int(ceros.sum()),
            'meses_solo_con_no_positivos': int((~positivos).sum()),
            'meses_interiores_sin_registros': int(huecos[producto]),
            'cv2_ventas_positivas_observadas': _cv2_positivos(cantidades),
        })
    d = pd.DataFrame(filas, columns=cols).sort_values('producto').reset_index(drop=True)
    m = d['meses_con_positivos']
    informe = InformePatronesPreliminares(
        detalle=d,
        productos=len(d),
        con_positivos_en_dos_o_mas_meses=int((m>=2).sum()),
        con_positivos_en_un_mes=int((m==1).sum()),
        sin_positivos_observados=int((m==0).sum()),
        con_movimientos_negativos=int((d['meses_con_negativos']>0).sum()),
        con_cantidades_cero=int((d['meses_con_ceros']>0).sum()),
        con_huecos_intermedios=int((d['meses_interiores_sin_registros']>0).sum()),
        con_huecos_y_positivos=int(((d['meses_interiores_sin_registros']>0)&(m>0)).sum()),
        con_cv2_descriptivo=int(d['cv2_ventas_positivas_observadas'].notna().sum()),
    )
    if informe.productos != cobertura.productos:
        raise ValueError('No coincide el número de productos de cobertura y series.')
    assert (informe.con_positivos_en_dos_o_mas_meses +
            informe.con_positivos_en_un_mes + informe.sin_positivos_observados
            == informe.productos)
    return informe


def _decimal_explicito(v: object) -> Decimal | None:
    if v is None or isinstance(v, bool) or isinstance(v, str) or not isinstance(v, (numbers.Number, Decimal)):
        return None
    try:
        numero = Decimal(str(v))
    except (ValueError, TypeError, InvalidOperation):
        return None
    return numero if numero.is_finite() else None


def clasificar_serie_mensual_verificada(
    valores_mensuales: Sequence[Decimal | int | float | None], *,
    cobertura_y_significado_confirmados: bool = False,
) -> ClasificacionVerificada:
    """ADI/CV² opcionales para una SERIE DE DEMANDA MENSUAL VERIFICADA.

    Cada valor debe ser un mes consecutivo conocido, incluso los ceros. No
    debe pasarse una lista que omita meses sin registros. La llamada requiere
    confirmación explícita de que son datos completos y que las cantidades
    significan demanda/ventas relevantes (sin negativos sin clasificar).

    ADI = cantidad de meses / meses con valor positivo.
    CV² = varianza poblacional de los valores estrictamente positivos / media².
    Umbrales convencionales: ADI 1.32 y CV² 0.49. Solo descriptivo.
    """
    n = len(valores_mensuales)
    if not cobertura_y_significado_confirmados:
        return ClasificacionVerificada(None, None, None, n, 0,
                                      'No se ha confirmado la cobertura y el significado de los movimientos.')
    cantidades = [_decimal_explicito(v) for v in valores_mensuales]
    if any(x is None for x in cantidades):
        return ClasificacionVerificada(None, None, None, n, 0,
                                      'Hay meses con cantidades desconocidas o valores no numéricos.')
    if any(x < 0 for x in cantidades):
        return ClasificacionVerificada(None, None, None, n, 0,
                                      'La serie contiene valores negativos que requieren interpretación comercial.')
    positivos = [x for x in cantidades if x > 0]
    if len(positivos) == 0:
        return ClasificacionVerificada(None, None, None, n, 0,
                                      'No hay meses con demanda positiva para caracterizar el patrón.')
    adi = n / len(positivos)
    cv2 = _cv2_positivos(positivos)
    if n < 4 or len(positivos) < 2:
        return ClasificacionVerificada(None, adi, cv2, n, len(positivos),
                                      'Hay muy pocas observaciones para dar una clasificación orientativa.')
    if cv2 is None:
        return ClasificacionVerificada(None, adi, cv2, n, len(positivos),
                                      'No se puede medir la variabilidad de las cantidades positivas.')
    if adi <= 1.32:
        patron = 'regular' if cv2 <= .49 else 'errática'
    else:
        patron = 'intermitente' if cv2 <= .49 else 'irregular'
    return ClasificacionVerificada(patron, adi, cv2, n, len(positivos),
                                  'Clasificación descriptiva; requiere validación temporal para predecir.')


def mostrar_patrones_corto(informe: InformePatronesPreliminares) -> None:
    """Resumen comprensible y sin nuevas preguntas en el modo sencillo."""
    print('\n6. ¿CÓMO SE COMPORTAN LAS VENTAS DE CADA PRODUCTO?')
    if not informe.productos:
        print('No hay productos con datos interpretables para estudiar su actividad.')
        return
    print(f'Productos con cantidades positivas en varios meses: {informe.con_positivos_en_dos_o_mas_meses:,}.'.replace(',', '.'))
    print(f'Productos con cantidades positivas en un solo mes: {informe.con_positivos_en_un_mes:,}.'.replace(',', '.'))
    if informe.sin_positivos_observados:
        print(f'Productos sin ninguna cantidad positiva registrada: {informe.sin_positivos_observados:,}.'.replace(',', '.'))
    if informe.con_movimientos_negativos:
        print(f'Productos con movimientos negativos por aclarar: {informe.con_movimientos_negativos:,}.'.replace(',', '.'))
    if informe.con_cantidades_cero:
        print(f'Productos con operaciones de cantidad cero: {informe.con_cantidades_cero:,}.'.replace(',', '.'))
    print(f'Productos con meses intermedios sin registros: {informe.con_huecos_intermedios:,}.'.replace(',', '.'))
    print('Todavía no conocemos si los meses sin registros significan cero ventas.')
    print('Por eso no hemos clasificado la demanda ni calculado intervalos entre ventas (ADI).')
    print('Cada producto se evaluará por separado: los de poco histórico no bloquean al resto.')
