"""Versión 2: métodos candidatos por producto.

Cada función recibe solo el histórico ANTERIOR al mes que se predice y devuelve la predicción de ese
mes, o None si el método no puede aplicarse con ese histórico. Las series son diccionarios
{índice de mes: unidades}; un valor None significa mes desconocido o pendiente (nunca un cero).
Los ceros confirmados son valores normales.

Las definiciones coinciden con las del experimento que justificó la versión 2
(prueba_prediccion/metodos.py y matriz.py), salvo `media_12`, que desde la etapa 2 exige 12 meses.

ARIMA: se usa UNA configuración fija inicial, no una búsqueda dentro de la familia ARIMA/SARIMA:
SARIMA(0,1,1)(0,1,1)12 con 36 meses seguidos o más y ARIMA(0,1,1) con 12 a 35 meses.
"""
from __future__ import annotations

import warnings
from math import isfinite

import numpy as np
import pandas as pd

# Orden fijo: también se usa para desempatar.
CANDIDATOS_POR_PRODUCTO = ('mes_anterior', 'anual', 'media_12', 'ses', 'ets', 'theta', 'arima_aerolinea')

# Historial consecutivo mínimo (en meses) que exige cada método.
MINIMOS = {'mes_anterior': 1, 'media_12': 12, 'ses': 3, 'ets': 4, 'theta': 6, 'arima_aerolinea': 12}

# Nombres para explicar los resultados a una persona sin conocimientos técnicos.
NOMBRE_METODO = {
    'mes_anterior': 'repetir las ventas del mes anterior',
    'anual': 'repetir las ventas del mismo mes del año anterior',
    'media_12': 'media de los últimos 12 meses',
    'ses': 'suavizado exponencial simple',
    'ets': 'suavizado exponencial con tendencia (ETS)',
    'theta': 'método Theta',
    'arima_aerolinea': 'ARIMA (configuración fija)',
    'global_xgb': 'XGBoost entrenado con todos los productos del archivo',
}


def tramo_final(serie: dict[int, float | None], mes: int) -> tuple[int | None, list[float]]:
    """Meses conocidos y consecutivos que terminan justo antes de `mes`.

    Un mes desconocido corta el tramo: no se rellena ni se salta.
    """
    if serie.get(mes - 1) is None:
        return None, []
    j = mes - 1
    while serie.get(j) is not None:
        j -= 1
    return j + 1, [float(serie[k]) for k in range(j + 1, mes)]


def _no_negativo(valor) -> float | None:
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return None
    return max(v, 0.0) if isfinite(v) else None


def ses(y: list[float]) -> float | None:
    if len(y) < MINIMOS['ses']:
        return None
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = SimpleExpSmoothing(np.asarray(y, float), initialization_method='estimated').fit()
            return _no_negativo(m.forecast(1)[0])
    except Exception:
        return None


def ets(y: list[float]) -> float | None:
    """Suavizado exponencial: estacionalidad anual desde 24 meses; tendencia amortiguada desde 6."""
    if len(y) < MINIMOS['ets']:
        return None
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    a = np.asarray(y, float)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if len(a) >= 24:
                m = ExponentialSmoothing(a, trend='add', damped_trend=True, seasonal='add',
                                         seasonal_periods=12, initialization_method='estimated').fit()
            elif len(a) >= 6:
                m = ExponentialSmoothing(a, trend='add', damped_trend=True,
                                         initialization_method='estimated').fit()
            else:
                m = ExponentialSmoothing(a, initialization_method='estimated').fit()
            return _no_negativo(m.forecast(1)[0])
    except Exception:
        return None


def theta(y: list[float]) -> float | None:
    """Método Theta; desestacionaliza solo con 24 meses y si el contraste detecta estacionalidad."""
    if len(y) < MINIMOS['theta']:
        return None
    from statsmodels.tsa.forecasting.theta import ThetaModel
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            s = pd.Series(np.asarray(y, float), index=pd.period_range('2000-01', periods=len(y), freq='M'))
            m = ThetaModel(s, period=12, deseasonalize=len(y) >= 24, use_test=True, method='auto').fit()
            return _no_negativo(m.forecast(1).iloc[0])
    except Exception:
        return None


def arima_aerolinea(y: list[float]) -> float | None:
    """SARIMA(0,1,1)(0,1,1)12 desde 36 meses; ARIMA(0,1,1) desde 12. Sigue tendencias."""
    if len(y) < MINIMOS['arima_aerolinea']:
        return None
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    a = np.asarray(y, float)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if len(a) >= 36:
                m = SARIMAX(a, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12))
            else:
                m = SARIMAX(a, order=(0, 1, 1))
            return _no_negativo(np.asarray(m.fit(disp=False, maxiter=60).forecast(1)).reshape(-1)[0])
    except Exception:
        return None


def ets_sin_estacionalidad(y: list[float]) -> float | None:
    """Solo para el diagnóstico del patrón anual: ETS con tendencia amortiguada y sin estacionalidad.
    Se calcula cuando hay 24 meses o más, para compararlo con el ETS estacional. No se usa para prever."""
    if len(y) < 24:
        return None
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = ExponentialSmoothing(np.asarray(y, float), trend='add', damped_trend=True,
                                     initialization_method='estimated').fit()
            return _no_negativo(m.forecast(1)[0])
    except Exception:
        return None


DIAGNOSTICO = 'diag_ets_sin_estacionalidad'


def predecir_candidatos(historia: dict[int, float | None], mes: int, *, diagnostico: bool = False) -> dict[str, float]:
    """Predicciones de los siete candidatos por producto para `mes`.

    `historia` solo debe contener meses anteriores a `mes`; se comprueba para evitar fugas.
    """
    if any(k >= mes for k in historia):
        raise ValueError('La historia contiene meses iguales o posteriores al mes previsto.')
    _, y = tramo_final(historia, mes)
    if not y:
        return {}
    out = {'mes_anterior': y[-1]}
    anual = historia.get(mes - 12)
    if anual is not None:
        out['anual'] = float(anual)
    if len(y) >= MINIMOS['media_12']:
        out['media_12'] = float(np.mean(y[-12:]))   # siempre con 12 meses conocidos
    for nombre, funcion in (('ses', ses), ('ets', ets), ('theta', theta), ('arima_aerolinea', arima_aerolinea)):
        v = funcion(y)
        if v is not None:
            out[nombre] = v
    if diagnostico:
        v = ets_sin_estacionalidad(y)
        if v is not None:
            out[DIAGNOSTICO] = v
    return out


def theta_desestacionaliza(y: list[float]) -> bool:
    """¿Ha detectado Theta estacionalidad anual (y por tanto la ha tenido en cuenta) con este historial?"""
    if len(y) < 24:
        return False
    from statsmodels.tsa.forecasting.theta import ThetaModel
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            s = pd.Series(np.asarray(y, float), index=pd.period_range('2000-01', periods=len(y), freq='M'))
            m = ThetaModel(s, period=12, deseasonalize=True, use_test=True, method='auto')
            m.fit()
            return bool(getattr(m, '_has_seasonality', False))
    except Exception:
        return False


def metodos_con_mes_del_ano(metodos: list[str], y: list[float]) -> list[str]:
    """Métodos de la lista que reciben información del mes del año con ese historial.

    - anual: repite el mismo mes del año anterior.
    - global_xgb: recibe el mes del año y, cuando se conocen, las ventas del mismo mes del año anterior.
    - ets: con 24 meses o más incluye un componente anual.
    - arima_aerolinea: con 36 meses o más incluye un componente anual.
    - theta: con 24 meses o más, solo si su contraste detecta estacionalidad.
    Que un método reciba esa información no demuestra que la previsión mejore por ello.
    """
    out = []
    for m in metodos:
        if m in ('anual', 'global_xgb') or (m == 'ets' and len(y) >= 24) or \
                (m == 'arima_aerolinea' and len(y) >= 36) or (m == 'theta' and theta_desestacionaliza(y)):
            out.append(m)
    return out
