"""Fase 4.3: ajuste del método seleccionado sobre el tramo mensual final conocido.

Consume el objetivo mensual de calidad y la selección de la fase 4.2. Comprueba
que las evaluaciones de selección correspondan al MISMO objetivo: no reutilizar
accidentalmente resultados de otro archivo. No infiere ceros y no genera ni
exporta predicciones futuras (eso corresponde a la fase 4.4).

Los hiperparámetros se reproducen literalmente de los experimentos 3.13–3.16;
antes de modificarlos aquí deben actualizarse los experimentos y la validación.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from math import isclose, isfinite
from typing import Any
import logging
import warnings

import pandas as pd

from .evaluacion_observaciones import ESTADOS_CON_VALOR, ESTADOS_PENDIENTES, _indice, _cantidad
from .objetivo_mensual import InformeObjetivoMensual
from .seleccion_modelo import InformeSeleccionModelos, METODOS, seleccionar_modelos

_MINIMOS = {'random_forest': 12, 'xgboost': 12, 'sarima': 24, 'prophet': 24,
            'mes_anterior': 1}
_COLUMNAS = ('producto', 'metodo', 'estado', 'ultimo_mes', 'mes_siguiente',
             'inicio_entrenamiento', 'meses_entrenamiento', 'cortes_validacion',
             'seleccion_provisional', 'motivo', 'prediccion_futura_generada')


@dataclass
class AjusteFinal:
    producto: str
    metodo: str | None
    estado: str
    ultimo_mes: date | None
    mes_siguiente: date | None
    inicio_entrenamiento: date | None
    meses_entrenamiento: int
    cortes_validacion: int
    seleccion_provisional: bool
    motivo: str
    prediccion_futura_generada: bool = False
    # Mantener el modelo solo en memoria; nunca serializarlo en CSV.
    modelo: Any = field(default=None, repr=False)
    valores_finales: tuple[float, ...] = field(default=(), repr=False)
    valor_referencia: float | None = field(default=None, repr=False)


@dataclass
class InformeReentrenamiento:
    resumen: pd.DataFrame
    ajustes: dict[str, AjusteFinal]
    ajustados: int
    referencias_preparadas: int
    no_preparados: int
    predicciones_futuras: int = 0


def _fecha(i: int) -> date:
    return date(i // 12, i % 12 + 1, 1)


def _validar_objetivo(objetivo: InformeObjetivoMensual) -> dict[str, dict[int, float | None]]:
    if not hasattr(objetivo, 'detalle') or not isinstance(objetivo.detalle, pd.DataFrame):
        raise ValueError('Falta el objetivo mensual validado.')
    tabla = objetivo.detalle
    campos = {'producto', 'mes', 'estado', 'unidades_observadas'}
    if not campos.issubset(tabla.columns) or tabla.empty:
        raise ValueError('Objetivo mensual vacío o incompleto.')
    if tabla.duplicated(['producto', 'mes']).any():
        raise ValueError('Objetivo mensual con meses duplicados.')
    series: dict[str, dict[int, float | None]] = {}
    for r in tabla.itertuples(index=False):
        if not isinstance(r.producto, str) or not r.producto.strip():
            raise ValueError('Identificador de producto inválido.')
        i = _indice(r.mes)  # fechas date y día 1, como los modelos originales.
        v = _cantidad(r.unidades_observadas)
        if r.estado in ESTADOS_CON_VALOR:
            if v is None or (r.estado == 'cero_explicito' and v != 0) or (
                    r.estado == 'positivo_registrado' and v <= 0):
                raise ValueError('Cantidad y estado mensual contradictorios.')
            try:
                n = float(v)
            except (OverflowError, ValueError) as exc:
                raise ValueError('Cantidad fuera del rango representable.') from exc
            if not isfinite(n):
                raise ValueError('Cantidad no finita.')
        elif r.estado in ESTADOS_PENDIENTES:
            if v is not None:
                raise ValueError('Mes desconocido o pendiente con valor asignado.')
            n = None
        else:
            raise ValueError('Estado mensual no reconocido.')
        serie = series.setdefault(r.producto, {})
        if i in serie:
            raise ValueError('Mes repetido tras normalizar la fecha.')
        serie[i] = n
    return series


def _comprobar_evaluaciones(evaluaciones: pd.DataFrame,
                           series: dict[str, dict[int, float | None]],
                           seleccionados: set[str]) -> None:
    """Previene seleccionar con evaluaciones antiguas del mismo código de producto."""
    cols = {'producto', 'mes_evaluado', 'real_provisional', 'metodo',
            'inicio_entrenamiento', 'fin_entrenamiento'}
    if not isinstance(evaluaciones, pd.DataFrame) or not cols.issubset(evaluaciones.columns):
        raise ValueError('No se han aportado las evaluaciones vinculadas a la selección.')
    if not seleccionados.issubset(set(evaluaciones['producto'].astype(str))):
        raise ValueError('Faltan productos seleccionados en las evaluaciones aportadas.')
    for r in evaluaciones.itertuples(index=False):
        p = str(r.producto)
        if p not in seleccionados:
            continue
        if p not in series:
            raise ValueError(f'{p}: el producto ya no existe en el objetivo mensual.')
        f = pd.Timestamp(r.mes_evaluado)
        if pd.isna(f) or f.day != 1:
            raise ValueError('Fecha inválida en evaluaciones guardadas.')
        i = f.year * 12 + f.month - 1
        real = series[p].get(i)
        try:
            observado = float(str(r.real_provisional).replace(',', '.'))
        except (ValueError, TypeError) as exc:
            raise ValueError('Real de evaluación inválido.') from exc
        if real is None or not isfinite(observado) or not isclose(
                real, observado, rel_tol=1e-12, abs_tol=0.000005):
            raise ValueError(f'{p}: las evaluaciones no corresponden al objetivo mensual actual.')
        fi = pd.Timestamp(r.inicio_entrenamiento)
        ff = pd.Timestamp(r.fin_entrenamiento)
        if pd.isna(fi) or pd.isna(ff) or fi.day != 1 or ff.day != 1 or not (
                fi <= ff and ff.to_period('M') == f.to_period('M') - 1):
            raise ValueError('Evaluación con fechas de entrenamiento inconsistentes.')
        inicio_i = fi.year * 12 + fi.month - 1
        fin_i = ff.year * 12 + ff.month - 1
        if any(series[p].get(k) is None for k in range(inicio_i, fin_i + 1)):
            raise ValueError(f'{p}: la evaluación original cruza una laguna desconocida.')


def _ajustar(metodo: str, datos: list[float], inicio: int) -> Any:
    """Entrena, sin generar un pronóstico. Hiperparámetros iguales a 3.13–3.16."""
    if metodo == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor
        x = [[datos[j-k] for k in (1, 2, 3)] for j in range(3, len(datos))]
        y = datos[3:]
        modelo = RandomForestRegressor(n_estimators=30, random_state=42,
                                       n_jobs=1, min_samples_leaf=2)
        modelo.fit(x, y)
        return modelo
    if metodo == 'xgboost':
        from xgboost import XGBRegressor
        x = [[datos[j-k] for k in (1, 2, 3)] for j in range(3, len(datos))]
        y = datos[3:]
        modelo = XGBRegressor(objective='reg:squarederror', n_estimators=60,
                 max_depth=2, learning_rate=0.1, subsample=1.0,
                 colsample_bytree=1.0, reg_lambda=1.0, random_state=42,
                 n_jobs=1, tree_method='hist', verbosity=0)
        modelo.fit(x, y)
        return modelo
    if metodo == 'sarima':
        import numpy as np
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            modelo = SARIMAX(np.asarray(datos, dtype=float), order=(0, 0, 0),
                    seasonal_order=(1, 0, 0, 12), trend='c',
                    enforce_stationarity=False, enforce_invertibility=False)
            ajuste = modelo.fit(disp=False, maxiter=80)
        if not ajuste.mle_retvals.get('converged', False):
            raise RuntimeError('SARIMA no convergió; no se considera ajustado.')
        return ajuste
    if metodo == 'prophet':
        from prophet import Prophet
        fechas = pd.to_datetime([_fecha(inicio + j) for j in range(len(datos))])
        entrenamiento = pd.DataFrame({'ds': fechas, 'y': datos})
        modelo = Prophet(yearly_seasonality=4, weekly_seasonality=False,
                daily_seasonality=False, seasonality_mode='additive',
                changepoint_prior_scale=0.05, uncertainty_samples=0,
                mcmc_samples=0)
        logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
        modelo.fit(entrenamiento)
        return modelo
    raise ValueError(f'Método sin algoritmo de ajuste: {metodo}')


def reentrenar_seleccionados(
    objetivo: InformeObjetivoMensual,
    seleccion: InformeSeleccionModelos,
    evaluaciones: pd.DataFrame,
) -> InformeReentrenamiento:
    """Ajusta los métodos propuestos sin prometer fiabilidad ni prever el futuro.

    No admite mezclas de históricos: compara cada real de evaluación con el
    objetivo actual. Por prudencia, solo prepara productos cuyo último mes
    coincide con el mes más reciente del conjunto y tiene dato conocido.
    Si hay un hueco, usa SOLO el tramo final consecutivo; no lo rellena.
    """
    series = _validar_objetivo(objetivo)
    if not hasattr(seleccion, 'resumen') or not isinstance(seleccion.resumen, pd.DataFrame):
        raise ValueError('Falta el informe de selección de la fase 4.2.')
    tabla = seleccion.resumen
    columnas = {'producto', 'metodo_sugerido', 'estado', 'cortes_comunes',
                'seleccion_definitiva'}
    if not columnas.issubset(tabla.columns) or tabla.duplicated('producto').any():
        raise ValueError('Selección incompleta o con productos duplicados.')
    if tabla.empty:
        return InformeReentrenamiento(pd.DataFrame(columns=_COLUMNAS), {}, 0, 0, 0)
    if tabla['producto'].isna().any() or not set(tabla['producto']).issubset(series):
        raise ValueError('La selección incluye productos ausentes del objetivo.')
    _comprobar_evaluaciones(evaluaciones, series, set(tabla['producto']))
    # No aceptar una selección de otro CSV o modificada después de validar.
    recalculada = seleccionar_modelos(
        evaluaciones, cortes_minimos=seleccion.cortes_minimos,
        maximo_cortes=seleccion.maximo_cortes,
        umbral_mejora_relativa=seleccion.umbral_mejora_relativa,
    ).resumen.set_index('producto')
    for r in tabla.itertuples(index=False):
        if r.producto not in recalculada.index:
            raise ValueError('La selección no corresponde a las evaluaciones aportadas.')
        correcto = recalculada.loc[r.producto]
        a = None if pd.isna(r.metodo_sugerido) else r.metodo_sugerido
        b = None if pd.isna(correcto.metodo_sugerido) else correcto.metodo_sugerido
        if a != b or r.estado != correcto.estado or int(r.cortes_comunes) != int(correcto.cortes_comunes):
            raise ValueError(f'{r.producto}: la selección no coincide con las evaluaciones.')
    global_max = max(max(s) for s in series.values())
    ajustes: dict[str, AjusteFinal] = {}
    for row in tabla.itertuples(index=False):
        producto = row.producto
        metodo = row.metodo_sugerido
        if pd.isna(metodo):
            metodo = None
        if metodo is not None and metodo not in METODOS:
            raise ValueError('Método sugerido desconocido.')
        s = series[producto]
        ultimo_i = max(s)
        fin = _fecha(ultimo_i)
        siguiente = _fecha(ultimo_i + 1)
        cortes = int(row.cortes_comunes)
        validacion_provisional = not bool(row.seleccion_definitiva)
        estado = ''
        motivo = ''
        modelo = None
        valor_referencia = None
        final = []
        ini = ultimo_i + 1
        if row.estado == 'evidencia_insuficiente' or metodo is None:
            estado, motivo = 'sin_seleccion', 'No hay método con evidencia comparativa suficiente.'
        elif ultimo_i != global_max:
            estado, motivo = 'historico_desactualizado', 'El producto no llega al último mes del conjunto.'
        elif s[ultimo_i] is None:
            estado, motivo = 'ultimo_mes_desconocido', 'El último mes no contiene un valor confirmado.'
        else:
            # No cruzar meses ausentes o pendientes, aunque haya datos anteriores.
            j = ultimo_i
            while j in s and s[j] is not None:
                j -= 1
            ini = j + 1
            final = [s[k] for k in range(ini, ultimo_i + 1)]
            if metodo == 'anual':
                # Para enero de 2026, referencia anual = enero de 2025.
                # No se exige continuidad de los once meses intermedios.
                valor_referencia = s.get(ultimo_i - 11)
                if valor_referencia is None:
                    estado, motivo = 'referencia_no_disponible', (
                        'No hay dato conocido del mismo mes del año anterior.')
                else:
                    estado, motivo = 'referencia_preparada', (
                        'Referencia anual disponible sin inventar meses intermedios.')
            elif metodo == 'mes_anterior':
                valor_referencia = s[ultimo_i]
                estado, motivo = 'referencia_preparada', (
                    'Referencia del último mes conocido disponible.')
            elif len(final) < _MINIMOS[metodo]:
                estado, motivo = 'historico_insuficiente', (
                    f'El tramo final conocido tiene {len(final)} meses; '
                    f'{metodo} requiere {_MINIMOS[metodo]} en la configuración actual.')
            else:
                try:
                    modelo = _ajustar(metodo, final, ini)
                    estado, motivo = 'ajustado', (
                        'Modelo ajustado con el tramo final conocido; selección todavía provisional.')
                except Exception as exc:
                    # Un fallo de ajuste nunca se transforma en una predicción válida.
                    estado, motivo = 'error_ajuste', f'{type(exc).__name__}: {exc}'
        ajuste = AjusteFinal(
            producto=producto, metodo=metodo, estado=estado, ultimo_mes=fin,
            mes_siguiente=siguiente, inicio_entrenamiento=_fecha(ini) if final else None,
            meses_entrenamiento=len(final), cortes_validacion=cortes,
            seleccion_provisional=validacion_provisional, motivo=motivo,
            modelo=modelo, valores_finales=tuple(final),
            valor_referencia=valor_referencia)
        ajustes[producto] = ajuste
    resumen = pd.DataFrame([{
        c: getattr(a, c) for c in _COLUMNAS
    } for a in ajustes.values()], columns=_COLUMNAS)
    return InformeReentrenamiento(
        resumen=resumen, ajustes=ajustes,
        ajustados=sum(a.estado == 'ajustado' for a in ajustes.values()),
        referencias_preparadas=sum(a.estado == 'referencia_preparada' for a in ajustes.values()),
        no_preparados=sum(a.estado not in {'ajustado','referencia_preparada'} for a in ajustes.values()),
        predicciones_futuras=0,
    )
