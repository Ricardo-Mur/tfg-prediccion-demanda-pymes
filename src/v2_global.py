"""Versión 2: XGBoost global, entrenado con todos los productos del archivo a la vez.

Cada fila de entrenamiento es un producto y un mes; sus variables solo usan meses anteriores.
Las ventas se dividen por el nivel reciente del producto (media de los últimos 12 meses conocidos)
para poder mezclar productos grandes y pequeños. Para predecir el mes M solo se entrena con filas
de meses anteriores a M. Mismas variables y parámetros que en el experimento
(prueba_prediccion/global_xgb.py).
"""
from __future__ import annotations

import numpy as np

from .v2_candidatos import tramo_final

PARAMETROS = dict(objective='reg:absoluteerror', n_estimators=300, max_depth=3, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5, random_state=42, verbosity=0)
FILAS_MINIMAS = 30


def rasgos(serie: dict[int, float | None], mes: int) -> tuple[list[float], float] | None:
    """Variables para predecir `mes` con meses anteriores y escala del producto."""
    _, y = tramo_final(serie, mes)
    if len(y) < 3:
        return None
    nivel = float(np.mean(y[-12:]))
    if nivel <= 0:
        return None
    hace_un_ano = serie.get(mes - 12)
    x = [y[-1] / nivel, y[-2] / nivel, y[-3] / nivel,
         (float(hace_un_ano) / nivel) if hace_un_ano is not None else np.nan,
         float(np.mean(y[-3:])) / nivel, (mes % 12) + 1, np.log(nivel), min(len(y), 24)]
    return x, nivel


def entrenar(series: dict[str, dict[int, float | None]], antes_de: int, *, hilos: int = 1):
    """Entrena con todas las filas de meses anteriores a `antes_de`. None si hay menos de 30 filas."""
    from xgboost import XGBRegressor
    X, Y = [], []
    for serie in series.values():
        for mes, valor in serie.items():
            if mes >= antes_de or valor is None:
                continue
            r = rasgos(serie, mes)
            if r is None:
                continue
            X.append(r[0])
            Y.append(float(valor) / r[1])
    if len(X) < FILAS_MINIMAS:
        return None
    return XGBRegressor(**PARAMETROS, n_jobs=hilos).fit(np.array(X, float), np.array(Y, float))


def predecir(modelo, serie: dict[int, float | None], mes: int) -> float | None:
    if modelo is None:
        return None
    historia = {k: v for k, v in serie.items() if k < mes}
    r = rasgos(historia, mes)
    if r is None:
        return None
    return max(float(modelo.predict(np.array([r[0]], float))[0]) * r[1], 0.0)
