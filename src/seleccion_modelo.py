"""Fase 4.2: recomendación EXPLORATORIA de método a partir de cortes ya evaluados.

No entrena modelos, no predice el futuro y no utiliza valores no observados.
Compara MAE y RMSE estrictamente sobre los mismos meses por producto.
El umbral relativo es configurable y experimental, NO un criterio universal
para recomendar modelos con garantías de precisión.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt

import pandas as pd

REFERENCIAS = ('mes_anterior', 'anual')
MODELOS = ('random_forest', 'xgboost', 'sarima', 'prophet')
METODOS = REFERENCIAS + MODELOS
COLUMNAS_REQUERIDAS = (
    'producto', 'mes_evaluado', 'real_provisional', 'metodo',
    'prediccion', 'inicio_entrenamiento', 'fin_entrenamiento',
)
COLUMNAS_RESUMEN = (
    'producto', 'estado', 'metodo_sugerido', 'mejor_referencia',
    'mejor_modelo_complejo', 'cortes_comunes', 'mae_sugerido',
    'rmse_sugerido', 'mae_referencia', 'rmse_referencia',
    'mejora_relativa_mejor_modelo', 'metodos_comparados',
    'metodos_excluidos', 'motivo', 'seleccion_definitiva',
    'prediccion_futura_generada',
)
COLUMNAS_METRICAS = ('producto', 'metodo', 'cortes_comunes', 'mae', 'rmse')


@dataclass
class InformeSeleccionModelos:
    """Resultado por producto y métricas de los métodos sobre fechas comunes."""
    resumen: pd.DataFrame
    metricas: pd.DataFrame
    umbral_mejora_relativa: float
    cortes_minimos: int
    maximo_cortes: int


def _fecha_mensual(serie: pd.Series, campo: str) -> pd.Series:
    # Se aceptan timestamps o cadenas ISO, pero no meses indeterminados.
    fechas = pd.to_datetime(serie, errors='coerce')
    if fechas.isna().any() or (fechas.dt.day != 1).any():
        raise ValueError(f'{campo}: se esperan fechas válidas de inicio de mes.')
    return fechas


def _numerico(serie: pd.Series, campo: str) -> pd.Series:
    # Compatible tanto con los CSV exportados (decimal coma) como con DataFrames.
    convertido = pd.to_numeric(serie.astype(str).str.replace(',', '.', regex=False), errors='coerce')
    if convertido.isna().any() or not converted_finite_nonnegative(convertido):
        raise ValueError(f'{campo}: existen valores ausentes, negativos o no finitos.')
    return convertido.astype(float)


def converted_finite_nonnegative(serie: pd.Series) -> bool:
    return bool(serie.map(lambda x: isfinite(float(x)) and x >= 0).all())


def _metricas(reales: list[float], predicciones: list[float]) -> tuple[float, float]:
    if not reales or len(reales) != len(predicciones):
        raise ValueError('Las predicciones y los reales no tienen fechas pareadas.')
    errores = [r - p for r, p in zip(reales, predicciones)]
    return (sum(abs(e) for e in errores) / len(errores),
            sqrt(sum(e * e for e in errores) / len(errores)))


def _mejor(datos: dict[str, tuple[float, float]], orden: tuple[str, ...]) -> str:
    # MAE principal, RMSE desempate, orden fijo para evitar elegir al azar.
    return min(datos, key=lambda m: (round(datos[m][0], 10),
                                     round(datos[m][1], 10), orden.index(m)))


def seleccionar_modelos(
    evaluaciones: pd.DataFrame,
    *,
    cortes_minimos: int = 3,
    maximo_cortes: int = 6,
    umbral_mejora_relativa: float = 0.10,
) -> InformeSeleccionModelos:
    """Compara métodos retrospectivamente y ofrece una recomendación PROVISIONAL.

    Regla de trabajo (ajustable y aún por validar): un modelo complejo solo se
    sugiere si reduce al menos un 10 % el MAE de la mejor referencia y no
    empeora su RMSE. Si no se demuestra esa mejora, se conserva la referencia.

    Los meses de comparación son los últimos hasta `maximo_cortes` en los que
    ambas referencias están disponibles; si no hay suficientes, se usa la
    referencia con más cortes. Todo método debe cubrir TODOS esos meses.
    No se compara un MAE calculado sobre fechas distintas del de otro método.
    """
    if type(cortes_minimos) is not int or cortes_minimos < 3:
        raise ValueError('Se exigen al menos tres cortes cronológicos.')
    if type(maximo_cortes) is not int or maximo_cortes < cortes_minimos:
        raise ValueError('maximo_cortes debe ser >= cortes_minimos.')
    if isinstance(umbral_mejora_relativa, bool) or not isinstance(umbral_mejora_relativa, (int, float)) or not isfinite(umbral_mejora_relativa) or not 0 <= umbral_mejora_relativa < 1:
        raise ValueError('El umbral relativo debe estar entre 0 (incluido) y 1 (excluido).')
    if not isinstance(evaluaciones, pd.DataFrame):
        raise TypeError('Las evaluaciones deben ser un DataFrame.')
    if not set(COLUMNAS_REQUERIDAS).issubset(evaluaciones.columns):
        raise ValueError('Faltan columnas obligatorias en evaluaciones_experimentales.csv.')

    if evaluaciones.empty:
        return InformeSeleccionModelos(pd.DataFrame(columns=COLUMNAS_RESUMEN),
            pd.DataFrame(columns=COLUMNAS_METRICAS), float(umbral_mejora_relativa),
            cortes_minimos, maximo_cortes)

    df = evaluaciones.loc[:, COLUMNAS_REQUERIDAS].copy()
    if df['producto'].isna().any() or df['metodo'].isna().any():
        raise ValueError('Hay productos o métodos sin identificar.')
    df['producto'] = df['producto'].map(str)
    df['metodo'] = df['metodo'].map(str)
    if df['producto'].map(lambda x: not x.strip()).any():
        raise ValueError('Hay identificadores de producto vacíos.')
    if not set(df['metodo']).issubset(METODOS):
        raise ValueError('El archivo contiene nombres de métodos no reconocidos.')

    for campo in ('mes_evaluado', 'inicio_entrenamiento', 'fin_entrenamiento'):
        df[campo] = _fecha_mensual(df[campo], campo)
    for campo in ('real_provisional', 'prediccion'):
        df[campo] = _numerico(df[campo], campo)

    if df.duplicated(['producto', 'mes_evaluado', 'metodo']).any():
        raise ValueError('Se ha repetido una predicción para producto, mes y método.')
    if (df['inicio_entrenamiento'] > df['fin_entrenamiento']).any():
        raise ValueError('Hay entrenamientos con fechas invertidas.')
    if (df['fin_entrenamiento'].dt.to_period('M') !=
            df['mes_evaluado'].dt.to_period('M') - 1).any():
        raise ValueError('Hay cortes sin entrenamiento anterior al mes evaluado.')
    if df.groupby(['producto', 'mes_evaluado'])['real_provisional'].nunique().gt(1).any():
        raise ValueError('Existen valores reales contradictorios para el mismo producto y mes.')

    filas, filas_metricas = [], []
    for producto, grupo in df.groupby('producto', sort=True):
        tabla = grupo.pivot(index='mes_evaluado', columns='metodo', values='prediccion')
        valores_reales = grupo.drop_duplicates('mes_evaluado').set_index('mes_evaluado')['real_provisional']
        refs = [m for m in REFERENCIAS if m in tabla.columns]
        fechas: list = []
        if len(refs) == 2:
            comunes = sorted(tabla.index[tabla[refs].notna().all(axis=1)].tolist())
            if len(comunes) >= cortes_minimos:
                fechas = comunes[-maximo_cortes:]
        if not fechas and refs:
            # Si las referencias no tienen suficientes fechas comunes,
            # se usa una sola; NO se comparan MAE entre ventanas distintas.
            refs_ordenadas = sorted(refs, key=lambda m: (-int(tabla[m].notna().sum()),
                                                        REFERENCIAS.index(m)))
            referencia_ancla = refs_ordenadas[0]
            fechas = sorted(tabla.index[tabla[referencia_ancla].notna()].tolist())[-maximo_cortes:]

        inicial = dict(producto=producto, estado='evidencia_insuficiente',
            metodo_sugerido=None, mejor_referencia=None, mejor_modelo_complejo=None,
            cortes_comunes=len(fechas), mae_sugerido=None, rmse_sugerido=None,
            mae_referencia=None, rmse_referencia=None,
            mejora_relativa_mejor_modelo=None, metodos_comparados='',
            metodos_excluidos='', motivo='No hay tres cortes para una referencia.',
            seleccion_definitiva=False, prediccion_futura_generada=False)
        if len(fechas) < cortes_minimos:
            filas.append(inicial)
            continue

        comparables = {m: tabla.loc[fechas, m].astype(float).tolist()
                       for m in METODOS if m in tabla.columns
                       and bool(tabla.loc[fechas, m].notna().all())}
        refs_validas = {m: comparables[m] for m in REFERENCIAS if m in comparables}
        if not refs_validas:
            filas.append(inicial)
            continue
        reales = valores_reales.loc[fechas].astype(float).tolist()
        puntuaciones = {m: _metricas(reales, pred) for m, pred in comparables.items()}
        for m, (mae, rmse) in puntuaciones.items():
            filas_metricas.append({'producto': producto, 'metodo': m,
                'cortes_comunes': len(fechas), 'mae': mae, 'rmse': rmse})
        puntuacion_refs = {m: puntuaciones[m] for m in REFERENCIAS if m in puntuaciones}
        puntuacion_modelos = {m: puntuaciones[m] for m in MODELOS if m in puntuaciones}
        mejor_ref = _mejor(puntuacion_refs, REFERENCIAS)
        mae_ref, rmse_ref = puntuaciones[mejor_ref]
        mejor_modelo = _mejor(puntuacion_modelos, MODELOS) if puntuacion_modelos else None
        mejora = None
        if mejor_modelo is not None:
            mae_modelo = puntuaciones[mejor_modelo][0]
            mejora = (mae_ref - mae_modelo) / mae_ref if mae_ref > 0 else 0.0
        # Un modelo debe pasar ambos filtros. Si varios pasan, vence el de menor MAE/RMSE.
        aprobados = {m: puntuaciones[m] for m in MODELOS if m in puntuaciones and
            mae_ref > 0 and puntuaciones[m][0] < mae_ref and
            (mae_ref - puntuaciones[m][0]) / mae_ref >= umbral_mejora_relativa and
            puntuaciones[m][1] <= rmse_ref + 1e-10}
        sugerido = _mejor(aprobados, MODELOS) if aprobados else mejor_ref
        resultado = inicial | {
            'estado': 'modelo_complejo_candidato' if aprobados else 'referencia_preferida',
            'metodo_sugerido': sugerido,
            'mejor_referencia': mejor_ref,
            'mejor_modelo_complejo': mejor_modelo,
            'mae_sugerido': puntuaciones[sugerido][0],
            'rmse_sugerido': puntuaciones[sugerido][1],
            'mae_referencia': mae_ref, 'rmse_referencia': rmse_ref,
            'mejora_relativa_mejor_modelo': mejora,
            'metodos_comparados': ','.join(m for m in METODOS if m in comparables),
            'metodos_excluidos': ','.join(m for m in METODOS if m in tabla.columns and m not in comparables),
            'motivo': ('Mejora experimental frente a referencia; requiere más validación.'
                       if aprobados else 'Sin mejora suficiente y estable frente a referencia.'),
        }
        filas.append(resultado)

    return InformeSeleccionModelos(
        resumen=pd.DataFrame(filas, columns=COLUMNAS_RESUMEN),
        metricas=pd.DataFrame(filas_metricas, columns=COLUMNAS_METRICAS),
        umbral_mejora_relativa=float(umbral_mejora_relativa),
        cortes_minimos=cortes_minimos, maximo_cortes=maximo_cortes)


def leer_evaluaciones_csv(ruta: str) -> pd.DataFrame:
    """Abre los CSV exportados por el prototipo (punto y coma, decimal coma)."""
    return pd.read_csv(ruta, sep=';', decimal=',', encoding='utf-8-sig',
                       dtype={'producto': 'string', 'metodo': 'string'})
