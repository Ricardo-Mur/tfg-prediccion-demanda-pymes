"""Versión 2 del motor de predicción (un mes vista, por producto).

Funcionamiento:
1. Ocho candidatos: mes anterior, mismo mes del año anterior, media de 12 meses, SES, ETS, Theta,
   ARIMA (0,1,1)(0,1,1)12 y un XGBoost global entrenado con todos los productos del archivo.
2. Para cada producto se comprueba cada candidato en los últimos 12 meses conocidos y consecutivos
   (cada predicción, solo con datos anteriores a su mes). La previsión es la media de los 3 candidatos
   con menos error medio absoluto (desempate: RMSE y, después, el orden fijo de los candidatos).
3. Esa misma regla se aplica a cada uno de los 12 meses anteriores para obtener el error que habrían
   tenido de verdad las previsiones de la aplicación. De ahí salen el estado, el rango orientativo
   y la comparación con repetir el mes anterior.

Estados:
  buen_respaldo        error acumulado de las previsiones de los meses comprobados hasta el 15 %, 12 meses
                       comprobados, ventas suficientes en ellos, previsión distinta de cero y error estable
                       (ver «Estabilidad»)
  orientativa          error hasta el 35 %, o error bajo con menos de 12 meses comprobados, previsión cero,
                       predicciones recientes peores o estabilidad no valorable
  no_recomendada       error superior al 35 %, ventas demasiado esporádicas para comprobar la previsión,
                       producto sin actividad reciente o producto retirado: no se ofrece cantidad
  datos_insuficientes  menos de 6 meses comprobados, ningún método evaluable, o histórico / estado
                       comercial sin confirmar (ver v2_completitud)

El error acumulado (WAPE) es la suma de las diferencias entre lo previsto y lo vendido dividida entre lo
vendido en esos meses. No es el porcentaje medio de error de cada mes. Si en los meses comprobados no hubo
ventas no se calcula.

Estabilidad (aprobada el 15-11-2026, ver PROPUESTA_ESTABILIDAD.md): los 12 meses comprobados se separan en los 6
más antiguos (A) y los 6 más recientes (B), con el mismo error acumulado en cada mitad (E_A, E_B). Si alguna
mitad tiene ventas en menos de 2 meses, la estabilidad no se puede valorar. Hay empeoramiento si E_B > 20 %, o
si E_B >= 2 * E_A y E_B > 10 %. Solo puede rebajar «buen respaldo» a «orientativa», y solo entonces se explica en
el mensaje. En las que ya eran «orientativa» no cambia nada ni añade texto (decisión del 15-11-2026: ese aviso
salía en la mayoría y no distinguía el riesgo); los errores de las dos mitades quedan en columnas técnicas.
No cambia la cantidad prevista ni los métodos. Describe el error observado en esos meses, que pueden ser épocas
del año distintas: no demuestra un cambio en las ventas.

Reglas que se mantienen de la versión 1: un mes desconocido o pendiente nunca se trata como cero ni se
rellena; un cero confirmado es un valor normal; solo se usa el tramo final de meses conocidos seguidos.

Revisiones (etapa 4): una anulación registrada en un mes posterior corrige las ventas de un mes anterior. El
histórico final ya trae la corrección, pero en la comprobación retrospectiva cada mes se prevé con lo que se
sabía entonces: al prever el mes m, una venta del mes k < m solo aparece corregida si la anulación se registró
antes de m. Así, ni las predicciones de prueba ni la elección de métodos usan anulaciones que todavía no se
conocían. Si la anulación dejó el mes k sin registros y el histórico no está confirmado, el mes k es desconocido
en el histórico final, pero antes del registro se conocía su venta: entonces el tramo de meses seguidos llegaba
más atrás, y así se reconstruye. El valor con el que se mide el error de cada mes es el definitivo.

«Definitivo» quiere decir conocido al cerrar el último mes utilizable. Una anulación registrada después (por
ejemplo, durante el mes incompleto que se va a prever) no se conocía entonces: se deshace en todo el cálculo
—candidatos, XGBoost global, referencias, elección de métodos y valores de comprobación— y no influye en la
previsión. Su fecha de registro se conserva tal cual; nunca se adelanta.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from math import isfinite
from pathlib import Path

import numpy as np
import pandas as pd

from . import v2_candidatos as C
from . import v2_global as G
from .objetivo_mensual import InformeObjetivoMensual
from .reentrenamiento_final import _validar_objetivo

CANDIDATOS = C.CANDIDATOS_POR_PRODUCTO + ('global_xgb',)
ESTADOS = ('buen_respaldo', 'orientativa', 'no_recomendada', 'datos_insuficientes')
NOMBRE_ESTADO = {'buen_respaldo': 'Predicción con buen respaldo', 'orientativa': 'Predicción orientativa',
                 'no_recomendada': 'Predicción no recomendada', 'datos_insuficientes': 'Datos insuficientes'}
COLUMNAS_RESUMEN = (
    'producto', 'estado', 'mes_previsto', 'prediccion', 'prediccion_exploratoria', 'rango_inferior',
    'rango_superior', 'error_historico', 'meses_comprobados', 'mae_version2', 'mae_mes_anterior',
    'supera_mes_anterior', 'mejora_vs_mes_anterior', 'metodos_combinados', 'candidatos_validos',
    'candidatos_excluidos', 'patron_anual', 'metodos_con_mes_del_ano', 'meses_con_ventas_comprobados',
    'unidades_comprobadas', 'estabilidad', 'error_mitad_antigua', 'error_mitad_reciente',
    'meses_seguidos_conocidos', 'motivo', 'mensaje')
ESTABILIDAD = ('no_aplica', 'estable', 'empeorada', 'no_valorable')
COLUMNAS_EXCLUIDOS = ('producto', 'metodo', 'motivo')
# Situaciones conocidas antes de calcular (v2_completitud) que no son falta de datos.
ESTADO_POR_SITUACION = {'sin_actividad_reciente': 'no_recomendada', 'producto_retirado': 'no_recomendada'}
COLUMNAS_HISTORIAL = ('producto', 'mes', 'real', 'prevision_version2', 'mes_anterior', 'metodos_combinados')


@dataclass(frozen=True)
class ConfiguracionV2:
    candidatos: tuple[str, ...] = CANDIDATOS
    cortes: int = 12                      # meses con los que se compara cada candidato
    combinar: int = 3                     # candidatos que se promedian
    cortes_minimos: int = 3               # sin ellos no se combina
    evaluaciones_minimas: int = 6         # meses comprobados para salir de «datos insuficientes»
    evaluaciones_buen_respaldo: int = 12  # meses comprobados exigidos para «buen respaldo»
    umbral_buen_respaldo: float = 0.15    # configurables: no son leyes universales
    umbral_no_recomendada: float = 0.35
    procesos: int = 1                     # procesos en paralelo para los candidatos por producto
    mejora_patron_anual: float = 0.10     # el ETS estacional debe errar un 10 % menos que sin estacionalidad
    meses_con_ventas_minimos: int = 4     # meses con ventas, entre los comprobados, para poder valorar el error
    prevision_minima_buen_respaldo: float = 0.5   # una previsión de cero unidades nunca tiene buen respaldo
    # Estabilidad: valores aprobados antes de evaluarla; no se ajustan con los resultados.
    aplicar_estabilidad: bool = True
    estabilidad_meses_mitad: int = 6
    estabilidad_umbral_reciente: float = 0.20     # condición 1: E_B > 20 %
    estabilidad_factor: float = 2.0               # condición 2: E_B >= 2 * E_A ...
    estabilidad_minimo_reciente: float = 0.10     # ... y E_B > 10 %
    estabilidad_meses_con_ventas: int = 2         # en cada mitad, para poder valorarla

    def __post_init__(self):
        if not self.candidatos or len(set(self.candidatos)) != len(self.candidatos) or \
                not set(self.candidatos).issubset(CANDIDATOS):
            raise ValueError('Candidatos no reconocidos o repetidos.')
        for nombre in ('cortes', 'combinar', 'cortes_minimos', 'evaluaciones_minimas',
                       'evaluaciones_buen_respaldo', 'procesos', 'meses_con_ventas_minimos',
                       'estabilidad_meses_mitad', 'estabilidad_meses_con_ventas'):
            v = getattr(self, nombre)
            if type(v) is not int or v < 1:
                raise ValueError(f'{nombre} debe ser un entero positivo.')
        if self.cortes_minimos > self.cortes or self.evaluaciones_minimas > self.cortes or \
                self.evaluaciones_buen_respaldo > self.cortes or \
                self.evaluaciones_minimas > self.evaluaciones_buen_respaldo:
            raise ValueError('Mínimos de evaluación incompatibles con el número de cortes.')
        if self.meses_con_ventas_minimos > self.evaluaciones_minimas:
            raise ValueError('meses_con_ventas_minimos no puede superar evaluaciones_minimas.')
        if not isfinite(self.prevision_minima_buen_respaldo) or self.prevision_minima_buen_respaldo < 0:
            raise ValueError('prevision_minima_buen_respaldo debe ser un número no negativo.')
        if not (0 < self.umbral_buen_respaldo < self.umbral_no_recomendada) or not isfinite(self.umbral_no_recomendada):
            raise ValueError('Los umbrales deben cumplir 0 < buen respaldo < no recomendada.')
        if type(self.aplicar_estabilidad) is not bool:
            raise ValueError('aplicar_estabilidad debe ser verdadero o falso.')
        if 2 * self.estabilidad_meses_mitad > self.cortes or \
                self.estabilidad_meses_con_ventas > self.estabilidad_meses_mitad:
            raise ValueError('Las mitades de la estabilidad no caben en los meses comprobados.')
        for nombre in ('estabilidad_umbral_reciente', 'estabilidad_factor', 'estabilidad_minimo_reciente'):
            v = getattr(self, nombre)
            if not isinstance(v, (int, float)) or not isfinite(v) or v <= 0:
                raise ValueError(f'{nombre} debe ser un número positivo.')


@dataclass
class InformeV2:
    mes_previsto: date
    resumen: pd.DataFrame
    historial: pd.DataFrame
    validacion: pd.DataFrame
    candidatos: pd.DataFrame
    configuracion: ConfiguracionV2
    por_estado: dict[str, int] = field(default_factory=dict)
    excluidos: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=COLUMNAS_EXCLUIDOS))


# ------------------------------------------------------------------ utilidades

def fecha(i: int) -> date:
    return date(i // 12, i % 12 + 1, 1)


def indice(d: date) -> int:
    return d.year * 12 + d.month - 1


def conocida_en(serie: dict[int, float | None], revisiones, mes: int) -> dict[int, float | None]:
    """La serie tal como se conocía al prever `mes`, con información hasta el mes anterior.

    `revisiones`: (mes de la venta, mes en que se registró la anulación, unidades anuladas) y, opcionalmente,
    un cuarto dato: True si el mes de la venta no tiene valor en la serie PORQUE todo lo que había se anuló.
    La serie ya trae todas las anulaciones aplicadas; aquí se deshacen las que en ese momento aún no se
    habían registrado. Un mes desconocido por cualquier otra causa sigue siendo desconocido.
    """
    s = {k: v for k, v in serie.items() if k < mes}
    for mes_venta, mes_registro, unidades, *vacio in revisiones or ():
        if mes_venta < mes <= mes_registro:
            if s.get(mes_venta) is not None:
                s[mes_venta] = s[mes_venta] + unidades
            elif vacio and vacio[0]:
                s[mes_venta] = float(unidades)
    return s


def _como_se_registro(serie: dict[int, float | None], revisiones) -> dict[int, float | None]:
    """La serie con todas las anulaciones posteriores deshechas: lo más que llegó a conocerse de cada mes."""
    return conocida_en(serie, [(k, max(serie) + 1, q, *v) for k, _, q, *v in revisiones], max(serie) + 1) if serie else {}


def _tabla_en(completa: pd.DataFrame, serie: dict, revisiones, mes: int) -> pd.DataFrame:
    """La tabla de candidatos tal como se habría visto al prever `mes`: solo el tramo de meses seguidos que se
    conocía entonces, con los valores de las ventas que se conocían entonces."""
    conocida = conocida_en(serie, revisiones, mes)
    ini, y = C.tramo_final(conocida, mes)
    if not y:
        return completa.iloc[0:0]
    t = completa[(completa.index >= ini + 1) & (completa.index <= mes)].copy()
    t['real'] = [float(conocida[k]) if k < mes and conocida.get(k) is not None else np.nan for k in t.index]
    return t


def _comprobar_revisiones(revisiones: dict, series: dict, ultimo: int) -> None:
    for p, lista in revisiones.items():
        if p not in series:
            raise ValueError(f'Hay revisiones de un producto que no está en las series: {p}')
        for r in lista:
            if len(r) not in (3, 4) or type(r[0]) is not int or type(r[1]) is not int or not r[0] < r[1] \
                    or r[0] > ultimo or not isfinite(float(r[2])) or float(r[2]) <= 0 \
                    or (len(r) == 4 and type(r[3]) is not bool):
                raise ValueError(f'Revisión no válida de {p}: {r}. Debe ser (mes de la venta, no posterior al '
                                 'último mes utilizable; mes de registro, posterior al de la venta; unidades '
                                 'positivas) y, si acaso, si el mes quedó vacío.')
            if len(r) == 4 and r[3] and series[p].get(r[0]) is not None:
                raise ValueError(f'Revisión no válida de {p}: {r}. El mes de la venta tiene valor en la serie.')


def _al_cierre(series: dict, revisiones: dict, ultimo: int) -> tuple[dict, dict]:
    """Series y revisiones tal como se conocían al cerrar el mes `ultimo`.

    Las anulaciones registradas después de ese mes se deshacen en las series y desaparecen de las revisiones:
    para este cálculo no existen. Las demás se conservan con su mes de registro real.
    """
    if not any(r[1] > ultimo for lista in revisiones.values() for r in lista):
        return series, revisiones
    series, antes = dict(series), {}
    for p, lista in revisiones.items():
        despues = [r for r in lista if r[1] > ultimo]
        if despues:
            series[p] = conocida_en(series[p], despues, ultimo + 1)
        # Un mes solo «quedó vacío» si sigue sin valor una vez deshecho lo posterior.
        resto = [(k, reg, q, bool(v and v[0]) and series[p].get(k) is None) for k, reg, q, *v in lista if reg <= ultimo]
        if resto:
            antes[p] = resto
    return series, antes


def series_desde_objetivo(objetivo: InformeObjetivoMensual) -> dict[str, dict[int, float | None]]:
    """Reutiliza la validación de la versión 1: meses pendientes -> None; ceros confirmados -> 0."""
    return _validar_objetivo(objetivo)


def _comprobar_series(series: dict, sin_meses: set | frozenset = frozenset()) -> None:
    """`sin_meses`: productos que pueden no tener ningún mes utilizable porque ya traen su situación."""
    if not isinstance(series, dict) or not series:
        raise ValueError('No hay series que predecir.')
    for p, s in series.items():
        if not isinstance(p, str) or not p.strip() or not isinstance(s, dict) or (not s and p not in sin_meses):
            raise ValueError('Serie o producto inválido.')
        for k, v in s.items():
            if type(k) is not int:
                raise ValueError('Los meses deben ser índices enteros.')
            if v is not None and (not isfinite(float(v)) or float(v) < 0):
                raise ValueError('Las cantidades deben ser finitas y no negativas.')


# ------------------------------------------------------------------ candidatos

def _candidatos_producto(args) -> list[tuple]:
    producto, serie, meses, candidatos, diag_desde, revisiones = args
    filas = []
    for m in meses:
        historia = conocida_en(serie, revisiones, m)
        diag = diag_desde is not None and m >= diag_desde
        for metodo, v in C.predecir_candidatos(historia, m, diagnostico=diag).items():
            if metodo in candidatos or metodo == C.DIAGNOSTICO:
                filas.append((producto, m, metodo, float(v)))
    return filas


def calcular_candidatos(series: dict[str, dict[int, float | None]], meses_por_producto: dict[str, list[int]],
                        cfg: ConfiguracionV2, *, catalogo: dict | None = None,
                        diagnostico: str = 'ultimos', revisiones: dict | None = None) -> pd.DataFrame:
    """Predicción de cada candidato en cada mes pedido. Cada predicción usa solo meses anteriores.

    `revisiones` (producto -> lista de revisiones, ver `conocida_en`): cada mes se prevé con la serie tal como
    se conocía entonces, también en el XGBoost global. La columna `real` es siempre el valor definitivo.

    `catalogo` son las series con las que se entrena el XGBoost global (por defecto, todas).
    `diagnostico` ('ultimos', 'todos' o 'ninguno'): meses en los que se calcula el ETS sin
    estacionalidad que sirve para comprobar si el patrón anual ayuda. Con 'ultimos', solo en los
    `cortes` meses anteriores al último mes pedido de cada producto.
    """
    if diagnostico not in ('ultimos', 'todos', 'ninguno'):
        raise ValueError('diagnostico debe ser ultimos, todos o ninguno.')
    catalogo = series if catalogo is None else catalogo
    revisiones = revisiones or {}

    def desde(meses):
        if diagnostico == 'ninguno':
            return None
        return min(meses) if diagnostico == 'todos' else max(meses) - cfg.cortes
    tareas = [(p, series[p], m, set(cfg.candidatos), desde(m), revisiones.get(p)) for p, m in meses_por_producto.items() if m]
    filas: list[tuple] = []
    if cfg.procesos > 1 and len(tareas) > 1:
        from multiprocessing import get_context
        with get_context('spawn').Pool(cfg.procesos) as pool:
            for r in pool.imap_unordered(_candidatos_producto, tareas, chunksize=1):
                filas += r
    else:
        for t in tareas:
            filas += _candidatos_producto(t)
    if 'global_xgb' in cfg.candidatos:
        necesitan: dict[int, list[str]] = {}
        for p, meses in meses_por_producto.items():
            for m in meses:
                necesitan.setdefault(m, []).append(p)
        for m in sorted(necesitan):
            # Productos con alguna anulación todavía no registrada al prever m: se entrenan con lo que se sabía.
            entonces = {p: conocida_en(catalogo[p], r, m) for p, r in revisiones.items()
                        if p in catalogo and any(k < m <= reg for k, reg, *_ in r)}
            modelo = G.entrenar({**catalogo, **entonces} if entonces else catalogo, m, hilos=cfg.procesos)
            for p in necesitan[m]:
                v = G.predecir(modelo, conocida_en(series[p], revisiones.get(p), m), m)
                if v is not None:
                    filas.append((p, m, 'global_xgb', v))
    d = pd.DataFrame(filas, columns=['producto', 'mes', 'metodo', 'prediccion'])
    d['real'] = [series[p].get(m) for p, m in zip(d.producto, d.mes)]
    d['real'] = d['real'].astype(float)
    return d.sort_values(['producto', 'mes', 'metodo']).reset_index(drop=True)


def tabla_producto(candidatos: pd.DataFrame) -> pd.DataFrame:
    """Filas = meses; columnas = candidatos + 'real' (NaN en el mes previsto)."""
    if candidatos.empty:
        return pd.DataFrame(columns=['real'])
    t = candidatos.pivot_table(index='mes', columns='metodo', values='prediccion', aggfunc='first')
    t['real'] = candidatos.drop_duplicates('mes').set_index('mes')['real']
    return t.sort_index()


# ------------------------------------------------------------------ decisión

def combinar(tabla: pd.DataFrame, mes: int, cfg: ConfiguracionV2) -> dict | None:
    """Media de los candidatos con menos error en los `cortes` meses anteriores a `mes`.

    Solo usa filas anteriores a `mes` (con su valor real) y las predicciones de la fila `mes`.
    Un candidato cuenta si tiene predicción en todos esos meses y en `mes`.
    """
    if mes not in tabla.index:
        return None
    h = tabla[(tabla.index >= mes - cfg.cortes) & (tabla.index < mes)]
    if len(h) < cfg.cortes_minimos or h['real'].isna().any():
        return None
    validos = [m for m in cfg.candidatos if m in tabla.columns and h[m].notna().all()
               and pd.notna(tabla.at[mes, m])]
    if not validos:
        return None
    e = h[validos].sub(h['real'], axis=0)
    mae, rmse = e.abs().mean(), np.sqrt((e ** 2).mean())
    orden = sorted(validos, key=lambda m: (round(mae[m], 10), round(rmse[m], 10), cfg.candidatos.index(m)))
    elegidos = orden[:cfg.combinar]
    return dict(prediccion=float(np.mean([tabla.at[mes, m] for m in elegidos])), elegidos=elegidos,
                cortes=len(h), mae={m: float(mae[m]) for m in validos})


def diagnostico_patron_anual(tabla: pd.DataFrame, mes: int, cfg: ConfiguracionV2) -> str:
    """¿Ayuda el patrón anual a prever este producto? Se responde con los meses anteriores a `mes`:
    compara el error del ETS con estacionalidad anual y el del mismo ETS sin ella."""
    h = tabla[(tabla.index >= mes - cfg.cortes) & (tabla.index < mes)]
    if 'ets' not in h.columns or C.DIAGNOSTICO not in h.columns:
        return 'sin_datos_suficientes'
    par = h[['ets', C.DIAGNOSTICO, 'real']].dropna()
    if len(par) < cfg.evaluaciones_minimas:
        return 'sin_datos_suficientes'
    con, sin = (par.ets - par.real).abs().mean(), (par[C.DIAGNOSTICO] - par.real).abs().mean()
    if sin <= 0:
        return 'no_ayuda'
    return 'ayuda' if con <= (1 - cfg.mejora_patron_anual) * sin else 'no_ayuda'


def _pct(x: float) -> str:
    return f'{round(100 * x)} %'


def _pct_fino(x: float) -> str:
    """Porcentaje con un decimal si hace falta (cerca de los umbrales, «10 %» y «10,4 %» no son lo mismo)."""
    t = f'{100 * x:.1f}'
    return (t[:-2] if t.endswith('.0') else t.replace('.', ',')) + ' %'


def valorar_estabilidad(real, prevision, cfg: ConfiguracionV2) -> dict:
    """Compara el error de las dos mitades de los últimos 2 * `estabilidad_meses_mitad` meses comprobados.

    `real` y `prevision` van en orden cronológico. Los errores se redondean a 10 decimales antes de compararlos
    con los umbrales, para que un 20 % exacto no pase por 20,000000000000004 %; los umbrales no cambian.
    Devuelve estabilidad ('estable', 'empeorada', 'no_valorable' o 'no_aplica'), los dos errores (vacíos si no
    se puede valorar) y, si no se puede valorar, qué mitad tiene pocas ventas ('antigua' o 'reciente').
    """
    k = cfg.estabilidad_meses_mitad
    out = dict(estabilidad='no_aplica', error_mitad_antigua=np.nan, error_mitad_reciente=np.nan, mitad_sin_ventas='')
    r, p = np.asarray(real, dtype=float), np.asarray(prevision, dtype=float)
    if len(r) != len(p) or len(r) < 2 * k:
        return out
    r, p = r[-2 * k:], p[-2 * k:]
    mitades = {'antigua': slice(0, k), 'reciente': slice(k, 2 * k)}
    pocas = [m for m, sl in mitades.items() if int((r[sl] > 0).sum()) < cfg.estabilidad_meses_con_ventas]
    if pocas:
        return out | dict(estabilidad='no_valorable', mitad_sin_ventas=pocas[0] if len(pocas) == 1 else 'ambas')
    ea, eb = (round(float(np.abs(r[sl] - p[sl]).sum() / r[sl].sum()), 10) for sl in mitades.values())
    empeora = (eb > cfg.estabilidad_umbral_reciente or
               (eb >= round(cfg.estabilidad_factor * ea, 10) and eb > cfg.estabilidad_minimo_reciente))
    return out | dict(estabilidad='empeorada' if empeora else 'estable', error_mitad_antigua=ea, error_mitad_reciente=eb)


def _num(x: float) -> str:
    if x >= 100:
        return f'{x:,.0f}'.replace(',', '.')
    return f'{x:.1f}'.replace('.', ',')


def motivos_exclusion(tabla: pd.DataFrame, mes: int, cfg: ConfiguracionV2, serie: dict, y: list[float]
                      ) -> dict[str, str]:
    """Por qué cada candidato no entra en la comparación del mes `mes` (solo los excluidos)."""
    h = tabla[(tabla.index >= mes - cfg.cortes) & (tabla.index < mes)]
    out = {}
    for m in cfg.candidatos:
        en_mes = mes in tabla.index and m in tabla.columns and pd.notna(tabla.at[mes, m])
        if en_mes and len(h) and h[m].notna().all():
            continue
        if en_mes:
            faltan = int(h[m].isna().sum())
            out[m] = (f'le falta la predicción de {faltan} de los {len(h)} meses de comparación '
                      '(al principio de esos meses no tenía historial suficiente)')
        elif m == 'anual' and serie.get(mes - 12) is None:
            out[m] = 'no se conocen las ventas del mismo mes del año anterior'
        elif m in C.MINIMOS and len(y) < C.MINIMOS[m]:
            out[m] = f'necesita {C.MINIMOS[m]} meses seguidos con ventas conocidas y hay {len(y)}'
        elif m == 'global_xgb' and len(y) < 3:
            out[m] = f'necesita 3 meses seguidos con ventas conocidas y hay {len(y)}'
        elif m == 'global_xgb' and float(np.mean(y[-12:])) <= 0:
            out[m] = 'no hubo ventas en los últimos meses, que es la escala con la que trabaja'
        elif m == 'global_xgb':
            out[m] = ('el archivo no tiene datos suficientes para entrenarlo '
                      f'(hacen falta {G.FILAS_MINIMAS} combinaciones de producto y mes)')
        else:
            out[m] = 'el cálculo no dio un resultado válido con estos datos'
    return out


def _lista(nombres: list[str]) -> str:
    return nombres[0] if len(nombres) == 1 else ', '.join(nombres[:-1]) + ' y ' + nombres[-1]


def evaluar_producto(producto: str, serie: dict[int, float | None], tabla: pd.DataFrame, mes: int,
                     cfg: ConfiguracionV2, situacion: tuple[str, str] | None = None,
                     revisiones: list | None = None) -> tuple[dict, list[dict], dict]:
    """Estado, previsión, rango y comparación de un producto para `mes`, con datos anteriores a `mes`.

    `serie` puede contener meses posteriores (evaluación retrospectiva): se descartan aquí.
    `situacion` (motivo, explicación): el producto no se prevé por una razón conocida antes de calcular
    (ver v2_completitud). Queda en «no recomendada» si es retirado o sin actividad reciente, y en
    «datos insuficientes» en los demás casos.
    `revisiones`: anulaciones registradas después del mes de su venta (ver `conocida_en`). En cada mes
    comprobado, la elección de métodos y la referencia del mes anterior usan lo que se sabía entonces.
    Devuelve (fila del resumen, historial de previsiones, validación). La validación incluye
    `excluidos`: los candidatos que no entraron en la comparación y por qué.
    """
    serie_final = {k: v for k, v in serie.items() if k < mes}
    serie = conocida_en(serie_final, revisiones, mes) if revisiones else serie_final
    tabla = tabla[tabla.index <= mes].copy()
    if mes in tabla.index:
        tabla.loc[mes, 'real'] = np.nan
    completa = tabla        # con revisiones puede traer meses anteriores al tramo final: se conocían en otros cortes
    ini, y = C.tramo_final(serie, mes)
    base = dict(producto=producto, mes_previsto=fecha(mes), prediccion=np.nan, prediccion_exploratoria=np.nan,
                rango_inferior=np.nan, rango_superior=np.nan, error_historico=np.nan, meses_comprobados=0,
                mae_version2=np.nan, mae_mes_anterior=np.nan, supera_mes_anterior=None,
                mejora_vs_mes_anterior=np.nan, metodos_combinados='', candidatos_validos=0, candidatos_excluidos='',
                patron_anual='sin_datos_suficientes', metodos_con_mes_del_ano='',
                meses_con_ventas_comprobados=np.nan, unidades_comprobadas=np.nan, estabilidad='no_aplica',
                error_mitad_antigua=np.nan, error_mitad_reciente=np.nan,
                meses_seguidos_conocidos=len(y), estado='datos_insuficientes', motivo='', mensaje='')
    validacion: dict = {}

    def insuficiente(motivo: str, mensaje: str):
        base.update(motivo=motivo, mensaje=mensaje)
        return base, [], validacion

    if situacion is not None:
        motivo, texto = situacion
        estado = ESTADO_POR_SITUACION.get(motivo, 'datos_insuficientes')
        base.update(estado=estado, motivo=motivo,
                    mensaje=f'No recomendamos una previsión. {texto}' if estado == 'no_recomendada' else texto)
        return base, [], validacion
    if not serie or max(serie) != mes - 1:
        return insuficiente('sin_registro_en_ultimo_mes',
                            f'No hay ventas registradas en {fecha(mes - 1):%m/%Y}, el último mes del archivo; '
                            'no se puede saber si el producto sigue a la venta.')
    if serie[mes - 1] is None:
        return insuficiente('ultimo_mes_pendiente',
                            f'Las ventas de {fecha(mes - 1):%m/%Y} están pendientes de revisión.')
    # Solo el tramo final de meses conocidos seguidos, igual que al prever el último mes del archivo
    # (importa en la evaluación retrospectiva, donde la tabla puede traer meses de tramos anteriores).
    tabla = tabla[(tabla.index >= max(ini + 1, mes - 2 * cfg.cortes)) & (tabla.index <= mes)]
    excluidos = motivos_exclusion(tabla, mes, cfg, serie, y)
    validacion = {'producto': producto, 'excluidos': excluidos}
    base['candidatos_excluidos'] = '; '.join(f'{m}: {r}' for m, r in excluidos.items())
    if revisiones:
        tabla = _tabla_en(completa, serie_final, revisiones, mes)
        tabla = tabla[tabla.index >= mes - 2 * cfg.cortes]
    final = combinar(tabla, mes, cfg)
    historial = []
    for m in range(mes - cfg.cortes, mes):
        # Solo se comprueban meses del tramo final, y solo si su valor definitivo se conoce.
        if m in tabla.index and pd.notna(completa.at[m, 'real']):
            c = combinar(_tabla_en(completa, serie_final, revisiones, m) if revisiones else tabla, m, cfg)
            if c is not None:
                anterior = conocida_en(serie_final, revisiones, m)[m - 1] if revisiones else serie[m - 1]
                historial.append(dict(producto=producto, mes=fecha(m), real=float(completa.at[m, 'real']),
                                      prevision_version2=c['prediccion'], mes_anterior=float(anterior),
                                      metodos_combinados='+'.join(c['elegidos'])))
    n = len(historial)
    necesarios = cfg.cortes_minimos + 1 + cfg.evaluaciones_minimas
    if final is None:
        return insuficiente('historial_corto',
                            f'Solo hay {len(y)} meses seguidos con ventas conocidas y ningún método se ha podido '
                            f'comprobar; hacen falta al menos {necesarios} para comprobar la previsión en '
                            f'{cfg.evaluaciones_minimas} meses.')
    validacion |= {'cortes': final['cortes'], **{f'mae_{k}': v for k, v in final['mae'].items()}}
    elegidos = final['elegidos']
    con_mes = C.metodos_con_mes_del_ano(elegidos, y)
    base.update(prediccion_exploratoria=final['prediccion'], metodos_combinados='+'.join(elegidos),
                candidatos_validos=len(final['mae']), patron_anual=diagnostico_patron_anual(tabla, mes, cfg),
                metodos_con_mes_del_ano='+'.join(con_mes), meses_comprobados=n)
    if n < cfg.evaluaciones_minimas:
        return (base | dict(motivo='pocos_meses_comprobados',
                            mensaje=f'Solo se ha podido comprobar la previsión en {n} meses; hacen falta '
                                    f'{cfg.evaluaciones_minimas}. Se necesitan unos {necesarios} meses seguidos '
                                    f'con ventas conocidas y hay {len(y)}.'), historial, validacion)

    h = pd.DataFrame(historial)
    e = h.real - h.prevision_version2
    suma_real, con_ventas = float(h.real.sum()), int((h.real > 0).sum())
    mae2, mae1 = float(e.abs().mean()), float((h.real - h.mes_anterior).abs().mean())
    pred = final['prediccion']
    lo, hi = max(pred + min(float(e.min()), 0.0), 0.0), pred + max(float(e.max()), 0.0)
    base.update(meses_con_ventas_comprobados=con_ventas, unidades_comprobadas=suma_real,
                rango_inferior=lo, rango_superior=hi, mae_version2=mae2, mae_mes_anterior=mae1)
    texto_rango = (f'Rango orientativo, según los errores de esos meses: entre {_num(lo)} y {_num(hi)} unidades '
                   '(no es un margen garantizado).')

    # Ventas nulas o demasiado esporádicas: el error no se puede valorar y no se da porcentaje.
    if con_ventas < cfg.meses_con_ventas_minimos:
        hubo = ('no hubo ninguna venta' if con_ventas == 0 else
                f'solo hubo ventas en {con_ventas} mes' + ('' if con_ventas == 1 else 'es'))
        mensaje = (f'No recomendamos una previsión. En los {n} meses comprobados {hubo}: con tan pocas ventas no se '
                   'puede comprobar si una previsión acierta, así que no se calcula un porcentaje de error. '
                   + texto_rango)
        base.update(estado='no_recomendada', motivo='ventas_demasiado_esporadicas', mensaje=mensaje)
        return base, historial, validacion

    error = float(e.abs().sum() / suma_real)
    supera = mae2 < mae1
    mejora = (1 - mae2 / mae1) if mae1 > 0 else np.nan
    if error <= cfg.umbral_buen_respaldo and n >= cfg.evaluaciones_buen_respaldo:
        estado, motivo = 'buen_respaldo', 'error_bajo'
        if pred < cfg.prevision_minima_buen_respaldo:
            estado, motivo = 'orientativa', 'prevision_cero'
    elif error <= cfg.umbral_no_recomendada:
        estado = 'orientativa'
        motivo = 'evidencia_limitada' if error <= cfg.umbral_buen_respaldo else 'error_moderado'
    else:
        estado, motivo = 'no_recomendada', 'error_elevado'

    # Estabilidad: solo rebaja «buen respaldo», y solo entonces se explica en el mensaje. En las que ya eran
    # «orientativa» se calcula para el diagnóstico técnico, sin cambiar el estado ni el texto.
    k = cfg.estabilidad_meses_mitad
    est = valorar_estabilidad(h.real.tolist(), h.prevision_version2.tolist(), cfg) \
        if cfg.aplicar_estabilidad and estado in ('buen_respaldo', 'orientativa') else valorar_estabilidad([], [], cfg)
    base.update(estabilidad=est['estabilidad'], error_mitad_antigua=est['error_mitad_antigua'],
                error_mitad_reciente=est['error_mitad_reciente'])
    if estado == 'buen_respaldo' and est['estabilidad'] == 'empeorada':
        estado, motivo = 'orientativa', 'predicciones_recientes_empeoradas'
    elif estado == 'buen_respaldo' and est['estabilidad'] == 'no_valorable':
        estado, motivo = 'orientativa', 'estabilidad_no_valorable'
    texto_estabilidad = ''
    if motivo == 'predicciones_recientes_empeoradas':
        texto_estabilidad = (f' Las predicciones recientes han empeorado: en los últimos {k} meses comprobados las '
                             f'diferencias suman el {_pct_fino(est["error_mitad_reciente"])} de lo vendido, frente al '
                             f'{_pct_fino(est["error_mitad_antigua"])} de los {k} anteriores. Es lo observado en esos '
                             'meses: puede deberse, por ejemplo, a que son épocas del año distintas, y no demuestra '
                             'que las ventas hayan cambiado.')
    elif motivo == 'estabilidad_no_valorable':
        cuales = {'antigua': f'los {k} primeros', 'reciente': f'los {k} últimos',
                  'ambas': f'las dos mitades de {k}'}[est['mitad_sin_ventas']]
        texto_estabilidad = (f' No se puede comprobar si el error se ha mantenido estable: en {cuales} meses '
                             f'comprobados hubo ventas en menos de {cfg.estabilidad_meses_con_ventas} meses.')

    texto_error = (f'En los últimos {n} meses comprobados, las diferencias entre lo previsto y lo vendido suman '
                   f'el {_pct(error)} de las unidades vendidas en esos meses.')
    if estado != 'no_recomendada' and (hi - lo) > pred:
        texto_rango += ' Es un rango amplio: la cifra es poco precisa.'
    texto_ref = (f'En esos meses acertó más que repetir las ventas del mes anterior ({_pct(mejora)} menos de error).'
                 if supera else 'En esos meses no acertó más que repetir las ventas del mes anterior.')
    nombres = [C.NOMBRE_METODO[m] for m in elegidos]
    texto_metodos = ''
    if len(elegidos) == 1:
        texto_metodos = (f' La previsión usa un único método ({nombres[0]}), porque los demás no se pudieron '
                         'comprobar en todos los meses.')
    elif len(elegidos) < cfg.combinar:
        texto_metodos = (f' La previsión combina solo {len(elegidos)} métodos ({_lista(nombres)}), porque los demás '
                         'no se pudieron comprobar en todos los meses.')
    texto_anual = ''
    if base['patron_anual'] == 'ayuda':
        texto_anual = (' En los meses comprobados, tener en cuenta el mes del año mejoró las previsiones de prueba'
                       + ('; la combinación incluye métodos que lo tienen en cuenta.' if con_mes else
                          ', pero ninguno de los métodos combinados lo tiene en cuenta.'))
    if estado == 'buen_respaldo':
        mensaje = f'Previsión: {_num(pred)} unidades. {texto_error} {texto_rango} {texto_ref}'
    elif estado == 'orientativa':
        extra = {'evidencia_limitada': f' El error ha sido bajo, pero solo hay {n} meses comprobados.',
                 'prevision_cero': ' Una previsión de cero unidades no se considera con buen respaldo, '
                                   'aunque el error haya sido bajo.',
                 'error_moderado': ' Conviene un margen de seguridad.',
                 'predicciones_recientes_empeoradas': '',
                 'estabilidad_no_valorable': ''}[motivo] + texto_estabilidad
        mensaje = f'Previsión orientativa: {_num(pred)} unidades. {texto_error}{extra} {texto_rango} {texto_ref}'
    else:
        mensaje = (f'No recomendamos una previsión para este producto. {texto_error} Las ventas han variado '
                   f'demasiado para ofrecer una cantidad respaldada. {texto_rango}')
        if not supera:
            mensaje += ' En estos meses, repetir las ventas del mes anterior funcionó igual o mejor.'
    mensaje += texto_metodos + texto_anual
    base.update(estado=estado, motivo=motivo, mensaje=mensaje, error_historico=error,
                prediccion=pred if estado != 'no_recomendada' else np.nan,
                supera_mes_anterior=supera, mejora_vs_mes_anterior=mejora)
    return base, historial, validacion


# ------------------------------------------------------------------ ejecución completa

def meses_necesarios(serie: dict[int, float | None], mes: int, cfg: ConfiguracionV2) -> list[int]:
    """Meses del tramo final (y el mes previsto) en los que hace falta la predicción de cada candidato."""
    ini, y = C.tramo_final({k: v for k, v in serie.items() if k < mes}, mes)
    if not y:
        return []
    return list(range(max(ini + 1, mes - 2 * cfg.cortes), mes + 1))


def ejecutar_v2(objetivo: InformeObjetivoMensual | dict, cfg: ConfiguracionV2 | None = None, *,
                productos: list[str] | None = None, ultimo_mes: int | None = None,
                situaciones: dict[str, tuple[str, str]] | None = None,
                revisiones: dict[str, list[tuple[int, int, float]]] | None = None) -> InformeV2:
    """Previsión del mes siguiente al último mes utilizable para cada producto.

    `productos` limita los productos que se prevén; el XGBoost global se entrena siempre con todos.
    `ultimo_mes`: último mes completo (por defecto, el último que aparece en las series). Las series no
    pueden traer meses posteriores: un mes sin cerrar no debe llegar al motor.
    `situaciones`: productos que no se prevén, con su motivo y su explicación (ver `v2_completitud`).
    `revisiones`: por producto, anulaciones registradas en un mes posterior al de su venta: (mes de la venta,
    mes de registro, unidades) con su mes de registro real. Las series ya las traen aplicadas; se usan para
    que ningún cálculo cuente con ellas antes de que se registraran. Las registradas después del último mes
    utilizable no se conocían al cerrarlo: se deshacen del todo y no influyen en la previsión.
    """
    cfg = cfg or ConfiguracionV2()
    situaciones = dict(situaciones or {})
    series = series_desde_objetivo(objetivo) if isinstance(objetivo, InformeObjetivoMensual) else objetivo
    _comprobar_series(series, frozenset(situaciones))
    if not set(situaciones).issubset(series):
        raise ValueError('Hay situaciones de productos que no están en las series.')
    for motivo_y_texto in situaciones.values():
        if len(motivo_y_texto) != 2 or not all(isinstance(x, str) and x for x in motivo_y_texto):
            raise ValueError('Cada situación debe traer un motivo y una explicación.')
    maximo = max(max(s) for s in series.values() if s)
    if ultimo_mes is None:
        ultimo = maximo
    else:
        if type(ultimo_mes) is not int:
            raise ValueError('ultimo_mes debe ser un índice de mes entero.')
        if maximo > ultimo_mes:
            raise ValueError('Las series contienen meses posteriores al último mes utilizable.')
        ultimo = ultimo_mes
    mes = ultimo + 1
    elegidos = sorted(series) if productos is None else list(productos)
    if not set(elegidos).issubset(series):
        raise ValueError('Se han pedido productos que no están en el archivo.')
    revisiones = {p: [tuple(x) for x in r] for p, r in (revisiones or {}).items() if r}
    _comprobar_revisiones(revisiones, series, ultimo)
    # Desde aquí, todo usa el histórico conocido al cerrar el último mes utilizable.
    series, revisiones = _al_cierre(series, revisiones, ultimo)
    pedidos = {}
    for p in elegidos:
        s = series[p]
        calcular = p not in situaciones and bool(s) and max(s) == ultimo and s.get(ultimo) is not None
        # Con revisiones, en cortes anteriores el tramo de meses seguidos pudo llegar más atrás que ahora.
        pedidos[p] = meses_necesarios(_como_se_registro(s, revisiones[p]) if p in revisiones else s, mes, cfg) if calcular else []
    cand = calcular_candidatos(series, pedidos, cfg, revisiones=revisiones)
    por_producto = {p: g for p, g in cand.groupby('producto')}
    filas, hist, val, excl = [], [], [], []
    for p in elegidos:
        tabla = tabla_producto(por_producto[p]) if p in por_producto else pd.DataFrame(columns=['real'])
        r, h, v = evaluar_producto(p, series[p], tabla, mes, cfg, situaciones.get(p), revisiones.get(p))
        filas.append(r); hist += h
        excl += [(p, m, motivo) for m, motivo in v.get('excluidos', {}).items()]
        if 'cortes' in v:
            val.append({k: x for k, x in v.items() if k != 'excluidos'})
    resumen = pd.DataFrame(filas, columns=COLUMNAS_RESUMEN)
    return InformeV2(mes_previsto=fecha(mes), resumen=resumen,
                     historial=pd.DataFrame(hist, columns=COLUMNAS_HISTORIAL),
                     validacion=pd.DataFrame(val), candidatos=cand, configuracion=cfg,
                     por_estado={e: int((resumen.estado == e).sum()) for e in ESTADOS},
                     excluidos=pd.DataFrame(excl, columns=COLUMNAS_EXCLUIDOS))


def exportar_v2(informe: InformeV2, carpeta: Path) -> dict[str, Path]:
    """CSV con punto y coma y coma decimal, como el resto del prototipo."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    rutas = {}
    for nombre, tabla in (('prevision_v2', informe.resumen), ('historial_v2', informe.historial),
                          ('validacion_v2', informe.validacion), ('candidatos_v2', informe.candidatos),
                          ('candidatos_excluidos_v2', informe.excluidos)):
        rutas[nombre] = carpeta / f'{nombre}.csv'
        tabla.to_csv(rutas[nombre], sep=';', decimal=',', encoding='utf-8-sig', index=False)
    rutas['configuracion'] = carpeta / 'configuracion_v2.json'
    rutas['configuracion'].write_text(json.dumps(
        asdict(informe.configuracion) | {'mes_previsto': str(informe.mes_previsto)}, ensure_ascii=False, indent=2),
        encoding='utf-8')
    return rutas


def mostrar_v2_corto(informe: InformeV2, ejemplos: int = 5) -> None:
    r = informe.resumen
    print(f'\nVERSIÓN 2: previsión de {informe.mes_previsto:%m/%Y} para {len(r)} productos')
    for e in ESTADOS:
        print(f'  {NOMBRE_ESTADO[e]}: {informe.por_estado.get(e, 0)}')
    for e in ('orientativa', 'no_recomendada', 'datos_insuficientes'):
        motivos = r[r.estado == e].motivo.value_counts()
        if len(motivos):
            print(f'  Motivos de «{NOMBRE_ESTADO[e].lower()}»: ' + '; '.join(f'{m}: {n}' for m, n in motivos.items()))
    for e in ESTADOS:
        sub = r[r.estado == e].head(ejemplos)
        for fila in sub.itertuples(index=False):
            print(f'  [{NOMBRE_ESTADO[e]}] {fila.producto}: {fila.mensaje}')
            avisos = getattr(fila, 'avisos_datos', '')      # diagnóstico de los datos (v2_diagnostico), si lo hay
            if isinstance(avisos, str) and avisos:
                print(f'      Sobre los datos: {avisos}')
