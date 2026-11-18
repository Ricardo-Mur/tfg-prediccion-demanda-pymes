"""Versión 2, etapa 3: diagnóstico de calidad y suficiencia de cada producto.

Reúne en una ficha por producto comprobaciones que ya existían, sin modificarlas:

  continuidad y lagunas      histórico confirmado de la versión 2 (`v2_completitud`): meses conocidos, meses que
                             siguen desconocidos y tramo final de meses seguidos que usa el motor
  operaciones pendientes     estados del objetivo mensual de la versión 1 (devoluciones y otras operaciones sin
                             interpretar), tal como llegan al histórico
  líneas repetidas y         `auditoria_avanzada.auditar_calidad_avanzada` de la versión 1, pidiéndole todos los
  cantidades muy altas       casos en lugar de una muestra, para repartirlos por producto
  ADI/CV²                    `patrones_demanda.clasificar_serie_mensual_verificada` de la versión 1, sobre el tramo
                             final y solo si el usuario ha confirmado que el histórico de esos meses está completo
  identificadores (etapa 4)  si el archivo trae identificador de línea o de factura, las líneas repetidas se
                             valoran con ellos (`v2_operaciones.repetidas_con_identificadores`)
  operaciones (etapa 4)      devoluciones, anulaciones, ajustes y negativos sin confirmar de cada producto
  suficiencia                meses seguidos que necesita el motor con su configuración
  compatibilidad de métodos  candidatos que el motor pudo comprobar y los que no (`candidatos_excluidos`)

El diagnóstico es informativo: no cambia ninguna cantidad, método, rango ni estado. Al resultado llegan dos
columnas, `patron_demanda` y `avisos_datos` (como mucho tres frases cortas, solo cuando hay algo que revisar).
El detalle queda en `diagnostico_v2.csv` y en `ficha_producto`.

Límites que conviene no olvidar:
- Una línea repetida (misma fecha, producto y cantidad) es un indicio, no un duplicado probado. Si las fechas
  del archivo no son diarias no se puede valorar. Con número de factura se podrá afinar (etapa 4).
- Una cantidad muy alta es un indicio, no un error: no se corrige ni se quita.
- ADI/CV² describe cómo se reparten las ventas; no dice si la previsión acierta.
- Los mínimos de meses son los de la configuración de esta aplicación, no reglas universales de suficiencia.
- Solo se usan fecha, producto y cantidad: ningún dato de clientes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .v2_exportacion import escribir_csv
from . import v2_candidatos as C
from .auditoria_avanzada import auditar_calidad_avanzada
from .motor_v2 import ConfiguracionV2, fecha
from .patrones_demanda import clasificar_serie_mensual_verificada
from .v2_completitud import HistoricoV2

PATRONES = ('regular', 'errática', 'intermitente', 'irregular')
TEXTO_PATRON = {
    'regular': 'Vende casi todos los meses y en cantidades parecidas.',
    'errática': 'Vende casi todos los meses, pero en cantidades muy distintas de un mes a otro.',
    'intermitente': 'Pasa meses sin vender; cuando vende, las cantidades son parecidas.',
    'irregular': 'Pasa meses sin vender y, cuando vende, las cantidades son muy distintas.',
}
SIN_CLASIFICAR = 'sin_clasificar'
COLUMNAS_DIAGNOSTICO = (
    'producto',
    # continuidad y lagunas
    'primera_venta', 'meses_del_periodo', 'meses_conocidos', 'meses_con_ventas', 'meses_seguidos_conocidos',
    'inicio_tramo_final', 'meses_desconocidos', 'lagunas', 'ultimo_mes_desconocido', 'meses_anteriores_sin_usar',
    'mes_que_corta_el_historial', 'motivo_del_corte',
    'meses_fuera_de_venta',
    # operaciones pendientes
    'meses_pendientes', 'ultimo_mes_pendiente',
    # líneas repetidas y cantidades muy altas
    'lineas', 'lineas_repetidas', 'unidades_repetidas', 'parte_unidades_repetidas', 'repetidas_valorable',
    'cantidades_muy_altas', 'muy_altas_en_tramo_final', 'mayor_cantidad_muy_alta', 'mes_mayor_cantidad_muy_alta',
    # etapa 4: identificadores y operaciones
    'lineas_con_id_repetido', 'lineas_repetidas_en_factura', 'unidades_devueltas', 'unidades_anuladas',
    'lineas_ajuste', 'negativos_pendientes',
    # ADI/CV²
    'patron_demanda', 'adi', 'cv2', 'motivo_patron',
    # suficiencia y métodos
    'suficiencia_historial', 'meses_que_faltan_para_prever', 'meses_que_faltan_para_buen_respaldo',
    'metodos_comprobados', 'metodos_no_comprobados',
    # ¿está confirmado que los archivos traen todas las ventas de los meses que se usan? (v2_recomendacion)
    'completitud_historico',
    'avisos', 'avisos_datos')
AVISOS = ('operaciones_pendientes', 'lagunas', 'lineas_repetidas', 'cantidades_muy_altas')


@dataclass(frozen=True)
class ConfiguracionDiagnostico:
    # Parte de las unidades del producto que tienen que estar en líneas repetidas para avisar en el resultado.
    # Por debajo, el dato queda solo en el diagnóstico técnico. Es un valor provisional.
    aviso_repetidas_minimo: float = 0.05
    max_avisos: int = 3
    # Las líneas repetidas solo se valoran si las fechas distinguen días. No lo hacen si, en esta parte de los
    # meses con varias líneas, todas llevan la misma fecha.
    meses_con_un_solo_dia: float = 0.8

    def __post_init__(self):
        for nombre in ('aviso_repetidas_minimo', 'meses_con_un_solo_dia'):
            v = getattr(self, nombre)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1:
                raise ValueError(f'{nombre} debe estar entre 0 y 1.')
        if type(self.max_avisos) is not int or self.max_avisos < 1:
            raise ValueError('max_avisos debe ser un entero positivo.')


@dataclass
class DiagnosticoV2:
    productos: pd.DataFrame                    # una fila por producto (COLUMNAS_DIAGNOSTICO)
    fechas_diarias: bool | None                # None: no se han pasado las líneas del archivo
    meses_para_prever: int
    meses_para_buen_respaldo: int
    configuracion: ConfiguracionDiagnostico
    por_aviso: dict[str, int] = field(default_factory=dict)
    por_patron: dict[str, int] = field(default_factory=dict)


# ------------------------------------------------------------------ utilidades

def _mes(i) -> str:
    return f'{fecha(int(i)):%m/%Y}'


def _plural(n: int, uno: str, varios: str) -> str:
    return f'{n} {uno if n == 1 else varios}'


def _faltan(n) -> str:
    return 'falta 1 mes' if int(n) == 1 else f'faltan {int(n)} meses'


def _decimal(x: float) -> str:
    return f'{x:.2f}'.replace('.', ',')


def _cantidad(x: float) -> str:
    return f'{x:,.0f}'.replace(',', '.') if float(x).is_integer() else f'{x:.1f}'.replace('.', ',')


def fechas_son_diarias(fechas, meses_con_un_solo_dia: float = 0.8) -> bool:
    """¿Distinguen las fechas del archivo unos días de otros?

    No, si en casi todos los meses con varias líneas todas llevan la misma fecha (por ejemplo, el último día
    del mes). Entonces dos ventas iguales del mismo mes no se pueden distinguir de una línea duplicada.
    """
    validas = pd.Series([f for f in fechas if isinstance(f, (pd.Timestamp, datetime, date)) and pd.notna(f)],
                        dtype='object')
    if validas.empty:
        return False
    validas = pd.to_datetime(validas)
    por_mes = validas.dt.day.groupby(validas.dt.to_period('M')).agg(['size', 'nunique'])
    varios = por_mes[por_mes['size'] >= 2]
    if varios.empty:
        return True
    return bool((varios['nunique'] == 1).mean() < meses_con_un_solo_dia)


# ------------------------------------------------------------------ líneas del archivo

def lineas_por_producto(datos: pd.DataFrame, cfg: ConfiguracionDiagnostico | None = None
                        ) -> tuple[pd.DataFrame, bool]:
    """Líneas repetidas y cantidades muy altas de cada producto, con la auditoría de la versión 1.

    `datos`: las líneas ya limpias (columnas fecha, producto y cantidad). No se modifica.
    Devuelve una tabla por producto y si las fechas son diarias. En las repetidas se cuenta lo que sobraría
    si lo fueran: de un grupo de tres líneas iguales, dos líneas y sus unidades.
    """
    cfg = cfg or ConfiguracionDiagnostico()
    if not {'fecha', 'producto', 'cantidad'}.issubset(datos.columns):
        raise ValueError('Faltan las columnas fecha, producto y cantidad.')
    base = datos.loc[:, ['fecha', 'producto', 'cantidad']].reset_index(drop=True)
    diarias = fechas_son_diarias(base.fecha, cfg.meses_con_un_solo_dia)
    producto = base.producto.map(lambda p: None if pd.isna(p) else str(p).strip())
    cantidad = pd.to_numeric(base.cantidad, errors='coerce')
    mes = pd.to_datetime(base.fecha, errors='coerce')
    mes = mes.dt.year * 12 + mes.dt.month - 1
    todas = max(len(base), 1)
    auditoria = auditar_calidad_avanzada(base, base, max_muestras=todas)
    filas: dict[str, dict] = {}

    def de(p: str) -> dict:
        return filas.setdefault(p, dict(lineas=0, lineas_repetidas=0, unidades_repetidas=0.0, unidades=0.0,
                                        cantidades_muy_altas=0, extremos=[]))
    for p, n in producto.dropna().value_counts().items():
        de(p)['lineas'] = int(n)
    positivas = cantidad.where(cantidad > 0, 0.0).fillna(0.0)
    for p, u in positivas.groupby(producto).sum().items():
        de(p)['unidades'] = float(u)
    for grupo in auditoria.muestra_grupos_misma_clave:          # posiciones desde 1
        p = producto.iat[grupo[0] - 1]
        q = cantidad.iat[grupo[0] - 1]
        if p is None or pd.isna(q) or q <= 0:
            continue                                            # devoluciones y ceros repetidos no inflan ventas
        de(p)['lineas_repetidas'] += len(grupo) - 1
        de(p)['unidades_repetidas'] += float(q) * (len(grupo) - 1)
    for e in auditoria.muestra_extremos:
        f = de(e['producto'])
        f['cantidades_muy_altas'] += 1
        m = mes.iat[e['fila_tabla'] - 1]
        f['extremos'].append((float(e['cantidad']), None if pd.isna(m) else int(m)))
    tabla = pd.DataFrame([{'producto': p, **f} for p, f in filas.items()],
                         columns=['producto', 'lineas', 'lineas_repetidas', 'unidades_repetidas', 'unidades',
                                  'cantidades_muy_altas', 'extremos'])
    return tabla, diarias


# ------------------------------------------------------------------ histórico

def _continuidad(hist: HistoricoV2) -> list[dict]:
    siguiente = hist.ultimo_mes + 1
    d = hist.detalle
    indice_mes = d.mes.map(lambda x: x.year * 12 + x.month - 1)
    por_producto = {p: g for p, g in d.assign(i=indice_mes).groupby('producto', sort=False)}
    primeras = hist.productos.set_index('producto').primera_venta
    filas = []
    for p, serie in hist.series.items():
        g = por_producto.get(p)
        primera = primeras.get(p)
        primera = None if primera is None or pd.isna(primera) else primera.year * 12 + primera.month - 1
        ini, y = C.tramo_final(serie, siguiente)
        fila = dict(producto=p, primera_venta=fecha(primera) if primera is not None else None,
                    meses_del_periodo=0, meses_conocidos=0, meses_con_ventas=0, meses_seguidos_conocidos=len(y),
                    inicio_tramo_final=fecha(ini) if y else None, meses_desconocidos=0, lagunas=0,
                    ultimo_mes_desconocido=None, meses_anteriores_sin_usar=0, mes_que_corta_el_historial=None,
                    motivo_del_corte='', meses_fuera_de_venta=0,
                    meses_pendientes=0, ultimo_mes_pendiente=None, _tramo=y, _ini=ini)
        if g is not None:
            g = g[g.i <= hist.ultimo_mes]
            pendientes = sorted(g.i[g.estado.str.startswith('pendiente')])
            fila.update(meses_pendientes=len(pendientes),
                        ultimo_mes_pendiente=fecha(pendientes[-1]) if pendientes else None)
        if primera is not None and g is not None:
            g = g[g.i >= primera].sort_values('i')
            estados = dict(zip(g.i, g.estado))
            conocidos = [m for m in range(primera, siguiente) if serie.get(m) is not None]
            pendientes = [m for m, e in estados.items() if e.startswith('pendiente')]
            desconocidos = [m for m, e in estados.items() if e == 'sin_registro']
            sin_valor = sorted(set(pendientes) | set(desconocidos))
            bloques = sum(1 for k, m in enumerate(sin_valor) if k == 0 or sin_valor[k - 1] != m - 1)
            fila.update(meses_del_periodo=siguiente - primera, meses_conocidos=len(conocidos),
                        meses_con_ventas=sum(1 for m in conocidos if serie[m] > 0),
                        meses_desconocidos=len(desconocidos), lagunas=bloques,
                        ultimo_mes_desconocido=fecha(max(desconocidos)) if desconocidos else None,
                        meses_anteriores_sin_usar=sum(1 for m in conocidos if not y or m < ini),
                        meses_fuera_de_venta=sum(1 for e in estados.values() if e == 'fuera_de_venta'))
            if y and fila['meses_anteriores_sin_usar']:
                # El mes anterior al tramo final es el que impide usar los meses más antiguos.
                corte = estados.get(ini - 1, 'sin_registro')
                fila.update(mes_que_corta_el_historial=fecha(ini - 1),
                            motivo_del_corte='operaciones_pendientes' if corte.startswith('pendiente') else 'sin_datos_confirmados')
        filas.append(fila)
    return filas


def _patron(hist: HistoricoV2, tramo: list[float], ini: int | None) -> tuple[str, float, float, str]:
    """ADI/CV² del tramo final, solo con el histórico de esos meses confirmado como completo."""
    if not tramo:
        return SIN_CLASIFICAR, np.nan, np.nan, 'No hay un tramo final de meses conocidos.'
    confirmado = all(hist.confirmacion.cubre(m) for m in range(ini, ini + len(tramo)))
    r = clasificar_serie_mensual_verificada(tramo, cobertura_y_significado_confirmados=confirmado)
    motivo = ('No se ha confirmado que el histórico de esos meses esté completo.' if not confirmado else r.motivo)
    return (r.patron or SIN_CLASIFICAR, np.nan if r.adi is None else r.adi, np.nan if r.cv2 is None else r.cv2,
            motivo)


# ------------------------------------------------------------------ diagnóstico completo

def diagnosticar_productos(hist: HistoricoV2, resumen: pd.DataFrame | None = None, datos: pd.DataFrame | None = None,
                           cfg: ConfiguracionV2 | None = None, cfg_diagnostico: ConfiguracionDiagnostico | None = None,
                           *, operaciones: pd.DataFrame | None = None, repetidas: pd.DataFrame | None = None
                           ) -> DiagnosticoV2:
    """Ficha de calidad y suficiencia de cada producto del histórico.

    `resumen`: resultado del motor (`InformeV2.resumen`), para saber qué métodos se pudieron comprobar.
    `datos`: líneas limpias del archivo (fecha, producto, cantidad), para las líneas repetidas y las
    cantidades muy altas. Sin ellas, esas columnas quedan vacías.
    `operaciones`: `ResultadoOperaciones.por_producto` (etapa 4). `repetidas`: resultado de
    `v2_operaciones.repetidas_con_identificadores`; si se pasa, el aviso de líneas repetidas se decide con los
    identificadores y no con la coincidencia de fecha, producto y cantidad.
    """
    cfg, dc = cfg or ConfiguracionV2(), cfg_diagnostico or ConfiguracionDiagnostico()
    para_prever = cfg.cortes_minimos + 1 + cfg.evaluaciones_minimas
    para_buen_respaldo = cfg.cortes_minimos + 1 + cfg.evaluaciones_buen_respaldo
    t = _continuidad(hist)
    lineas, diarias = (lineas_por_producto(datos, dc) if datos is not None else (None, None))
    por_linea = {} if lineas is None else lineas.set_index('producto').to_dict('index')
    motor = {} if resumen is None else resumen.set_index('producto')[
        ['candidatos_validos', 'candidatos_excluidos', 'estado']
        + (['completitud_historico'] if 'completitud_historico' in resumen.columns else [])].to_dict('index')
    por_op = {} if operaciones is None else operaciones.set_index('producto').to_dict('index')
    por_rep = {} if repetidas is None else repetidas.set_index('producto').to_dict('index')
    con_id_linea = repetidas is not None and bool(repetidas.con_id_linea.any())
    con_id_factura = repetidas is not None and bool(repetidas.con_id_factura.any())
    filas = []
    for f in t:
        tramo, ini, p = f.pop('_tramo'), f.pop('_ini'), f['producto']
        n = len(tramo)
        rep, op = por_rep.get(p, {}), por_op.get(p, {})
        f['lineas_con_id_repetido'] = rep.get('lineas_con_id_repetido', 0) if con_id_linea else np.nan
        f['lineas_repetidas_en_factura'] = rep.get('lineas_repetidas_en_factura', 0) if con_id_factura else np.nan
        for c in ('unidades_devueltas', 'unidades_anuladas', 'lineas_ajuste', 'negativos_pendientes'):
            f[c] = op.get(c, 0) if operaciones is not None else np.nan
        venta = rep.get('unidades_de_venta', 0) or 0
        ids = None if repetidas is None else dict(
            con_id_linea=con_id_linea, repetidas_en_factura=rep.get('lineas_repetidas_en_factura', 0),
            parte_en_factura=(rep.get('unidades_repetidas_en_factura', 0) / venta) if venta > 0 else 0.0)
        # líneas del archivo
        li = por_linea.get(p)
        if li is None:
            f.update(lineas=np.nan, lineas_repetidas=np.nan, unidades_repetidas=np.nan, parte_unidades_repetidas=np.nan,
                     repetidas_valorable=False, cantidades_muy_altas=np.nan, muy_altas_en_tramo_final=np.nan,
                     mayor_cantidad_muy_alta=np.nan, mes_mayor_cantidad_muy_alta=None)
            en_tramo = []
        else:
            en_tramo = [(q, m) for q, m in li['extremos'] if n and m is not None and ini <= m < ini + n]
            mayor = max(en_tramo or li['extremos'], default=None)
            f.update(lineas=li['lineas'], lineas_repetidas=li['lineas_repetidas'],
                     unidades_repetidas=li['unidades_repetidas'],
                     parte_unidades_repetidas=li['unidades_repetidas'] / li['unidades'] if li['unidades'] > 0 else np.nan,
                     repetidas_valorable=bool(diarias), cantidades_muy_altas=li['cantidades_muy_altas'],
                     muy_altas_en_tramo_final=len(en_tramo),
                     mayor_cantidad_muy_alta=mayor[0] if mayor else np.nan,
                     mes_mayor_cantidad_muy_alta=fecha(mayor[1]) if mayor and mayor[1] is not None else None)
        # ADI/CV²
        f['patron_demanda'], f['adi'], f['cv2'], f['motivo_patron'] = _patron(hist, tramo, ini)
        # suficiencia y métodos
        f['suficiencia_historial'] = ('sin_tramo_final' if n == 0 else 'insuficiente' if n < para_prever else
                                      'permite_orientativa' if n < para_buen_respaldo else 'permite_buen_respaldo')
        f['meses_que_faltan_para_prever'] = max(para_prever - n, 0)
        f['meses_que_faltan_para_buen_respaldo'] = max(para_buen_respaldo - n, 0)
        m = motor.get(p)
        calculado = m is not None and (m['candidatos_validos'] > 0 or bool(m['candidatos_excluidos']))
        f['metodos_comprobados'] = int(m['candidatos_validos']) if calculado else np.nan
        f['metodos_no_comprobados'] = len(cfg.candidatos) - int(m['candidatos_validos']) if calculado else np.nan
        f['completitud_historico'] = (m or {}).get('completitud_historico', '')
        # avisos para el resultado: cortos y solo si hay algo que revisar
        avisos = _avisos(f, n, dc, ids)
        f['avisos'] = '+'.join(c for c, _ in avisos)
        f['avisos_datos'] = ' '.join([texto for _, texto in avisos if texto][:dc.max_avisos])
        filas.append(f)
    productos = pd.DataFrame(filas, columns=list(COLUMNAS_DIAGNOSTICO))
    por_aviso = {a: int(productos.avisos.str.split('+').map(lambda x: a in x).sum()) for a in AVISOS}
    por_patron = {k: int(v) for k, v in productos.patron_demanda.value_counts().items()}
    return DiagnosticoV2(productos, diarias, para_prever, para_buen_respaldo, dc, por_aviso, por_patron)


def _avisos(f: dict, n: int, dc: ConfiguracionDiagnostico, ids: dict | None = None) -> list[tuple[str, str]]:
    """(código, frase) de cada aviso. Una frase vacía significa que otra frase ya lo explica.

    `ids`: si el archivo trae identificadores, las líneas repetidas se valoran con ellos.
    """
    avisos = []
    k, pend = f['meses_anteriores_sin_usar'], f['meses_pendientes']
    cortado = bool(n and k)
    corta_un_pendiente = cortado and f['motivo_del_corte'] == 'operaciones_pendientes'
    if pend:
        texto = '' if corta_un_pendiente else (
            f'{_plural(pend, "mes tiene", "meses tienen")} devoluciones u otras operaciones sin interpretar y no se '
            f'{"usa" if pend == 1 else "usan"} (el último, {f["ultimo_mes_pendiente"]:%m/%Y}).')
        avisos.append(('operaciones_pendientes', texto))
    if cortado:
        sin_usar = '1 mes anterior' if k == 1 else f'{k} meses anteriores'
        if corta_un_pendiente:
            causa = (f'{f["mes_que_corta_el_historial"]:%m/%Y} tiene devoluciones u otras operaciones sin interpretar '
                     f'y deja sin usar {sin_usar} con datos' + (f' (hay {pend} meses así)' if pend > 1 else ''))
        else:
            causa = f'{f["mes_que_corta_el_historial"]:%m/%Y} no tiene datos confirmados y deja sin usar {sin_usar} con datos'
        avisos.append(('lagunas', f'Solo se usan los {_plural(n, "mes seguido", "meses seguidos")} más recientes: {causa}.'))
    parte = f['parte_unidades_repetidas']
    if ids is not None and ids['con_id_linea']:
        r = int(f['lineas_con_id_repetido'])
        if r:
            avisos.append(('lineas_repetidas',
                           f'{_plural(r, "línea tiene", "líneas tienen")} el mismo identificador que otra: es probable '
                           'que la misma línea esté dos veces en el archivo. Conviene revisarlo.'))
    elif ids is not None:
        r = int(ids['repetidas_en_factura'])
        if r and ids['parte_en_factura'] >= dc.aviso_repetidas_minimo:
            avisos.append(('lineas_repetidas',
                           f'{_plural(r, "línea repite", "líneas repiten")} producto, cantidad y lote dentro de la misma '
                           f'factura ({round(100 * ids["parte_en_factura"])} % de las unidades). Pueden ser entregas '
                           'distintas o líneas duplicadas: conviene revisarlo.'))
    elif f['repetidas_valorable'] and f['lineas_repetidas'] and pd.notna(parte) and parte >= dc.aviso_repetidas_minimo:
        r = int(f['lineas_repetidas'])
        avisos.append(('lineas_repetidas',
                       f'{_plural(r, "línea repite", "líneas repiten")} fecha, producto y cantidad '
                       f'({round(100 * parte)} % de las unidades). Pueden ser ventas distintas o líneas duplicadas: '
                       'conviene revisarlo.'))
    if f['muy_altas_en_tramo_final'] and pd.notna(f['muy_altas_en_tramo_final']):
        a = int(f['muy_altas_en_tramo_final'])
        avisos.append(('cantidades_muy_altas',
                       f'{_plural(a, "venta muy alta", "ventas muy altas")} para este producto (la mayor, '
                       f'{_cantidad(f["mayor_cantidad_muy_alta"])} unidades en {f["mes_mayor_cantidad_muy_alta"]:%m/%Y}). '
                       f'{"Puede ser correcta" if a == 1 else "Pueden ser correctas"}, pero '
                       f'{"podría" if a == 1 else "podrían"} afectar a la precisión de la previsión.'))
    return avisos


# ------------------------------------------------------------------ presentación

def integrar_en_resumen(resumen: pd.DataFrame, diagnostico: DiagnosticoV2) -> pd.DataFrame:
    """Añade al resultado `patron_demanda` y `avisos_datos`. No toca ninguna otra columna."""
    extra = diagnostico.productos[['producto', 'patron_demanda', 'avisos_datos']]
    return resumen.drop(columns=['patron_demanda', 'avisos_datos'], errors='ignore').merge(extra, on='producto', how='left')


def ficha_producto(diagnostico: DiagnosticoV2, producto: str) -> str:
    """Texto de la ficha de un producto, para la pantalla y para la futura interfaz."""
    t = diagnostico.productos
    f = t[t.producto == producto]
    if f.empty:
        raise KeyError(f'Producto desconocido: {producto}')
    f = f.iloc[0]
    lineas = [f'Diagnóstico de {producto}']
    if f.primera_venta is None or pd.isna(f.primera_venta):
        return '\n'.join(lineas + ['  No tiene ninguna venta que se pueda interpretar.'])
    n = int(f.meses_seguidos_conocidos)
    lineas.append(f'  Historial: ventas desde {f.primera_venta:%m/%Y}. De {_plural(int(f.meses_del_periodo), "mes", "meses")}, '
                  f'{int(f.meses_conocidos)} con ventas conocidas ({int(f.meses_con_ventas)} con alguna venta).')
    if n:
        lineas.append(f'  Meses seguidos que se usan: {n}, desde {f.inicio_tramo_final:%m/%Y}.')
    else:
        lineas.append('  Meses seguidos que se usan: ninguno (no se conocen las ventas del último mes).')
    if f.meses_desconocidos:
        lineas.append(f'  Lagunas: {_plural(int(f.meses_desconocidos), "mes", "meses")} sin datos confirmados, en '
                      f'{_plural(int(f.lagunas), "tramo", "tramos")} (el último, {f.ultimo_mes_desconocido:%m/%Y}).'
                      )
    else:
        lineas.append('  Lagunas: ninguna.')
    if n and f.meses_anteriores_sin_usar:
        causa = ('tiene operaciones sin interpretar' if f.motivo_del_corte == 'operaciones_pendientes'
                 else 'no tiene datos confirmados')
        lineas.append(f'  Historial cortado: {f.mes_que_corta_el_historial:%m/%Y} {causa}; '
                      f'{_plural(int(f.meses_anteriores_sin_usar), "mes anterior queda", "meses anteriores quedan")} sin usar.')
    if f.meses_fuera_de_venta:
        lineas.append(f'  Fuera de venta: {_plural(int(f.meses_fuera_de_venta), "mes", "meses")} (producto retirado).')
    lineas.append(f'  Operaciones pendientes: {_plural(int(f.meses_pendientes), "mes", "meses")}'
                  + (f' (el último, {f.ultimo_mes_pendiente:%m/%Y}).' if f.meses_pendientes else '.'))
    if pd.isna(f.lineas):
        lineas.append('  Líneas repetidas y cantidades muy altas: sin revisar (no se han pasado las líneas del archivo).')
    else:
        if pd.notna(f.lineas_con_id_repetido):
            k = int(f.lineas_con_id_repetido)
            lineas.append('  Líneas repetidas: cada línea tiene su propio identificador; ninguna está dos veces.' if k == 0
                          else f'  Líneas repetidas: {_plural(k, "línea comparte", "líneas comparten")} identificador con '
                               'otra. Es probable que estén duplicadas.')
        elif not f.repetidas_valorable:
            lineas.append('  Líneas repetidas: no se puede valorar, porque las fechas del archivo no son diarias.')
        elif f.lineas_repetidas:
            lineas.append(f'  Líneas repetidas: {int(f.lineas_repetidas)} de {int(f.lineas)} repiten fecha, producto y '
                          f'cantidad ({round(100 * f.parte_unidades_repetidas)} % de las unidades). Es un indicio, no '
                          'un duplicado probado.')
        else:
            lineas.append('  Líneas repetidas: ninguna.')
        if pd.notna(f.lineas_repetidas_en_factura) and f.lineas_repetidas_en_factura:
            lineas.append(f'  En la misma factura: {_plural(int(f.lineas_repetidas_en_factura), "línea repite", "líneas repiten")} '
                          'producto, cantidad y lote. Puede ser correcto.')
        if f.cantidades_muy_altas:
            lineas.append(f'  Cantidades muy altas: {int(f.cantidades_muy_altas)} '
                          f'({int(f.muy_altas_en_tramo_final)} en los meses que se usan); la mayor, '
                          f'{_cantidad(f.mayor_cantidad_muy_alta)} unidades en {f.mes_mayor_cantidad_muy_alta:%m/%Y}. '
                          'No se corrigen.')
        else:
            lineas.append('  Cantidades muy altas: ninguna.')
    if pd.notna(f.unidades_devueltas):
        partes = []
        if f.unidades_devueltas:
            partes.append(f'{_cantidad(f.unidades_devueltas)} unidades devueltas, contadas aparte (no se restan de las ventas)')
        if f.unidades_anuladas:
            partes.append(f'{_cantidad(f.unidades_anuladas)} unidades anuladas, quitadas de su venta')
        if f.lineas_ajuste:
            partes.append(f'{_plural(int(f.lineas_ajuste), "línea de ajuste que no cuenta", "líneas de ajuste que no cuentan")}')
        if f.negativos_pendientes:
            partes.append(f'{_plural(int(f.negativos_pendientes), "línea negativa", "líneas negativas")} sin confirmar')
        lineas.append('  Devoluciones y anulaciones: ' + ('; '.join(partes) if partes else 'ninguna') + '.')
    if f.patron_demanda in TEXTO_PATRON:
        lineas.append(f'  Patrón de ventas: {f.patron_demanda}. {TEXTO_PATRON[f.patron_demanda]} '
                      f'(ADI {_decimal(f.adi)}; CV² {_decimal(f.cv2)}). Describe las ventas; no dice si la previsión acierta.')
    else:
        lineas.append(f'  Patrón de ventas: sin clasificar. {f.motivo_patron}')
    a, b = diagnostico.meses_para_prever, diagnostico.meses_para_buen_respaldo
    texto = {'sin_tramo_final': 'no hay meses seguidos hasta el último mes.',
             'insuficiente': f'{_faltan(f.meses_que_faltan_para_prever)} para poder comprobar una previsión.',
             'permite_orientativa': f'permiten comprobar una previsión; {_faltan(f.meses_que_faltan_para_buen_respaldo)} '
                                    'para que pueda llegar a tener buen respaldo.',
             'permite_buen_respaldo': 'son suficientes para comprobar la previsión en 12 meses.'}[f.suficiencia_historial]
    lineas.append(f'  Suficiencia: {_plural(n, "mes seguido", "meses seguidos")}; {texto} (Con la configuración actual hacen falta {a} y {b}; '
                  'tener meses suficientes no garantiza una buena previsión.)')
    completitud = getattr(f, 'completitud_historico', '')
    if isinstance(completitud, str) and completitud and completitud != 'confirmado':
        lineas.append('  Completitud: no está confirmado que los archivos contengan todas las ventas de los meses que se usan. '
                      'Mientras no lo esté, la previsión no se presentará con buen respaldo: como mucho, orientativa.')
    if pd.notna(f.metodos_comprobados):
        total = int(f.metodos_comprobados + f.metodos_no_comprobados)
        k = int(f.metodos_comprobados)
        lineas.append(f'  Métodos: {"se ha" if k == 1 else "se han"} podido comprobar {k} de {total}.')
    else:
        lineas.append('  Métodos: no se ha calculado ninguno para este producto.')
    return '\n'.join(lineas)


def exportar_diagnostico(diagnostico: DiagnosticoV2, carpeta: Path) -> dict[str, Path]:
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / 'diagnostico_v2.csv'
    escribir_csv(diagnostico.productos, ruta)
    return {'diagnostico_v2': ruta}


def mostrar_diagnostico_corto(diagnostico: DiagnosticoV2, ejemplo: str | None = None) -> None:
    t = diagnostico.productos
    print(f'\nDIAGNÓSTICO DE LOS DATOS DE CADA PRODUCTO ({len(t)} productos)')
    nombres = {'operaciones_pendientes': 'con meses pendientes de interpretar',
               'lagunas': 'con meses anteriores sin usar por lagunas',
               'lineas_repetidas': 'con líneas repetidas que conviene revisar',
               'cantidades_muy_altas': 'con ventas muy altas en los meses que se usan'}
    for a in AVISOS:
        print(f'  Productos {nombres[a]}: {diagnostico.por_aviso.get(a, 0)}')
    if diagnostico.fechas_diarias is False:
        print('  Las fechas del archivo no son diarias: no se pueden valorar las líneas repetidas.')
    patrones = ', '.join(f'{p}: {diagnostico.por_patron.get(p, 0)}' for p in PATRONES)
    print(f'  Patrón de ventas: {patrones}; sin clasificar: {diagnostico.por_patron.get(SIN_CLASIFICAR, 0)}')
    print('  Es información sobre los datos: no cambia las previsiones ni su estado. Detalle en diagnostico_v2.csv.')
    if ejemplo is not None:
        print('\n' + ficha_producto(diagnostico, ejemplo))
