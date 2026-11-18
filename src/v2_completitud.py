"""Versión 2, etapa 1: completitud del histórico, último mes, ceros confirmados y estado comercial.

Este módulo NO modifica el objetivo mensual de la versión 1 (`objetivo_mensual.py`): lo lee y construye,
aparte, el histórico que usa el motor de la versión 2.

Reglas:
- Un mes sin operaciones solo pasa a «cero confirmado» si el usuario confirma que el histórico está
  completo para ese mes. Sin esa confirmación sigue siendo desconocido.
- Nunca hay ceros antes de la primera venta positiva conocida del producto.
- Los huecos entre dos meses con operaciones válidas del producto se convierten (el producto se vendía
  antes y después). Los meses posteriores a su última operación válida solo se convierten si el usuario
  confirma que el producto ha seguido a la venta.
- Los meses con operaciones pendientes de interpretar (por ejemplo, devoluciones) no se convierten.
- Si el usuario no confirma que el último mes está completo, ese mes no se usa para ningún producto.
- Llevar muchos meses sin ventas no convierte a un producto en retirado: el estado comercial solo lo fija
  el usuario (activo confirmado, retirado confirmado o desconocido). «Sin ventas recientes» es un hecho
  de los datos y se informa aparte.
- Un producto retirado puede llevar la fecha (mes) desde la que está retirado. Los meses entre su última
  operación y esa fecha estuvo a la venta (pueden ser ceros confirmados); desde esa fecha queda fuera de
  venta. Sin fecha, los meses posteriores a su última operación siguen como desconocidos (interpretación
  prudente: pudo seguir a la venta parte de ese tiempo).
- Un cero confirmado significa que no hubo ventas registradas, no que no hubiera demanda.
"""
from __future__ import annotations

import calendar
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .v2_exportacion import escribir_csv
from .objetivo_mensual import InformeObjetivoMensual
from .reentrenamiento_final import _validar_objetivo

RESPUESTAS = ('si', 'no', 'no_lo_se')
ESTADOS_COMERCIALES = ('activo_confirmado', 'retirado_confirmado', 'desconocido')
ESTADOS_CON_VALOR = ('positivo_registrado', 'cero_explicito', 'cero_confirmado')
EVIDENCIA_DE_VENTA = ('positivo_registrado', 'cero_explicito')   # el producto figura en el archivo con una operación válida
MOTIVO_HUECO = 'hueco_entre_operaciones_con_historico_confirmado'
MOTIVO_ACTIVO = 'mes_posterior_a_la_ultima_operacion_de_producto_confirmado_como_activo'
MOTIVO_ANTES_RETIRADA = 'mes_anterior_a_la_fecha_de_retirada_confirmada'
# Meses en los que todo lo que había eran ventas anuladas por completo (etapa 4). La anulación demuestra que
# esas operaciones no cuentan, no que no hubiera otras ventas: son un cero solo con el histórico confirmado.
MOTIVO_ANULADO_CONFIRMADO = 'ventas_del_mes_anuladas_e_historico_confirmado'
MOTIVO_ANULADO_SIN_CONFIRMAR = 'ventas_del_mes_anuladas_sin_confirmar_el_historico'
AVISO_CEROS = ('Los ceros confirmados indican que no hubo ventas registradas en esos meses. No demuestran que '
               'no hubiera demanda: pudo faltar el producto en el almacén.')
COLUMNAS_DETALLE = ('producto', 'mes', 'estado', 'unidades', 'origen', 'motivo')
COLUMNAS_PRODUCTOS = (
    'producto', 'estado_comercial', 'origen_estado_comercial', 'retirado_desde', 'primera_venta', 'ultima_venta',
    'ultima_operacion_valida', 'meses_sin_ventas', 'sin_ventas_recientes', 'meses_con_ventas',
    'ceros_del_archivo', 'ceros_confirmados', 'meses_desconocidos', 'meses_pendientes',
    'llega_al_ultimo_mes', 'situacion', 'explicacion')
COLUMNAS_CONVERSIONES = ('producto', 'mes', 'motivo')


def fecha(i: int) -> date:
    return date(i // 12, i % 12 + 1, 1)


def indice(d: date) -> int:
    return d.year * 12 + d.month - 1


def _mes_texto(i: int) -> str:
    return f'{i % 12 + 1:02d}/{i // 12}'


# ------------------------------------------------------------------ confirmaciones

@dataclass(frozen=True)
class ConfirmacionCompletitud:
    """Lo que el usuario ha confirmado sobre sus archivos. Los valores por defecto son los prudentes."""
    completo: str = 'no_lo_se'                 # ¿contienen todas las ventas del periodo?
    periodo_inicio: date | None = None         # primer mes confirmado
    periodo_fin: date | None = None            # último mes confirmado
    meses_excluidos: tuple[date, ...] = ()     # meses del periodo que el usuario deja fuera
    ultimo_mes_completo: str = 'no_lo_se'      # ¿están ya todas las ventas del último mes del archivo?
    fecha_hora: str = ''                       # cuándo se respondió
    pregunta_completitud: str = ''             # texto literal mostrado
    pregunta_ultimo_mes: str = ''

    def __post_init__(self):
        if self.completo not in RESPUESTAS or self.ultimo_mes_completo not in RESPUESTAS:
            raise ValueError('Las respuestas deben ser si, no o no_lo_se.')
        for d in (self.periodo_inicio, self.periodo_fin, *self.meses_excluidos):
            if d is not None and (type(d) is not date or d.day != 1):
                raise ValueError('Los meses deben indicarse con el día 1.')
        if self.completo == 'si':
            if self.periodo_inicio is None or self.periodo_fin is None:
                raise ValueError('Una confirmación de completitud necesita el periodo al que se refiere.')
            if self.periodo_inicio > self.periodo_fin:
                raise ValueError('El periodo confirmado termina antes de empezar.')

    def cubre(self, mes: int) -> bool:
        """¿Ha confirmado el usuario que el archivo contiene todas las ventas de ese mes?"""
        if self.completo != 'si':
            return False
        return (indice(self.periodo_inicio) <= mes <= indice(self.periodo_fin)
                and fecha(mes) not in self.meses_excluidos)

    def a_dict(self) -> dict:
        return {'completo': self.completo,
                'periodo_inicio': str(self.periodo_inicio) if self.periodo_inicio else None,
                'periodo_fin': str(self.periodo_fin) if self.periodo_fin else None,
                'meses_excluidos': [str(m) for m in self.meses_excluidos],
                'ultimo_mes_completo': self.ultimo_mes_completo, 'fecha_hora': self.fecha_hora,
                'pregunta_completitud': self.pregunta_completitud, 'pregunta_ultimo_mes': self.pregunta_ultimo_mes}

    @classmethod
    def desde_dict(cls, d: dict) -> 'ConfirmacionCompletitud':
        f = lambda x: date.fromisoformat(x) if x else None
        return cls(completo=d['completo'], periodo_inicio=f(d.get('periodo_inicio')),
                   periodo_fin=f(d.get('periodo_fin')),
                   meses_excluidos=tuple(date.fromisoformat(m) for m in d.get('meses_excluidos', [])),
                   ultimo_mes_completo=d['ultimo_mes_completo'], fecha_hora=d.get('fecha_hora', ''),
                   pregunta_completitud=d.get('pregunta_completitud', ''),
                   pregunta_ultimo_mes=d.get('pregunta_ultimo_mes', ''))


@dataclass(frozen=True)
class EstadoComercial:
    """Estado comercial ACTUAL de un producto. Solo lo fija el usuario (o una columna que él haya confirmado).

    `retirado_desde`: para un producto retirado, primer mes en que ya no estaba a la venta (opcional).
    """
    estado: str = 'desconocido'
    origen: str = 'sin confirmar'
    retirado_desde: date | None = None

    def __post_init__(self):
        if self.estado not in ESTADOS_COMERCIALES:
            raise ValueError('Estado comercial no reconocido.')
        if self.retirado_desde is not None:
            if self.estado != 'retirado_confirmado':
                raise ValueError('Solo un producto retirado puede tener fecha de retirada.')
            if type(self.retirado_desde) is not date or self.retirado_desde.day != 1:
                raise ValueError('La fecha de retirada debe indicarse con el día 1 del mes.')


@dataclass(frozen=True)
class GrupoProductos:
    clave: str
    descripcion: str
    productos: tuple[str, ...]


@dataclass
class HistoricoV2:
    series: dict[str, dict[int, float | None]]   # lo que recibe el motor (None = desconocido o pendiente)
    detalle: pd.DataFrame                         # un mes por fila, con su estado y su origen
    productos: pd.DataFrame                       # situación de cada producto
    conversiones: pd.DataFrame                    # cada mes convertido a cero y por qué
    situaciones: dict[str, tuple[str, str]]       # producto -> (motivo, explicación) si no procede prever
    ultimo_mes_archivo: int
    ultimo_mes: int                               # último mes utilizable (completo)
    confirmacion: ConfirmacionCompletitud
    meses_sin_ventas_recientes: int
    estados_comerciales: dict[str, EstadoComercial] = field(default_factory=dict)


# ------------------------------------------------------------------ lectura del objetivo de la versión 1

def _leer_objetivo(objetivo: InformeObjetivoMensual) -> dict[str, dict[int, tuple[str, float | None]]]:
    """producto -> {mes: (estado, unidades)}. Valida con la función de la versión 1 y no modifica nada."""
    valores = _validar_objetivo(objetivo)
    filas: dict[str, dict[int, tuple[str, float | None]]] = {}
    for r in objetivo.detalle[['producto', 'mes', 'estado']].itertuples(index=False):
        m = indice(r.mes)
        filas.setdefault(r.producto, {})[m] = (r.estado, valores[r.producto][m])
    return filas


def ultimo_mes_del_archivo(objetivo: InformeObjetivoMensual) -> int:
    return max(indice(m) for m in objetivo.detalle['mes'])


def primer_mes_del_archivo(objetivo: InformeObjetivoMensual) -> int:
    return min(indice(m) for m in objetivo.detalle['mes'])


def _ultimo_utilizable(ultimo_archivo: int, confirmacion: ConfirmacionCompletitud) -> int:
    return ultimo_archivo if confirmacion.ultimo_mes_completo == 'si' else ultimo_archivo - 1


def _basicos(filas: dict[int, tuple[str, float | None]], ultimo: int) -> dict:
    positivos = [m for m, (e, _) in filas.items() if e == 'positivo_registrado' and m <= ultimo]
    evidencia = [m for m, (e, _) in filas.items() if e in EVIDENCIA_DE_VENTA and m <= ultimo]
    return {'primera_venta': min(positivos) if positivos else None,
            'ultima_venta': max(positivos) if positivos else None,
            'ultima_operacion_valida': max(evidencia) if evidencia else None}


# ------------------------------------------------------------------ grupos para confirmar el estado comercial

def proponer_grupos(objetivo: InformeObjetivoMensual, confirmacion: ConfirmacionCompletitud,
                    cortes: tuple[int, ...] = (3, 6, 12)) -> list[GrupoProductos]:
    """Agrupa los productos sin operaciones en los últimos meses, para preguntar por grupos y no uno a uno.

    Solo entran productos con alguna venta conocida y sin operaciones válidas en el último mes utilizable.
    """
    if not cortes or list(cortes) != sorted(set(cortes)) or cortes[0] < 1:
        raise ValueError('Los cortes de los grupos deben ser crecientes y positivos.')
    ultimo = _ultimo_utilizable(ultimo_mes_del_archivo(objetivo), confirmacion)
    cubos: dict[int, list[str]] = {i: [] for i in range(len(cortes) + 1)}
    for p, filas in _leer_objetivo(objetivo).items():
        b = _basicos(filas, ultimo)
        if b['primera_venta'] is None or b['ultima_operacion_valida'] >= ultimo:
            continue
        meses = ultimo - b['ultima_operacion_valida']
        cubos[next((i for i, c in enumerate(cortes) if meses <= c), len(cortes))].append(p)
    grupos, anterior = [], 0
    for i, c in enumerate(cortes):
        if cubos[i]:
            texto = 'desde hace 1 mes' if (anterior, c) == (0, 1) else f'desde hace entre {anterior + 1} y {c} meses'
            grupos.append(GrupoProductos(f'sin_operaciones_{anterior + 1}_{c}',
                                         f'Sin ventas registradas {texto}', tuple(sorted(cubos[i]))))
        anterior = c
    if cubos[len(cortes)]:
        grupos.append(GrupoProductos(f'sin_operaciones_mas_de_{cortes[-1]}',
                                     f'Sin ventas registradas desde hace más de {cortes[-1]} meses',
                                     tuple(sorted(cubos[len(cortes)]))))
    return grupos


def aplicar_respuestas(grupos: list[GrupoProductos], respuestas: dict[str, str],
                       excepciones: dict[str, str] | None = None, *,
                       retirada_por_grupo: dict[str, date] | None = None,
                       retirada_por_producto: dict[str, date] | None = None,
                       ultimas: dict[str, int] | None = None) -> dict[str, EstadoComercial]:
    """Convierte las respuestas por grupo y las excepciones individuales en estados comerciales.

    Un grupo sin respuesta, o con respuesta «desconocido», deja sus productos como desconocidos.
    `retirada_por_grupo` (clave de grupo -> mes) se aplica a los productos retirados de ese grupo que no
    sean excepción; `retirada_por_producto` (código -> mes), a las excepciones retiradas.
    Sin fecha, el producto retirado queda sin fecha de retirada.
    `ultimas` (código -> último mes con operación): si se indica, la fecha del grupo no se aplica a los
    productos con operaciones en ese mes o después; quedan retirados sin fecha.
    """
    excepciones = excepciones or {}
    retirada_por_grupo, retirada_por_producto = retirada_por_grupo or {}, retirada_por_producto or {}
    conocidos = {p for g in grupos for p in g.productos}
    claves = {g.clave for g in grupos}
    if not set(respuestas).issubset(claves) or not set(retirada_por_grupo).issubset(claves):
        raise ValueError('Hay respuestas para grupos que no existen.')
    if not set(excepciones).issubset(conocidos):
        raise ValueError('Hay excepciones para productos que no están en ningún grupo.')
    if any(respuestas.get(c) != 'retirado_confirmado' for c in retirada_por_grupo):
        raise ValueError('Solo un grupo retirado puede tener fecha de retirada.')
    if any(excepciones.get(p) != 'retirado_confirmado' for p in retirada_por_producto):
        raise ValueError('Solo una excepción retirada puede tener fecha de retirada.')
    estados: dict[str, EstadoComercial] = {}
    for g in grupos:
        estado = respuestas.get(g.clave, 'desconocido')
        for p in g.productos:
            if p in excepciones:
                estados[p] = EstadoComercial(excepciones[p], 'excepción indicada por el usuario',
                                             retirada_por_producto.get(p))
            elif estado != 'desconocido':
                origen = f'confirmación por grupo: {g.descripcion.lower()}'
                fecha_grupo = retirada_por_grupo.get(g.clave) if estado == 'retirado_confirmado' else None
                if fecha_grupo is not None and ultimas is not None and ultimas.get(p, -1) >= indice(fecha_grupo):
                    fecha_grupo = None
                    origen += ' (la fecha del grupo no se aplica: tiene ventas en ese mes o después)'
                estados[p] = EstadoComercial(estado, origen, fecha_grupo)
    return estados


def ultimas_operaciones(objetivo: InformeObjetivoMensual, confirmacion: ConfirmacionCompletitud) -> dict[str, int]:
    """Último mes con una operación válida de cada producto, dentro de los meses utilizables."""
    ultimo = _ultimo_utilizable(ultimo_mes_del_archivo(objetivo), confirmacion)
    out = {}
    for p, filas in _leer_objetivo(objetivo).items():
        u = _basicos(filas, ultimo)['ultima_operacion_valida']
        if u is not None:
            out[p] = u
    return out


# ------------------------------------------------------------------ construcción del histórico

def construir_historico(objetivo: InformeObjetivoMensual,
                        confirmacion: ConfirmacionCompletitud | None = None,
                        estados: dict[str, EstadoComercial] | None = None, *,
                        meses_sin_ventas_recientes: int = 12,
                        meses_anulados=None) -> HistoricoV2:
    """Histórico que usa el motor, aplicando solo lo que el usuario ha confirmado.

    `meses_anulados`: (producto, mes) en los que todo lo que había eran ventas anuladas por completo (etapa 4).
    Se tratan como cualquier mes sin registros (cero solo si el histórico está confirmado), con un motivo propio.

    Sin confirmaciones devuelve las mismas series que la versión 1 salvo el último mes del archivo, que no
    se usa mientras no se confirme que está completo.
    """
    confirmacion = confirmacion or ConfirmacionCompletitud()
    estados = dict(estados or {})
    if type(meses_sin_ventas_recientes) is not int or meses_sin_ventas_recientes < 1:
        raise ValueError('meses_sin_ventas_recientes debe ser un entero positivo.')
    todo = _leer_objetivo(objetivo)
    if not set(estados).issubset(todo):
        raise ValueError('Hay estados comerciales de productos que no están en el archivo.')
    ultimo_archivo = max(max(f) for f in todo.values())
    ultimo = _ultimo_utilizable(ultimo_archivo, confirmacion)
    n = meses_sin_ventas_recientes
    for p, e in estados.items():
        if e.retirado_desde is None:
            continue
        r, u = indice(e.retirado_desde), _basicos(todo[p], ultimo)['ultima_operacion_valida']
        if (u is not None and r <= u) or r > ultimo + 1:
            raise ValueError(f'{p}: la fecha de retirada debe ser posterior a su última operación '
                             f'({_mes_texto(u) if u is not None else "ninguna"}) y como muy tarde {_mes_texto(ultimo + 1)}.')
    anulados = frozenset(meses_anulados or ())
    detalle, conversiones, productos = [], [], []
    series: dict[str, dict[int, float | None]] = {}
    situaciones: dict[str, tuple[str, str]] = {}

    for p in sorted(todo):
        filas = todo[p]
        b = _basicos(filas, ultimo)
        primera, ultima_venta, ultima_valida = b['primera_venta'], b['ultima_venta'], b['ultima_operacion_valida']
        comercial = estados.get(p, EstadoComercial())
        nuevo: dict[int, tuple[str, float | None, str, str]] = {}   # mes -> (estado, unidades, origen, motivo)
        for m in range(min(filas), max(max(filas), ultimo) + 1):
            est, val = filas.get(m, ('sin_registro', None))
            if m > ultimo:
                if m in filas:   # último mes del archivo no confirmado como completo: no se usa
                    nuevo[m] = ('mes_incompleto', None, 'archivo', 'ultimo_mes_no_confirmado_como_completo')
                continue
            if est != 'sin_registro':
                nuevo[m] = (est, val, 'archivo', '')
                continue
            if primera is None or m < primera:
                nuevo[m] = ('sin_registro', None, 'archivo', 'anterior_a_la_primera_venta')
            elif not confirmacion.cubre(m):
                nuevo[m] = ('sin_registro', None, 'archivo', 'historico_no_confirmado_para_este_mes')
            elif m < ultima_valida:
                nuevo[m] = ('cero_confirmado', 0.0, 'confirmacion', MOTIVO_HUECO)
            elif comercial.estado == 'activo_confirmado':
                nuevo[m] = ('cero_confirmado', 0.0, 'confirmacion', MOTIVO_ACTIVO)
            elif comercial.estado == 'retirado_confirmado':
                if comercial.retirado_desde is None:
                    nuevo[m] = ('sin_registro', None, 'archivo', 'retirado_sin_fecha_de_retirada')
                elif m < indice(comercial.retirado_desde):
                    nuevo[m] = ('cero_confirmado', 0.0, 'confirmacion', MOTIVO_ANTES_RETIRADA)
                else:
                    nuevo[m] = ('fuera_de_venta', None, 'confirmacion', 'producto_retirado_desde_ese_mes')
            else:
                nuevo[m] = ('sin_registro', None, 'archivo', 'estado_comercial_sin_confirmar')
            if (p, m) in anulados:
                # Todo lo que había ese mes eran ventas anuladas: se distingue de un mes sin ningún documento.
                est_m, val_m, origen_m, motivo_m = nuevo[m]
                if est_m == 'cero_confirmado':
                    motivo_m = MOTIVO_ANULADO_CONFIRMADO
                elif est_m == 'sin_registro':
                    motivo_m = (MOTIVO_ANULADO_SIN_CONFIRMAR if motivo_m == 'historico_no_confirmado_para_este_mes'
                                else f'{motivo_m}; ventas_del_mes_anuladas')
                nuevo[m] = (est_m, val_m, origen_m, motivo_m)
        for m, (est, val, origen, motivo) in sorted(nuevo.items()):
            detalle.append((p, fecha(m), est, val, origen, motivo))
            if est == 'cero_confirmado':
                conversiones.append((p, fecha(m), motivo))
        series[p] = {m: (v if e in ESTADOS_CON_VALOR else None) for m, (e, v, _, _) in nuevo.items() if m <= ultimo}

        cuenta = lambda *e: sum(1 for m, x in nuevo.items() if x[0] in e and m <= ultimo)
        llega = series[p].get(ultimo) is not None
        estado_ultimo = nuevo.get(ultimo, (None,))[0]
        sin_recientes = ultima_venta is None or ultimo - ultima_venta >= n
        situacion = None
        recientes = [series[p].get(m) for m in range(ultimo - n + 1, ultimo + 1)]
        if comercial.estado == 'retirado_confirmado':
            if comercial.retirado_desde is not None:
                situacion = ('producto_retirado', f'Producto retirado desde {comercial.retirado_desde:%m/%Y}, según '
                                                  'has confirmado: no se calcula previsión de ventas.')
            else:
                situacion = ('producto_retirado', 'Producto retirado, según has confirmado (sin fecha de retirada): '
                                                  'no se calcula previsión de ventas.')
        elif primera is None:
            if any(m > ultimo for m in filas):
                situacion = ('sin_ventas_conocidas', f'Sus únicas ventas son de {_mes_texto(ultimo_archivo)}, '
                                                     'un mes que no se ha confirmado como completo.')
            else:
                situacion = ('sin_ventas_conocidas', 'No tiene ninguna venta registrada que se pueda interpretar.')
        elif not llega:
            if estado_ultimo is not None and estado_ultimo.startswith('pendiente'):
                situacion = ('ultimo_mes_pendiente', f'Las operaciones de {_mes_texto(ultimo)} están pendientes de '
                                                     'interpretar (por ejemplo, devoluciones).')
            elif not confirmacion.cubre(ultimo):
                situacion = ('historico_sin_confirmar',
                             f'No tiene ventas registradas en {_mes_texto(ultimo)} y no se ha confirmado que el '
                             'histórico esté completo: no se sabe si no vendió o si faltan datos.')
            else:
                situacion = ('estado_comercial_sin_confirmar',
                             f'No tiene ventas registradas desde {_mes_texto(ultima_valida)} y no se ha confirmado '
                             'si ha seguido a la venta.')
        elif sin_recientes and all(v is not None for v in recientes):
            # Los ceros escritos en el archivo son ventas cero, pero no prueban que el producto estuviera a la
            # venta: el mensaje dice «sin ventas registradas» y no lo da por activo ni por retirado.
            if comercial.estado == 'activo_confirmado':
                texto = (f'Sigue a la venta, según has confirmado, pero no tiene ventas registradas en los últimos '
                         f'{n} meses: no hay ventas con las que comprobar una previsión.')
            else:
                texto = (f'Sin ventas registradas en los últimos {n} meses (el archivo recoge ventas cero). No se ha '
                         'confirmado si sigue a la venta o está retirado. No hay ventas con las que comprobar una '
                         'previsión.')
            situacion = ('sin_actividad_reciente', texto)
        elif sin_recientes:
            situacion = ('historico_sin_confirmar',
                         f'No tiene ventas en los últimos {n} meses y algunos de esos meses son desconocidos: '
                         'no se sabe si no vendió o si faltan datos.')
        if situacion:
            situaciones[p] = situacion
        productos.append((p, comercial.estado, comercial.origen, comercial.retirado_desde,
                          fecha(primera) if primera is not None else None,
                          fecha(ultima_venta) if ultima_venta is not None else None,
                          fecha(ultima_valida) if ultima_valida is not None else None,
                          (ultimo - ultima_venta) if ultima_venta is not None else None, sin_recientes,
                          cuenta('positivo_registrado'), cuenta('cero_explicito'), cuenta('cero_confirmado'),
                          cuenta('sin_registro'),
                          sum(1 for m, x in nuevo.items() if x[0].startswith('pendiente') and m <= ultimo),
                          llega, situacion[0] if situacion else 'se_calcula', situacion[1] if situacion else ''))

    return HistoricoV2(
        series=series, detalle=pd.DataFrame(detalle, columns=COLUMNAS_DETALLE),
        productos=pd.DataFrame(productos, columns=COLUMNAS_PRODUCTOS),
        conversiones=pd.DataFrame(conversiones, columns=COLUMNAS_CONVERSIONES),
        situaciones=situaciones, ultimo_mes_archivo=ultimo_archivo, ultimo_mes=ultimo,
        confirmacion=confirmacion, meses_sin_ventas_recientes=n, estados_comerciales=estados)


# ------------------------------------------------------------------ indicio sobre el último mes

@dataclass(frozen=True)
class IndicioUltimoMes:
    """Ayuda para la pregunta sobre el último mes. Nunca decide: la respuesta es del usuario."""
    mes: date
    valorable: bool
    indicio: str                 # 'posible_mes_parcial', 'sin_indicios' o 'no_valorable'
    ultimo_dia_con_ventas: int | None
    dias_del_mes: int
    dias_con_ventas: int
    texto: str


def diagnosticar_ultimo_mes(fechas, margen_dias: int = 7) -> IndicioUltimoMes:
    """Compara el último día con operaciones del último mes con lo habitual en los meses anteriores.

    Si las fechas no son diarias (por ejemplo, todas las operaciones llevan el último día del mes), no se
    puede valorar nada. Que haya operaciones hasta el final del mes tampoco demuestra que estén todas.
    """
    validas = pd.Series([f for f in fechas if isinstance(f, (pd.Timestamp, datetime, date)) and pd.notna(f)],
                        dtype='object')
    if validas.empty:
        raise ValueError('No hay fechas interpretables.')
    validas = pd.to_datetime(validas)
    periodo = validas.dt.to_period('M')
    ultimo = periodo.max()
    dias_mes = calendar.monthrange(ultimo.year, ultimo.month)[1]
    mes = date(ultimo.year, ultimo.month, 1)
    del_mes = validas[periodo == ultimo]
    ultimo_dia, dias_con = int(del_mes.dt.day.max()), int(del_mes.dt.day.nunique())
    anteriores = validas[periodo < ultimo]
    dias_por_mes = anteriores.groupby(periodo[periodo < ultimo]).agg(lambda x: x.dt.day.nunique())
    if anteriores.empty or dias_por_mes.median() < 4:
        return IndicioUltimoMes(mes, False, 'no_valorable', ultimo_dia, dias_mes, dias_con,
                                'Las fechas del archivo no permiten valorar si el último mes está completo '
                                '(no hay fechas diarias o no hay meses anteriores con los que comparar).')
    falta = anteriores.groupby(periodo[periodo < ultimo]).agg(
        lambda x: calendar.monthrange(x.iloc[0].year, x.iloc[0].month)[1] - x.dt.day.max())
    habitual = float(falta.median())
    if dias_mes - ultimo_dia > habitual + margen_dias:
        return IndicioUltimoMes(mes, True, 'posible_mes_parcial', ultimo_dia, dias_mes, dias_con,
                                f'La última operación de {mes:%m/%Y} es del día {ultimo_dia} y el mes tiene '
                                f'{dias_mes} días. En los meses anteriores las ventas llegan casi hasta el final: '
                                'el mes podría estar a medias.')
    return IndicioUltimoMes(mes, True, 'sin_indicios', ultimo_dia, dias_mes, dias_con,
                            f'Hay operaciones de {mes:%m/%Y} hasta el día {ultimo_dia}, como en los meses '
                            'anteriores. Eso no demuestra que estén todas las ventas del mes.')


# ------------------------------------------------------------------ registro auditable

def registro_auditable(hist: HistoricoV2, *, conjunto: dict | None = None,
                       grupos: list[GrupoProductos] | None = None, respuestas: dict[str, str] | None = None,
                       excepciones: dict[str, str] | None = None, indicio: IndicioUltimoMes | None = None,
                       retirada_por_grupo: dict[str, date] | None = None,
                       retirada_por_producto: dict[str, date] | None = None) -> dict:
    """Todo lo necesario para saber, más adelante, qué se confirmó y qué meses cambió esa confirmación."""
    c, prod = hist.confirmacion, hist.productos
    respuestas, excepciones = respuestas or {}, excepciones or {}
    return {
        'version_registro': 1,
        'conjunto_de_datos': conjunto,
        'confirmacion': c.a_dict(),
        'ultimo_mes_del_archivo': str(fecha(hist.ultimo_mes_archivo)),
        'ultimo_mes_utilizado': str(fecha(hist.ultimo_mes)),
        'mes_previsto': str(fecha(hist.ultimo_mes + 1)),
        'indicio_ultimo_mes': None if indicio is None else {
            'indicio': indicio.indicio, 'ultimo_dia_con_ventas': indicio.ultimo_dia_con_ventas,
            'dias_del_mes': indicio.dias_del_mes, 'texto': indicio.texto},
        'estado_comercial': {
            'grupos': [{'clave': g.clave, 'descripcion': g.descripcion, 'productos': len(g.productos),
                        'respuesta': respuestas.get(g.clave, 'desconocido')} for g in (grupos or [])],
            'excepciones': [{'producto': p, 'estado': e,
                             'retirado_desde': str((retirada_por_producto or {})[p]) if p in (retirada_por_producto or {}) else None}
                            for p, e in sorted(excepciones.items())],
            'productos': {p: {'estado': e.estado, 'origen': e.origen,
                              'retirado_desde': str(e.retirado_desde) if e.retirado_desde else None}
                          for p, e in sorted(hist.estados_comerciales.items())},
            'fechas_de_retirada_por_grupo': {c: str(f) for c, f in sorted((retirada_por_grupo or {}).items())},
            'resumen': {e: int((prod.estado_comercial == e).sum()) for e in ESTADOS_COMERCIALES},
        },
        'resultado': {
            'productos': int(len(prod)),
            'meses_convertidos_a_cero': int(len(hist.conversiones)),
            'productos_con_meses_convertidos': int(hist.conversiones['producto'].nunique()),
            'meses_convertidos_por_motivo': {k: int(v) for k, v in hist.conversiones['motivo'].value_counts().items()},
            'meses_que_siguen_desconocidos': int(prod.meses_desconocidos.sum()),
            'meses_pendientes_de_interpretar': int(prod.meses_pendientes.sum()),
            'productos_sin_ventas_recientes': int(prod.sin_ventas_recientes.sum()),
            'productos_por_situacion': {k: int(v) for k, v in prod.situacion.value_counts().items()},
        },
        'meses_sin_ventas_recientes': hist.meses_sin_ventas_recientes,
        'aviso': AVISO_CEROS,
    }


def guardar_registro(hist: HistoricoV2, carpeta: Path, **datos) -> dict[str, Path]:
    """Escribe la confirmación (JSON) y los CSV de meses convertidos, meses y productos."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    rutas = {'confirmaciones': carpeta / 'confirmaciones_v2.json',
             'ceros_confirmados': carpeta / 'ceros_confirmados_v2.csv',
             'meses': carpeta / 'meses_v2.csv', 'productos': carpeta / 'productos_v2.csv'}
    rutas['confirmaciones'].write_text(
        json.dumps(registro_auditable(hist, **datos), ensure_ascii=False, indent=2), encoding='utf-8')
    for clave, tabla in (('ceros_confirmados', hist.conversiones), ('meses', hist.detalle),
                         ('productos', hist.productos)):
        escribir_csv(tabla, rutas[clave])
    return rutas


class ConfirmacionNoReutilizable(ValueError):
    """Unas confirmaciones guardadas que no se pueden reutilizar con los archivos actuales."""


def comprobar_huellas(actual: str | None, guardada: str | None, que: str = 'Las confirmaciones guardadas') -> None:
    """Unas confirmaciones guardadas solo valen para los mismos archivos y las mismas columnas con los que se
    hicieron. Se rechazan si falta la huella de los archivos actuales, si falta la de la confirmación o si no
    coinciden. No hay excepciones: ante la duda, se vuelve a preguntar."""
    if not actual:
        raise ConfirmacionNoReutilizable(f'{que} no se pueden reutilizar: no se ha podido identificar los archivos actuales.')
    if not guardada:
        raise ConfirmacionNoReutilizable(f'{que} no se pueden reutilizar: no indican a qué archivos corresponden.')
    if actual != guardada:
        raise ConfirmacionNoReutilizable(f'{que} corresponden a otros archivos o a otras columnas.')


def cargar_registro(ruta: Path, *, huella_conjunto: str | None = None
                    ) -> tuple[ConfirmacionCompletitud, dict[str, EstadoComercial]]:
    """Recupera una confirmación guardada.

    `huella_conjunto`: huella de los archivos actuales. Para REUTILIZAR la confirmación hay que indicarla siempre
    (aunque esté vacía): se rechaza si falta, si la confirmación no trae la suya o si no coinciden. Con None no
    se comprueba nada: solo sirve para leer el archivo, nunca para reutilizarlo con otros datos.
    """
    d = json.loads(Path(ruta).read_text(encoding='utf-8'))
    guardada = (d.get('conjunto_de_datos') or {}).get('huella_conjunto')
    if huella_conjunto is not None:
        comprobar_huellas(huella_conjunto, guardada, 'Las confirmaciones de completitud guardadas')
    estados = {p: EstadoComercial(e['estado'], e['origen'],
                                  date.fromisoformat(e['retirado_desde']) if e.get('retirado_desde') else None)
               for p, e in d.get('estado_comercial', {}).get('productos', {}).items()}
    return ConfirmacionCompletitud.desde_dict(d['confirmacion']), estados


def mostrar_completitud_corta(hist: HistoricoV2) -> None:
    c, prod = hist.confirmacion, hist.productos
    print('\nHISTÓRICO QUE SE VA A UTILIZAR')
    if c.completo == 'si':
        print(f'  Histórico confirmado como completo de {c.periodo_inicio:%m/%Y} a {c.periodo_fin:%m/%Y}.')
    else:
        print('  No se ha confirmado que el histórico esté completo: los meses sin ventas siguen como desconocidos.')
    if hist.ultimo_mes < hist.ultimo_mes_archivo:
        print(f'  El último mes del archivo ({_mes_texto(hist.ultimo_mes_archivo)}) no se usa porque no se ha '
              'confirmado que esté completo.')
    print(f'  Último mes utilizado: {_mes_texto(hist.ultimo_mes)}. Mes que se prevé: {_mes_texto(hist.ultimo_mes + 1)}.')
    print(f'  Meses convertidos a cero ventas: {len(hist.conversiones)} '
          f'(en {hist.conversiones["producto"].nunique()} productos).')
    print(f'  Meses que siguen desconocidos: {int(prod.meses_desconocidos.sum())}; '
          f'pendientes de interpretar: {int(prod.meses_pendientes.sum())}.')
    for e, texto in (('activo_confirmado', 'confirmados como activos'),
                     ('retirado_confirmado', 'confirmados como retirados'),
                     ('desconocido', 'con estado comercial sin confirmar')):
        print(f'  Productos {texto}: {int((prod.estado_comercial == e).sum())}')
    print(f'  Productos sin ventas en los últimos {hist.meses_sin_ventas_recientes} meses: '
          f'{int(prod.sin_ventas_recientes.sum())} (no se consideran retirados por ello).')
    con_fecha = int(prod.retirado_desde.notna().sum())
    if (prod.estado_comercial == 'retirado_confirmado').any():
        print(f'  De los retirados, con fecha de retirada: {con_fecha}. Sin fecha, los meses posteriores a su '
              'última venta siguen como desconocidos.')
    if len(hist.conversiones):
        print('  ' + AVISO_CEROS)
