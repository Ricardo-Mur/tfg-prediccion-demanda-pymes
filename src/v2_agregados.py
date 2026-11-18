"""Versión 2, etapa 5: previsión por familia y del total de los productos del archivo.

Cada nivel tiene su propia serie mensual, su propia comprobación retrospectiva, sus métricas y su estado de
fiabilidad. Se calculan con el mismo motor que los productos (`motor_v2`), sin modificarlo: la serie de una
familia o del total se trata como una serie más, y su estado sale de sus propios meses comprobados, no de los
estados de sus productos.

Niveles:
  total         todos los productos del archivo. Existe siempre.
  familia       un grupo por cada familia. Solo si el usuario ha confirmado una columna como agrupación o ha
                importado una correspondencia producto-familia.
  sin_familia   los productos que no tienen familia asignada. NO es una familia: se calcula aparte para que las
                familias y este resto sumen el total, y se presenta siempre separado y con ese nombre.

Cómo se construye la serie de un grupo:
- El valor de un mes es la suma de las unidades registradas de sus productos ese mes.
- Ese valor solo se da por conocido si (a) el usuario ha confirmado que los archivos contienen todas las ventas
  de ese mes y (b) ningún producto del grupo tiene ese mes operaciones pendientes de interpretar.
- Si falla alguna de las dos, el total del mes es DESCONOCIDO. No se rellena con cero y la suma de lo que hay no
  se presenta como total: se guarda aparte, marcada como parcial, y el motor no la usa.
- Un mes desconocido corta el tramo que usa el motor, igual que en un producto: no se salta ni se rellena.

Qué aporta un producto sin ventas registradas en un mes:
- Si el mes está confirmado como completo, aporta cero unidades a la suma: el archivo trae todas las ventas y de
  ese producto no hay ninguna. Esto no dice nada de su serie de producto, donde ese mes puede seguir siendo
  desconocido porque no se sabe si estaba a la venta. Son preguntas distintas: «¿cuánto se vendió en total?» se
  puede responder; «¿habría vendido algo ese producto?», no.
- Si el mes no está confirmado, no se sabe si faltan ventas: el total es desconocido.

Las previsiones de cada nivel son independientes: la suma de las previsiones de las familias no tiene por qué
coincidir con la previsión del total, y no se ajustan entre sí.

Tres cosas que no se mezclan:
  total exacto          lo vendido por el grupo en un mes cuyo total se conoce (`agregados_meses_v2.csv`, columna `total`).
  suma parcial          lo que suman los registros de un mes cuyo total NO se conoce (columna `suma_parcial`). No es
                        un total, no se usa en ningún cálculo y nunca aparece en la columna `total`.
  previsión agregada    lo que se prevé para el mes siguiente (`agregados_v2.csv`), con su estado. Se calcula solo con
                        totales exactos.

Operaciones pendientes: la regla es estricta. Si un producto tiene en un mes una operación cuyo significado puede
cambiar el total, ese mes no tiene total hasta que se aclare; no hay umbrales para aceptar una suma aproximada.
`pendientes_del_total_v2.csv` dice qué líneas son, de qué producto y mes, a qué grupos dejan sin total y qué hay
que aclarar de cada una.

Archivo de familias: sus columnas se reconocen por cabeceras conocidas. Si no queda claro, sin ninguna duda, cuál
es el producto y cuál la familia, hace falta que el usuario lo confirme expresamente; nunca se asignan por su
posición, y sin confirmación el archivo no se usa.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .v2_exportacion import escribir_csv
from . import motor_v2 as M
from . import v2_completitud as K
from .v2_columnas import PERSONALES, _cabecera_normalizada

NIVELES = ('total', 'familia', 'sin_familia')
SIN_FAMILIA = '(sin familia)'
NOMBRE_TOTAL = 'Total'
CLAVE_TOTAL, CLAVE_SIN_FAMILIA, PREFIJO_FAMILIA = 'total', 'sin_familia', 'familia:'
TIPOS_DE_AGRUPACION = ('', 'familia', 'marca_o_proveedor', 'otra', 'archivo')
TEXTO_TIPO = {'familia': 'familia', 'archivo': 'familia', 'marca_o_proveedor': 'marca o proveedor', 'otra': 'grupo', '': 'familia'}
TEXTO_SIN_GRUPO = {'familia': ('Productos sin familia asignada', 'no es una familia'),
                   'marca o proveedor': ('Productos sin marca o proveedor', 'no es una marca ni un proveedor'),
                   'grupo': ('Productos sin grupo asignado', 'no es un grupo')}
MOTIVO_SIN_CONFIRMAR = 'historico_sin_confirmar_para_ese_mes'
MOTIVO_PENDIENTES = 'operaciones_pendientes_de_interpretar'
MOTIVO_SIN_CORRESPONDENCIAS = 'no da familia a ningún producto de las ventas'
# `total`: el total exacto del mes, solo cuando se conoce. `suma_parcial`: lo que suman los registros cuando el total
# NO se conoce; nunca se usa como total ni entra en ninguna previsión. Las dos columnas nunca tienen valor a la vez.
COLUMNAS_MESES = ('nivel', 'grupo', 'mes', 'total', 'suma_parcial', 'total_conocido', 'motivo', 'productos_con_ventas',
                  'productos_con_operaciones_pendientes')
COLUMNAS_PENDIENTES = ('mes', 'producto', 'grupos_sin_total', 'motivo', 'lineas', 'unidades', 'filas', 'que_revisar')
CLASES_PENDIENTES = ('negativo_pendiente', 'serie_sin_confirmar', 'positivo_sin_confirmar', 'no_interpretable')
QUE_REVISAR = {
    'negativo_pendiente': 'Línea con cantidad negativa sin aclarar: indica si es una anulación, una devolución o un ajuste.',
    'serie_sin_confirmar': 'La serie «{serie}» está sin confirmar: indica qué contiene.',
    'positivo_sin_confirmar': 'Línea positiva de una serie mixta o de devoluciones: indica si es una venta o un ajuste.',
    'no_interpretable': 'La cantidad de la línea no se puede leer: corrígela en el archivo.'}
MAX_FILAS_EN_EL_AVISO = 10
COLUMNAS_PROPIAS = ('nivel', 'grupo', 'productos', 'meses_con_total_conocido', 'meses_sin_total_conocido',
                    'ultimo_mes_sin_total', 'aviso_historial', 'unidades_ultimos_12_meses')
MAX_PRODUCTOS_EN_EL_REGISTRO = 50
# Cabeceras conocidas del archivo producto-familia. Una cabecera exacta manda sobre un fragmento.
CABECERAS_PRODUCTO = frozenset({'producto', 'productos', 'articulo', 'articulos', 'codigo', 'codart', 'codarticulo', 'codproducto',
                                'codigoarticulo', 'codigoproducto', 'codigodearticulo', 'codigodeproducto', 'referencia', 'sku',
                                'item', 'itemid', 'product', 'productid', 'productcode', 'productoid', 'articuloid', 'idproducto',
                                'idarticulo', 'idart'})
CABECERAS_FAMILIA = frozenset({'familia', 'familias', 'categoria', 'categorias', 'grupo', 'seccion', 'departamento', 'family',
                               'category', 'subfamilia', 'gama', 'codfamilia', 'codigofamilia', 'idfamilia', 'familiaid',
                               'familiadeproducto', 'familiaproducto', 'categoriadeproducto', 'categoriaproducto'})
FRAGMENTOS_PRODUCTO = ('producto', 'product', 'articulo', 'article', 'sku', 'referencia', 'codart')
FRAGMENTOS_FAMILIA = ('famil', 'categ', 'grupo', 'group', 'seccion', 'depart', 'gama')
IDENTIFICACIONES = ('cabeceras reconocidas de forma inequívoca', 'indicadas expresamente por el usuario',
                    'propuesta del programa confirmada por el usuario')
AVISO_NIVELES = ('Cada nivel se comprueba por separado. La suma de las previsiones de las familias no tiene por qué coincidir '
                 'con la previsión del total.')


def _mes(i: int) -> str:
    return f'{i % 12 + 1:02d}/{i // 12}'


# ------------------------------------------------------------------ series agregadas

def registrado_por_producto(hist: K.HistoricoV2) -> dict[str, dict[int, float | None]]:
    """Lo que hay registrado de cada producto en cada mes utilizable: unidades, o None si el mes tiene operaciones
    pendientes de interpretar. Los meses sin ningún registro no aparecen."""
    out: dict[str, dict[int, float | None]] = {p: {} for p in hist.series}
    for r in hist.detalle.itertuples(index=False):
        m = K.indice(r.mes)
        if m > hist.ultimo_mes:
            continue
        if r.estado in K.ESTADOS_CON_VALOR:
            out.setdefault(r.producto, {})[m] = float(r.unidades)
        elif r.estado.startswith('pendiente'):
            out.setdefault(r.producto, {})[m] = None
    return out


def agregar_series(registrado: dict[str, dict[int, float | None]], productos, ultimo: int,
                   cubre: Callable[[int], bool] = lambda m: True) -> tuple[dict[int, float | None], list[tuple]]:
    """Serie mensual de un grupo de productos y el detalle de cada mes.

    `registrado`: producto -> mes -> unidades (None = hay operaciones sin interpretar ese mes). Un mes que no
    aparece es un mes sin ventas registradas de ese producto.
    `cubre(mes)`: ¿está confirmado que los archivos contienen todas las ventas de ese mes?
    Devuelve la serie (None = total desconocido) y, por mes, (mes, suma registrada, total conocido, motivo,
    productos con ventas, productos con operaciones pendientes). La serie empieza en el primer mes con algún
    registro del grupo.
    """
    suma: dict[int, float] = {}
    con_ventas: dict[int, int] = {}
    pendientes: dict[int, int] = {}
    for p in productos:
        for m, v in registrado.get(p, {}).items():
            if m > ultimo:
                continue
            suma.setdefault(m, 0.0)
            if v is None:
                pendientes[m] = pendientes.get(m, 0) + 1
            else:
                suma[m] += float(v)
                con_ventas[m] = con_ventas.get(m, 0) + (v > 0)
    if not suma:
        return {}, []
    serie, filas = {}, []
    for m in range(min(suma), ultimo + 1):
        motivo = '' if cubre(m) else MOTIVO_SIN_CONFIRMAR
        if not motivo and pendientes.get(m):
            motivo = MOTIVO_PENDIENTES
        serie[m] = None if motivo else suma.get(m, 0.0)
        filas.append((m, suma.get(m, 0.0), not motivo, motivo, con_ventas.get(m, 0), pendientes.get(m, 0)))
    return serie, filas


@dataclass
class Grupo:
    clave: str                         # identificador interno, único: 'total', 'sin_familia' o 'familia:<nombre>'
    nivel: str                         # 'total', 'familia' o 'sin_familia'
    nombre: str                        # lo que ve el usuario
    productos: tuple[str, ...]
    serie: dict[int, float | None]
    meses: list[tuple]                 # ver `agregar_series`
    pendientes: dict = field(default_factory=dict)     # mes -> productos con operaciones pendientes ese mes


def construir_grupos(hist: K.HistoricoV2, familias: dict[str, str] | None = None) -> list[Grupo]:
    """El total y, si hay familias, un grupo por familia y otro con los productos sin familia.

    `familias`: producto -> familia. Un producto que no está, o cuya familia está vacía o es «(sin familia)», va
    al grupo de productos sin familia, que no es una familia.
    """
    registrado = registrado_por_producto(hist)
    productos = sorted(hist.series)
    cubre = hist.confirmacion.cubre

    def grupo(clave, nivel, nombre, del_grupo):
        serie, meses = agregar_series(registrado, del_grupo, hist.ultimo_mes, cubre)
        pendientes: dict[int, list[str]] = {}
        for q in del_grupo:
            for m, v in registrado.get(q, {}).items():
                if v is None and m <= hist.ultimo_mes:
                    pendientes.setdefault(m, []).append(q)
        return Grupo(clave, nivel, nombre, tuple(del_grupo), serie, meses, {m: tuple(sorted(v)) for m, v in pendientes.items()})

    grupos = [grupo(CLAVE_TOTAL, 'total', NOMBRE_TOTAL, productos)]
    if familias:
        por_familia: dict[str, list[str]] = {}
        sin = []
        for p in productos:
            f = familias.get(p)
            f = '' if f is None or pd.isna(f) else str(f).strip()
            if f and f != SIN_FAMILIA:
                por_familia.setdefault(f, []).append(p)
            else:
                sin.append(p)
        grupos += [grupo(PREFIJO_FAMILIA + f, 'familia', f, por_familia[f]) for f in sorted(por_familia)]
        if sin:
            grupos.append(grupo(CLAVE_SIN_FAMILIA, 'sin_familia', SIN_FAMILIA, sin))
    return grupos


# ------------------------------------------------------------------ previsión de cada nivel

@dataclass
class InformeAgregados:
    resumen: pd.DataFrame              # una fila por grupo: COLUMNAS_PROPIAS y las columnas del motor
    meses: pd.DataFrame                # COLUMNAS_MESES: la serie de cada grupo, con lo que no se conoce y por qué
    historial: pd.DataFrame            # comprobación retrospectiva de cada grupo
    candidatos: pd.DataFrame
    tipo_de_agrupacion: str = ''
    origen_familias: str = ''
    coherencia: dict = field(default_factory=dict)
    registro: dict = field(default_factory=dict)
    # Operaciones que impiden conocer el total de algún mes (COLUMNAS_PENDIENTES), para que se puedan aclarar.
    pendientes: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=list(COLUMNAS_PENDIENTES)))


def _situacion(g: Grupo, ultimo: int) -> tuple[str, str] | None:
    """Por qué no se puede prever un grupo, si se sabe antes de calcular."""
    que = 'el total' if g.nivel == 'total' else 'el total de este grupo'
    if not g.serie:
        return 'sin_ventas_registradas', 'No hay ventas registradas que se puedan sumar.'
    if all(v is None for v in g.serie.values()) and all(f[3] == MOTIVO_SIN_CONFIRMAR for f in g.meses):
        return ('historico_sin_confirmar',
                'No se ha confirmado que los archivos contengan todas las ventas. La suma de lo registrado podría ser solo '
                f'una parte, así que no se da como {que} ni se prevé.')
    fila = g.meses[-1]
    if fila[2]:
        return None
    if fila[3] == MOTIVO_PENDIENTES:
        return ('ultimo_mes_con_operaciones_pendientes',
                f'No se conoce {que} de {_mes(ultimo)}: {fila[5]} producto(s) tienen ese mes operaciones pendientes de '
                'interpretar (por ejemplo, devoluciones). La suma de lo demás sería parcial y no se usa.')
    return ('ultimo_mes_sin_confirmar',
            f'No se ha confirmado que los archivos contengan todas las ventas de {_mes(ultimo)}: no se conoce {que} de ese mes.')


def _aviso_historial(g: Grupo) -> str:
    """Si algún mes no tiene total conocido, dice cuál es el más reciente, por qué y desde cuándo se usa el historial."""
    desconocidos = [f for f in g.meses if not f[2]]
    if not desconocidos or all(not f[2] for f in g.meses):
        return ''
    m, _, _, motivo, _, pendientes = desconocidos[-1]
    causa = (f'{pendientes} producto(s) tienen ese mes operaciones pendientes de interpretar' if motivo == MOTIVO_PENDIENTES else
             'no se ha confirmado que los archivos contengan todas las ventas de ese mes')
    otros = f' (y de otros {len(desconocidos) - 1} meses anteriores)' if len(desconocidos) > 1 else ''
    if m == g.meses[-1][0]:      # el último mes: ya lo dice el mensaje del grupo
        return f'Tampoco se conoce el total de otros {len(desconocidos) - 1} meses anteriores.' if len(desconocidos) > 1 else ''
    return (f'No se conoce el total de {_mes(m)}{otros}: {causa}. El historial que se usa empieza en {_mes(m + 1)}: el hueco no '
            'se rellena con cero ni con la suma de lo que hay.')


def operaciones_que_impiden_el_total(grupos: list[Grupo], lineas: pd.DataFrame | None = None) -> pd.DataFrame:
    """Qué operaciones dejan sin total algún mes, para poder revisarlas.

    `lineas`: una fila por línea del archivo, con `fila` (desde 0), `producto`, `mes`, `cantidad`, `clase` (la de
    `v2_operaciones`) y `serie`. Sin ella solo se puede decir qué productos y meses son.
    Devuelve una fila por producto, mes y motivo: los grupos que se quedan sin total ese mes, cuántas líneas son,
    cuántas unidades suman, sus números de fila (desde 1, como en `operaciones_v2.csv`) y qué hay que aclarar.
    """
    total = next((g for g in grupos if g.nivel == 'total'), None)
    if total is None or not total.pendientes:
        return pd.DataFrame(columns=list(COLUMNAS_PENDIENTES))
    grupo_de = {p: g.nombre for g in grupos if g.nivel != 'total' for p in g.productos}
    por_clave = None
    if lineas is not None and len(lineas):
        pend = lineas[lineas.clase.isin(CLASES_PENDIENTES)]
        por_clave = {k: v for k, v in pend.groupby(['producto', 'mes'])}
    filas = []
    for m in sorted(total.pendientes):
        for p in total.pendientes[m]:
            afectados = ', '.join([grupo_de[p]] if p in grupo_de else []) + (', ' if p in grupo_de else '') + NOMBRE_TOTAL
            del_mes = None if por_clave is None else por_clave.get((p, m))
            if del_mes is None or del_mes.empty:
                filas.append((K.fecha(m), p, afectados, 'mes_pendiente_en_la_preparacion_de_los_datos', np.nan, np.nan, '',
                              'Revisa las operaciones de ese producto en ese mes.'))
                continue
            for (clase, serie), g in del_mes.groupby(['clase', del_mes.serie.fillna('').astype(str)], sort=True):
                numeros = sorted(int(f) + 1 for f in g.fila)
                texto = ', '.join(map(str, numeros[:MAX_FILAS_EN_EL_AVISO])) + (' ...' if len(numeros) > MAX_FILAS_EN_EL_AVISO else '')
                filas.append((K.fecha(m), p, afectados, clase, len(g), float(g.cantidad.sum()), texto,
                              QUE_REVISAR[clase].format(serie=serie)))
    return pd.DataFrame(filas, columns=list(COLUMNAS_PENDIENTES))


def _revisiones_del_grupo(g: Grupo, revisiones: pd.DataFrame | None, ultimo: int) -> list[tuple[int, int, float]]:
    """Anulaciones registradas en un mes posterior al de su venta, de los productos del grupo. En cada corte, el
    total de un mes es el que se conocía entonces."""
    if revisiones is None or revisiones.empty:
        return []
    del_grupo = revisiones[revisiones.producto.isin(set(g.productos))]
    out = []
    for (venta, registro), unidades in del_grupo.groupby(['mes_venta', 'mes_registro']).unidades.sum().items():
        venta, registro = int(venta), int(registro)
        if venta < registro and venta <= ultimo and g.serie.get(venta) is not None and unidades > 0:
            out.append((venta, registro, float(unidades)))
    return out


def prever_grupos(grupos: list[Grupo], cfg: M.ConfiguracionV2 | None = None, ultimo_mes: int | None = None, *,
                  revisiones: pd.DataFrame | None = None, tipo_de_agrupacion: str = '', origen_familias: str = '',
                  lineas: pd.DataFrame | None = None) -> InformeAgregados:
    """Previsión, comprobación retrospectiva y estado de cada grupo, con el motor sin modificar.

    El total se calcula por su cuenta. Las familias y el grupo sin familia se calculan juntos, también por su
    cuenta: el XGBoost global de cada nivel se entrena solo con las series de ese nivel.
    `revisiones`: tabla de `ResultadoOperaciones.revisiones` (producto, mes_venta, mes_registro, unidades).
    `lineas`: las líneas del archivo con su clase, para decir qué operaciones impiden conocer algún total
    (ver `operaciones_que_impiden_el_total`). No interviene en ninguna previsión.
    """
    cfg = cfg or M.ConfiguracionV2()
    if tipo_de_agrupacion not in TIPOS_DE_AGRUPACION:
        raise ValueError('Tipo de agrupación no reconocido.')
    if len({g.clave for g in grupos}) != len(grupos):
        raise ValueError('Hay grupos repetidos.')
    con_serie = [g for g in grupos if g.serie]
    ultimo = ultimo_mes if ultimo_mes is not None else max((max(g.serie) for g in con_serie), default=None)
    resumenes, historiales, candidatos = [], [], []
    for nivel in (('total',), ('familia', 'sin_familia')):
        del_nivel = [g for g in con_serie if g.nivel in nivel]
        if not del_nivel:
            continue
        situaciones = {g.clave: s for g in del_nivel if (s := _situacion(g, ultimo)) is not None}
        rev = {g.clave: r for g in del_nivel if g.clave not in situaciones and (r := _revisiones_del_grupo(g, revisiones, ultimo))}
        informe = M.ejecutar_v2({g.clave: g.serie for g in del_nivel}, cfg, ultimo_mes=ultimo, situaciones=situaciones, revisiones=rev)
        resumenes.append(informe.resumen); historiales.append(informe.historial); candidatos.append(informe.candidatos)
    por_clave = {g.clave: g for g in grupos}
    filas = []
    motor = pd.concat(resumenes, ignore_index=True) if resumenes else pd.DataFrame(columns=list(M.COLUMNAS_RESUMEN))
    calculados = set(motor.producto)
    for g in grupos:
        conocidos = [f for f in g.meses if f[2]]
        desconocidos = [f for f in g.meses if not f[2]]
        ultimos = [g.serie.get(m) for m in range((ultimo or 0) - 11, (ultimo or 0) + 1)] if g.serie else []
        propias = dict(clave=g.clave, nivel=g.nivel, grupo=g.nombre, productos=len(g.productos),
                       meses_con_total_conocido=len(conocidos), meses_sin_total_conocido=len(desconocidos),
                       ultimo_mes_sin_total=K.fecha(desconocidos[-1][0]) if desconocidos else None,
                       aviso_historial=_aviso_historial(g),
                       unidades_ultimos_12_meses=float(sum(ultimos)) if ultimos and all(v is not None for v in ultimos) else np.nan)
        if g.clave not in calculados:           # un grupo sin ninguna venta registrada
            vacio = {c: np.nan for c in M.COLUMNAS_RESUMEN}
            vacio.update(producto=g.clave, estado='datos_insuficientes', motivo='sin_ventas_registradas',
                         mensaje='No hay ventas registradas que se puedan sumar.', meses_comprobados=0)
            motor = pd.concat([motor, pd.DataFrame([vacio])], ignore_index=True)
        filas.append(propias)
    resumen = pd.DataFrame(filas).merge(motor.rename(columns={'producto': 'clave'}), on='clave', how='left')
    cambio = {'total': 'para el total', 'familia': 'para este grupo', 'sin_familia': 'para este grupo'}
    resumen['mensaje'] = [str(t).replace('para este producto', cambio[n]) for t, n in zip(resumen.mensaje, resumen.nivel)]
    orden = {n: k for k, n in enumerate(NIVELES)}
    resumen = resumen.assign(_o=resumen.nivel.map(orden), _u=-resumen.unidades_ultimos_12_meses.fillna(-1)).sort_values(
        ['_o', '_u', 'grupo']).drop(columns=['_o', '_u', 'clave']).reset_index(drop=True)
    nombre = {g.clave: (g.nivel, g.nombre) for g in grupos}
    # El total exacto y la suma parcial van en columnas distintas: una suma parcial nunca aparece como total.
    meses = pd.DataFrame([(g.nivel, g.nombre, K.fecha(m), s if c else np.nan, np.nan if c else s, c, motivo, v, pend)
                          for g in grupos for m, s, c, motivo, v, pend in g.meses], columns=list(COLUMNAS_MESES))

    def con_nombre(tabla: pd.DataFrame) -> pd.DataFrame:
        if tabla.empty:
            return tabla.assign(nivel=[], grupo=[]).drop(columns='producto', errors='ignore')
        t = tabla.assign(nivel=tabla.producto.map(lambda c: nombre[c][0]), grupo=tabla.producto.map(lambda c: nombre[c][1]))
        return t[['nivel', 'grupo'] + [c for c in tabla.columns if c != 'producto']]

    historial = con_nombre(pd.concat(historiales, ignore_index=True) if historiales else pd.DataFrame(columns=list(M.COLUMNAS_HISTORIAL)))
    cand = con_nombre(pd.concat(candidatos, ignore_index=True) if candidatos else pd.DataFrame(columns=['producto', 'mes', 'metodo', 'prediccion', 'real']))
    # Para poder aclararlo: qué productos tienen operaciones pendientes en los meses en que no se conoce el total.
    total = next((g for g in grupos if g.nivel == 'total'), None)
    registro = {'meses_sin_total_conocido': [
        {'mes': str(K.fecha(m)), 'motivo': motivo, 'productos_con_operaciones_pendientes': len(total.pendientes.get(m, ())),
         'productos': list(total.pendientes.get(m, ())[:MAX_PRODUCTOS_EN_EL_REGISTRO])}
        for m, _, conocido, motivo, _, _ in total.meses if not conocido]} if total is not None else {}
    return InformeAgregados(resumen=resumen, meses=meses, historial=historial, candidatos=cand, tipo_de_agrupacion=tipo_de_agrupacion,
                            origen_familias=origen_familias, coherencia=coherencia_entre_niveles(resumen), registro=registro,
                            pendientes=operaciones_que_impiden_el_total(grupos, lineas))


def coherencia_entre_niveles(resumen: pd.DataFrame) -> dict:
    """Compara la previsión del total con la suma de las previsiones de las familias y del grupo sin familia.

    Solo es una suma completa si TODOS esos grupos tienen cantidad recomendada. Si falta alguno, se dice cuántos
    faltan y la suma no se presenta como un total.
    """
    partes = resumen[resumen.nivel.isin(['familia', 'sin_familia'])]
    total = resumen[resumen.nivel == 'total']
    prevision_total = float(total.prediccion.iloc[0]) if len(total) and pd.notna(total.prediccion.iloc[0]) else None
    if partes.empty:
        return {'hay_familias': False, 'prevision_del_total': prevision_total}
    con = partes[partes.prediccion.notna()]
    completa = len(con) == len(partes)
    suma = float(con.prediccion.sum()) if len(con) else None
    return {'hay_familias': True, 'grupos': int(len(partes)), 'grupos_con_prevision': int(len(con)), 'suma_completa': bool(completa),
            'suma_de_los_grupos_con_prevision': suma, 'prevision_del_total': prevision_total,
            'diferencia_relativa': (suma / prevision_total - 1) if completa and suma is not None and prevision_total else None}


# ------------------------------------------------------------------ familias: columna confirmada o archivo importado

def leer_tabla_de_familias(ruta: Path) -> pd.DataFrame:
    """Lee el archivo producto-familia (CSV o Excel) con todas sus columnas como texto. No decide qué es cada una."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise ValueError(f'No existe el archivo de familias: {ruta}')
    if ruta.suffix.lower() in ('.xlsx', '.xlsm', '.xls'):
        cabeceras = [str(c).strip() for c in pd.read_excel(ruta, header=None, nrows=1, dtype=str).iloc[0]]
        tabla = pd.read_excel(ruta, dtype=str)
    else:
        # El separador se decide con la primera línea, entre los habituales: no se adivina con letras del texto.
        try:
            texto, codificacion = ruta.read_text(encoding='utf-8-sig'), 'utf-8-sig'
        except UnicodeDecodeError:
            texto, codificacion = ruta.read_text(encoding='latin-1'), 'latin-1'
        primera = texto.split('\n', 1)[0]
        separador = max((';', ',', '\t', '|'), key=primera.count)
        if primera.count(separador) == 0:
            raise ValueError('El archivo de familias debe tener al menos dos columnas: el producto y su familia.')
        cabeceras = [c.strip().strip('"').strip() for c in primera.rstrip('\r').split(separador)]
        tabla = pd.read_csv(ruta, sep=separador, dtype=str, encoding=codificacion)
    if tabla.shape[1] < 2:
        raise ValueError('El archivo de familias debe tener al menos dos columnas: el producto y su familia.')
    # Las cabeceras se miran tal como vienen: si hay dos iguales no se sabría a cuál se refiere el usuario.
    if len(set(cabeceras)) != len(cabeceras):
        raise ValueError('El archivo de familias tiene cabeceras repetidas. Si no tiene fila de cabecera, añádele una.')
    tabla.columns = [str(c).strip() for c in tabla.columns]
    return tabla


@dataclass
class ColumnasDeFamilias:
    """Qué columna del archivo de familias identifica el producto y cuál la familia.

    Solo se usan sin preguntar si la identificación es inequívoca. Si no lo es, `propuesta_*` es lo que el
    programa propondría, pero hace falta la confirmación expresa del usuario; sin ella, el archivo no se usa.
    """
    tabla: pd.DataFrame               # el archivo (si no tenía fila de cabecera, con las columnas renombradas)
    producto: str | None              # columna del producto, solo si es inequívoca
    familia: str | None               # columna de la familia, solo si es inequívoca
    inequivoca: bool
    motivo: str                       # por qué no es inequívoca
    propuesta_producto: str | None
    propuesta_familia: str | None
    columnas: pd.DataFrame            # columna, valores_distintos, codigos_de_las_ventas, parece
    sin_cabecera: bool = False


def _valores(columna: pd.Series) -> pd.Series:
    v = columna.astype('string').str.strip()
    return v[v.notna() & (v != '')]


def identificar_columnas(tabla: pd.DataFrame, productos_de_las_ventas=None) -> ColumnasDeFamilias:
    """Propone qué columna es el producto y cuál la familia, y dice si la identificación es inequívoca.

    Por la cabecera: una cabecera conocida exacta (producto, ProductoID, artículo, código... / familia, categoría,
    grupo...) o, si no hay ninguna exacta, una que contenga esas palabras y no parezca de clientes o de personas.
    Es inequívoca solo si hay exactamente una columna para el producto y una para la familia.
    Nunca se asignan columnas por su posición.
    `productos_de_las_ventas`: códigos de producto del archivo de ventas. Con ellos se comprueba, además, que la
    columna del producto contiene códigos de las ventas; si no los contiene y otra columna sí, no es inequívoca.
    También permiten notar que el archivo no tiene fila de cabecera (su primera fila ya es un producto).
    """
    ventas = None if productos_de_las_ventas is None else {str(x).strip() for x in productos_de_las_ventas}
    sin_cabecera = bool(ventas) and any(c in ventas for c in tabla.columns)
    if sin_cabecera:
        nombres = [f'Columna {k + 1}' for k in range(tabla.shape[1])]
        primera = pd.DataFrame([list(tabla.columns)], columns=nombres)
        tabla = pd.concat([primera, tabla.set_axis(nombres, axis=1)], ignore_index=True)
    claves = {c: _cabecera_normalizada(c) for c in tabla.columns}
    personal = {c: any(f in k for f in PERSONALES) for c, k in claves.items()}
    exacta_p = [c for c, k in claves.items() if k in CABECERAS_PRODUCTO and not sin_cabecera]
    exacta_f = [c for c, k in claves.items() if k in CABECERAS_FAMILIA and not sin_cabecera]
    con_f = [c for c, k in claves.items() if any(f in k for f in FRAGMENTOS_FAMILIA) and c not in exacta_p and not sin_cabecera]
    con_p = [c for c, k in claves.items() if any(f in k for f in FRAGMENTOS_PRODUCTO) and c not in con_f and c not in exacta_f
             and not sin_cabecera]
    seguras_p = exacta_p or [c for c in con_p if not personal[c]]
    seguras_f = exacta_f or [c for c in con_f if not personal[c]]
    coincidencias = {c: (int(_valores(tabla[c]).drop_duplicates().isin(ventas).sum()) if ventas else None) for c in tabla.columns}
    parece = {c: ('producto' if c in seguras_p or c in con_p else 'familia' if c in seguras_f or c in con_f else
                  'cliente o persona' if personal[c] else '') for c in tabla.columns}
    columnas = pd.DataFrame([(c, int(_valores(tabla[c]).nunique()), coincidencias[c], parece[c]) for c in tabla.columns],
                            columns=['columna', 'valores_distintos', 'codigos_de_las_ventas', 'parece'])
    producto = seguras_p[0] if len(seguras_p) == 1 else None
    familia = seguras_f[0] if len(seguras_f) == 1 else None
    motivos = []
    if sin_cabecera:
        motivos.append('el archivo no parece tener fila de cabecera (su primera fila ya contiene un código de producto)')
    for nombre, seguras, dudosas in (('el producto', seguras_p, con_p), ('la familia', seguras_f, con_f)):
        if len(seguras) > 1:
            motivos.append(f'varias columnas podrían ser {nombre} (' + ', '.join(f'«{c}»' for c in seguras) + ')')
        elif not seguras and not sin_cabecera:
            de_clientes = [c for c in dudosas if personal[c]]
            motivos.append(f'«{de_clientes[0]}» podría ser {nombre}, pero su nombre también puede referirse a clientes o a personas'
                           if len(de_clientes) == 1 else f'ninguna cabecera identifica con seguridad {nombre}')
    # Con los códigos de las ventas: la columna del producto tiene que contenerlos.
    mejor = None
    if ventas:
        con_codigos = sorted((c for c in tabla.columns if coincidencias[c]), key=lambda c: -coincidencias[c])
        if con_codigos and (len(con_codigos) == 1 or coincidencias[con_codigos[0]] > coincidencias[con_codigos[1]]):
            mejor = con_codigos[0]
        if producto is not None and not coincidencias[producto] and mejor is not None:
            motivos.append(f'«{producto}» no contiene ningún código de producto de las ventas y «{mejor}» sí')
    # Propuestas: lo inequívoco; si no, el único candidato (aunque sea dudoso); si no, lo que apuntan los códigos.
    propuesta_p = producto if producto is not None and not (ventas and not coincidencias[producto] and mejor) else None
    if propuesta_p is None:
        propuesta_p = mejor if mejor is not None else (con_p[0] if len(con_p) == 1 else None)
    candidatas_f = [c for c in (seguras_f or con_f) if c != propuesta_p]
    propuesta_f = familia if familia is not None and familia != propuesta_p else (candidatas_f[0] if len(candidatas_f) == 1 else None)
    if propuesta_f is None and propuesta_p is not None and tabla.shape[1] == 2:
        propuesta_f = next(c for c in tabla.columns if c != propuesta_p)
    inequivoca = not motivos and producto is not None and familia is not None and producto != familia
    return ColumnasDeFamilias(tabla=tabla, producto=producto if inequivoca else None, familia=familia if inequivoca else None,
                              inequivoca=inequivoca, motivo='; '.join(motivos), propuesta_producto=propuesta_p,
                              propuesta_familia=propuesta_f, columnas=columnas, sin_cabecera=sin_cabecera)


def correspondencia(tabla: pd.DataFrame, columna_producto: str, columna_familia: str, ruta: Path,
                    identificacion: str = IDENTIFICACIONES[1]) -> tuple[dict[str, str], dict]:
    """producto -> familia con las dos columnas ya identificadas. Solo se leen esas dos columnas.

    Un producto que aparece con dos familias distintas queda sin familia: no se elige una. Un producto con la
    familia vacía queda sin familia. Devuelve también un recuento, la huella del archivo y cómo se identificaron
    las columnas, para dejar constancia de dónde salen las familias.
    """
    if identificacion not in IDENTIFICACIONES:
        raise ValueError('Identificación de las columnas no reconocida.')
    for c in (columna_producto, columna_familia):
        if c not in tabla.columns:
            raise ValueError(f'El archivo de familias no tiene ninguna columna «{c}». Tiene: ' + ', '.join(f'«{x}»' for x in tabla.columns))
    if columna_producto == columna_familia:
        raise ValueError('La columna del producto y la de la familia no pueden ser la misma.')
    t = pd.DataFrame({'producto': tabla[columna_producto].astype('string').str.strip(), 'familia': tabla[columna_familia].astype('string').str.strip()})
    t = t[t.producto.notna() & (t.producto != '')]
    t['familia'] = t.familia.fillna('')
    distintas = t.drop_duplicates().groupby('producto').familia.nunique()
    conflictos = set(distintas[distintas > 1].index)
    mapa = {r.producto: r.familia for r in t.drop_duplicates('producto').itertuples(index=False)
            if r.producto not in conflictos and r.familia and r.familia != SIN_FAMILIA}
    ruta = Path(ruta)
    info = {'archivo': ruta.name, 'huella_sha256': hashlib.sha256(ruta.read_bytes()).hexdigest(), 'filas': int(len(t)),
            'columnas_del_archivo': [str(c) for c in tabla.columns],
            'columna_del_producto': str(columna_producto), 'columna_de_la_familia': str(columna_familia),
            'identificacion_de_las_columnas': identificacion,
            'columnas_reconocidas_por_la_cabecera': identificacion == IDENTIFICACIONES[0], 'productos': int(t.producto.nunique()),
            'productos_con_familia': len(mapa), 'familias': len(set(mapa.values())),
            'productos_con_dos_familias_distintas': len(conflictos),
            'productos_con_la_familia_vacia': int(t.producto.nunique() - len(mapa) - len(conflictos))}
    return mapa, info


def leer_correspondencia(ruta: Path, productos_de_las_ventas=None, *, columna_producto: str | None = None,
                         columna_familia: str | None = None) -> tuple[dict[str, str], dict]:
    """Lee el archivo producto-familia sin hacer preguntas.

    Con `columna_producto` y `columna_familia`, usa esas columnas: las ha indicado el usuario. Sin ellas, solo
    sigue si las cabeceras identifican las dos columnas de forma inequívoca. Si no, lanza un error que dice por
    qué y qué se propondría: nunca se toman columnas por su posición ni se usa una propuesta sin confirmar.
    """
    tabla = leer_tabla_de_familias(ruta)
    if (columna_producto is None) != (columna_familia is None):
        raise ValueError('Hay que indicar las dos columnas: la del producto y la de la familia.')
    if columna_producto is not None:
        return correspondencia(tabla, columna_producto, columna_familia, ruta, IDENTIFICACIONES[1])
    c = identificar_columnas(tabla, productos_de_las_ventas)
    if not c.inequivoca:
        raise ValueError('No se puede saber con seguridad qué columna del archivo de familias es el producto y cuál la familia: '
                         f'{c.motivo}. Hay que indicarlas expresamente.')
    return correspondencia(c.tabla, c.producto, c.familia, ruta, IDENTIFICACIONES[0])


def columnas_guardadas(ruta_registro: Path, huella_del_archivo: str) -> tuple[str, str] | None:
    """Columnas que el usuario indicó o confirmó en una ejecución anterior para ESTE mismo archivo de familias
    (misma huella), según `agregados_v2.json`. None si no hay o es otro archivo."""
    try:
        d = json.loads(Path(ruta_registro).read_text(encoding='utf-8'))['detalle']['archivo_de_familias']
    except (OSError, KeyError, ValueError, TypeError):
        return None
    # Sin las dos huellas, o si no coinciden, no se reutiliza nada.
    if not huella_del_archivo or not d.get('huella_sha256') or d.get('huella_sha256') != huella_del_archivo \
            or d.get('identificacion_de_las_columnas') not in IDENTIFICACIONES[1:]:
        return None
    return d['columna_del_producto'], d['columna_de_la_familia']


def asignar_familias(mapa: dict[str, str], productos) -> tuple[dict[str, str], dict]:
    """Familia de cada producto del archivo de ventas según la correspondencia. Devuelve también cuántos
    productos quedan con y sin familia y cuántos códigos de la correspondencia no están en las ventas."""
    productos = list(productos)
    familias = {p: mapa[p] for p in productos if p in mapa}
    return familias, {'productos_del_archivo_de_ventas': len(productos), 'con_familia': len(familias),
                      'sin_familia': len(productos) - len(familias),
                      'codigos_de_la_correspondencia_que_no_estan_en_las_ventas': len(set(mapa) - set(productos))}


# ------------------------------------------------------------------ presentación

def exportar_agregados(informe: InformeAgregados, carpeta: Path) -> dict[str, Path]:
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    rutas = {}
    # En el archivo de candidatos el mes va como fecha, igual que en los demás (dentro del programa es un número).
    candidatos = informe.candidatos
    if len(candidatos):
        candidatos = candidatos.assign(mes=[M.fecha(int(m)) for m in candidatos.mes])
    for nombre, tabla in (('agregados_v2', informe.resumen), ('agregados_meses_v2', informe.meses),
                          ('agregados_historial_v2', informe.historial), ('agregados_candidatos_v2', candidatos)):
        rutas[nombre] = carpeta / f'{nombre}.csv'
        escribir_csv(tabla, rutas[nombre])
    if len(informe.pendientes):
        rutas['pendientes_del_total_v2'] = carpeta / 'pendientes_del_total_v2.csv'
        escribir_csv(informe.pendientes, rutas['pendientes_del_total_v2'])
    rutas['agregados_registro'] = carpeta / 'agregados_v2.json'
    r = informe.resumen
    rutas['agregados_registro'].write_text(json.dumps({
        'version_registro': 1,
        'niveles': {n: int((r.nivel == n).sum()) for n in NIVELES},
        'tipo_de_agrupacion': informe.tipo_de_agrupacion, 'origen_de_las_familias': informe.origen_familias,
        'estados_por_nivel': {n: {e: int(((r.nivel == n) & (r.estado == e)).sum()) for e in M.ESTADOS} for n in NIVELES},
        'coherencia_entre_niveles': informe.coherencia, 'detalle': informe.registro,
        'operaciones_que_impiden_algun_total': int(len(informe.pendientes)),
        'nota': ('El grupo «(sin familia)» reúne los productos sin familia asignada: no es una familia. Un mes cuyo total no se '
                 'conoce (histórico sin confirmar u operaciones pendientes) no se rellena con cero ni se sustituye por la suma '
                 'parcial: en agregados_meses_v2.csv, `total` es el total exacto y `suma_parcial`, lo que suman los registros de '
                 'un mes sin total, que no se usa. ' + AVISO_NIVELES)}, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    return rutas


def _mostrar_pendientes(pendientes: pd.DataFrame, maximo: int = 8) -> None:
    """Qué operaciones impiden conocer el total, en pantalla, para que se puedan aclarar."""
    if pendientes.empty:
        return
    print(f'  OPERACIONES QUE IMPIDEN CONOCER EL TOTAL: {len(pendientes)}, de {pendientes.producto.nunique()} producto(s) en '
          f'{pendientes.mes.nunique()} mes(es). Mientras no se aclaren, esos meses no tienen total.')
    for f in pendientes.head(maximo).itertuples(index=False):
        cuantas = '' if pd.isna(f.lineas) else f'{int(f.lineas)} línea(s), {f.unidades:g} unidades' + (f', fila(s) {f.filas}' if f.filas else '') + '. '
        print(f'    {f.mes:%m/%Y}, producto {f.producto} (sin total: {f.grupos_sin_total}). {cuantas}{f.que_revisar}')
    if len(pendientes) > maximo:
        print(f'    ... y {len(pendientes) - maximo} más.')
    print('    Están todas en pendientes_del_total_v2.csv. Para aclararlas, vuelve a ejecutar el programa y responde qué son; una '
          'línea concreta se puede indicar por su número de fila.')


def _linea(f) -> str:
    return f'[{M.NOMBRE_ESTADO[f.estado]}] {f.mensaje}'


def mostrar_agregados_corto(informe: InformeAgregados, ejemplos: int = 5) -> None:
    r = informe.resumen
    total = r[r.nivel == 'total']
    familias, sin = r[r.nivel == 'familia'], r[r.nivel == 'sin_familia']
    tipo = TEXTO_TIPO[informe.tipo_de_agrupacion]
    print('\nPREVISIÓN DEL TOTAL' + (f' Y POR {tipo.upper()}' if len(familias) else ''))
    for f in total.itertuples(index=False):
        print(f'  Total de los {f.productos} productos del archivo. {_linea(f)}')
        if f.aviso_historial:
            print(f'    {f.aviso_historial}')
    _mostrar_pendientes(informe.pendientes)
    if not len(familias):
        print(f'  No hay ninguna {tipo} confirmada: se prevé por producto y el total. Para prever por familias, confirma una '
              'columna como familia o indica un archivo con la familia de cada producto (--familias).')
        return
    print(f'  Agrupación: {informe.origen_familias}. {len(familias)} grupo(s).')
    for e in M.ESTADOS:
        n = int((familias.estado == e).sum())
        if n:
            print(f'  {M.NOMBRE_ESTADO[e]}: {n}')
    for f in familias.head(ejemplos).itertuples(index=False):
        print(f'  {f.grupo} ({f.productos} productos). {_linea(f)}' + (f' {f.aviso_historial}' if f.aviso_historial else ''))
    if len(familias) > ejemplos:
        print(f'  ... y {len(familias) - ejemplos} más, en agregados_v2.csv.')
    for f in sin.itertuples(index=False):
        nombre, aclaracion = TEXTO_SIN_GRUPO[tipo]
        print(f'  {nombre} ({f.productos}; {aclaracion}). {_linea(f)}')
    c = informe.coherencia
    if c.get('suma_completa') and c.get('prevision_del_total'):
        print(f'  La suma de las previsiones de los grupos es {M._num(c["suma_de_los_grupos_con_prevision"])} y la previsión directa '
              f'del total, {M._num(c["prevision_del_total"])} ({100 * c["diferencia_relativa"]:+.1f} %).')
    elif c.get('grupos_con_prevision') is not None:
        print(f'  Solo {c["grupos_con_prevision"]} de los {c["grupos"]} grupos tienen una cantidad recomendada: su suma no es un total.')
    print('  ' + AVISO_NIVELES)
