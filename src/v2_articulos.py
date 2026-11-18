"""Versión 2, etapa 4 (segunda parte): estado comercial, agrupaciones y fecha de alta de los artículos.

Parte del inventario de columnas (`v2_columnas`). Aquí se analiza qué dicen las columnas que describen al
artículo y se aplican solo si el usuario confirma qué significan:

  estado del artículo   una columna que marca los artículos (por ejemplo, «activo»). Se muestra qué valores
                        tiene, cuántos productos hay con cada uno y cuándo vendieron por última vez. Lo que
                        signifique cada valor (sigue a la venta, retirado, otra cosa) lo dice el usuario.
  agrupaciones          columnas con un valor fijo por producto (familia, marca, proveedor...). Se propone
                        cuáles pueden servir para agrupar; ninguna se usa sin confirmar.
  fecha de alta         una fecha fija por producto. Solo se anota si el usuario confirma que es la de alta.

Reglas:
- El nombre de una columna no decide nada. «Activo» puede querer decir que el artículo sigue a la venta o
  cualquier otra cosa (por ejemplo, que se publica en la tienda web).
- El estado es el actual: se toma el valor más reciente de cada producto. Un producto que solo aparece en
  archivos sin esa columna queda sin dato, no «sin marcar».
- Un producto que la columna da por retirado y tiene ventas en el último mes utilizable no se da por
  retirado, salvo que el usuario lo confirme expresamente.
- Un retirado por la columna no tiene fecha de retirada: los meses posteriores a su última venta siguen como
  desconocidos (regla de `v2_completitud`). Las reglas de ceros y de completitud no cambian.
- Las agrupaciones no cambian ninguna previsión: se guardan para el diagnóstico y para la previsión por
  familias.
- Nada de esto usa columnas de clientes. De una columna que en unos archivos describe al producto y en otros
  al cliente solo se leen los archivos en los que describe al producto.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from .v2_exportacion import escribir_csv
from . import v2_completitud as K
from .v2_columnas import InventarioColumnas, _a_fecha, _texto

SIGNIFICADOS_ESTADO = ('activo', 'retirado', 'desconocido')
TIPOS_AGRUPACION = ('familia', 'marca_o_proveedor', 'otra')
# El usuario identifica expresamente cuál es el importe neto (el final, ya con descuentos) y cuál el bruto (antes
# de descuentos). Nunca se deduce de los valores. Como total de la línea o como precio de una unidad.
TIPOS_IMPORTE = ('importe_neto', 'importe_bruto', 'precio_neto', 'precio_bruto', 'coste', 'descuento')
TIPOS_NETO, TIPOS_BRUTO = ('importe_neto', 'precio_neto'), ('importe_bruto', 'precio_bruto')
TIPOS_IMPORTE_ANTERIORES = ('importe_de_venta', 'precio_unitario')    # no decían si era neto o bruto: ya no valen
TEXTO_IMPORTE = {'importe_neto': 'importe neto de la línea', 'importe_bruto': 'importe bruto de la línea',
                 'precio_neto': 'precio neto de una unidad', 'precio_bruto': 'precio de una unidad antes de descuentos',
                 'coste': 'coste', 'descuento': 'descuento'}
TIPOS_EXISTENCIAS = ('existencias_al_exportar', 'existencias_en_cada_venta')
SIN_MARCA = ''              # el producto aparece, sin valor, en archivos donde la columna sí tiene datos
SIN_GRUPO = '(sin dato)'
TEXTO_SIGNIFICADO = {'activo': 'sigue a la venta', 'retirado': 'retirado', 'desconocido': 'sin confirmar'}
TEXTO_AGRUPACION = {'familia': 'familia o categoría', 'marca_o_proveedor': 'marca o proveedor', 'otra': 'otra agrupación'}


def _valor_texto(v: str) -> str:
    return 'sin marcar' if v == SIN_MARCA else f'«{v}»'


# ------------------------------------------------------------------ confirmaciones

@dataclass(frozen=True)
class ConfirmacionColumnas:
    """Lo que el usuario ha confirmado sobre las columnas del artículo, los importes y las existencias.

    Sin confirmar no se usa nada: los valores por defecto dejan todo como estaba.
    """
    estado: tuple[tuple[str, str, str], ...] = ()            # (columna, valor, significado); valor '' = sin marcar
    retirados_con_ventas_recientes: bool = False             # ¿retirados también los que vendieron el último mes?
    excepciones_estado: tuple[tuple[str, str], ...] = ()     # (producto, estado comercial)
    agrupaciones: tuple[tuple[str, str], ...] = ()           # (columna, tipo de agrupación)
    agrupacion_principal: str = ''                           # columna que se usará para la previsión por familias
    fechas_de_alta: tuple[str, ...] = ()                     # columnas confirmadas como fecha de alta del artículo
    importes: tuple[tuple[str, str], ...] = ()               # (columna, tipo de importe)
    existencias: tuple[tuple[str, str], ...] = ()            # (columna, tipo de existencias)
    fecha_hora: str = ''
    huella_conjunto: str = ''
    # Columnas de importes confirmadas con una versión anterior, que no decía si eran netas o brutas: no se usan.
    importes_sin_identificar: tuple[str, ...] = ()

    def __post_init__(self):
        if any(len(x) != 3 or x[2] not in SIGNIFICADOS_ESTADO for x in self.estado):
            raise ValueError('Cada valor de una columna de estado debe ser activo, retirado o desconocido.')
        if len({(c, v) for c, v, _ in self.estado}) != len(self.estado):
            raise ValueError('Hay valores de estado repetidos.')
        if type(self.retirados_con_ventas_recientes) is not bool:
            raise ValueError('retirados_con_ventas_recientes debe ser verdadero o falso.')
        if any(len(x) != 2 or x[1] not in K.ESTADOS_COMERCIALES for x in self.excepciones_estado):
            raise ValueError('Excepción de estado no reconocida.')
        if any(len(x) != 2 or x[1] not in TIPOS_AGRUPACION for x in self.agrupaciones):
            raise ValueError('Tipo de agrupación no reconocido.')
        if self.agrupacion_principal and self.agrupacion_principal not in dict(self.agrupaciones):
            raise ValueError('La agrupación principal debe ser una de las confirmadas.')
        if any(len(x) != 2 or x[1] not in TIPOS_IMPORTE for x in self.importes):
            raise ValueError('Tipo de importe no reconocido.')
        if len({c for c, _ in self.importes}) != len(self.importes):
            raise ValueError('Una columna de importes solo puede tener un significado.')
        for tipos, nombre in ((TIPOS_NETO, 'neto'), (TIPOS_BRUTO, 'bruto')):
            if sum(1 for _, t in self.importes if t in tipos) > 1:
                raise ValueError(f'Solo una columna puede ser el importe {nombre}.')
        if any(len(x) != 2 or x[1] not in TIPOS_EXISTENCIAS for x in self.existencias):
            raise ValueError('Tipo de existencias no reconocido.')

    @property
    def vacia(self) -> bool:
        return not (any(s != 'desconocido' for _, _, s in self.estado) or self.agrupaciones or self.fechas_de_alta
                    or self.importes or self.existencias)

    @property
    def neto(self) -> tuple[str, str] | None:
        """(columna, tipo) del importe neto identificado por el usuario, o None."""
        return next(((c, t) for c, t in self.importes if t in TIPOS_NETO), None)

    @property
    def bruto(self) -> tuple[str, str] | None:
        return next(((c, t) for c, t in self.importes if t in TIPOS_BRUTO), None)

    def significado(self, columna: str, valor: str) -> str:
        return {(c, v): s for c, v, s in self.estado}.get((columna, valor), 'desconocido')

    def a_dict(self) -> dict:
        return {'estado': [list(x) for x in self.estado],
                'retirados_con_ventas_recientes': self.retirados_con_ventas_recientes,
                'excepciones_estado': [list(x) for x in self.excepciones_estado],
                'agrupaciones': [list(x) for x in self.agrupaciones], 'agrupacion_principal': self.agrupacion_principal,
                'fechas_de_alta': list(self.fechas_de_alta), 'importes': [list(x) for x in self.importes],
                'existencias': [list(x) for x in self.existencias], 'fecha_hora': self.fecha_hora,
                'huella_conjunto': self.huella_conjunto, 'importes_sin_identificar': list(self.importes_sin_identificar)}

    @classmethod
    def desde_dict(cls, d: dict) -> 'ConfirmacionColumnas':
        t = lambda clave: tuple(tuple(x) for x in d.get(clave, []))
        # Una confirmación anterior decía «importe de venta» o «precio por unidad» sin aclarar si era neto o bruto.
        # No se interpreta: esas columnas quedan sin confirmar y hay que identificarlas de nuevo.
        antiguos = tuple(c for c, tipo in t('importes') if tipo in TIPOS_IMPORTE_ANTERIORES)
        return cls(estado=t('estado'), retirados_con_ventas_recientes=bool(d.get('retirados_con_ventas_recientes', False)),
                   excepciones_estado=t('excepciones_estado'), agrupaciones=t('agrupaciones'),
                   agrupacion_principal=d.get('agrupacion_principal', ''), fechas_de_alta=tuple(d.get('fechas_de_alta', [])),
                   importes=tuple(x for x in t('importes') if x[1] not in TIPOS_IMPORTE_ANTERIORES),
                   existencias=t('existencias'), fecha_hora=d.get('fecha_hora', ''), huella_conjunto=d.get('huella_conjunto', ''),
                   importes_sin_identificar=tuple(d.get('importes_sin_identificar', [])) + antiguos)


# ------------------------------------------------------------------ líneas

def _lineas(inventario: InventarioColumnas, datos: pd.DataFrame) -> pd.DataFrame:
    """Producto, cantidad, fecha (con día si se reconoció), mes y archivo de cada línea."""
    n = len(datos)
    fecha = inventario.extras['fecha_diaria'] if inventario.tiene('fecha_diaria') else _a_fecha(datos['fecha'].reset_index(drop=True))
    fecha = pd.to_datetime(pd.Series(list(fecha), index=range(n)))
    principal = _a_fecha(datos['fecha'].reset_index(drop=True))
    fuente = inventario.fuente if inventario.fuente is not None else pd.Series(0, index=range(n))
    return pd.DataFrame({'producto': _texto(datos['producto'].reset_index(drop=True)),
                         'cantidad': pd.to_numeric(datos['cantidad'].reset_index(drop=True), errors='coerce'),
                         'fecha': fecha.fillna(principal),
                         'mes': (principal.dt.year * 12 + principal.dt.month - 1),
                         'fuente': pd.Series(list(fuente), index=range(n))})


def _columnas_de(inventario: InventarioColumnas, papel: str) -> list[str]:
    t = inventario.tabla
    if inventario.valores is None or t.empty:
        return []
    return [c for c in t[t.papel == papel].columna if c in inventario.valores.columns]


def _mas_reciente(lineas: pd.DataFrame, valor: pd.Series) -> tuple[dict[str, str], int, int]:
    """Valor de cada producto en su fecha más reciente. Devuelve (valores, productos cuyo valor cambia entre
    líneas, productos sin un único valor en la fecha más reciente)."""
    d = lineas[['producto', 'fecha']].assign(v=valor).dropna(subset=['producto', 'v'])
    if d.empty:
        return {}, 0, 0
    cambian = int((d.groupby('producto').v.nunique() > 1).sum())
    ultima = d[d.fecha == d.groupby('producto').fecha.transform('max')]
    por_producto = ultima.groupby('producto').v.agg(['nunique', 'first'])
    unicos = por_producto[por_producto['nunique'] == 1]['first']
    return unicos.to_dict(), cambian, int((por_producto['nunique'] > 1).sum())


# ------------------------------------------------------------------ estado del artículo

@dataclass
class AnalisisEstado:
    columna: str
    utilizable: bool
    motivo: str                       # por qué no se puede usar como estado del artículo
    valores: pd.DataFrame             # valor, productos, vendieron_el_ultimo_mes, vendieron_en_los_ultimos_3, mas_de_12_meses_sin_ventas
    por_producto: dict[str, str]      # valor más reciente de cada producto ('' = sin marcar)
    sin_dato: int                     # productos que solo aparecen en archivos sin esta columna
    cambian: int                      # productos cuyo valor no es siempre el mismo
    primer_mes: int | None
    ultimo_mes: int | None
    ultimo_mes_archivo: int


COLUMNAS_VALORES_ESTADO = ('valor', 'productos', 'vendieron_el_ultimo_mes', 'vendieron_en_los_ultimos_3',
                           'mas_de_12_meses_sin_ventas')


def analizar_estado(inventario: InventarioColumnas, datos: pd.DataFrame) -> list[AnalisisEstado]:
    """Qué dice cada columna de estado del artículo. No decide qué significa: eso lo confirma el usuario."""
    lineas = _lineas(inventario, datos)
    ultimo = int(lineas.mes.max()) if len(lineas) else 0
    ultima_venta = lineas[lineas.cantidad > 0].groupby('producto').mes.max()
    out = []
    for c in _columnas_de(inventario, 'estado_articulo'):
        t = inventario.valores[c]
        con = t.notna()
        en_archivos_con_dato = lineas.fuente.isin(set(lineas.fuente[con]))
        detalle = inventario.por_fuente.get(c, {})
        meses = lineas.mes[con]
        base = dict(columna=c, sin_dato=int(lineas.producto.nunique() - lineas.producto[en_archivos_con_dato].nunique()),
                    primer_mes=int(meses.min()) if len(meses) else None, ultimo_mes=int(meses.max()) if len(meses) else None,
                    ultimo_mes_archivo=ultimo)
        valor = t.fillna(SIN_MARCA).where(en_archivos_con_dato)
        por_producto, cambian, dudosos = _mas_reciente(lineas, valor)
        v = pd.Series(por_producto, dtype='object')
        u = ultima_venta.reindex(v.index)
        filas = [(x, int((v == x).sum()), int((u[v == x] >= ultimo).sum()), int((u[v == x] >= ultimo - 2).sum()),
                  int(((u[v == x] < ultimo - 12) | u[v == x].isna()).sum()))
                 for x in sorted(v.unique(), key=lambda x: (x == SIN_MARCA, x))]
        tabla = pd.DataFrame(filas, columns=list(COLUMNAS_VALORES_ESTADO))
        motivo = ''
        if 'la_linea' in detalle.values():
            motivo = 'Cambia entre líneas del mismo producto: no describe al artículo.'
        elif len(tabla) < 2:
            motivo = 'Todos los productos tienen el mismo valor: no distingue unos artículos de otros.'
        elif len(tabla) > 6:
            motivo = f'Tiene {len(tabla)} valores distintos: no parece una marca de estado.'
        out.append(AnalisisEstado(utilizable=not motivo, motivo=motivo, valores=tabla, por_producto=por_producto,
                                  cambian=cambian, **base))
    return out


def estados_desde_columnas(analisis: list[AnalisisEstado], confirmacion: ConfirmacionColumnas,
                           ultimas: dict[str, int], ultimo_mes: int) -> tuple[dict[str, K.EstadoComercial], dict]:
    """Estado comercial que se propone para cada producto según lo que el usuario confirmó de las columnas.

    `ultimas`: último mes con una operación válida de cada producto. Un producto que la columna da por retirado
    y tiene operaciones en el último mes utilizable no se da por retirado, salvo confirmación expresa.
    Si dos columnas se contradicen, el producto queda sin estado. Las excepciones individuales mandan.
    Devuelve los estados y un recuento de lo aplicado y de lo que no se ha aplicado.
    """
    propuestos: dict[str, tuple[str, str]] = {}
    contradictorios: set[str] = set()
    for a in analisis:
        if not a.utilizable:
            continue
        for p, v in a.por_producto.items():
            s = confirmacion.significado(a.columna, v)
            if s == 'desconocido':
                continue
            origen = f'columna «{a.columna}» ({_valor_texto(v)} = {TEXTO_SIGNIFICADO[s]}, según confirmó el usuario)'
            if p in propuestos and propuestos[p][0] != s:
                contradictorios.add(p)
            propuestos.setdefault(p, (s, origen))
    estados: dict[str, K.EstadoComercial] = {}
    cuenta = dict(activos=0, retirados=0, retirados_con_ventas_recientes_sin_aplicar=0, contradictorios=len(contradictorios),
                  excepciones=0)
    for p, (s, origen) in propuestos.items():
        if p in contradictorios:
            continue
        if s == 'activo':
            estados[p] = K.EstadoComercial('activo_confirmado', origen)
            cuenta['activos'] += 1
        elif ultimas.get(p, -1) >= ultimo_mes and not confirmacion.retirados_con_ventas_recientes:
            cuenta['retirados_con_ventas_recientes_sin_aplicar'] += 1
        else:
            estados[p] = K.EstadoComercial('retirado_confirmado', origen)
            cuenta['retirados'] += 1
    for p, e in confirmacion.excepciones_estado:
        cuenta['excepciones'] += 1
        if e == 'desconocido':
            estados.pop(p, None)
        else:
            estados[p] = K.EstadoComercial(e, 'excepción indicada por el usuario')
    return estados, cuenta


# ------------------------------------------------------------------ agrupaciones

@dataclass
class AnalisisAtributo:
    columna: str
    archivos_con_dato: int            # archivos en los que la columna describe al producto
    archivos: int                     # archivos en los que la columna tiene algún dato
    grupos: int
    productos_con_dato: int
    productos: int
    parte_de_las_unidades: float      # parte de las unidades vendidas que corresponde a productos con dato
    mayor_grupo: float                # parte de los productos con dato que reúne el grupo más grande
    grupos_de_un_producto: int
    cambian: int                      # productos con más de un valor
    por_producto: dict[str, str]
    propuesta: str                    # 'posible_agrupacion' o 'poco_util'
    motivo: str


def analizar_atributos(inventario: InventarioColumnas, datos: pd.DataFrame) -> list[AnalisisAtributo]:
    """Columnas con un valor fijo por producto y si pueden servir para agrupar. Ninguna se usa sin confirmar."""
    lineas = _lineas(inventario, datos)
    productos = int(lineas.producto.nunique())
    unidades = lineas[lineas.cantidad > 0].groupby('producto').cantidad.sum()
    out = []
    for c in _columnas_de(inventario, 'atributo_del_producto'):
        por_producto, cambian, _ = _mas_reciente(lineas, inventario.valores[c])
        detalle = inventario.por_fuente.get(c, {})
        tam = pd.Series(por_producto, dtype='object').value_counts()
        con_dato = len(por_producto)
        mayor = float(tam.iloc[0] / con_dato) if con_dato else 0.0
        de_uno = int((tam == 1).sum())
        if len(tam) < 2:
            propuesta, motivo = 'poco_util', 'Todos los productos con dato tienen el mismo valor.'
        elif cambian > 0.05 * con_dato:
            propuesta, motivo = 'poco_util', f'{cambian} producto(s) no tienen siempre el mismo valor.'
        elif mayor > 0.9:
            propuesta, motivo = 'poco_util', f'Un solo grupo reúne el {100 * mayor:.0f} % de los productos.'
        elif de_uno > 0.5 * con_dato:
            propuesta, motivo = 'poco_util', 'La mayoría de los grupos tendría un solo producto.'
        else:
            propuesta, motivo = 'posible_agrupacion', ''
        out.append(AnalisisAtributo(
            columna=c, archivos_con_dato=sum(1 for v in detalle.values() if v in ('el_articulo', 'valor_unico')),
            archivos=len(detalle), grupos=int(len(tam)), productos_con_dato=con_dato, productos=productos,
            parte_de_las_unidades=float(unidades.reindex(list(por_producto)).sum() / unidades.sum()) if unidades.sum() else 0.0,
            mayor_grupo=mayor, grupos_de_un_producto=de_uno, cambian=cambian, por_producto=por_producto,
            propuesta=propuesta, motivo=motivo))
    return sorted(out, key=lambda a: (a.propuesta != 'posible_agrupacion', -a.productos_con_dato))


def grupos_de_productos(analisis: list[AnalisisAtributo], columna: str, productos) -> dict[str, str]:
    """Grupo de cada producto según una columna confirmada. Los que no tienen dato van a «(sin dato)»."""
    a = next((x for x in analisis if x.columna == columna), None)
    if a is None:
        raise ValueError(f'No hay ninguna agrupación analizada con la columna {columna}.')
    return {p: a.por_producto.get(p, SIN_GRUPO) for p in productos}


COLUMNAS_FAMILIAS = ('grupo', 'productos', 'con_prevision', 'no_recomendada', 'datos_insuficientes',
                     'unidades_ultimos_12_meses', 'parte_de_las_unidades')


def resumen_por_grupo(grupos: dict[str, str], resumen: pd.DataFrame, series: dict[str, dict], ultimo_mes: int) -> pd.DataFrame:
    """Productos, estados y unidades recientes de cada grupo. Es un resumen: no hay previsión por grupo todavía."""
    r = resumen.set_index('producto').estado if len(resumen) else pd.Series(dtype='object')
    filas = {}
    for p, g in grupos.items():
        f = filas.setdefault(g, dict(grupo=g, productos=0, con_prevision=0, no_recomendada=0, datos_insuficientes=0,
                                     unidades_ultimos_12_meses=0.0))
        f['productos'] += 1
        estado = r.get(p)
        if estado in ('buen_respaldo', 'orientativa'):
            f['con_prevision'] += 1
        elif estado in ('no_recomendada', 'datos_insuficientes'):
            f[estado] += 1
        f['unidades_ultimos_12_meses'] += sum(v for m, v in series.get(p, {}).items() if v is not None and ultimo_mes - 11 <= m <= ultimo_mes)
    t = pd.DataFrame(list(filas.values()), columns=list(COLUMNAS_FAMILIAS[:-1]))
    total = float(t.unidades_ultimos_12_meses.sum())
    t['parte_de_las_unidades'] = t.unidades_ultimos_12_meses / total if total else 0.0
    return t.sort_values(['unidades_ultimos_12_meses', 'grupo'], ascending=[False, True]).reset_index(drop=True)


# ------------------------------------------------------------------ fecha de alta

def fechas_de_alta(inventario: InventarioColumnas, datos: pd.DataFrame, columna: str) -> dict[str, date]:
    """Fecha más antigua de cada producto en una columna confirmada como fecha de alta del artículo."""
    if inventario.valores is None or columna not in inventario.valores.columns:
        raise ValueError(f'No se conoce la columna {columna}.')
    lineas = _lineas(inventario, datos)
    f = pd.to_datetime(inventario.valores[columna].str.slice(0, 10), errors='coerce')
    d = pd.DataFrame({'p': lineas.producto, 'f': f}).dropna()
    return {p: v.date() for p, v in d.groupby('p').f.min().items()}


# ------------------------------------------------------------------ tabla por producto y registro

@dataclass
class ColumnasDelArticulo:
    """Todo lo analizado, para preguntar una sola vez y para exportar."""
    estado: list[AnalisisEstado] = field(default_factory=list)
    atributos: list[AnalisisAtributo] = field(default_factory=list)
    fechas: list[str] = field(default_factory=list)           # columnas que parecen una fecha fija por producto
    importes: list = field(default_factory=list)              # v2_importes.AnalisisImporte
    existencias: list = field(default_factory=list)           # v2_importes.AnalisisExistencias
    # (neto, bruto) -> cobertura y coherencia de las dos columnas que el usuario indique (v2_importes.cobertura_neto_bruto)
    comparar_importes: object = None

    @property
    def hay_algo(self) -> bool:
        return bool(self.estado or self.atributos or self.fechas or self.importes or self.existencias)


def analizar_columnas(inventario: InventarioColumnas, datos: pd.DataFrame) -> ColumnasDelArticulo:
    from . import v2_importes as IM
    if not inventario.alineado:
        return ColumnasDelArticulo()
    return ColumnasDelArticulo(estado=analizar_estado(inventario, datos), atributos=analizar_atributos(inventario, datos),
                               fechas=_columnas_de(inventario, 'fecha_del_articulo'),
                               importes=IM.analizar_importes(inventario, datos),
                               existencias=IM.analizar_existencias(inventario, datos),
                               comparar_importes=lambda neto, bruto: IM.cobertura_neto_bruto(inventario, datos, neto, bruto))


def tabla_de_articulos(analisis: ColumnasDelArticulo, confirmacion: ConfirmacionColumnas, inventario: InventarioColumnas,
                       datos: pd.DataFrame, productos, estados: dict[str, K.EstadoComercial] | None = None,
                       precios: pd.DataFrame | None = None, existencias: pd.DataFrame | None = None) -> pd.DataFrame:
    """Una fila por producto con lo que aportan las columnas confirmadas. Sin confirmar nada, solo el código."""
    t = pd.DataFrame({'producto': sorted(productos)})
    for a in analisis.estado:
        if a.utilizable and any(c == a.columna and s != 'desconocido' for c, _, s in confirmacion.estado):
            t[f'valor_en_{a.columna}'] = t.producto.map(lambda p: {None: 'sin dato', SIN_MARCA: 'sin marcar'}.get(a.por_producto.get(p), a.por_producto.get(p)))
    if estados is not None and len(t.columns) > 1:
        t['estado_comercial'] = t.producto.map(lambda p: estados[p].estado if p in estados else 'desconocido')
        t['origen_del_estado'] = t.producto.map(lambda p: estados[p].origen if p in estados else 'sin confirmar')
    for columna, tipo in confirmacion.agrupaciones:
        g = grupos_de_productos(analisis.atributos, columna, t.producto)
        t[f'{tipo}_{columna}' if columna != confirmacion.agrupacion_principal else 'grupo'] = t.producto.map(g)
    for columna in confirmacion.fechas_de_alta:
        altas = fechas_de_alta(inventario, datos, columna)
        t['alta_del_articulo'] = t.producto.map(altas)
    for extra in (precios, existencias):
        if extra is not None and len(extra.columns) > 1:
            t = t.merge(extra, on='producto', how='left')
    return t


# Importes y precios: se redondean solo al escribirlos. Los cálculos usan todos los decimales.
DECIMALES_AL_EXPORTAR = {'importe': 2, 'precio_medio': 4, 'precio_medio_reciente': 4, 'precio_medio_anterior': 4,
                         'cambio_de_precio': 4, 'descuento_medio': 4}


def exportar_articulos(tabla: pd.DataFrame, confirmacion: ConfirmacionColumnas, analisis: ColumnasDelArticulo,
                       carpeta: Path, *, familias: pd.DataFrame | None = None, precios_por_mes: pd.DataFrame | None = None,
                       resumen_estado: dict | None = None) -> dict[str, Path]:
    """Escribe lo confirmado y lo que se ha obtenido de ello. Sin confirmaciones solo deja constancia de lo
    que se reconoció y no se usa."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    rutas = {'columnas_confirmadas': carpeta / 'columnas_confirmadas_v2.json'}
    registro = {
        'version_registro': 1, 'confirmacion': confirmacion.a_dict(),
        'estado_del_articulo': [{
            'columna': a.columna, 'utilizable': a.utilizable, 'motivo': a.motivo, 'productos_sin_dato': a.sin_dato,
            'productos_que_cambian_de_valor': a.cambian, 'primer_mes': str(K.fecha(a.primer_mes)) if a.primer_mes is not None else None,
            'ultimo_mes': str(K.fecha(a.ultimo_mes)) if a.ultimo_mes is not None else None,
            'valores': [{**f, 'significado': confirmacion.significado(a.columna, f['valor'])}
                        for f in a.valores.to_dict('records')]} for a in analisis.estado],
        'estados_propuestos': resumen_estado or {},
        'agrupaciones': [{
            'columna': a.columna, 'grupos': a.grupos, 'productos_con_dato': a.productos_con_dato, 'productos': a.productos,
            'parte_de_las_unidades': round(a.parte_de_las_unidades, 4), 'mayor_grupo': round(a.mayor_grupo, 4),
            'grupos_de_un_producto': a.grupos_de_un_producto, 'productos_que_cambian_de_valor': a.cambian,
            'archivos_en_los_que_describe_al_producto': a.archivos_con_dato, 'archivos_con_dato': a.archivos,
            'propuesta': a.propuesta, 'motivo': a.motivo,
            'confirmada_como': dict(confirmacion.agrupaciones).get(a.columna)} for a in analisis.atributos],
        'importes': [x.a_dict() | {'confirmada_como': dict(confirmacion.importes).get(x.columna)} for x in analisis.importes],
        'existencias': [x.a_dict() | {'confirmada_como': dict(confirmacion.existencias).get(x.columna)} for x in analisis.existencias],
        'nota': ('Nada se usa sin confirmar. Las agrupaciones, los importes y las existencias no cambian ninguna previsión. '
                 'Unas existencias con un único valor por producto son, como mucho, una foto del momento de exportar: no '
                 'permiten saber si faltó género en meses pasados.')}
    rutas['columnas_confirmadas'].write_text(json.dumps(registro, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    if len(tabla.columns) > 1:
        rutas['articulos'] = carpeta / 'articulos_v2.csv'
        escribir_csv(tabla, rutas['articulos'], decimales=DECIMALES_AL_EXPORTAR)
    for nombre, t in (('familias', familias), ('precios', precios_por_mes)):
        if t is not None and len(t):
            rutas[nombre] = carpeta / f'{nombre}_v2.csv'
            escribir_csv(t, rutas[nombre], decimales=DECIMALES_AL_EXPORTAR)
    return rutas


def cargar_confirmacion(ruta: Path, huella_conjunto: str | None = None) -> ConfirmacionColumnas:
    """Recupera unas confirmaciones guardadas.

    `huella_conjunto`: huella de los archivos actuales. Para REUTILIZARLAS hay que indicarla siempre (aunque esté
    vacía): se rechazan si falta, si no traen la suya o si no coinciden. Con None solo se lee el archivo.
    """
    d = json.loads(Path(ruta).read_text(encoding='utf-8'))
    c = ConfirmacionColumnas.desde_dict(d['confirmacion'])
    if huella_conjunto is not None:
        K.comprobar_huellas(huella_conjunto, c.huella_conjunto, 'Las confirmaciones de columnas guardadas')
    return c


def mostrar_articulos_corto(analisis: ColumnasDelArticulo, confirmacion: ConfirmacionColumnas, resumen_estado: dict | None = None,
                            familias: pd.DataFrame | None = None) -> None:
    if not analisis.hay_algo:
        return
    print('\nESTADO, AGRUPACIONES, IMPORTES Y EXISTENCIAS')
    if confirmacion.vacia:
        print('  No se ha confirmado ninguna de estas columnas: no se usa ninguna.')
        return
    if resumen_estado:
        print(f'  Estado del artículo según la columna confirmada: {resumen_estado["activos"]} siguen a la venta y '
              f'{resumen_estado["retirados"]} están retirados.')
        if resumen_estado['retirados_con_ventas_recientes_sin_aplicar']:
            print(f'  {resumen_estado["retirados_con_ventas_recientes_sin_aplicar"]} figuran como retirados pero tienen ventas en el '
                  'último mes: no se dan por retirados.')
        if resumen_estado['contradictorios']:
            print(f'  {resumen_estado["contradictorios"]} tienen estados contradictorios en dos columnas: quedan sin confirmar.')
    for columna, tipo in confirmacion.agrupaciones:
        a = next(x for x in analisis.atributos if x.columna == columna)
        print(f'  Agrupación «{columna}» ({TEXTO_AGRUPACION[tipo]}): {a.grupos} grupos para {a.productos_con_dato} de '
              f'{a.productos} productos' + (' (la principal).' if columna == confirmacion.agrupacion_principal else '.'))
    if familias is not None and len(familias):
        print('  Grupos con más unidades en los últimos 12 meses: '
              + '; '.join(f'{f.grupo} ({100 * f.parte_de_las_unidades:.0f} %)' for f in familias.head(3).itertuples(index=False)) + '.')
        print('  Las agrupaciones no cambian ninguna previsión: se guardan para la previsión por familias.')
