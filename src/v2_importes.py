"""Versión 2, etapa 4 (segunda parte): importes, precios, descuentos y existencias.

Qué se hace con cada columna reconocida como importe o como existencias:

  qué parece            importe total de la línea, precio de una unidad o porcentaje, según cómo se comporta
                        frente a la cantidad. Es una propuesta: lo que es lo confirma el usuario.
  cobertura             líneas y meses con dato. Unos importes que solo cubren unos meses sirven para describir
                        el precio reciente, no como variable de todo el histórico.
  coherencia            signo del importe frente al de la cantidad; líneas vendidas con importe cero o sin importe.
  precio medio          con el importe neto confirmado: importe entre unidades, por producto y mes; cambio del
                        precio reciente frente al anterior; descuento medio si además hay un importe bruto.

Neto y bruto:
- El usuario dice expresamente qué columna es el importe neto (el final, ya con descuentos) y cuál el bruto
  (antes de descuentos). Nunca se deduce de los valores ni de las sumas: una columna bruta incompleta puede
  sumar menos que la neta.
- De cada una se comprueba la cobertura. El descuento solo se calcula con las líneas que traen los dos datos.
- Los valores solo sirven para comprobar la coherencia: si el bruto es menor que el neto en más del 1 % de las
  líneas que traen los dos, no se calcula ningún descuento y se avisa. No se intercambian.

Importe desconocido e importe cero:
- Una línea vendida sin dato de importe (celda vacía, en un archivo que sí trae esa columna) tiene importe
  DESCONOCIDO. Una línea con un 0 escrito tiene importe CERO (puede ser un regalo o una muestra).
- Ninguna de las dos entra en el precio medio, y se cuentan por separado. Las dos siguen contando como
  unidades vendidas.

Anulaciones:
- El precio se calcula con las ventas que quedan después de aplicar las anulaciones confirmadas, igual que las
  unidades brutas: una venta anulada por completo no entra; una anulada en parte entra con las unidades que
  quedan y con su importe menos el importe de la anulación. Si la línea de anulación no trae importe (o lo trae
  con el signo contrario), se quita la parte proporcional.
- Las devoluciones, los ajustes y las líneas sin confirmar no son ventas: no entran.

Existencias:
- Una columna con un único valor por producto es, como mucho, una foto: lo que había al exportar el archivo.
- Una columna que cambia entre líneas podría ser lo que había en el momento de cada venta; solo se conocería
  en los días con ventas.
- En ningún caso se afirma que faltó género en un mes pasado: con estos datos no se puede demostrar.

Nada de esto cambia una previsión. Los precios solo se calculan hasta el último mes utilizable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import v2_completitud as K
from .v2_articulos import TIPOS_BRUTO, TIPOS_NETO, ConfirmacionColumnas, _columnas_de, _lineas, _mas_reciente
from .v2_columnas import InventarioColumnas, _cabecera_normalizada, _numeros

MESES_MINIMOS_COMO_VARIABLE = 24       # meses con importes para plantearse usarlos como variable del histórico
COBERTURA_MINIMA_COMO_VARIABLE = 0.9   # parte de las líneas de esos meses que deben traer importe
MESES_PRECIO_RECIENTE = 3
COHERENCIA_MINIMA_BRUTO = 0.99         # parte de las líneas con neto y bruto en las que el bruto no es menor
CLASES_DE_VENTA = ('venta', 'venta_anulada_en_parte')
AVISO_EXISTENCIAS = ('Unas existencias actuales no son un histórico: no permiten saber si faltó género en meses '
                     'pasados.')
TEXTO_PARECE = {'importe_de_linea': 'parece el importe total de la línea', 'precio_unitario': 'parece el precio de una unidad',
                'porcentaje': 'parece un porcentaje', 'indeterminado': 'no se distingue si es un total o un precio por unidad',
                'no_numerico': 'no contiene números'}


def _mes(i: int | None) -> str:
    return '' if i is None else f'{i % 12 + 1:02d}/{i // 12}'


# ------------------------------------------------------------------ importes

@dataclass
class AnalisisImporte:
    columna: str
    que_parece: str
    lineas_con_dato: int
    cobertura: float                  # parte de todas las líneas
    primer_mes: int | None
    ultimo_mes: int | None
    meses_con_dato: int
    meses_del_archivo: int
    signo_coherente: float            # líneas con dato y cantidad distinta de cero cuyo signo coincide con el de la cantidad
    ceros_con_venta: int              # líneas con cantidad positiva y un 0 escrito
    sin_dato_con_venta: int           # líneas con cantidad positiva y la celda vacía, en archivos que traen la columna
    apto_como_variable: bool
    motivo: str                       # por qué no sirve como variable del histórico

    def a_dict(self) -> dict:
        d = asdict(self)
        d['primer_mes'], d['ultimo_mes'] = _mes(self.primer_mes), _mes(self.ultimo_mes)
        return d


def _que_parece(nombre: str, numeros: pd.Series, lineas: pd.DataFrame) -> str:
    clave = _cabecera_normalizada(nombre)
    validos = numeros.dropna()
    if any(f in clave for f in ('dto', 'descuento', 'discount', 'iva', 'porc')) and len(validos) and validos.between(0, 100).all():
        return 'porcentaje'
    d = pd.DataFrame({'p': lineas.producto, 'q': lineas.cantidad, 'v': numeros}).dropna()
    d = d[(d.q > 0) & (d.v > 0)]
    d['u'] = (d.v / d.q).round(2)
    d['v'] = d.v.round(2)
    g = d.groupby('p').agg(n=('q', 'size'), q=('q', 'nunique'), v=('v', 'nunique'), u=('u', 'nunique'))
    g = g[(g.n >= 3) & (g.q >= 2)]          # productos con cantidades distintas: ahí se nota la diferencia
    if len(g) < 5:
        return 'indeterminado'
    total, unidad = float((g.u < g.v).mean()), float((g.v < g.u).mean())
    return 'importe_de_linea' if total >= 0.6 else 'precio_unitario' if unidad >= 0.6 else 'indeterminado'


def _en_alcance(lineas: pd.DataFrame, con: pd.Series) -> pd.Series:
    """Líneas de los archivos en los que la columna trae algún dato. Fuera de ellos la columna no existe: una
    celda vacía ahí no es un importe desconocido, es un periodo sin importes."""
    return lineas.fuente.isin(set(lineas.fuente[con]))


def analizar_importes(inventario: InventarioColumnas, datos: pd.DataFrame) -> list[AnalisisImporte]:
    lineas = _lineas(inventario, datos)
    meses_archivo = int(lineas.mes.nunique())
    out = []
    for c in _columnas_de(inventario, 'importe'):
        t = inventario.valores[c]
        con = t.notna()
        numeros = _numeros(t)
        meses = lineas.mes[con]
        q = lineas.cantidad
        base = dict(columna=c, lineas_con_dato=int(con.sum()), cobertura=float(con.mean()) if len(con) else 0.0,
                    primer_mes=int(meses.min()) if len(meses) else None, ultimo_mes=int(meses.max()) if len(meses) else None,
                    meses_con_dato=int(meses.nunique()), meses_del_archivo=meses_archivo,
                    sin_dato_con_venta=int((_en_alcance(lineas, con) & ~con & (q > 0)).sum()))
        if numeros[con].notna().mean() < 0.99:
            out.append(AnalisisImporte(que_parece='no_numerico', signo_coherente=0.0, ceros_con_venta=0, apto_como_variable=False,
                                       motivo='No contiene números.', **base))
            continue
        comparables = con & (q != 0) & (numeros != 0)
        signo = float((np.sign(numeros[comparables]) == np.sign(q[comparables])).mean()) if comparables.any() else 1.0
        en_sus_meses = lineas.mes.isin(set(meses))
        cubre = float(con[en_sus_meses].mean()) if en_sus_meses.any() else 0.0
        apto = base['meses_con_dato'] >= MESES_MINIMOS_COMO_VARIABLE and cubre >= COBERTURA_MINIMA_COMO_VARIABLE
        motivo = ''
        if base['meses_con_dato'] < MESES_MINIMOS_COMO_VARIABLE:
            motivo = (f'Solo hay datos de {_mes(base["primer_mes"])} a {_mes(base["ultimo_mes"])} ({base["meses_con_dato"]} de los '
                      f'{meses_archivo} meses): sirve para describir el precio reciente, no como variable de todo el histórico.')
        elif not apto:
            motivo = f'En los meses con dato, solo lo trae el {100 * cubre:.0f} % de las líneas.'
        out.append(AnalisisImporte(que_parece=_que_parece(c, numeros, lineas), signo_coherente=signo,
                                   ceros_con_venta=int((con & (q > 0) & (numeros == 0)).sum()),
                                   apto_como_variable=apto, motivo=motivo, **base))
    return out


def _importe_de_linea(inventario: InventarioColumnas, lineas: pd.DataFrame, columna: str, tipo: str) -> pd.Series:
    """Importe total de cada línea según la columna confirmada. Un precio por unidad se multiplica por la cantidad.
    Vacío (NaN) = desconocido; 0 = cero escrito en el archivo."""
    v = _numeros(inventario.valores[columna])
    return v if tipo.startswith('importe') else v * lineas.cantidad


def cobertura_neto_bruto(inventario: InventarioColumnas, datos: pd.DataFrame, neto: tuple[str, str],
                         bruto: tuple[str, str]) -> dict:
    """Cobertura y coherencia de las columnas que el usuario ha identificado como importe neto y bruto.

    No decide cuál es cuál: eso lo ha dicho el usuario. Solo cuenta en cuántas líneas vendidas está cada dato y,
    donde están los dos, en cuántas el bruto no es menor que el neto.
    """
    lineas = _lineas(inventario, datos)
    n, b = _importe_de_linea(inventario, lineas, *neto), _importe_de_linea(inventario, lineas, *bruto)
    vendidas = lineas.cantidad > 0
    ambos = vendidas & n.notna() & b.notna()
    correcto = float((b[ambos] >= n[ambos] - 0.005).mean()) if ambos.any() else 1.0
    return dict(columna_neto=neto[0], columna_bruto=bruto[0], lineas_vendidas=int(vendidas.sum()),
                con_neto=int((vendidas & n.notna()).sum()), con_bruto=int((vendidas & b.notna()).sum()), con_los_dos=int(ambos.sum()),
                solo_neto=int((vendidas & n.notna() & b.isna()).sum()), solo_bruto=int((vendidas & n.isna() & b.notna()).sum()),
                bruto_no_menor_que_neto=correcto, coherente=correcto >= COHERENCIA_MINIMA_BRUTO)


@dataclass
class ResultadoPrecios:
    por_producto: pd.DataFrame        # una fila por producto con ventas en el periodo con importes (COLUMNAS_PRECIO_PRODUCTO)
    por_mes: pd.DataFrame             # COLUMNAS_PRECIO_MES
    columna_neto: str                 # columna que el usuario identificó como importe neto
    columna_bruto: str                # columna que identificó como importe bruto ('' si no hay)
    resumen: dict


COLUMNAS_PRECIO_PRODUCTO = ('producto', 'meses_con_precio', 'precio_medio_reciente', 'precio_medio_anterior',
                            'cambio_de_precio', 'descuento_medio', 'parte_de_las_unidades_con_precio',
                            'lineas_con_importe_cero', 'unidades_con_importe_cero', 'lineas_con_importe_desconocido',
                            'unidades_con_importe_desconocido', 'lineas_con_importe_negativo')
COLUMNAS_PRECIO_MES = ('producto', 'mes', 'unidades', 'importe', 'precio_medio', 'unidades_con_importe_cero',
                       'unidades_con_importe_desconocido')


def _ventas_efectivas(lineas: pd.DataFrame, importes: dict[str, pd.Series], operaciones: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    """Líneas de venta con las unidades y los importes que quedan después de las anulaciones confirmadas.

    Sin `operaciones` (clase de cada línea, de `v2_operaciones`), son ventas las líneas con cantidad positiva.
    Con ellas: solo las ventas confirmadas; una anulada en parte lleva las unidades que quedan y su importe menos
    el de la anulación (o la parte proporcional, si la anulación no trae un importe válido).
    """
    d = lineas[['producto', 'mes', 'fuente']].assign(unidades=lineas.cantidad, **importes)
    cuenta = dict(anulaciones_aplicadas=0, anulaciones_sin_importe_valido=0)
    if operaciones is None:
        return d[d.unidades > 0], cuenta
    if len(operaciones) != len(lineas):
        raise ValueError('Las operaciones no corresponden a estas líneas.')
    clase = operaciones.clase.to_numpy()
    original = lineas.cantidad.to_numpy(dtype=float)
    d['unidades'] = operaciones.unidades_brutas.to_numpy(dtype=float)
    de_linea = {nombre: v.to_numpy(dtype=float) for nombre, v in importes.items()}       # lo que trae cada línea
    queda = {nombre: v.copy() for nombre, v in de_linea.items()}                          # lo que queda en cada venta
    for i in np.flatnonzero((clase == 'anulacion') & operaciones.fila_de_la_venta.notna().to_numpy()):
        v = int(operaciones.fila_de_la_venta.iat[i])
        if original[v] <= 0:
            continue
        cuenta['anulaciones_aplicadas'] += 1
        for nombre in importes:
            if np.isnan(de_linea[nombre][v]):
                continue                                   # la venta no trae ese importe: sigue desconocido
            de_la_anulacion = de_linea[nombre][i]
            if np.isnan(de_la_anulacion) or de_la_anulacion >= 0:
                # Sin un importe válido en la anulación (vacío, cero o con el signo contrario): parte proporcional.
                queda[nombre][v] -= de_linea[nombre][v] * (-original[i]) / original[v]
                cuenta['anulaciones_sin_importe_valido'] += nombre == 'neto'
            else:
                queda[nombre][v] += de_la_anulacion         # el importe de la anulación ya viene en negativo
    for nombre in importes:
        d[nombre] = queda[nombre]
    return d[np.isin(clase, CLASES_DE_VENTA) & (d.unidades > 1e-9)], cuenta


def resumen_de_precios(inventario: InventarioColumnas, datos: pd.DataFrame, confirmacion: ConfirmacionColumnas,
                       ultimo_mes: int, *, operaciones: pd.DataFrame | None = None) -> ResultadoPrecios | None:
    """Precio medio por producto y mes con el importe neto que el usuario ha identificado. None si no hay ninguno.

    `operaciones`: `ResultadoOperaciones.lineas` (clase de cada línea). Con ella, el precio se calcula con las
    ventas que quedan después de las anulaciones confirmadas; las devoluciones, los ajustes y las líneas sin
    confirmar no entran.
    El importe bruto, si el usuario lo ha identificado, solo sirve para el descuento, que se calcula con las
    líneas que traen los dos datos. Si el bruto es menor que el neto en más del 1 % de esas líneas, no se calcula.
    Solo cuentan los archivos en los que la columna del neto trae datos y los meses hasta `ultimo_mes`.
    """
    neto, bruto = confirmacion.neto, confirmacion.bruto
    if neto is None or neto[0] not in inventario.valores.columns:
        return None
    if bruto is not None and bruto[0] not in inventario.valores.columns:
        bruto = None
    lineas = _lineas(inventario, datos)
    importes = {'neto': _importe_de_linea(inventario, lineas, *neto)}
    if bruto is not None:
        importes['bruto'] = _importe_de_linea(inventario, lineas, *bruto)
    alcance = _en_alcance(lineas, importes['neto'].notna())
    d, cuenta = _ventas_efectivas(lineas, importes, operaciones)
    d = d[alcance.reindex(d.index) & (d.mes <= ultimo_mes) & d.producto.notna()]
    if 'bruto' not in d:
        d = d.assign(bruto=np.nan)
    d = d.assign(tipo=np.select([d.neto.isna(), d.neto > 1e-9, d.neto.abs() <= 1e-9], ['desconocido', 'con_precio', 'cero'], 'negativo'))
    # Descuento: solo con las líneas que traen neto y bruto, y solo si el bruto es coherente con el neto.
    ambos = d[(d.tipo == 'con_precio') & d.bruto.notna() & (d.bruto > 1e-9)]
    con_los_dos = d[d.neto.notna() & d.bruto.notna()]
    coherencia = float((con_los_dos.bruto >= con_los_dos.neto - 0.005).mean()) if len(con_los_dos) else 1.0
    hay_descuento = bruto is not None and len(ambos) > 0 and coherencia >= COHERENCIA_MINIMA_BRUTO
    motivo_sin_descuento = ''
    if bruto is not None and not hay_descuento:
        motivo_sin_descuento = ('ninguna línea trae el importe neto y el bruto a la vez' if not len(ambos) else
                                f'el importe bruto es menor que el neto en el {100 * (1 - coherencia):.0f} % de las líneas que '
                                'traen los dos: puede que las columnas no sean lo que se ha indicado')
    precio = d[d.tipo == 'con_precio']
    por_mes = precio.groupby(['producto', 'mes']).agg(unidades=('unidades', 'sum'), importe=('neto', 'sum'))
    for tipo, nombre in (('cero', 'unidades_con_importe_cero'), ('desconocido', 'unidades_con_importe_desconocido')):
        por_mes = por_mes.join(d[d.tipo == tipo].groupby(['producto', 'mes']).unidades.sum().rename(nombre), how='outer')
    por_mes = por_mes.fillna(0.0)
    por_mes['precio_medio'] = (por_mes.importe / por_mes.unidades).where(por_mes.unidades > 0)
    por_mes = por_mes.reset_index()
    filas = []
    for p, g in d.groupby('producto'):
        cp = por_mes[(por_mes.producto == p) & (por_mes.unidades > 0)].sort_values('mes')
        reciente, anterior = cp.tail(MESES_PRECIO_RECIENTE), cp.iloc[:-MESES_PRECIO_RECIENTE]
        pr = float(reciente.importe.sum() / reciente.unidades.sum()) if len(reciente) else np.nan
        pa = float(anterior.importe.sum() / anterior.unidades.sum()) if len(anterior) else np.nan
        descuento = np.nan
        if hay_descuento:
            a = ambos[ambos.producto == p]
            descuento = float(1 - a.neto.sum() / a.bruto.sum()) if len(a) else np.nan
        por_tipo = g.groupby('tipo').unidades.agg(['size', 'sum'])
        lin = lambda t: int(por_tipo['size'].get(t, 0))
        uni = lambda t: float(por_tipo['sum'].get(t, 0.0))
        filas.append((p, int(len(cp)), pr, pa, (pr / pa - 1) if pa and pa > 0 and not np.isnan(pr) else np.nan, descuento,
                      uni('con_precio') / g.unidades.sum() if g.unidades.sum() > 0 else np.nan,
                      lin('cero'), uni('cero'), lin('desconocido'), uni('desconocido'), lin('negativo')))
    por_producto = pd.DataFrame(filas, columns=list(COLUMNAS_PRECIO_PRODUCTO))
    por_mes['mes'] = por_mes.mes.map(lambda i: K.fecha(int(i)))
    cambio = por_producto.cambio_de_precio.dropna()
    resumen = dict(columna_neto=neto[0], columna_bruto=bruto[0] if bruto else '',
                   productos_con_precio=int((por_producto.meses_con_precio > 0).sum()),
                   primer_mes=_mes(int(d.mes.min())) if len(d) else '', ultimo_mes=_mes(int(d.mes.max())) if len(d) else '',
                   meses_con_importes=int(precio.mes.nunique()),
                   productos_con_precio_reciente_y_anterior=int(len(cambio)),
                   productos_con_cambio_de_precio_mayor_del_5_por_ciento=int((cambio.abs() > 0.05).sum()),
                   lineas_de_venta=int(len(d)), lineas_con_precio=int((d.tipo == 'con_precio').sum()),
                   lineas_con_importe_cero=int((d.tipo == 'cero').sum()),
                   lineas_con_importe_desconocido=int((d.tipo == 'desconocido').sum()),
                   lineas_con_importe_negativo=int((d.tipo == 'negativo').sum()),
                   lineas_con_importe_bruto=int(d.bruto.notna().sum()) if bruto else 0,
                   lineas_con_neto_y_bruto=int(len(con_los_dos)) if bruto else 0,
                   bruto_no_menor_que_neto=coherencia if bruto else None,
                   descuento_calculado=bool(hay_descuento), motivo_sin_descuento=motivo_sin_descuento,
                   lineas_con_descuento=int((ambos.bruto > ambos.neto + 0.005).sum()) if hay_descuento else 0, **cuenta)
    return ResultadoPrecios(por_producto, por_mes[list(COLUMNAS_PRECIO_MES)], neto[0], bruto[0] if bruto else '', resumen)


# ------------------------------------------------------------------ existencias

@dataclass
class AnalisisExistencias:
    columna: str
    tipo: str                         # 'un_valor_por_producto' o 'cambia_entre_lineas'
    productos_con_dato: int
    productos: int
    valores_distintos: int
    primer_mes: int | None
    ultimo_mes: int | None
    explicacion: str
    por_producto: dict[str, float]    # dato más reciente de cada producto

    def a_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != 'por_producto'}
        d['primer_mes'], d['ultimo_mes'] = _mes(self.primer_mes), _mes(self.ultimo_mes)
        return d


def analizar_existencias(inventario: InventarioColumnas, datos: pd.DataFrame) -> list[AnalisisExistencias]:
    """Qué son, a lo sumo, las columnas reconocidas como existencias. Nunca un histórico de roturas de stock."""
    lineas = _lineas(inventario, datos)
    describe = inventario.tabla.set_index('columna')['describe'].to_dict() if len(inventario.tabla) else {}
    out = []
    for c in _columnas_de(inventario, 'existencias'):
        numeros = _numeros(inventario.valores[c])
        por_producto, _, _ = _mas_reciente(lineas, numeros)
        meses = lineas.mes[numeros.notna()]
        fijo = describe.get(c, '') in ('el_articulo', 'valor_unico')
        explicacion = ('Un único valor por producto: como mucho, lo que había al exportar el archivo, o un dato fijo del '
                       'artículo que no es una cantidad en almacén. ' if fijo else
                       'Cambia entre líneas del mismo producto: podría ser lo que había en el momento de cada venta, y solo '
                       'se conocería en los días con ventas. ') + AVISO_EXISTENCIAS
        out.append(AnalisisExistencias(
            columna=c, tipo='un_valor_por_producto' if fijo else 'cambia_entre_lineas', productos_con_dato=len(por_producto),
            productos=int(lineas.producto.nunique()), valores_distintos=int(numeros.nunique()),
            primer_mes=int(meses.min()) if len(meses) else None, ultimo_mes=int(meses.max()) if len(meses) else None,
            explicacion=explicacion, por_producto={p: float(v) for p, v in por_producto.items()}))
    return out


def existencias_por_producto(analisis: list[AnalisisExistencias], confirmacion: ConfirmacionColumnas) -> pd.DataFrame | None:
    """Dato de existencias más reciente de cada producto, solo de las columnas confirmadas."""
    tipos = dict(confirmacion.existencias)
    t = None
    for a in analisis:
        if a.columna not in tipos:
            continue
        nombre = 'existencias_al_exportar' if tipos[a.columna] == 'existencias_al_exportar' else 'existencias_en_la_ultima_venta'
        parte = pd.DataFrame({'producto': list(a.por_producto), nombre: list(a.por_producto.values())})
        t = parte if t is None else t.merge(parte, on='producto', how='outer', suffixes=('', f'_{a.columna}'))
    return t


def mostrar_importes_corto(precios: ResultadoPrecios | None, existencias: pd.DataFrame | None) -> None:
    if precios is not None:
        r = precios.resumen
        print(f'  Precios con «{r["columna_neto"]}» (importe neto, según has indicado): {r["productos_con_precio"]} productos con '
              f'precio, de {r["primer_mes"]} a {r["ultimo_mes"]} ({r["meses_con_importes"]} meses).')
        if r['productos_con_precio_reciente_y_anterior']:
            print(f'  En {r["productos_con_cambio_de_precio_mayor_del_5_por_ciento"]} de los '
                  f'{r["productos_con_precio_reciente_y_anterior"]} productos con precio reciente y anterior, el precio medio ha '
                  'cambiado más de un 5 %.')
        if r['lineas_con_importe_cero']:
            print(f'  {r["lineas_con_importe_cero"]} líneas vendidas tienen importe cero (pueden ser regalos o muestras): '
                  'cuentan como unidades vendidas y no entran en el precio medio.')
        if r['lineas_con_importe_desconocido']:
            print(f'  {r["lineas_con_importe_desconocido"]} líneas vendidas no traen importe: su importe es desconocido, no cero. '
                  'Tampoco entran en el precio medio.')
        if r['lineas_con_importe_negativo']:
            print(f'  {r["lineas_con_importe_negativo"]} líneas vendidas tienen un importe negativo: no entran en el precio medio.')
        if r['columna_bruto']:
            print(f'  «{r["columna_bruto"]}» (importe bruto, según has indicado) está en {r["lineas_con_neto_y_bruto"]} de las '
                  f'{r["lineas_con_precio"] + r["lineas_con_importe_cero"] + r["lineas_con_importe_negativo"]} líneas con importe neto.')
            if r['descuento_calculado']:
                print(f'  El descuento se calcula solo con esas líneas: {r["lineas_con_descuento"]} llevan descuento.')
            else:
                print(f'  No se calcula ningún descuento: {r["motivo_sin_descuento"]}.')
        print('  Los precios no cambian ninguna previsión.')
    if existencias is not None and len(existencias):
        print(f'  Existencias anotadas para {len(existencias)} productos. ' + AVISO_EXISTENCIAS)
