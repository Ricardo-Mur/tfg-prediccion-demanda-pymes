"""Versión 2, etapa 4: reconocimiento e inventario de las columnas adicionales del archivo.

La versión 1 solo usa fecha, producto y cantidad (y, si las reconoce, una columna de identificador y otra de
tipo). Aquí se revisan todas las demás columnas, sin modificar nada:

  qué parece cada una        por el nombre de la cabecera y por comprobaciones sobre su contenido
  cobertura                  parte de las líneas con dato y primer y último mes en que lo hay
  coherencia                 que un identificador de línea no se repita, que las líneas de una factura compartan
                             fecha, que una fecha diaria caiga en el mismo mes que la fecha principal...
  qué uso se le da           operaciones, diagnóstico, pendiente de confirmar, no usar o sin datos

Reglas:
- Un nombre de cabecera es un indicio. El papel propuesto solo se usa si además el contenido es coherente y
  la candidata es única; si hay dos candidatas para lo mismo, no se usa ninguna.
- Las columnas de clientes (y las que parecen contener teléfonos, correos o documentos de identidad) no se
  usan como variables ni se exportan. Como excepción, el código de cliente se usa en memoria para emparejar
  una anulación con su venta (`v2_operaciones`); nunca se escribe en ningún archivo.
- El inventario no escribe valores de ninguna columna: solo nombres, recuentos y fechas.
- Familias, estado del artículo, importes y existencias se reconocen, pero quedan «pendientes de confirmar»:
  no se usan hasta que el usuario confirme qué significan (`v2_articulos`, `v2_importes`).
- De cada columna se comprueba además qué describe: el artículo (cada producto tiene siempre el mismo valor),
  el cliente (cada cliente tiene siempre el mismo valor) o la línea. Se comprueba archivo por archivo, porque
  una misma cabecera puede no significar lo mismo en todos. Una columna que describe al cliente se trata como
  dato de cliente aunque su nombre no lo diga.
- El nombre «stock» no basta: una columna que repite la cantidad de la línea no son existencias, y un valor
  fijo por producto es, como mucho, una foto del momento de exportar. Ninguna de las dos permite saber si
  faltó género en meses pasados.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .v2_exportacion import escribir_csv
from .calidad import _sin_valor
from .configuracion_operaciones import CABECERAS_IDENTIFICADOR, CABECERAS_TIPO, _cabecera_normalizada

ROLES = ('id_linea', 'id_factura', 'serie', 'tipo_movimiento', 'fecha_diaria', 'codigo_cliente', 'dato_de_cliente',
         'lote', 'descripcion', 'estado_articulo', 'existencias', 'importe', 'atributo_del_producto', 'otra_fecha',
         'fecha_del_articulo', 'cantidad_repetida', 'sin_datos', 'no_reconocida')
USO = {'id_linea': 'operaciones', 'id_factura': 'operaciones', 'serie': 'operaciones',
       'tipo_movimiento': 'operaciones', 'fecha_diaria': 'diagnostico', 'lote': 'operaciones',
       'codigo_cliente': 'solo_para_emparejar_anulaciones', 'dato_de_cliente': 'no_usar_dato_personal',
       'descripcion': 'no_se_usa', 'estado_articulo': 'pendiente_de_confirmar',
       'existencias': 'pendiente_de_confirmar', 'importe': 'pendiente_de_confirmar',
       'atributo_del_producto': 'pendiente_de_confirmar', 'otra_fecha': 'no_se_usa', 'sin_datos': 'sin_datos',
       'fecha_del_articulo': 'pendiente_de_confirmar', 'cantidad_repetida': 'no_se_usa', 'no_reconocida': 'no_se_usa'}
OPERATIVOS = ('id_linea', 'id_factura', 'serie', 'tipo_movimiento', 'fecha_diaria', 'codigo_cliente', 'lote')
# Columnas que solo se usan si el usuario confirma qué son. Sus valores se guardan en memoria para analizarlas.
PENDIENTES = ('estado_articulo', 'atributo_del_producto', 'importe', 'existencias', 'fecha_del_articulo')
DESCRIBE = ('el_articulo', 'el_cliente', 'la_linea', 'valor_unico', 'articulo_o_cliente', 'segun_el_archivo', '')
COLUMNAS_INVENTARIO = ('columna', 'papel', 'uso', 'se_usa', 'lineas_con_dato', 'cobertura', 'primer_mes', 'ultimo_mes',
                       'valores_distintos', 'describe', 'comprobacion')

CAB_LINEA = frozenset({'idlin', 'idlinea', 'linea', 'numlinea', 'nlinea', 'lineid', 'lineno', 'linenumber',
                       'idlineafactura', 'numerolinea'})
CAB_FACTURA = CABECERAS_IDENTIFICADOR | frozenset({'idfacv', 'idfac', 'numfac', 'albaran', 'numalbaran',
                                                   'numeroalbaran', 'documento', 'numdocumento'})
CAB_SERIE = frozenset({'serie', 'series', 'seriefactura', 'seriedocumento', 'seriedoc'})
CAB_LOTE = frozenset({'lote', 'numlote', 'batch', 'batchno'})
# Fragmentos de cabecera que apuntan a datos de clientes o de personas. Ante la duda, la columna no se usa.
PERSONALES = ('cli', 'customer', 'nif', 'dni', 'cif', 'nom', 'dir', 'address', 'tel', 'phone', 'movil', 'mail', 'pob',
              'postal', 'provincia', 'country', 'pais', 'contact', 'razonsocial', 'rep', 'comercial', 'vendedor',
              'ruta', 'zona', 'param')
CODIGO_CLIENTE = frozenset({'codcli', 'idcli', 'idcliente', 'clienteid', 'customerid', 'codigocliente', 'codcliente',
                            'numcliente', 'cliente', 'customer'})
RE_TELEFONO = re.compile(r'^\+?\d[\d\s.-]{7,}\d$')
RE_CORREO = re.compile(r'^[^@\s]+@[^@\s]+\.[a-z]{2,}$', re.I)
RE_DOCUMENTO = re.compile(r'^[xyzXYZ]?\d{7,8}[A-Za-z]$')


@dataclass
class InventarioColumnas:
    tabla: pd.DataFrame            # una fila por columna adicional (COLUMNAS_INVENTARIO), sin valores
    extras: pd.DataFrame           # columnas operativas reconocidas, alineadas con las líneas; solo en memoria
    lineas: int
    alineado: bool                 # False: no se pudieron alinear las columnas con las líneas y no se usa ninguna
    aviso: str = ''
    # Etapa 4, segunda parte. Solo en memoria; nunca se exportan desde aquí.
    valores: pd.DataFrame | None = None      # texto de las columnas pendientes de confirmar (estado, atributos, importes...)
    fuente: pd.Series | None = None          # archivo del que viene cada línea (0, 1, 2...)
    por_fuente: dict = field(default_factory=dict)   # columna -> {archivo: qué describe en ese archivo}

    def tiene(self, papel: str) -> bool:
        return papel in self.extras.columns


# ------------------------------------------------------------------ utilidades

def _vacio(s: pd.Series) -> pd.Series:
    return s.map(_sin_valor)


def _a_fecha(s: pd.Series) -> pd.Series:
    """Fechas inequívocas: objetos de fecha o texto AAAA-MM-DD. Lo demás queda sin interpretar."""
    def una(v):
        if isinstance(v, (pd.Timestamp, datetime)):
            return pd.Timestamp(v).normalize()
        if isinstance(v, date):
            return pd.Timestamp(v)
        if isinstance(v, str) and re.match(r'^\d{4}-\d{2}-\d{2}', v.strip()):
            try:
                return pd.Timestamp(v.strip()[:10])
            except ValueError:
                return pd.NaT
        return pd.NaT
    return pd.to_datetime(s.map(una))


def _texto(s: pd.Series) -> pd.Series:
    return s.map(lambda v: None if _sin_valor(v) else str(v).strip())


def _parece_personal(valores: pd.Series) -> bool:
    muestra = valores.dropna().astype(str).str.strip()
    if muestra.empty:
        return False
    muestra = muestra.sample(min(len(muestra), 2000), random_state=0)
    digitos = muestra.str.replace(r'\D', '', regex=True)
    telefono = muestra.str.match(RE_TELEFONO) & (digitos.str.len() == 9) & digitos.str[:1].isin(list('6789'))
    # Un correo o un documento de identidad bastan en pocas líneas; un número de 9 cifras puede ser un código,
    # así que solo cuenta como teléfono si lo parecen casi todos los valores.
    return bool((muestra.str.match(RE_CORREO) | muestra.str.match(RE_DOCUMENTO)).mean() > 0.02 or telefono.mean() > 0.8)


def _papel_por_cabecera(clave: str) -> str | None:
    if clave in CAB_LINEA:
        return 'id_linea'
    if any(f in clave for f in PERSONALES):
        return 'codigo_cliente' if clave.rstrip('0123456789') in CODIGO_CLIENTE else 'dato_de_cliente'
    if clave in CAB_FACTURA:
        return 'id_factura'
    if clave in CAB_SERIE:
        return 'serie'
    if clave in CABECERAS_TIPO:
        return 'tipo_movimiento'
    if clave in CAB_LOTE:
        return 'lote'
    if 'activo' in clave or 'active' in clave or 'descatalog' in clave or clave in ('baja', 'estadoarticulo'):
        return 'estado_articulo'
    if 'stock' in clave or 'existencia' in clave:
        return 'existencias'
    if clave in ('neto', 'bruto', 'pvp', 'total', 'dto', 'iva', 'margen', 'base') or \
            any(f in clave for f in ('precio', 'importe', 'price', 'amount', 'descuento', 'discount', 'coste', 'cost', 'tarifa')):
        return 'importe'
    if 'desc' in clave or 'alias' in clave or 'description' in clave or 'denominacion' in clave:
        return 'descripcion'
    if any(f in clave for f in ('famil', 'categ', 'seccion', 'gama', 'depart', 'grupo')):
        return 'atributo_del_producto'
    return None


def unir_originales(fuentes: list) -> tuple[pd.DataFrame, list[set[int]]]:
    """Une por nombre de cabecera las tablas originales de las fuentes, en el mismo orden que la versión 1.

    Devuelve la tabla y, para cada fuente, las posiciones de sus columnas ya usadas (fecha, producto...).
    Si una cabecera se repite en un archivo, la segunda y siguientes se numeran.
    """
    partes = []
    for f in fuentes:
        t = f.datos_originales.reset_index(drop=True).copy()
        vistos: dict[str, int] = {}
        nombres = []
        for c in t.columns:
            c = str(c)
            vistos[c] = vistos.get(c, 0) + 1
            nombres.append(c if vistos[c] == 1 else f'{c} ({vistos[c]})')
        t.columns = nombres
        partes.append(t)
    return pd.concat(partes, ignore_index=True, sort=False), [set(f.columnas.values()) for f in fuentes]


# ------------------------------------------------------------------ qué describe una columna

def _constante(valor: pd.Series, clave: pd.Series) -> float | None:
    """Parte de las claves (productos o clientes) con dos líneas o más que tienen siempre el mismo valor.

    None si no hay al menos 4 claves así: con menos no se puede comprobar nada.
    """
    par = pd.DataFrame({'k': clave, 'v': valor}).dropna()
    g = par.groupby('k').v.agg(['nunique', 'size'])
    g = g[g['size'] >= 2]
    return float((g['nunique'] == 1).mean()) if len(g) >= 4 else None


def _que_describe(t: pd.Series, con: pd.Series, producto: pd.Series, cliente: pd.Series | None,
                  fuente: pd.Series) -> tuple[str, dict[int, str]]:
    """Qué describe la columna en cada archivo con dato y en conjunto.

    'el_articulo': cada producto tiene siempre el mismo valor y los clientes no. 'el_cliente': al revés.
    'la_linea': cambia entre líneas del mismo producto. 'valor_unico': no hay nada que comprobar.
    Si la columna solo tiene un valor y está vacía en parte de las líneas (una marca), lo que se comprueba
    es si cada producto está siempre marcado o siempre sin marcar.
    """
    por_fuente: dict[int, str] = {}
    for f in sorted(fuente[con].unique()):
        del_archivo = fuente == f
        m = con & del_archivo
        valor = t[m]
        if valor.nunique() <= 1:
            parte = float(m.sum() / del_archivo.sum())
            if not 0.01 < parte < 0.99:
                por_fuente[int(f)] = 'valor_unico'
                continue
            valor, m = con[del_archivo].map({True: 'marcado', False: 'sin marcar'}), del_archivo
        cp = _constante(valor, producto[m])
        cc = _constante(valor, cliente[m]) if cliente is not None else None
        art, cli = cp is not None and cp >= 0.99, cc is not None and cc >= 0.99
        por_fuente[int(f)] = ('articulo_o_cliente' if art and cli else 'el_articulo' if art else
                              'el_cliente' if cli else 'la_linea')
    distintos = set(por_fuente.values()) - {'valor_unico'}
    return ('valor_unico' if not distintos else distintos.pop() if len(distintos) == 1 else 'segun_el_archivo'), por_fuente


def _numeros(t: pd.Series) -> pd.Series:
    """Texto a número, aceptando la coma decimal. Lo que no es un número queda vacío."""
    return pd.to_numeric(t.astype('string').str.replace(',', '.', regex=False), errors='coerce').astype('float64')


# ------------------------------------------------------------------ inventario

def inventariar_columnas(originales: pd.DataFrame, datos: pd.DataFrame, usadas: set[str] | frozenset = frozenset(),
                         fuente=None) -> InventarioColumnas:
    """Revisa las columnas de `originales` que no son fecha, producto ni cantidad.

    `originales`: todas las columnas del archivo, una fila por línea.
    `datos`: las líneas ya limpias de la versión 1 (fecha, producto, cantidad), en el mismo orden.
    `usadas`: nombres de las columnas que ya se usan como fecha, producto, cantidad o descripción.
    `fuente`: archivo del que viene cada línea (por defecto, todas del mismo).
    """
    n = len(datos)
    vacio = InventarioColumnas(pd.DataFrame(columns=list(COLUMNAS_INVENTARIO)), pd.DataFrame(index=range(n)), n, False,
                               valores=pd.DataFrame(index=range(n)), fuente=pd.Series(0, index=range(n)))
    if len(originales) != n or (fuente is not None and len(fuente) != n):
        vacio.aviso = 'Las columnas adicionales no se han podido alinear con las líneas: no se usa ninguna.'
        return vacio
    originales = originales.reset_index(drop=True)
    fuente = pd.Series(0, index=range(n)) if fuente is None else pd.Series(list(fuente), index=range(n)).astype(int)
    fecha = _a_fecha(datos['fecha'].reset_index(drop=True))
    mes = fecha.dt.to_period('M')
    producto = _texto(datos['producto'].reset_index(drop=True))
    cantidad = pd.to_numeric(datos['cantidad'].reset_index(drop=True), errors='coerce')
    # Código de cliente provisional (el más completo de los que lo parecen), solo para comprobar qué describe cada columna.
    cliente = None
    for c in originales.columns:
        if c not in usadas and _papel_por_cabecera(_cabecera_normalizada(str(c))) == 'codigo_cliente':
            candidata = _texto(originales[c])
            if cliente is None or candidata.notna().sum() > cliente.notna().sum():
                cliente = candidata
    filas, candidatas = [], {}
    valores = pd.DataFrame(index=range(n))
    por_fuente: dict[str, dict[int, str]] = {}
    for c in originales.columns:
        if c in usadas or str(c).startswith('__tfg_'):
            continue
        s = originales[c]
        con = ~_vacio(s)
        k = int(con.sum())
        fila = dict(columna=str(c), papel='sin_datos', lineas_con_dato=k, cobertura=k / n if n else 0.0,
                    primer_mes=None, ultimo_mes=None, valores_distintos=0, describe='', comprobacion='')
        if k:
            t = _texto(s)
            meses = mes[con].dropna()
            fila.update(primer_mes=meses.min().to_timestamp().date() if len(meses) else None,
                        ultimo_mes=meses.max().to_timestamp().date() if len(meses) else None,
                        valores_distintos=int(t.nunique()))
            fila['papel'], fila['comprobacion'], fila['describe'], detalle = _papel(
                str(c), s, t, con, fecha, producto, cantidad, cliente, fuente)
            if fila['papel'] in PENDIENTES:
                # Solo se guardan los valores de los archivos en los que la columna no describe al cliente.
                sirve = fuente.map(lambda f: detalle.get(int(f), '') not in ('el_cliente', 'articulo_o_cliente'))
                if fila['papel'] == 'atributo_del_producto':
                    # Si el nombre ya dice que es una agrupación, también vale el archivo en el que cada cliente
                    # compra productos de un solo grupo (valor fijo por producto y por cliente).
                    validos = {'el_articulo', 'valor_unico'} | (
                        {'articulo_o_cliente'} if _papel_por_cabecera(_cabecera_normalizada(str(c))) == 'atributo_del_producto' else set())
                    sirve = fuente.map(lambda f: detalle.get(int(f), '') in validos)
                valores[str(c)] = t.where(sirve)
                por_fuente[str(c)] = detalle
        filas.append(fila)
        if fila['papel'] in OPERATIVOS:
            candidatas.setdefault(fila['papel'], []).append(fila)
    extras = pd.DataFrame(index=range(n))
    for papel, cands in candidatas.items():
        if papel == 'codigo_cliente' and len(cands) > 1:
            # Varias columnas de código de cliente: solo se usa una si todas dicen lo mismo donde coinciden.
            cands = sorted(cands, key=lambda f: -f['lineas_con_dato'])
            principal = _texto(originales[cands[0]['columna']])
            iguales = []
            for f in cands[1:]:
                otra = _texto(originales[f['columna']])
                ambas = principal.notna() & otra.notna()
                iguales.append(float((principal[ambas] == otra[ambas]).mean()) if ambas.any() else 0.0)
            if min(iguales) >= 0.99:
                cands[0]['comprobacion'] = (f'Hay {len(cands)} columnas de código de cliente y coinciden donde ambas '
                                            'tienen dato: se usa la más completa, solo para buscar correspondencias.')
                for f in cands[1:]:
                    f['papel'], f['comprobacion'] = 'dato_de_cliente', 'Repite el código de cliente de otra columna más completa.'
                cands = cands[:1]
            else:
                for f in cands:
                    f['comprobacion'] = (f'Hay {len(cands)} columnas que parecen código de cliente y no coinciden entre sí '
                                         f'(coincidencia mínima: {100 * min(iguales):.0f} %). No se usa ninguna hasta saber '
                                         'cuál es: las anulaciones solo se buscan dentro del mismo documento.')
                    f['_sin_usar'] = True
                continue
        if len(cands) > 1:
            for f in cands:
                f['comprobacion'] = (f'Hay {len(cands)} columnas que parecen «{papel}»: no se usa ninguna hasta '
                                     'saber cuál es. ' + f['comprobacion']).strip()
                f['_sin_usar'] = True
            continue
        col = originales[cands[0]['columna']]
        extras[papel] = _a_fecha(col) if papel == 'fecha_diaria' else _texto(col)
    if 'id_linea' in extras:
        # El identificador de línea se comprueba dentro de su documento: serie, factura y fecha.
        clave = pd.DataFrame({'serie': extras['serie'] if 'serie' in extras else '',
                              'factura': extras['id_factura'] if 'id_factura' in extras else '',
                              'fecha': extras['fecha_diaria'] if 'fecha_diaria' in extras else fecha,
                              'linea': extras['id_linea']}).fillna({'serie': '', 'factura': ''})
        repetidas = int(clave[clave.linea.notna()].duplicated(keep=False).sum())
        ambito = 'dentro de su documento (serie, factura y fecha)' if 'id_factura' in extras else 'en la misma fecha'
        for f in filas:
            if f['papel'] == 'id_linea' and not f.get('_sin_usar'):
                f['comprobacion'] = (f'Identificador de línea: no se repite {ambito}.' if repetidas == 0 else
                                     f'Identificador de línea: {repetidas} líneas lo repiten {ambito}.')
    for f in filas:
        f['uso'] = USO[f['papel']]
        f['se_usa'] = f['papel'] in OPERATIVOS and not f.pop('_sin_usar', False)
    tabla = pd.DataFrame(filas, columns=list(COLUMNAS_INVENTARIO))
    return InventarioColumnas(tabla, extras, n, True, valores=valores, fuente=fuente, por_fuente=por_fuente)


def _texto_archivos(detalle: dict[int, str], clase: str) -> str:
    return f'{sum(1 for v in detalle.values() if v == clase)} de los {len(detalle)} archivos con dato'


def _papel(nombre: str, s: pd.Series, t: pd.Series, con: pd.Series, fecha: pd.Series, producto: pd.Series,
           cantidad: pd.Series, cliente: pd.Series | None, fuente: pd.Series) -> tuple[str, str, str, dict[int, str]]:
    """Papel propuesto para una columna con datos, la comprobación que lo respalda y qué describe."""
    clave = _cabecera_normalizada(nombre)
    por_cabecera = _papel_por_cabecera(clave)
    if por_cabecera in ('codigo_cliente', 'dato_de_cliente'):
        return por_cabecera, 'Columna de clientes o de personas: no se usa como variable ni se exporta.', '', {}
    if por_cabecera in (None, 'descripcion', 'atributo_del_producto', 'estado_articulo') and _parece_personal(t):
        return 'dato_de_cliente', 'Parece contener teléfonos, correos o documentos de identidad: no se usa ni se exporta.', '', {}
    k = int(con.sum())
    if por_cabecera in OPERATIVOS:
        describe, detalle = '', {}
    else:
        describe, detalle = _que_describe(t, con, producto, cliente, fuente)
    del_cliente = (describe == 'el_cliente'
                   or (describe == 'segun_el_archivo' and 'el_cliente' in detalle.values() and 'el_articulo' not in detalle.values()))
    como_fecha = _a_fecha(s)
    if como_fecha.notna().sum() >= 0.9 * k:
        ambas = como_fecha.notna() & fecha.notna()
        mismo_mes = float((como_fecha[ambas].dt.to_period('M') == fecha[ambas].dt.to_period('M')).mean()) if ambas.any() else 0.0
        if mismo_mes >= 0.99 and como_fecha[ambas].nunique() > fecha[ambas].nunique():
            return 'fecha_diaria', (f'Fecha con día: cae en el mismo mes que la fecha principal en el '
                                    f'{100 * mismo_mes:.1f} % de las líneas.'), '', {}
        if del_cliente:
            return 'dato_de_cliente', ('Fecha fija por cliente (cada cliente tiene siempre la misma): es un dato del '
                                       'cliente. No se usa ni se exporta.'), describe, detalle
        if describe == 'el_articulo':
            primera = pd.DataFrame({'p': producto[ambas], 'f': fecha[ambas], 'v': como_fecha[ambas]}).groupby('p').agg(
                f=('f', 'min'), v=('v', 'min'))
            antes = float((primera.v <= primera.f + pd.offsets.MonthEnd(0)).mean())
            return 'fecha_del_articulo', (f'Fecha fija por producto, anterior o igual a su primera venta en el {100 * antes:.0f} % '
                                          'de los productos: podría ser la fecha de alta del artículo. No se usa hasta '
                                          'confirmar qué es.'), describe, detalle
        return 'otra_fecha', (f'Es una fecha, pero solo coincide con el mes de la venta en el {100 * mismo_mes:.0f} % '
                              'de las líneas.'), describe, detalle
    if por_cabecera == 'id_linea':
        repetidos = int(t[con].duplicated(keep=False).sum())
        return 'id_linea', ('Identificador de línea: no se repite en ninguna línea.' if repetidos == 0 else
                            f'Identificador de línea: {repetidos} líneas comparten identificador con otra.'), '', {}
    if por_cabecera == 'id_factura':
        g = pd.DataFrame({'id': t[con], 'f': fecha[con]}).dropna().groupby('id').f.nunique()
        varias = int((g > 1).sum())
        return 'id_factura', (f'Identificador de documento: {len(g)} documentos; '
                              + ('todas las líneas de cada uno comparten fecha.' if varias == 0 else
                                 f'{varias} tienen líneas con fechas distintas (el número puede repetirse entre series o años).')), '', {}
    if por_cabecera in ('serie', 'tipo_movimiento', 'lote'):
        return por_cabecera, f'{int(t.nunique())} valores distintos.', '', {}
    if por_cabecera == 'descripcion':
        return por_cabecera, 'Reconocida por su nombre. No se usa.', describe, detalle
    if del_cliente:
        return 'dato_de_cliente', ('Cada cliente tiene siempre el mismo valor y los productos no: es un dato del '
                                   'cliente, aunque el nombre no lo diga. No se usa ni se exporta.'), describe, detalle
    del_articulo = describe in ('el_articulo', 'valor_unico')
    if por_cabecera == 'existencias':
        numeros = _numeros(t[con])
        if numeros.notna().mean() >= 0.99:
            igual = float(np.isclose(numeros, cantidad[con], equal_nan=False).mean())
            if igual >= 0.99:
                return 'cantidad_repetida', (f'Su nombre sugiere existencias, pero coincide con la cantidad de la línea en el '
                                             f'{100 * igual:.0f} % de las líneas: es la cantidad vendida, no lo que había en el '
                                             'almacén. No se usa.'), describe, detalle
            if del_articulo:
                return 'existencias', ('Un único valor por producto: como mucho es una foto (las existencias al exportar el '
                                       'archivo) o un dato fijo del artículo. No es un histórico de existencias y no permite '
                                       'saber si faltó género en meses pasados.'), describe, detalle
            return 'existencias', ('Cambia entre líneas del mismo producto: podría ser lo que había en el almacén en el momento '
                                   'de cada venta. Hay que confirmarlo; aun así, solo se conocería en los días con ventas.'), describe, detalle
        if del_articulo and describe == 'el_articulo':
            return 'atributo_del_producto', ('Su nombre sugiere existencias, pero no es una cantidad: es un dato fijo del '
                                             'artículo. No dice cuánto género había.'), describe, detalle
        return 'no_reconocida', 'Su nombre sugiere existencias, pero no contiene cantidades: no se usa.', describe, detalle
    if por_cabecera == 'estado_articulo':
        como = {'el_articulo': 'Cada producto tiene siempre el mismo valor (o está siempre sin marcar).',
                'valor_unico': 'Todas las líneas con dato tienen el mismo valor.',
                'la_linea': 'Cambia entre líneas del mismo producto: puede que no describa al artículo.',
                'segun_el_archivo': 'No se comporta igual en todos los archivos.'}.get(describe, '')
        return por_cabecera, f'Reconocida por su nombre. {como} No se usa hasta confirmar qué significa.'.replace('  ', ' '), describe, detalle
    if por_cabecera == 'importe':
        return por_cabecera, 'Reconocida por su nombre. No se usa hasta confirmar qué significa.', describe, detalle
    if por_cabecera is None and describe == 'la_linea':
        numeros = _numeros(t[con])
        if numeros.notna().mean() >= 0.99 and float(np.isclose(numeros, cantidad[con]).mean()) >= 0.99:
            return 'cantidad_repetida', 'Repite la cantidad de la línea: no aporta nada nuevo. No se usa.', describe, detalle
    # Sin nombre reconocible (o con nombre de agrupación): ¿es un atributo del producto?
    par = pd.DataFrame({'p': producto[con], 'v': t[con]}).dropna()
    productos, distintos = par.p.nunique(), par.v.nunique()
    if describe == 'el_articulo' and productos >= 4 and 2 <= distintos <= max(2, productos // 2):
        return 'atributo_del_producto', (f'Cada producto tiene siempre el mismo valor ({distintos} valores para '
                                         f'{productos} productos). Podría servir para agrupar, si se confirma qué es.'), describe, detalle
    if describe == 'segun_el_archivo' and 'el_articulo' in detalle.values():
        return 'atributo_del_producto', (f'No significa lo mismo en todos los archivos: en {_texto_archivos(detalle, "el_articulo")} '
                                         'cada producto tiene siempre el mismo valor; en los demás describe al cliente o a la '
                                         'línea. Solo se tendrían en cuenta los archivos en los que describe al producto, y solo '
                                         'si se confirma qué es.'), describe, detalle
    if por_cabecera is None and describe == 'articulo_o_cliente':
        return 'no_reconocida', ('Tiene un valor fijo por producto y también por cliente: no se distingue a cuál de los dos '
                                 'describe. No se usa.'), describe, detalle
    if por_cabecera == 'atributo_del_producto':
        if describe == 'la_linea':
            return 'no_reconocida', ('Su nombre sugiere una agrupación, pero cambia entre líneas del mismo producto: '
                                     'no se usa.'), describe, detalle
        return por_cabecera, 'Reconocida por su nombre. No se usa hasta confirmar qué significa.', describe, detalle
    return 'no_reconocida', 'No se reconoce: no se usa.', describe, detalle


def inventariar_resultado(resultado: dict) -> InventarioColumnas:
    """Inventario a partir de lo que devuelve la carga de la versión 1 (`ejecutar_terminal`)."""
    datos = resultado['limpio'].datos
    fuentes = resultado.get('fuentes') or []
    if not fuentes:
        inv = inventariar_columnas(pd.DataFrame(index=range(len(datos))), datos)
        inv.aviso = 'No se conocen los archivos de origen: no se revisan columnas adicionales.'
        return inv
    originales, usadas_por_fuente = unir_originales(fuentes)
    usadas = set()
    for f, posiciones in zip(fuentes, usadas_por_fuente):
        usadas |= {str(f.datos_originales.columns[i]) for i in posiciones}
    fuente = np.concatenate([np.full(len(f.datos_originales), i) for i, f in enumerate(fuentes)])
    inv = inventariar_columnas(originales, datos, usadas, fuente if len(fuente) == len(datos) else None)
    if inv.alineado:
        # Comprobación de que las filas siguen el mismo orden que las líneas limpias de la versión 1.
        original = pd.concat([_texto(f.datos_originales.iloc[:, f.columnas['producto']]) for f in fuentes], ignore_index=True)
        limpio = _texto(datos['producto'].reset_index(drop=True))
        if not (original.fillna('') == limpio.fillna('')).all():
            inv = InventarioColumnas(inv.tabla.assign(se_usa=False), pd.DataFrame(index=range(len(datos))), len(datos), False,
                                     'Las columnas adicionales no siguen el orden de las líneas: no se usa ninguna.',
                                     valores=pd.DataFrame(index=range(len(datos))), fuente=pd.Series(0, index=range(len(datos))))
    return inv


# ------------------------------------------------------------------ presentación

def exportar_inventario(inventario: InventarioColumnas, carpeta: Path) -> dict[str, Path]:
    """Solo nombres de columna, recuentos y fechas: ningún valor del archivo."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / 'columnas_v2.csv'
    escribir_csv(inventario.tabla, ruta)
    return {'columnas_v2': ruta}


def mostrar_inventario_corto(inventario: InventarioColumnas) -> None:
    t = inventario.tabla
    print(f'\nCOLUMNAS ADICIONALES DEL ARCHIVO ({len(t)})')
    if inventario.aviso:
        print(f'  {inventario.aviso}')
    nombres = {'id_linea': 'identificador de línea', 'id_factura': 'identificador de factura o documento',
               'serie': 'serie', 'tipo_movimiento': 'tipo de movimiento', 'fecha_diaria': 'fecha con día',
               'codigo_cliente': 'código de cliente (solo para emparejar anulaciones; no se exporta)', 'lote': 'lote'}
    usadas = t[t.se_usa]
    for f in usadas.itertuples(index=False):
        print(f'  Se usa «{f.columna}» como {nombres[f.papel]}: datos en el {100 * f.cobertura:.0f} % de las líneas. '
              f'{f.comprobacion}')
    if usadas.empty:
        print('  No se ha reconocido ninguna columna de operaciones.')
    for uso, texto in (('pendiente_de_confirmar', 'reconocidas, sin usar hasta confirmar qué significan'),
                       ('no_usar_dato_personal', 'de clientes o personas, que no se usan ni se exportan'),
                       ('sin_datos', 'vacías')):
        n = int(((t.uso == uso) & ~t.se_usa).sum())
        if n:
            print(f'  Columnas {texto}: {n}')
    con_fecha = t[t.primer_mes.notna()]
    inicio, fin = (con_fecha.primer_mes.min(), con_fecha.ultimo_mes.max()) if len(con_fecha) else (None, None)
    for f in usadas.itertuples(index=False):
        if f.primer_mes is not None and (f.primer_mes > inicio or f.ultimo_mes < fin):
            print(f'  Atención: «{f.columna}» solo tiene datos de {f.primer_mes:%m/%Y} a {f.ultimo_mes:%m/%Y}. '
                  'Fuera de esos meses no se puede usar.')
        elif f.cobertura < 0.99:
            print(f'  Atención: «{f.columna}» falta en el {100 * (1 - f.cobertura):.0f} % de las líneas.')
    print('  Detalle en columnas_v2.csv (sin valores del archivo).')
