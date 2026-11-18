"""Pantallas de la revisión agrupada (paso 4): una tabla por sección en lugar de una pregunta cada vez.

Solo dibujan y recogen lo que el usuario decide. No calculan ni deciden nada: las reglas siguen en el programa
congelado, que recibe las decisiones como respuestas a sus propias preguntas (`decisiones.py`).

Todo empieza en «No lo sé». Ninguna anulación, devolución o ajuste viene marcada: el único atajo es aceptar las
propuestas de series que hace el propio programa, y hay que pulsarlo.

Cada pantalla devuelve lo que ha pedido el usuario:
  ('confirmar', valor)     guarda la sección y deja que el programa siga
  ('atras', sección)       volver a una sección anterior
  ('una_a_una',)           contestar esta sección con las preguntas del programa, una a una
  None                     nada todavía
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from interfaz import decisiones as DEC

ORDEN = ('series', 'lineas', 'columnas', 'completitud', 'productos')
NOMBRES = {'series': 'Series', 'lineas': 'Líneas', 'columnas': 'Otras columnas', 'completitud': 'Datos completos',
           'productos': 'Productos', 'antes': 'Antes de calcular'}

PAPEL = {'ventas': 'Ventas a clientes', 'devoluciones': 'Devoluciones o abonos', 'ajustes': 'Ajustes (no son ventas)',
         'mixta': 'Mixta (ventas y otras operaciones)', 'desconocida': 'No lo sé'}
POSITIVAS = {'ventas': 'Ventas a clientes', 'ajustes': 'Ajustes', 'desconocido': 'No lo sé'}
FILA_POSITIVA = {'venta': 'Venta', 'ajuste': 'Ajuste', 'desconocido': 'No lo sé'}
TRATAMIENTO = {'anulacion': 'Anulación', 'devolucion': 'Devolución', 'ajuste': 'Ajuste', 'desconocido': 'No lo sé'}
ESTADO = {'activo_confirmado': 'Sigue a la venta', 'retirado_confirmado': 'Retirado', 'desconocido': 'No lo sé'}
COMO_EL_GRUPO = 'Como el grupo'
MAX_UNA_A_UNA = 20             # en «Antes de calcular», hasta cuántas líneas pendientes se enseñan con su propio selector
SIN_FECHA = 'No lo sé'

LEYENDA_NEGATIVAS = (
    '**Anulación**: esa venta no llegó a existir; se quita de la venta a la que anula.  \n'
    '**Devolución**: la venta existió y después se devolvió género; la venta se mantiene y la devolución se cuenta aparte.  \n'
    '**Ajuste**: no es venta ni devolución; no cuenta.  \n'
    '**No lo sé**: el mes de ese producto queda pendiente y no se usa.')


def _invertir(d: dict) -> dict:
    return {v: k for k, v in d.items()}


def _clave(seccion: str, vuelta: int, *resto) -> str:
    return '_'.join(['rv', seccion, str(vuelta), *map(str, resto)])


def barra(actual: str, decisiones: DEC.Decisiones, carpeta, hay_series: bool = True) -> None:
    """Las secciones de la revisión y en qué punto está cada una."""
    partes = []
    pasada = actual in ORDEN or actual == 'antes'
    for seccion in [s for s in ORDEN if hay_series or s != 'series'] + ['antes']:
        if seccion == actual:
            partes.append(f'**● {NOMBRES[seccion]}**')
            pasada = False
        elif pasada and (decisiones.lista(seccion, carpeta) or seccion in decisiones.una_a_una):
            partes.append(f'✓ {NOMBRES[seccion]}')
        elif pasada:
            partes.append(f'– {NOMBRES[seccion]}')         # el programa no ha preguntado por ella: con estos datos no hace falta
        else:
            partes.append(f'○ {NOMBRES[seccion]}')
    st.markdown(' · '.join(partes))


def texto_del_programa(contexto: str) -> None:
    with st.expander('Detalles técnicos'):
        st.caption('Lo que escribe el programa antes de esta sección, tal cual:')
        st.text(contexto.strip() or '(nada)')


def botones(seccion: str, vuelta: int, anterior: str | None, texto: str = 'Confirmar →') -> tuple[bool, bool, bool]:
    """Volver, contestar una a una y confirmar. Devuelve qué se ha pulsado."""
    atras_col, una_col, ok_col = st.columns([2, 3, 2])
    atras = anterior is not None and atras_col.button(f'← {NOMBRES[anterior]}', key=_clave(seccion, vuelta, 'atras'))
    una = una_col.button('Contestar esta parte pregunta a pregunta', key=_clave(seccion, vuelta, 'una'),
                         help='Las mismas preguntas del programa, una cada vez, como en la versión anterior.')
    ok = ok_col.button(texto, type='primary', key=_clave(seccion, vuelta, 'ok'))
    return atras, una, ok


# ------------------------------------------------------------------ 4.1 series

def series(datos: dict, actual: dict | None, vuelta: int, contexto: str, anterior=None):
    st.subheader('Qué hay en cada serie de documentos')
    n = len(datos['series'])
    st.write(f'Tus archivos tienen {n} {"serie" if n == 1 else "series"}. Una serie que no confirmes no se usa: '
             'sus meses quedan sin conocer.')
    con_propuesta = [f for f in datos['series'] if f['propuesta']]
    estado_clave = _clave('series', vuelta, 'valores')
    if estado_clave not in st.session_state:
        st.session_state[estado_clave] = {'aceptar': bool(actual and actual['aceptar']),
                                          'papel': dict(actual['papel']) if actual else {}, 'editor': 0}
    valores = st.session_state[estado_clave]
    if con_propuesta and st.button(f'Aceptar {"la propuesta" if len(con_propuesta) == 1 else f"las {len(con_propuesta)} propuestas"} del programa',
                                   key=_clave('series', vuelta, 'aceptar')):
        valores['aceptar'] = True
        valores['papel'] |= {f['serie']: f['propuesta'] for f in con_propuesta}
        valores['editor'] += 1
        st.rerun()
    papel = {}
    opciones = list(PAPEL.values())
    for f in datos['series']:
        with st.container(border=True):
            texto, eleccion = st.columns([3, 2])
            texto.markdown(f'**Serie «{f["serie"]}»**: {f["lineas"]} {"línea" if f["lineas"] == 1 else "líneas"} '
                           f'(positivas: {f["positivas"]}; negativas: {f["negativas"]}), '
                           f'de {f["desde"]} a {f["hasta"]}.  \n'
                           + (f'Propuesta del programa: **{PAPEL[f["propuesta"]].lower()}** ({f["motivo"]}).' if f['propuesta']
                              else f'Sin propuesta: {f["motivo"]}.'))
            q = eleccion.selectbox(f'Qué es la serie «{f["serie"]}»', opciones,
                                   index=opciones.index(PAPEL[valores['papel'].get(f['serie'], 'desconocida')]),
                                   key=_clave('series', vuelta, 'papel', f['serie'], valores['editor']))
            papel[f['serie']] = _invertir(PAPEL)[q]
    aceptar = valores['aceptar'] and all(papel[f['serie']] == f['propuesta'] for f in con_propuesta)
    if valores['aceptar'] and not aceptar:
        st.caption('Has cambiado alguna propuesta: cada serie queda como la indiques tú.')
    elif aceptar:
        st.caption('Propuestas del programa aceptadas.')
    st.caption('**Mixta**: tiene ventas y otras operaciones; de sus líneas negativas te preguntaré en la siguiente pantalla.')
    texto_del_programa(contexto)
    atras, una, ok = botones('series', vuelta, anterior)
    if una:
        return ('una_a_una',)
    if atras:
        return ('atras', anterior)
    if ok:
        if not any(p in ('ventas', 'mixta') for p in papel.values()):
            st.error('Ninguna serie es de ventas ni mixta. Sin ellas no queda ninguna venta que usar y no hay nada que calcular.')
            return None
        return ('confirmar', {'aceptar': aceptar, 'papel': papel})
    return None


# ------------------------------------------------------------------ 4.2 líneas

def _fila(x: dict) -> dict:
    venta = x.get('venta')
    return {'Fila': x['fila'], 'Fecha': x['fecha'], 'Producto': x['producto'], 'Cantidad': x['cantidad'], 'Serie': x['serie'],
            'Documento': x['documento'], 'Posible venta': f'fila {venta["fila"]} · {venta["fecha"]} · {venta["cantidad"]:g}' if venta else ''}


def lineas(datos: dict, actual: dict | None, vuelta: int, contexto: str, anterior=None):
    n = sum(len(g['lineas']) for g in datos['positivas'] + datos['negativas'])
    st.subheader(f'Líneas que el programa no sabe interpretar ({n})')
    st.write('No se restan ni se descartan sin saber qué son. Elige qué son para todo el grupo y, si alguna línea es '
             'distinta, cámbiala en su fila.')
    actual = actual or {'positivas': {}, 'negativos': {}, 'filas': {}}
    positivas, negativos, filas, errores = {}, {}, {}, []

    for g in datos['positivas']:
        lineas_g = g['lineas']
        with st.container(border=True):
            st.markdown(f'**Serie «{g["serie"]}»: {len(lineas_g)} {"línea positiva o cero" if len(lineas_g) == 1 else "líneas positivas o cero"}.** '
                        'Pueden ser ventas o ajustes positivos. Si no lo sabes, quedan pendientes y sus meses no se usan.')
            opciones = list(POSITIVAS.values())
            elegido = st.selectbox('Todo el grupo', opciones, index=opciones.index(POSITIVAS[actual['positivas'].get(g['serie'], 'desconocido')]),
                                   key=_clave('lineas', vuelta, 'pos', g['serie']))
            positivas[g['serie']] = _invertir(POSITIVAS)[elegido]
            con_cantidad = [x for x in lineas_g if x['cantidad'] > 0]
            with st.expander(f'Ver {"la línea" if len(lineas_g) == 1 else f"las {len(lineas_g)} líneas"}', expanded=len(lineas_g) <= 5):
                if con_cantidad:
                    filas |= _tabla_de_lineas(con_cantidad, actual['filas'], FILA_POSITIVA, vuelta, f'pos_{g["serie"]}')
                ceros = [x for x in lineas_g if x['cantidad'] <= 0]
                if ceros:
                    st.caption(f'{len(ceros)} con cantidad cero: siguen lo que digas para el grupo.')
                    st.dataframe(pd.DataFrame([_fila(x) for x in ceros]), hide_index=True, width='stretch')

    if datos['negativas']:
        with st.expander('Qué significa cada opción', expanded=True):
            st.markdown(LEYENDA_NEGATIVAS)
    for g in datos['negativas']:
        lineas_g = g['lineas']
        opciones = [TRATAMIENTO[t] for t in TRATAMIENTO if not (g['sin_anulacion'] and t == 'anulacion')]
        with st.container(border=True):
            st.markdown(f'**En {len(lineas_g)} {"línea" if len(lineas_g) == 1 else "líneas"} {g["texto"]}.**')
            if g['sin_anulacion']:
                st.caption('Una anulación necesita una venta a la que aplicarse: aquí no se puede elegir.')
            previo = TRATAMIENTO[actual['negativos'].get(g['grupo'], 'desconocido')]
            elegido = st.selectbox('Todo el grupo', opciones, index=opciones.index(previo) if previo in opciones else opciones.index('No lo sé'),
                                   key=_clave('lineas', vuelta, 'neg', g['grupo']))
            negativos[g['grupo']] = _invertir(TRATAMIENTO)[elegido]
            with st.expander(f'Ver {"la línea" if len(lineas_g) == 1 else f"las {len(lineas_g)} líneas"}', expanded=len(lineas_g) <= 5):
                de_grupo = _tabla_de_lineas(lineas_g, actual['filas'], {k: v for k, v in TRATAMIENTO.items() if not (g['sin_anulacion'] and k == 'anulacion')},
                                            vuelta, f'neg_{g["grupo"]}')
                if g['sin_anulacion'] and 'anulacion' in de_grupo.values():
                    errores.append(f'En el grupo «{g["texto"]}» no puede haber anulaciones.')
                filas |= de_grupo

    sin_aclarar = sum(1 for g in datos['positivas'] for x in g['lineas']
                      if filas.get(x['fila'], positivas[g['serie']] if x['fila'] not in filas else None) in ('desconocido',)) + \
        sum(1 for g in datos['negativas'] for x in g['lineas'] if filas.get(x['fila'], negativos[g['grupo']]) == 'desconocido')
    st.caption(f'Sin aclarar: {sin_aclarar} de {n} líneas. Lo que quede sin aclarar no se usa; antes de calcular verás a qué afecta.')
    texto_del_programa(contexto)
    atras, una, ok = botones('lineas', vuelta, anterior)
    if una:
        return ('una_a_una',)
    if atras:
        return ('atras', anterior)
    if ok:
        if errores:
            for e in errores:
                st.error(e)
            return None
        return ('confirmar', {'positivas': positivas, 'negativos': negativos, 'filas': filas})
    return None


def _tabla_de_lineas(lineas_g: list[dict], previas: dict, opciones: dict, vuelta: int, nombre: str) -> dict:
    """Tabla de un grupo con la columna «Esta línea». Devuelve las filas marcadas una a una: {fila: tratamiento}."""
    etiquetas = [COMO_EL_GRUPO] + list(opciones.values())
    tabla = pd.DataFrame([_fila(x) | {'Esta línea': opciones.get(previas.get(x['fila']), COMO_EL_GRUPO)
                                       if x['fila'] in previas else COMO_EL_GRUPO} for x in lineas_g])
    editada = st.data_editor(tabla, hide_index=True, width='stretch', key=_clave('lineas', vuelta, 'tabla', nombre),
                             disabled=[c for c in tabla.columns if c != 'Esta línea'],
                             column_config={'Esta línea': st.column_config.SelectboxColumn('Esta línea', options=etiquetas, required=True),
                                            'Cantidad': st.column_config.NumberColumn(format='%g')})
    inverso = _invertir(opciones)
    return {int(f): inverso[e] for f, e in zip(editada['Fila'], editada['Esta línea']) if e != COMO_EL_GRUPO}


# ------------------------------------------------------------------ 4.3 otras columnas

def columnas(vuelta: int, contexto: str, anterior=None):
    st.subheader('Otras columnas del archivo')
    reconocidas = next((l.strip() for l in contexto.splitlines() if 'He reconocido columnas' in l), '')
    if reconocidas:
        st.write(reconocidas)
    st.write('Ninguna cambia las cantidades previstas. **Si no las revisas, no se usan.** Si quieres revisarlas, el programa '
             'te preguntará por cada una, como hasta ahora.')
    texto_del_programa(contexto)
    atras_col, revisar_col, seguir_col = st.columns([2, 2, 3])
    if anterior is not None and atras_col.button(f'← {NOMBRES[anterior]}', key=_clave('columnas', vuelta, 'atras')):
        return ('atras', anterior)
    if revisar_col.button('Revisarlas', key=_clave('columnas', vuelta, 'revisar')):
        return ('confirmar', {'revisar': True})
    if seguir_col.button('Seguir sin usarlas →', type='primary', key=_clave('columnas', vuelta, 'seguir')):
        return ('confirmar', {'revisar': False})
    return None


# ------------------------------------------------------------------ 4.4 datos completos

ULTIMO = {'si': 'Sí, el mes está cerrado', 'no': 'No, faltan ventas', 'no_lo_se': 'No lo sé'}
COMPLETO = {'si': 'Sí, todo el periodo', 'parte': 'Solo una parte', 'no': 'No', 'no_lo_se': 'No lo sé'}


def completitud(datos: dict, actual: dict | None, vuelta: int, contexto: str, anterior=None):
    st.subheader('¿Están completos tus datos?')
    primer, ultimo = datos['primer_mes'], datos['ultimo_mes_archivo']
    actual = actual or {'ultimo': 'no_lo_se', 'completo': 'no_lo_se', 'desde': '', 'hasta': ''}
    st.write(f'El último mes de tus archivos es **{DEC.mes_corto(ultimo)}**.' + (f' {datos["indicio"]}' if datos['indicio'] else ''))
    with st.container(border=True):
        opciones = list(ULTIMO.values())
        r1 = st.radio(f'¿Están ya TODAS las ventas de {DEC.mes_corto(ultimo)}?', opciones, index=opciones.index(ULTIMO[actual['ultimo']]),
                      horizontal=True, key=_clave('completitud', vuelta, 'ultimo'))
        r1 = _invertir(ULTIMO)[r1]
        if r1 != 'si':
            st.caption(f'No se usará {DEC.mes_corto(ultimo)}. La previsión será para ese mes.')
    utilizable = ultimo if r1 == 'si' else ultimo - 1
    r2, desde, hasta = 'no_lo_se', '', ''
    if utilizable >= primer:
        with st.container(border=True):
            opciones = list(COMPLETO.values())
            r2 = st.radio(f'¿Contienen tus archivos TODAS las ventas de {DEC.mes_corto(primer)} a {DEC.mes_corto(utilizable)}?', opciones,
                          index=opciones.index(COMPLETO[actual['completo']]), horizontal=True, key=_clave('completitud', vuelta, 'completo'))
            r2 = _invertir(COMPLETO)[r2]
            st.caption('Si respondes que sí, cuando un producto no tenga ventas en un mes contaré cero ventas (nunca antes de su '
                       'primera venta). Si no lo sabes, esos meses seguirán como desconocidos.')
            if r2 == 'parte':
                meses = [DEC.mes(i) for i in range(primer, utilizable + 1)]
                etiqueta = {DEC.mes(i): DEC.mes_corto(i) for i in range(primer, utilizable + 1)}
                a, b = st.columns(2)
                desde = a.selectbox('Primer mes completo', meses, index=meses.index(actual['desde']) if actual['desde'] in meses else 0,
                                    format_func=etiqueta.get, key=_clave('completitud', vuelta, 'desde'))
                hasta = b.selectbox('Último mes completo', meses, index=meses.index(actual['hasta']) if actual['hasta'] in meses else len(meses) - 1,
                                    format_func=etiqueta.get, key=_clave('completitud', vuelta, 'hasta'))
    else:
        st.caption('No quedan meses por los que preguntar si están completos.')
    texto_del_programa(contexto)
    atras, una, ok = botones('completitud', vuelta, anterior)
    if una:
        return ('una_a_una',)
    if atras:
        return ('atras', anterior)
    if ok:
        if r2 == 'parte' and DEC.indice(desde) > DEC.indice(hasta):
            st.error('El primer mes completo no puede ser posterior al último.')
            return None
        return ('confirmar', {'ultimo': r1, 'completo': r2, 'desde': desde if r2 == 'parte' else '', 'hasta': hasta if r2 == 'parte' else ''})
    return None


# ------------------------------------------------------------------ 4.5 productos

def productos(datos: dict, actual: dict | None, vuelta: int, contexto: str, descripciones: dict | None = None, anterior=None,
              cambiados: bool = False):
    todos = [x for g in datos['grupos'] for x in g['productos']]
    st.subheader(f'Productos sin ventas en los últimos meses ({len(todos)})')
    st.write('Si siguen a la venta, sus meses sin ventas cuentan como cero. Si no lo sabes, esos meses no se usan. '
             '**Ningún producto se da por retirado si no lo dices.**')
    if cambiados:
        st.info('La lista de productos ha cambiado con tus últimas respuestas: revísala. Lo que ya habías decidido se conserva.')
    actual = actual or {'grupos': {}, 'desde_grupo': {}, 'productos': {}, 'desde_producto': {}}
    ultimo = datos['ultimo_mes']
    marcar = _clave('productos', vuelta, 'marcar')
    a, b, _ = st.columns([2, 2, 3])
    if a.button('Todos siguen a la venta', key=_clave('productos', vuelta, 'todos_si')):
        st.session_state[marcar] = 'activo_confirmado'
    if b.button('Todos: no lo sé', key=_clave('productos', vuelta, 'todos_no')):
        st.session_state[marcar] = 'desconocido'
    en_bloque = st.session_state.get(marcar)
    grupos, desde_grupo, excepciones, desde_producto, errores = {}, {}, {}, {}, []
    for g in datos['grupos']:
        meses = [DEC.mes(i) for i in range(g['desde_minimo'], g['desde_maximo'] + 1)]
        etiqueta = {DEC.mes(i): DEC.mes_corto(i) for i in range(g['desde_minimo'], g['desde_maximo'] + 1)}
        with st.container(border=True):
            st.markdown(f'**{g["descripcion"]}**: {len(g["productos"])} {"producto" if len(g["productos"]) == 1 else "productos"}.')
            izquierda, derecha = st.columns(2)
            opciones = list(ESTADO.values())
            previo = en_bloque or actual['grupos'].get(g['clave'], 'desconocido')
            elegido = izquierda.selectbox(f'¿Siguen a la venta hoy (hasta {DEC.mes_corto(ultimo)})?', opciones, index=opciones.index(ESTADO[previo]),
                                          key=_clave('productos', vuelta, 'grupo', g['clave'], en_bloque or ''))
            grupos[g['clave']] = _invertir(ESTADO)[elegido]
            if grupos[g['clave']] == 'retirado_confirmado':
                previo_mes = actual['desde_grupo'].get(g['clave'], '')
                fecha = derecha.selectbox('Retirados desde', [''] + meses, index=([''] + meses).index(previo_mes) if previo_mes in meses else 0,
                                          format_func=lambda m: etiqueta.get(m, SIN_FECHA), key=_clave('productos', vuelta, 'desde', g['clave']),
                                          help='Si no lo sabes o cada uno es distinto, déjalo en «No lo sé»: los meses posteriores a su última '
                                               'venta quedarán como desconocidos.')
                if fecha:
                    desde_grupo[g['clave']] = fecha
            filas = []
            for x in g['productos']:
                previo = actual['productos'].get(x['codigo'])
                filas.append({'Producto': x['codigo']} | ({'Descripción': (descripciones or {}).get(x['codigo'], '')} if descripciones else {})
                             | {'Última venta': DEC.mes_corto(x['ultima']) if x['ultima'] is not None else '',
                                'Sin ventas desde hace': (f'{ultimo - x["ultima"]} {"mes" if ultimo - x["ultima"] == 1 else "meses"}'
                                                          if x['ultima'] is not None else ''),
                                'Este producto': ESTADO[previo] if previo else COMO_EL_GRUPO,
                                'Retirado desde': actual['desde_producto'].get(x['codigo'], '')})
            tabla = pd.DataFrame(filas)
            with st.expander(f'Ver {"el producto" if len(filas) == 1 else f"los {len(filas)} productos"}', expanded=len(filas) <= 10):
                editada = st.data_editor(
                    tabla, hide_index=True, width='stretch', key=_clave('productos', vuelta, 'tabla', g['clave']),
                    disabled=[c for c in tabla.columns if c not in ('Este producto', 'Retirado desde')],
                    column_config={'Este producto': st.column_config.SelectboxColumn('Este producto', options=[COMO_EL_GRUPO] + list(ESTADO.values()), required=True),
                                   'Retirado desde': st.column_config.SelectboxColumn('Retirado desde', options=[''] + meses,
                                                                                      help='Solo si lo marcas como retirado. Vacío: no lo sé.')})
            minimo = {x['codigo']: x for x in g['productos']}
            for codigo, este, fecha in zip(editada['Producto'], editada['Este producto'], editada['Retirado desde']):
                if este == COMO_EL_GRUPO:
                    continue
                if not minimo[codigo]['admite_excepcion']:
                    errores.append(f'El código «{codigo}» no se puede indicar uno a uno al programa (lleva comas o espacios al principio o al final).')
                    continue
                excepciones[codigo] = _invertir(ESTADO)[este]
                if excepciones[codigo] == 'retirado_confirmado' and fecha:
                    if DEC.indice(fecha) < minimo[codigo]['desde_minimo']:
                        errores.append(f'{codigo} tiene ventas en {DEC.mes_corto(minimo[codigo]["ultima"])}: no puede estar retirado desde '
                                       f'{DEC.mes_corto(DEC.indice(fecha))}.')
                    else:
                        desde_producto[codigo] = fecha
    texto_del_programa(contexto)
    atras, una, ok = botones('productos', vuelta, anterior)
    if una:
        return ('una_a_una',)
    if atras:
        return ('atras', anterior)
    if ok:
        if errores:
            for e in errores:
                st.error(e)
            return None
        st.session_state.pop(marcar, None)
        return ('confirmar', {'grupos': grupos, 'desde_grupo': desde_grupo, 'productos': excepciones, 'desde_producto': desde_producto})
    return None


# ------------------------------------------------------------------ 4.6 antes de calcular

def efecto_de_la_linea(linea: dict, datos: dict) -> str:
    """Qué deja sin conocer una línea pendiente, con lo que han calculado las funciones del propio programa."""
    if not linea['en_el_periodo']:
        return 'Es posterior al último mes que se usa: no afecta.'
    partes = []
    if datos['total']['situacion'] and datos['total']['situacion'][0] == 'historico_sin_confirmar':
        pass                    # sin confirmar que los datos estén completos, ningún mes tiene total: esta línea no cambia eso
    elif any(str(linea['fila']) in str(f['filas']).split(', ') for f in datos['impiden_el_total']):
        partes.append(f'No se conoce el total de {DEC.mes_corto(linea["mes"])}.')
    p = datos['productos'].get(linea['producto'])
    if p is not None:
        partes.append(f'{linea["producto"]} solo puede usar {p["meses_seguidos"]} meses seguidos' + (f', desde {p["desde"]}.' if p['desde'] else '.'))
    return ' '.join(partes)


def resumen(decisiones: DEC.Decisiones, carpeta) -> list[str]:
    """Una línea por sección con lo decidido."""
    out = []
    s = DEC.leer_datos(carpeta, 'series')
    if s and s['con_serie'] and decisiones.series:
        papel = decisiones.series['papel']
        conf = [f['serie'] for f in s['series'] if (decisiones.series['aceptar'] and f['propuesta']) or papel.get(f['serie'], 'desconocida') != 'desconocida']
        out.append(f'Series: {len(conf)} de {len(s["series"])} confirmadas.')
    l = DEC.leer_datos(carpeta, 'lineas')
    if l and decisiones.lineas:
        d = decisiones.lineas
        total = sin = 0
        for g in l['positivas']:
            for x in g['lineas']:
                total += 1
                sin += d['filas'].get(x['fila'], d['positivas'].get(g['serie'], 'desconocido')) == 'desconocido'
        for g in l['negativas']:
            for x in g['lineas']:
                total += 1
                sin += d['filas'].get(x['fila'], d['negativos'].get(g['grupo'], 'desconocido')) == 'desconocido'
        if total:
            out.append(f'Líneas: {total - sin} aclaradas' + (f', {sin} sin aclarar.' if sin else '.'))
    if decisiones.columnas is not None:
        out.append('Otras columnas: ' + ('revisadas una a una.' if decisiones.columnas.get('revisar') else 'no se usan.'))
    c = DEC.leer_datos(carpeta, 'antes_de_calcular')
    if c:
        if c['completo'] == 'si':
            out.append(f'Datos completos: sí, de {c["periodo"][0][5:7]}/{c["periodo"][0][:4]} a {c["periodo"][1][5:7]}/{c["periodo"][1][:4]}.')
        else:
            out.append('Datos completos: sin confirmar; los meses sin ventas siguen como desconocidos.')
        activos, retirados = c['estados']['activo_confirmado'], c['estados']['retirado_confirmado']
        if activos or retirados:
            out.append(f'Productos: {len(activos)} {"sigue" if len(activos) == 1 else "siguen"} a la venta'
                       + (f', {len(retirados)} {"retirado" if len(retirados) == 1 else "retirados"} ({", ".join(retirados[:10])}).' if retirados else '.'))
    return out


def antes_de_calcular(datos: dict, decisiones: DEC.Decisiones, carpeta, vuelta: int, problemas: list[str]):
    """Resumen, lo que queda sin aclarar y su efecto, y corregirlo aquí mismo.
    Devuelve ('calcular',), ('corregir', {fila: tratamiento}), ('revisar', sección), ('una_a_una_todo',) o None."""
    st.subheader('Antes de calcular')
    if problemas:
        st.error('**Lo que ha registrado el programa no coincide con lo que has decidido.** No se calcula nada así.\n\n- '
                 + '\n- '.join(problemas))
        if st.button('Contestar todo pregunta a pregunta', type='primary', key=_clave('antes', vuelta, 'una_todo')):
            return ('una_a_una_todo',)
        return None
    for linea in resumen(decisiones, carpeta):
        st.markdown(f'- {linea}')

    pendientes = [l for l in datos['pendientes'] if l['en_el_periodo']]
    lineas_datos = DEC.leer_datos(carpeta, 'lineas') or {'positivas': [], 'negativas': []}
    sin_anulacion = {x['fila'] for g in lineas_datos['negativas'] if g['sin_anulacion'] for x in g['lineas']}
    accion = None
    if pendientes or datos['series_sin_confirmar']:
        with st.container(border=True):
            st.markdown(f'#### Queda{"" if len(pendientes) == 1 else "n"} {len(pendientes)} '
                        f'{"operación" if len(pendientes) == 1 else "operaciones"} sin aclarar')
            sin_confirmar = bool(datos['total']['situacion']) and datos['total']['situacion'][0] == 'historico_sin_confirmar'
            if datos['total']['situacion']:
                st.write(datos['total']['situacion'][1])
            if datos['total']['aviso'] and not sin_confirmar:
                st.write(datos['total']['aviso'])
            if not sin_confirmar and any(m['motivo'] == 'operaciones_pendientes_de_interpretar' for m in datos['total']['meses_sin_total']):
                st.write(f'Con esto, el total solo puede usar **{datos["total"]["meses_seguidos"]} meses seguidos**'
                         + (f' (desde {datos["total"]["desde"]}).' if datos['total']['desde'] else '.')
                         + f' Para dar una previsión, el programa tiene que comprobarla en al menos {datos["evaluaciones_minimas"]} meses: '
                         'con poco historial, puede quedarse sin previsión.')
                st.caption('Aclarar una operación no garantiza que el total tenga previsión: eso depende de lo que el programa pueda '
                           'comprobar con los meses que queden. Lo sabrás al calcular.')
            negativas = [l for l in pendientes if l['clase'] == 'negativo_pendiente']
            positivas = [l for l in pendientes if l['clase'] == 'positivo_sin_confirmar']
            raras = [l for l in pendientes if l['clase'] == 'no_interpretable']
            cambios = {}
            for titulo, grupo, opciones in (('Líneas con cantidad negativa', negativas, TRATAMIENTO),
                                            ('Líneas positivas de series mixtas o de devoluciones', positivas, FILA_POSITIVA)):
                if not grupo:
                    continue
                st.markdown(f'**{titulo}**')
                if len(grupo) <= MAX_UNA_A_UNA:
                    # Pocas: cada una con su selector al lado, para corregirla sin buscarla en una tabla.
                    for l in grupo:
                        validas = [v for k, v in opciones.items() if not (k == 'anulacion' and l['fila'] in sin_anulacion)]
                        texto, eleccion = st.columns([3, 1])
                        texto.markdown(f'Fila {l["fila"]} · {l["producto"]} · {l["fecha"]} · {l["cantidad"]:g} unidades'
                                       + (f' · serie {l["serie"]}' if l['serie'] else '') + f'  \n{efecto_de_la_linea(l, datos)}')
                        q = eleccion.selectbox('Qué es', validas, index=validas.index('No lo sé'), key=_clave('antes', vuelta, 'fila', l['fila']))
                        if q != 'No lo sé':
                            cambios[int(l['fila'])] = _invertir(opciones)[q]
                    continue
                tabla = pd.DataFrame([_fila(l) | {'Qué es': 'No lo sé', 'Qué deja sin conocer': efecto_de_la_linea(l, datos)} for l in grupo])
                editada = st.data_editor(tabla, hide_index=True, width='stretch', key=_clave('antes', vuelta, titulo[:8]),
                                         disabled=[c for c in tabla.columns if c != 'Qué es'],
                                         column_config={'Qué es': st.column_config.SelectboxColumn('Qué es', options=list(opciones.values()), required=True),
                                                        'Qué deja sin conocer': st.column_config.TextColumn(width='large'),
                                                        'Cantidad': st.column_config.NumberColumn(format='%g')})
                for f, q in zip(editada['Fila'], editada['Qué es']):
                    if q != 'No lo sé':
                        cambios[int(f)] = _invertir(opciones)[q]
            if raras:
                st.caption(f'{len(raras)} línea(s) con una cantidad que no se puede leer: corrígelas en el archivo.')
            for s in datos['series_sin_confirmar']:
                st.write(f'La serie «{s["serie"]}» está sin confirmar: {s["lineas"]} líneas de {s["productos"]} productos, de {s["desde"]} a {s["hasta"]}.')
            a, b = st.columns([2, 3])
            if cambios and a.button('Aplicar los cambios', type='primary', key=_clave('antes', vuelta, 'aplicar')):
                malas = [f for f, t in cambios.items() if t == 'anulacion' and f in sin_anulacion]
                if malas:
                    st.error(f'La fila {malas[0]} no tiene ninguna venta a la que aplicar una anulación.')
                else:
                    accion = ('corregir', cambios)
            if datos['series_sin_confirmar'] and b.button('Revisar las series', key=_clave('antes', vuelta, 'series')):
                accion = ('revisar', 'series')
            st.caption('Al aplicar, tus respuestas se vuelven a pasar al programa desde el principio, sin preguntarte nada más.')
    else:
        st.success('No queda ninguna operación sin aclarar.')
    st.markdown('**Revisar:**')
    columnas = st.columns(4)
    for col, seccion in zip(columnas, ('series', 'lineas', 'completitud', 'productos')):
        if DEC.leer_datos(carpeta, {'series': 'series', 'lineas': 'lineas', 'completitud': 'completitud', 'productos': 'productos'}[seccion]) \
                and col.button(NOMBRES[seccion], key=_clave('antes', vuelta, 'rev', seccion)):
            accion = ('revisar', seccion)
    if st.button('Calcular previsiones →', type='primary', key=_clave('antes', vuelta, 'calcular')):
        accion = accion or ('calcular',)
    return accion
