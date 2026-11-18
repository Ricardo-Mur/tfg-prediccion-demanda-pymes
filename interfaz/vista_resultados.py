"""Pantallas de resultados (bloque 3): productos, ficha, familias, total, avisos y descargas.

Solo dibuja. Lo que enseña lo prepara `interfaz/resultados.py` a partir de los archivos del programa congelado.
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from interfaz import resultados as R

AZUL, NARANJA = '#3A5DA8', '#D9822B'
POR_QUE = {'buen_respaldo': 'Por qué tiene buen respaldo', 'orientativa': 'Por qué es orientativa',
           'no_recomendada': 'Por qué no se recomienda', 'datos_insuficientes': 'Qué falta'}
AYUDA_ERROR = ('Las diferencias entre lo previsto y lo vendido en los meses comprobados, como porcentaje de las unidades '
               'vendidas en esos meses.')
AYUDA_MESES = 'Meses pasados en los que se ha comparado la previsión que habría dado la aplicación con lo que se vendió.'


def miles(n: int) -> str:
    return f'{int(n):,}'.replace(',', '.')


# ------------------------------------------------------------------ piezas

def tarjetas_de_estado(recuento: dict[str, int], prefijo: str = '') -> None:
    for columna, estado in zip(st.columns(4), R.ESTADOS):
        columna.metric(f'{prefijo}{R.etiqueta_de_estado(estado)}', miles(recuento.get(estado, 0)))
        columna.caption(R.BREVE[estado])


def grafico(ficha: R.Ficha) -> None:
    datos = R.serie_para_el_grafico(ficha)
    if not len(datos) or datos.unidades.notna().sum() < 2:
        return
    series = [s for s in ('Vendido', 'Lo que se habría previsto') if s in set(datos.serie)]
    if ficha.estado == 'no_recomendada' and len(series) > 1:     # lo previsto en meses pasados no es una cantidad para el mes que viene
        st.info(R.NOTA_GRAFICO_NO_RECOMENDADA.replace('{mes}', ficha.mes))
    base = alt.Chart(datos).encode(
        x=alt.X('mes:T', title=None, axis=alt.Axis(format='%m/%Y', labelAngle=0, tickCount=8, grid=False)),
        y=alt.Y('unidades:Q', title='Unidades', scale=alt.Scale(zero=True)),
        color=alt.Color('serie:N', title=None, scale=alt.Scale(domain=series, range=[AZUL, NARANJA][:len(series)]),
                        legend=alt.Legend(orient='top') if len(series) > 1 else None),
        tooltip=[alt.Tooltip('mes:T', title='Mes', format='%m/%Y'), alt.Tooltip('serie:N', title='Qué es'),
                 alt.Tooltip('unidades:Q', title='Unidades', format=',.1f')])
    lineas = base.mark_line(strokeWidth=2)
    puntos = base.transform_filter(alt.datum.unidades != None).mark_point(size=42, filled=True).encode(  # noqa: E711
        shape=alt.Shape('serie:N', legend=None, scale=alt.Scale(domain=series, range=['circle', 'diamond'][:len(series)])))
    st.altair_chart((lineas + puntos).properties(height=260), width='stretch')
    pie = 'Lo vendido cada mes'
    if len(series) > 1:
        pie += f' y lo que la aplicación habría previsto en los {ficha.meses_comprobados} meses comprobados'
    st.caption(pie + '. Un mes que no se conoce queda en blanco: no se dibuja como cero.')


def dibujar_ficha(ficha: R.Ficha, descripcion: str = '', clave: str = '') -> None:
    """La tarjeta de un producto, de una familia o del total. Sin cantidad ni rango si no se recomienda."""
    with st.container(border=True):
        titulo, estado = st.columns([3, 2])
        titulo.markdown(f'#### {ficha.nombre}' + (f' · {descripcion}' if descripcion else ''))
        estado.markdown(f'#### {R.etiqueta_de_estado(ficha.estado)}')
        if ficha.con_cantidad:
            cifra = f'**Previsión para {ficha.mes}: {ficha.cantidad} unidades**'
            if ficha.estado == 'orientativa':
                st.warning(f'{cifra}\n\n{ficha.frase}')
            else:
                st.success(f'{cifra}\n\n{ficha.frase}')
            if ficha.rango:
                st.write(f'**Rango orientativo:** entre {ficha.rango[0]} y {ficha.rango[1]} unidades. {R.NOTA_RANGO}'
                         + (' Es un rango amplio: la cifra es poco precisa.' if ficha.rango_amplio else ''))
        elif ficha.estado == 'no_recomendada':
            st.error(f'**Sin cantidad recomendada para {ficha.mes}.** {ficha.frase}')
        else:
            st.info(f'**Sin previsión para {ficha.mes}.** {ficha.frase}')
        for aviso in ficha.avisos:
            st.warning(aviso)

        st.markdown(f'**{POR_QUE[ficha.estado]}**')
        frases = ficha.explicacion or ([ficha.motivo] if ficha.motivo else [])
        for frase in frases:
            st.write(frase)
        if ficha.estado == 'buen_respaldo':
            st.caption(R.NOTA_BUEN_RESPALDO)
        if ficha.error is not None or ficha.meses_comprobados:
            a, b, _ = st.columns([1, 1, 2])
            a.metric('Error histórico', ficha.error or '—', help=AYUDA_ERROR)
            b.metric('Meses comprobados', ficha.meses_comprobados, help=AYUDA_MESES)
        grafico(ficha)

        if ficha.sobre_los_datos:
            st.markdown('**Sobre los datos**')
            for frase in ficha.sobre_los_datos:
                st.write(f'- {frase}')
        if ficha.comparacion:
            with st.expander('Cómo le fue frente a repetir el mes anterior (dato del pasado)'):
                st.write(ficha.comparacion)
                st.caption(R.ADVERTENCIA_MES_ANTERIOR)
        if ficha.tecnico or ficha.cifra_no_recomendada:
            with st.expander('Detalle técnico'):
                if ficha.cifra_no_recomendada:          # el aviso, antes de cualquier cifra del texto del programa
                    st.warning(f'{R.AVISO_TECNICO}\n\n{ficha.cifra_no_recomendada}')
                for nombre, texto in ficha.tecnico:
                    st.write(f'**{nombre}.** {texto}')


def tabla_que_se_pulsa(tabla: pd.DataFrame, clave: str) -> int | None:
    """Enseña la tabla y devuelve la fila elegida con el ratón, si la hay."""
    evento = st.dataframe(tabla.drop(columns=['_estado']), hide_index=True, width='stretch', on_select='rerun',
                          selection_mode='single-row', key=clave)
    try:
        filas = list(evento.selection.rows)
    except AttributeError:
        filas = []
    return filas[0] if filas else None


# ------------------------------------------------------------------ pestañas

def pestana_productos(r: R.Resultados, descripciones: dict[str, str]) -> None:
    tarjetas_de_estado(R.recuento(r.prevision))
    st.caption('Las previsiones «no recomendadas» y con «datos insuficientes» no llevan cantidad: con ellas no conviene decidir una compra.')
    tabla = R.tabla_de_productos(r, descripciones)
    buscar, mostrar = st.columns([2, 3])
    texto = buscar.text_input('Buscar', placeholder='Código o nombre', key='r_buscar')
    elegidos = mostrar.multiselect('Mostrar', R.ESTADOS, format_func=R.etiqueta_de_estado, placeholder='Todos los estados', key='r_estados')
    visible = R.filtrar(tabla, texto, elegidos).reset_index(drop=True)
    st.caption(f'{miles(len(visible))} de {miles(len(tabla))} productos.' + ('' if len(visible) else ' Ninguno coincide con la búsqueda.'))
    fila = tabla_que_se_pulsa(visible, 'r_tabla') if len(visible) else None
    codigos = visible['Producto'].tolist()
    if not codigos:
        return
    pulsado = codigos[fila] if fila is not None and fila < len(codigos) else None
    if pulsado is not None and st.session_state.get('r_ultimo_pulsado') != pulsado:
        st.session_state['r_ultimo_pulsado'] = pulsado
        st.session_state['r_producto'] = pulsado                     # la fila pulsada manda sobre el desplegable
    if st.session_state.get('r_producto') not in codigos:
        st.session_state.pop('r_producto', None)
    estados = dict(zip(tabla['Producto'], tabla['_estado']))
    elegido = st.selectbox('Ver la ficha de', codigos, index=None, key='r_producto', placeholder='Elige un producto o pulsa una fila de la tabla',
                           format_func=lambda c: f'{c}' + (f' · {descripciones[c]}' if descripciones.get(c) else '') + f' — {R.NOMBRE[estados[c]]}')
    if elegido is not None:
        dibujar_ficha(R.ficha_de_producto(r, elegido), descripciones.get(elegido, ''))


def avisos_de_niveles() -> None:
    with st.expander('Cómo leer las familias y el total', expanded=False):
        for frase in R.NIVELES:
            st.write(f'- {frase}')


def pestana_familias(r: R.Resultados) -> None:
    info = R.familias_disponibles(r)
    grupos = r.familias
    if not info['hay'] or not len(grupos):
        st.info('**No hay previsiones por familia**, porque el programa no sabe a qué familia pertenece cada producto.')
        st.write('Hay dos formas de indicarlo:')
        st.write('- En el paso 1, subir un archivo con la familia de cada producto.')
        st.write('- En el paso de preguntas, revisar las otras columnas del archivo y confirmar cuál es la familia.')
        return
    st.write(f'Agrupación: {info["origen"]}.' if info['origen'] else 'Agrupación confirmada por ti.')
    st.caption(' '.join(R.NIVELES[:2]))
    tarjetas_de_estado(R.recuento(grupos), prefijo='')
    tabla = R.tabla_de_grupos(grupos)
    fila = tabla_que_se_pulsa(tabla, 'r_tabla_familias')
    con_cantidad = int(grupos.estado.isin(R.CON_CANTIDAD).sum())
    if 0 < con_cantidad < len(grupos):
        st.caption(f'Solo {con_cantidad} de los {len(grupos)} grupos tienen una cantidad recomendada: su suma no es un total.')
    if info['sin_familia']:
        st.caption(f'«{R.SIN_FAMILIA}» reúne los productos sin familia asignada: no es una familia.')
    nombres = tabla['Familia'].tolist()
    pulsado = nombres[fila] if fila is not None and fila < len(nombres) else None
    if pulsado is not None and st.session_state.get('r_ultima_familia_pulsada') != pulsado:
        st.session_state['r_ultima_familia_pulsada'] = pulsado
        st.session_state['r_familia'] = pulsado
    if st.session_state.get('r_familia') not in nombres:
        st.session_state.pop('r_familia', None)
    elegida = st.selectbox('Ver la ficha de', nombres, index=None, key='r_familia', placeholder='Elige una familia o pulsa una fila de la tabla')
    if elegida is not None:
        nivel = str(grupos[grupos.grupo.astype(str) == elegida].nivel.iloc[0])
        dibujar_ficha(R.ficha_de_grupo(r, nivel, elegida))
        sin_total = R.meses_sin_total(r, nivel, elegida)
        if sin_total:
            meses_sin_total(sin_total)


def meses_sin_total(filas: list[dict]) -> None:
    with st.expander(f'Meses cuyo total no se conoce ({len(filas)})'):
        st.dataframe(pd.DataFrame(filas), hide_index=True, width='stretch')
        st.caption('Un mes sin total no se rellena con cero ni con la suma de lo que hay: se queda sin dato.')


def pestana_total(r: R.Resultados) -> None:
    total = r.total
    if total is None:
        st.info('No hay previsión del total en los resultados.')
        return
    st.caption('El total es una serie propia: se prevé y se comprueba aparte. ' + R.NIVELES[1])
    dibujar_ficha(R.ficha_de_grupo(r, 'total', str(total.grupo)))
    sin_total = R.meses_sin_total(r, 'total', str(total.grupo))
    if sin_total:
        meses_sin_total(sin_total)


def pestana_avisos(r: R.Resultados, revisar_respuestas) -> None:
    st.write('Lo que puede mejorar el resultado. Casi todo se resuelve respondiendo a las preguntas pendientes.')
    if st.button('Revisar mis respuestas', key='r_revisar_avisos'):
        revisar_respuestas()
    c = R.situacion_de_completitud(r)
    st.subheader('Completitud')
    for frase in (c['texto'], c['ultimo_mes']):
        if frase:
            st.write(frase)
    if c['sin_confirmar']:
        st.warning(f'{miles(c["sin_confirmar"])} previsión(es) se han calculado con meses que no has confirmado como completos: ninguna de '
                   'ellas pasa de «orientativa».'
                   + (f' {miles(c["rebajadas"])} tendrían «buen respaldo» con los registros disponibles.' if c['rebajadas'] else ''))
    pendientes = R.operaciones_pendientes(r)
    st.subheader(f'Operaciones pendientes ({len(pendientes)})')
    if len(pendientes):
        st.write('Mientras no se aclaren, sus meses no se usan y el total de esos meses no se conoce.')
        st.dataframe(pendientes, hide_index=True, width='stretch')
        st.caption('El número de fila sirve para indicar una línea concreta en las preguntas («¿Alguna línea del grupo es otra cosa?»).')
    else:
        st.write('No queda ninguna operación sin interpretar.')
    cortados = R.productos_con_historial_cortado(r)
    if len(cortados):
        st.subheader(f'Productos que pierden historial ({len(cortados)})')
        st.write('Un mes sin aclarar deja sin usar los meses anteriores: los métodos trabajan con meses seguidos.')
        st.dataframe(cortados, hide_index=True, width='stretch')
    calidad = R.calidad(r)
    st.subheader(f'Calidad ({len(calidad)})')
    if len(calidad):
        st.write('Productos con líneas que coinciden con otra o con cantidades muy altas. Son avisos, no errores: no se ha quitado nada.')
        st.dataframe(calidad, hide_index=True, width='stretch')
    else:
        st.write('No hay líneas repetidas ni cantidades muy altas que revisar.')
    columnas = R.columnas_usadas(r)
    st.subheader(f'Columnas ({len(columnas)})')
    if len(columnas):
        st.dataframe(columnas, hide_index=True, width='stretch')
        st.caption('Solo nombres: aquí no se enseña ningún valor. Las columnas de clientes o de personas no se usan en las previsiones ni se exportan.')


def pestana_descargar(zip_de_resultados) -> None:
    st.write('Un ZIP con los archivos que ha escrito el programa, sin cambios.')
    st.download_button('Descargar los resultados (ZIP)', data=zip_de_resultados(), file_name='previsiones.zip', mime='application/zip',
                       type='primary', key='r_descargar')
    st.markdown('**Los tres más útiles**')
    for nombre, para_que in R.ARCHIVOS_UTILES:
        st.write(f'- `{nombre}`: {para_que}')
    st.caption(R.NOTA_EXCEL)
    st.caption(R.NOTA_CIFRAS)
    st.caption('Los archivos siguen en la copia temporal de esta sesión hasta que pulses «Terminar y borrar mis archivos».')


def pantalla(r: R.Resultados, *, descripciones: dict[str, str], zip_de_resultados, revisar_respuestas) -> None:
    # Algo más de ancho que el resto de las pantallas, para que la tabla quepa sin desplazarla.
    st.markdown('<style>.block-container, [data-testid="stMainBlockContainer"] {max-width: 1080px;}</style>', unsafe_allow_html=True)
    st.header(f'Previsión para {R.mes_largo(r.mes_previsto)}')
    c = R.situacion_de_completitud(r)
    if c['sin_confirmar']:
        st.warning('**No has confirmado que tus archivos traigan todas las ventas:** ninguna previsión pasa de «orientativa».')
        if st.button('Revisar respuestas', key='r_revisar'):
            revisar_respuestas()
    productos, familias, total, avisos, descargar = st.tabs(['Productos', 'Familias', 'Total', 'Avisos y pendientes', 'Descargar'])
    with productos:
        pestana_productos(r, descripciones)
    with familias:
        pestana_familias(r)
    with total:
        pestana_total(r)
    with avisos:
        pestana_avisos(r, revisar_respuestas)
    with descargar:
        pestana_descargar(zip_de_resultados)
