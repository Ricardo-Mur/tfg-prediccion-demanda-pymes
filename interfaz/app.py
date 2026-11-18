"""Interfaz de la versión 2: inicio, carga, columnas, revisión, preguntas, cálculo y resultados.

Uso, desde la carpeta del proyecto:   streamlit run interfaz/app.py      (o iniciar_interfaz.bat)

Es una capa aparte: no calcula ni decide nada, y no modifica el programa congelado. Las pantallas usan las
funciones de carga del programa (`interfaz/carga.py`). Los archivos del usuario se guardan en un espacio de
ejecución propio, fuera del proyecto (`interfaz/espacio.py`), que se borra al terminar.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

PROYECTO = Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True
if str(PROYECTO) not in sys.path:
    sys.path.insert(0, str(PROYECTO))

from interfaz import carga as C                                     # noqa: E402
from interfaz import decisiones as DEC                              # noqa: E402
from interfaz import revision as RV                                 # noqa: E402
from interfaz import ejemplo                                        # noqa: E402
from interfaz import resultados as R                                # noqa: E402
from interfaz import vista_resultados                               # noqa: E402
from interfaz.ejecucion import Ejecucion, registrar                 # noqa: E402
from interfaz.espacio import Espacio, EspacioNoValido, espacios_abandonados   # noqa: E402
from interfaz.guion import Paso                                     # noqa: E402
from interfaz.preguntas import (Dialogo, NoSePuedeResponder, carga_confirmada, columnas_que_admite_el_programa,   # noqa: E402
                                es_de_la_carga, etiqueta_de, ninguna_serie_confirmada, opciones_de_lista, plan_de_cambios,
                                presentar, vista)

PASOS = ('Inicio', 'Cargar', 'Columnas', 'Revisión', 'Aclarar datos', 'Resultados')
POR_DEFECTO = {'paso': 0, 'espacio': None, 'subidas': 0, 'elecciones': {}, 'leidos': {}, 'columnas': {}, 'propuestas': {},
               'resultado': None, 'resumen': None, 'es_ejemplo': False, 'aviso': '', 'confirmar_ejemplo': False, 'familias': '',
               'dialogo': None, 'respuestas': [], 'guion': None, 'calculo': '', 'vuelta': 0, 'resultados': None,
               'decisiones': None}
PREFIJOS_DE_CONTROLES = ('col_', 'fila_', 'hoja_', 'subida_', 'quitar_', 'p_', 'cambiar_', 'familias_', 'r_', 'rv_')
HORAS_SIN_ACTIVIDAD = 24
# Segundos sin gastar procesador ni escribir nada a partir de los que se avisa de que el programa parece bloqueado.
# Sin psutil no se puede medir el procesador y solo se miran los registros: hace falta esperar más.
SIN_ACTIVIDAD = {'preguntas': 60, 'preguntas_sin_psutil': 180, 'calculo': 180}
CORREGIR_CABECERAS = (
    '**Estas columnas no se pueden cambiar aquí.** El programa las cambia de una en una y, en este archivo, no queda '
    'ninguna columna libre para hacer el intercambio.\n\n'
    'Cómo resolverlo:\n\n'
    '1. Abre una copia del archivo y pon a cada columna el título de lo que contiene: «Fecha», «Producto», «Cantidad».\n'
    '2. Guarda la copia.\n'
    '3. Vuelve al paso 1 con «← Atrás», quita este archivo y sube la copia.\n\n'
    'Tu archivo original no se ha modificado.')
MESES = ('enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio', 'agosto', 'septiembre', 'octubre', 'noviembre',
         'diciembre')


def miles(n: int) -> str:
    return f'{int(n):,}'.replace(',', '.')


def cuantos(n: int, uno: str, varios: str) -> str:
    """«1 línea» o «6 líneas»."""
    return f'{miles(n)} {uno if int(n) == 1 else varios}'


# ------------------------------------------------------------------ estado de la sesión

@st.cache_resource
def _reloj_de_limpieza() -> dict:
    return {'ultima': 0.0}


def limpiar_sesiones_antiguas() -> int:
    """Borra los espacios de sesiones que nadie cerró: sin ejecución en curso y sin actividad desde hace 24 horas.
    Se repite, como mucho, una vez por hora mientras la aplicación está abierta, no solo al arrancarla. Una sesión
    en uso anota su actividad en cada pantalla (`Espacio.tocar`), así que no se borra."""
    reloj = _reloj_de_limpieza()
    if time.time() - reloj['ultima'] < 3600:
        return 0
    reloj['ultima'] = time.time()
    borrados = 0
    for viejo in espacios_abandonados(horas=HORAS_SIN_ACTIVIDAD):
        try:
            viejo.eliminar()
            borrados += 1
        except (EspacioNoValido, OSError):
            pass
    return borrados


def estado():
    for clave, valor in POR_DEFECTO.items():
        if clave not in st.session_state:
            st.session_state[clave] = type(valor)() if isinstance(valor, (dict, list)) else valor
    return st.session_state


def espacio() -> Espacio:
    s = estado()
    if s.espacio and Path(s.espacio).is_dir():
        actual = Espacio(Path(s.espacio))
        actual.tocar()
        return actual
    nuevo = Espacio.crear()
    s.espacio = str(nuevo.ruta)
    return nuevo


def invalidar(desde: str) -> None:
    """Si cambia algo de un paso, lo de los pasos siguientes deja de valer y se vuelve a pedir.
    `desde`: 'archivos', 'columnas' o 'preguntas'."""
    s = estado()
    if desde == 'archivos':
        s.leidos, s.columnas, s.propuestas = {}, {}, {}
    if desde in ('archivos', 'columnas'):
        s.resultado = s.resumen = None
    if s.dialogo is not None:
        s.dialogo.cerrar()
    elif s.espacio and Path(s.espacio).is_dir():
        Ejecucion(Espacio(Path(s.espacio))).cancelar()
    s.dialogo, s.respuestas, s.guion, s.calculo, s.resultados, s.decisiones = None, [], None, '', None, None
    for clave in [k for k in st.session_state if str(k).startswith(('r_', 'rv_'))]:   # controles de resultados y de la revisión
        st.session_state.pop(clave, None)


def borrar_todo() -> None:
    """Borra los archivos del usuario y todo lo demás de la sesión, y vuelve al inicio. Si algo no se puede borrar,
    queda anotado fuera del estado que se reinicia, para que el usuario lo vea hasta que se resuelva."""
    s = estado()
    sin_borrar = list(st.session_state.get('sin_borrar', []))
    if s.espacio and Path(s.espacio).is_dir():
        actual = Espacio(Path(s.espacio))
        try:
            Ejecucion(actual).cancelar()
            actual.eliminar()
        except (EspacioNoValido, OSError):
            sin_borrar.append(str(actual.ruta))
    for clave in [k for k in st.session_state if k in POR_DEFECTO or str(k).startswith(PREFIJOS_DE_CONTROLES)]:
        st.session_state.pop(clave, None)
    st.session_state['sin_borrar'] = sin_borrar


def reintentar_borrado() -> None:
    quedan = []
    for ruta in st.session_state.get('sin_borrar', []):
        if Path(ruta).exists():
            try:
                Ejecucion(Espacio(Path(ruta))).cancelar()
                Espacio(Path(ruta)).eliminar()
            except (EspacioNoValido, OSError):
                quedan.append(ruta)
    st.session_state['sin_borrar'] = quedan


def cargar_ejemplo() -> None:
    actual = espacio()
    ejemplo.tres_archivos(actual.entradas)
    estado().es_ejemplo = True
    invalidar('archivos')


def opciones_de_calculo() -> dict:
    s = estado()
    return {'procesos': max(1, min(4, os.cpu_count() or 1))} | ({'familias': s.familias} if s.familias else {})


def ir(paso: int) -> None:
    estado().paso = paso
    st.rerun()


# ------------------------------------------------------------------ lectura de los archivos de la sesión

def leido(archivo: Path) -> C.ArchivoLeido:
    s = estado()
    eleccion = s.elecciones.get(archivo.name, {})
    firma = (archivo.stat().st_size, archivo.stat().st_mtime_ns, repr(sorted(eleccion.items(), key=str)))
    guardado = s.leidos.get(archivo.name)
    if guardado is None or guardado[0] != firma:
        guardado = (firma, C.leer_archivo(archivo, eleccion.get('hoja'), eleccion.get('filas')))
        s.leidos[archivo.name] = guardado
    return guardado[1]


def tablas_de_la_sesion() -> list[C.Tabla]:
    """Todas las hojas de ventas, en el orden en que las lee el programa."""
    return [t for a in espacio().archivos_de_ventas() for t in leido(a).tablas]


def columnas_de(tabla: C.Tabla) -> dict[str, str]:
    """Columna elegida para cada dato (por su nombre). Empieza con la propuesta del programa."""
    s = estado()
    if tabla.nombre not in s.columnas:
        posiciones, revisar = C.propuesta_de_columnas(tabla.datos)
        s.propuestas[tabla.nombre] = revisar
        s.columnas[tabla.nombre] = {campo: str(tabla.datos.columns[i]) for campo, i in posiciones.items()}
    return s.columnas[tabla.nombre]


def problemas_de_columnas(tabla: C.Tabla, elegidas: dict[str, str]) -> list[str]:
    nombres = [str(c) for c in tabla.datos.columns]
    problemas = [f'Falta elegir la columna de «{C.NOMBRES[c].lower()}».' for c in C.CAMPOS if c not in elegidas]
    usadas = [v for v in elegidas.values()]
    for repetida in sorted({v for v in usadas if usadas.count(v) > 1}):
        problemas.append(f'La columna «{repetida}» está elegida para dos datos distintos.')
    for campo in C.CAMPOS:
        if campo in elegidas:
            bien, explicacion = C.columna_valida(tabla.datos, nombres.index(elegidas[campo]), campo)
            if not bien:
                problemas.append(f'«{elegidas[campo]}» no sirve como {C.NOMBRES[campo].lower()}: {explicacion}')
    if not problemas:                   # ¿se le pueden indicar al programa, que las cambia de una en una?
        posiciones, _ = C.propuesta_de_columnas(tabla.datos)
        try:
            plan_de_cambios({c: nombres[i] for c, i in posiciones.items()}, elegidas, nombres, *columnas_que_admite_el_programa(tabla.datos))
        except NoSePuedeResponder:
            problemas.append(CORREGIR_CABECERAS)
    return problemas


# ------------------------------------------------------------------ pantallas

def pantalla_inicio() -> None:
    st.title('Previsión de ventas del mes siguiente')
    st.write('Sube tus ventas y te digo, producto a producto, cuánto se puede prever para el mes que viene y hasta qué '
             'punto conviene fiarse de esa cifra.')
    izquierda, derecha = st.columns(2)
    with izquierda:
        st.subheader('Qué necesitas')
        st.write('Uno o varios archivos Excel o CSV con una fila por venta: **fecha**, **producto** y **cantidad**.')
    with derecha:
        st.subheader('Qué hago con tus archivos')
        st.write('No los modifico y **no salen de este ordenador**. Trabajo con una copia temporal que se borra al terminar.')
    with st.expander('Qué puedes esperar'):
        st.write('Es normal que muchos productos no tengan una previsión recomendada.')
        st.write('Esta aplicación se evaluó con las ventas de 8 tiendas de una misma cadena de supermercados de Estados '
                 'Unidos (11.520 previsiones de artículos). En ese estudio, 4 de cada 100 previsiones tuvieron «buen '
                 'respaldo», 41 fueron «orientativas» y 54 no se recomendaron.')
        st.caption('Son los resultados de ese estudio, con esas tiendas. No dicen qué pasará con los datos de otra empresa.')
    st.divider()
    uno, dos = st.columns(2)
    if uno.button('Empezar con mis archivos', type='primary', width='stretch'):
        ir(1)
    s = estado()
    if dos.button('Probar con datos de ejemplo', width='stretch'):
        if s.es_ejemplo:
            ir(1)
        if espacio().archivos_de_ventas():
            s.confirmar_ejemplo = True                  # hay archivos del usuario: el ejemplo no se mezcla con ellos
            st.rerun()
        cargar_ejemplo()
        ir(1)
    if s.confirmar_ejemplo:
        st.warning('Ya has subido archivos tuyos. Los datos de ejemplo no se mezclan con ellos: para probar el ejemplo hay '
                   'que borrar antes tus archivos de esta sesión.')
        borrar, seguir = st.columns(2)
        if borrar.button('Borrar mis archivos y cargar el ejemplo'):
            borrar_todo()
            cargar_ejemplo()
            ir(1)
        if seguir.button('Seguir con mis archivos'):
            s.confirmar_ejemplo = False
            ir(1)


def pantalla_cargar() -> None:
    s = estado()
    st.header('Sube tus archivos de ventas')
    if s.es_ejemplo:
        st.info('Estás usando los **datos de ejemplo**: tres archivos inventados de una tienda con 10 productos. No se '
                'mezclan con archivos tuyos: para usar los tuyos, quita antes el ejemplo.')
        if st.button('Quitar el ejemplo y empezar con mis archivos'):
            borrar_todo()
            ir(1)
    subidos = None if s.es_ejemplo else st.file_uploader('Excel (.xlsx) o CSV. Puedes subir varios a la vez.', type=['csv', 'xlsx'],
                               accept_multiple_files=True, key=f'subida_{s.subidas}')
    if subidos:
        actual = espacio()
        for archivo in subidos:
            contenido = archivo.getvalue()
            try:
                existente = next((p for p in actual.archivos_de_ventas() if p.name == Path(archivo.name).name), None)
                if existente is not None and existente.read_bytes() == contenido:
                    continue
                if existente is not None:
                    actual.quitar_ventas(existente.name)
                actual.guardar_ventas(archivo.name, contenido)
            except ValueError as e:
                s.aviso = str(e)
        invalidar('archivos')
        s.subidas += 1                                  # vacía el cuadro de subida: la lista de abajo es la que cuenta
        st.rerun()

    archivos = espacio().archivos_de_ventas()
    if not archivos:
        st.caption('Todavía no has subido ningún archivo.')
    todos_listos = bool(archivos)
    for k, archivo in enumerate(archivos):
        a = leido(archivo)
        todos_listos = todos_listos and a.listo
        with st.container(border=True):
            titulo, quitar = st.columns([5, 1])
            titulo.markdown(f'**{archivo.name}**')
            if not s.es_ejemplo and quitar.button('Quitar', key=f'quitar_{k}'):
                espacio().quitar_ventas(archivo.name)
                s.elecciones.pop(archivo.name, None)
                invalidar('archivos')
                st.rerun()
            if a.error:
                st.error(a.error)
                continue
            for t in a.tablas:
                donde = f'Hoja «{t.hoja}»' if t.hoja and len(a.tablas) + len(a.hojas_sin_incorporar) > 1 else 'Archivo'
                if not t.falta:
                    st.write(f'✔ {donde}: {cuantos(len(t.datos), "fila leída", "filas leídas")}.')
                elif t.falta == 'vacia':
                    st.error(f'{donde}: no tiene ninguna fila de datos debajo de la cabecera.')
                else:
                    st.warning(f'{donde}: no localizo con seguridad la fila con los nombres de las columnas.')
                    opciones = [f'Fila {f + 1}' for f in t.filas_posibles] + ['Otra fila']
                    eleccion = st.radio('¿En qué fila están los nombres de las columnas?', opciones, index=None,
                                        key=f'fila_{k}_{t.hoja}', horizontal=True)
                    numero = st.number_input('Número de fila', 1, 100, 1, key=f'fila_num_{k}_{t.hoja}') if eleccion == 'Otra fila' else None
                    if st.button('Usar esta fila', key=f'fila_ok_{k}_{t.hoja}', disabled=eleccion is None):
                        fila = int(numero) - 1 if eleccion == 'Otra fila' else t.filas_posibles[opciones.index(eleccion)]
                        s.elecciones.setdefault(archivo.name, {}).setdefault('filas', {})[t.hoja] = fila
                        invalidar('archivos')
                        st.rerun()
                    st.caption('Si no lo sabes, abre el archivo para comprobarlo o quítalo de la lista. No lo uso sin saberlo.')
            if a.hojas_para_elegir:
                st.warning('Este Excel tiene varias hojas y no reconozco con seguridad cuál es la de ventas.')
                hoja = st.radio('¿Cuál tiene las ventas?', a.hojas_para_elegir, index=None, key=f'hoja_{k}')
                if st.button('Usar esta hoja', key=f'hoja_ok_{k}', disabled=hoja is None):
                    s.elecciones.setdefault(archivo.name, {})['hoja'] = hoja
                    invalidar('archivos')
                    st.rerun()
            if a.hojas_sin_incorporar and not a.hojas_para_elegir:
                st.warning('No se han incorporado estas hojas, porque no parecen tablas de ventas: '
                           + ', '.join(f'«{h}»' for h in a.hojas_sin_incorporar)
                           + '. Si alguna tiene ventas, el histórico no estará completo.')
    if not s.es_ejemplo:
        familias_de_la_sesion()
    st.caption('Tus archivos originales no se modifican. Trabajo con una copia temporal en este ordenador.')
    atras, _, sigue = st.columns([1, 3, 1])
    if atras.button('← Atrás'):
        ir(0)
    if sigue.button('Continuar →', type='primary', disabled=not todos_listos):
        ir(2)
    if archivos and not todos_listos:
        st.caption('Para continuar, resuelve lo que falta en los archivos marcados o quítalos.')


def pantalla_columnas() -> None:
    s = estado()
    tablas = tablas_de_la_sesion()
    if not tablas:
        ir(1)
    st.header('¿Son estas las columnas?')
    st.write('Necesito saber cuál es la **fecha**, cuál el **producto** y cuál la **cantidad**. La descripción es opcional.')
    hay_problemas = False
    for i, t in enumerate(tablas):
        elegidas = columnas_de(t)
        antes = dict(elegidas)
        nombres = [str(c) for c in t.datos.columns]
        with st.container(border=True):
            st.markdown(f'**{t.nombre}**')
            for campo in C.CAMPOS_TODOS:
                opcional = campo not in C.CAMPOS
                etiqueta = C.NOMBRES[campo] + (' (opcional)' if opcional and '(opcional)' not in C.NOMBRES[campo] else '')
                eleccion = st.selectbox(etiqueta, nombres, index=nombres.index(elegidas[campo]) if elegidas.get(campo) in nombres else None,
                                        placeholder='Sin descripción' if opcional else 'Elige la columna', key=f'col_{i}_{campo}')
                if eleccion is None:
                    elegidas.pop(campo, None)
                else:
                    elegidas[campo] = eleccion
                if campo in s.propuestas.get(t.nombre, ()) and elegidas.get(campo) == antes.get(campo):
                    st.caption('⚠ Revisa esta especialmente: hay otra columna que se le parece.')
            if elegidas != antes:
                invalidar('columnas')
            problemas = problemas_de_columnas(t, elegidas)
            hay_problemas = hay_problemas or bool(problemas)
            for problema in problemas:
                st.warning(problema)
            if not problemas:
                vista, ocultas = C.vista_previa(t.datos, {c: nombres.index(v) for c, v in elegidas.items()})
                st.caption('Así se leen las primeras filas. Solo se enseñan las columnas que has elegido.')
                if len(vista.columns):
                    st.dataframe(vista, hide_index=True, width='stretch')
                for oculta in ocultas:
                    st.warning(f'No enseño los valores de «{oculta}»: por su nombre o por su contenido puede ser un dato de '
                               'clientes o de personas. Comprueba que es la columna correcta.')
    atras, no_se, sigue = st.columns([1, 2, 2])
    if atras.button('← Atrás'):
        ir(1)
    if no_se.button('No lo sé'):
        st.info('Sin confirmación no se analizan columnas que podrían ser incorrectas. Puedes abrir el archivo para '
                'comprobarlo y volver después: no se ha modificado nada.')
    if sigue.button('Sí, son estas columnas →', type='primary', disabled=hay_problemas):
        with st.spinner('Preparando tus ventas, sin modificar los archivos...'):
            posiciones = [{c: [str(x) for x in t.datos.columns].index(v) for c, v in columnas_de(t).items()} for t in tablas]
            s.resultado = C.preparar(tablas, posiciones)
            s.resumen = C.resumen(s.resultado)
        ir(3)


def pantalla_revision() -> None:
    s = estado()
    if s.resumen is None:
        ir(2)
    r = s.resumen
    st.header('Revisión de tus datos')
    st.info('**Esta revisión es preliminar.** Describe lo que hay en tus archivos tal como están. Hasta que no respondas a '
            'las preguntas del paso siguiente (qué son las líneas negativas, si los datos están completos) no se sabe qué '
            'meses se pueden usar ni qué productos se pueden prever.')
    periodo = ('—' if r['primer_dia'] is None else
               f'{MESES[r["primer_dia"].month - 1]} de {r["primer_dia"].year} a {MESES[r["ultimo_dia"].month - 1]} de {r["ultimo_dia"].year}')
    a, b, c = st.columns(3)
    a.metric('Registros leídos', miles(r['registros']))
    b.metric('Productos', miles(r['productos']))
    c.metric('Meses con registros', f'{r["meses_con_registros"]} de {r["meses_del_intervalo"]}')
    st.caption('Son registros, no ventas confirmadas: todavía puede haber devoluciones y anulaciones sin interpretar.')
    st.write(f'**Periodo:** {periodo}.' + (f' Se han unido {r["tablas"]} tablas de {r["archivos"]} archivos.' if r['tablas'] > 1 else ''))

    st.subheader('Lo que conviene que sepas')
    avisos = []
    if r['campos_sin_resolver']:
        avisos.append((f'{cuantos(r["campos_sin_resolver"], "dato", "datos")} de {cuantos(r["registros_sin_resolver"], "fila", "filas")} '
                       'no se han podido leer (fecha, producto o cantidad).', 'No los relleno ni supongo un valor. Pueden limitar las previsiones.'))
    if r['negativas']:
        avisos.append((f'{cuantos(r["negativas"], "línea", "líneas")} con cantidad negativa.',
                       'Pueden ser devoluciones, anulaciones o ajustes. No las resto ni las descarto: te pregunto qué son en el paso siguiente.'))
    if r['ceros']:
        avisos.append((f'{cuantos(r["ceros"], "línea", "líneas")} con cantidad cero.', 'Se conservan tal cual.'))
    if r['filas_identicas']:
        avisos.append((f'{cuantos(r["filas_identicas"], "fila idéntica a otra", "filas idénticas")}.', 'Pueden ser duplicados o no. No las quito.'))
    if r['muy_altas']:
        avisos.append((f'{cuantos(r["muy_altas"], "cantidad muy alta", "cantidades muy altas")} para su producto.', 'No las corrijo ni las quito.'))
    if r['nombres_distintos']:
        avisos.append((f'{cuantos(r["nombres_distintos"], "producto aparece", "productos aparecen")} con descripciones distintas.', 'Se tratan como un solo producto, por su código.'))
    sin_registros = r['meses_del_intervalo'] - r['meses_con_registros']
    if sin_registros > 0:
        avisos.append((f'{cuantos(sin_registros, "mes del periodo no tiene", "meses del periodo no tienen")} ningún registro en tus archivos.',
                       'No los cuento como cero ventas: de momento son meses sin datos.'))
    if r['con_huecos']:
        avisos.append((f'{cuantos(r["con_huecos"], "producto tiene", "productos tienen")} meses sin ventas entre su primera y su última venta.',
                       'Hasta que confirmes si tus archivos están completos, esos meses son desconocidos, no ceros.'))
    if r['solapes']:
        avisos.append((f'Hay periodos que se solapan entre {r["solapes"]} pares de archivos.', 'Un solapamiento no demuestra que haya duplicados.'))
    if r['coincidencias_entre_archivos']:
        avisos.append((f'{cuantos(r["coincidencias_entre_archivos"], "venta coincide", "ventas coinciden")} en fecha, producto y cantidad entre archivos distintos.',
                       'Conviene revisarlas: pueden estar dos veces.'))
    for que, que_hago in avisos:
        st.warning(f'**{que}** {que_hago}')
    if not avisos:
        st.success('No he visto nada que revisar en las fechas, los productos ni las cantidades.')

    st.subheader('¿Hay meses suficientes?')
    if r['meses_con_registros'] < 4:
        st.warning('**Histórico demasiado corto** para comprobar previsiones mensuales con fiabilidad. Aunque haya muchas '
                   'filas, hacen falta ventas de varios meses.')
    st.write('Productos según los meses en que aparecen en tus archivos:')
    for columna, (etiqueta, clave) in zip(st.columns(4), (('1 mes', 'en_un_mes'), ('2 o 3 meses', 'en_dos_o_tres'),
                                                           ('4 a 11 meses', 'en_cuatro_a_once'), ('12 o más', 'en_doce_o_mas'))):
        columna.metric(etiqueta, miles(r[clave]))
    minimo, respaldo = meses_necesarios()
    st.caption(f'Para evaluar una previsión hacen falta unos {minimo} meses seguidos con ventas conocidas, y {respaldo} para que '
               'pueda tener «buen respaldo». Esto cuenta los meses en que aparece cada producto: todavía no decide qué '
               'productos se pueden prever.')

    with st.expander(f'Otras columnas de tus archivos ({len(r["columnas"])}): cuáles se usan y cuáles no'):
        if r['aviso_columnas']:
            st.warning(r['aviso_columnas'])
        if r['columnas']:
            st.dataframe(r['columnas'], hide_index=True, width='stretch')
        st.caption('Solo nombres y recuentos: aquí no se enseña ningún valor de tus archivos. Las columnas de clientes o de '
                   'personas no se usan en las previsiones ni se exportan.')
    atras, _, sigue = st.columns([1, 3, 1])
    if atras.button('← Atrás'):
        ir(2)
    if sigue.button('Continuar →', type='primary'):
        ir(4)


@st.cache_data
def meses_necesarios() -> tuple[int, int]:
    """Meses seguidos que pide la configuración del programa: para evaluar y para poder tener buen respaldo."""
    from src.motor_v2 import ConfiguracionV2
    cfg = ConfiguracionV2()
    return cfg.cortes_minimos + 1 + cfg.evaluaciones_minimas, cfg.cortes_minimos + 1 + cfg.evaluaciones_buen_respaldo


def familias_de_la_sesion() -> None:
    """Archivo opcional con la familia de cada producto. Sus columnas las pregunta el programa en el paso 4."""
    s = estado()
    with st.expander('Tengo un archivo con la familia de cada producto (opcional)', expanded=bool(s.familias)):
        if s.familias:
            st.write(f'✔ **{s.familias}**. En el paso de preguntas se comprueba cuál de sus columnas es el producto y cuál la familia.')
            if st.button('Quitar este archivo', key='familias_quitar'):
                ruta = espacio().familias / s.familias
                if ruta.is_file() and not ruta.is_symlink():
                    ruta.unlink()
                s.familias = ''
                invalidar('preguntas')
                st.rerun()
        else:
            subido = st.file_uploader('Excel o CSV con una columna de producto y otra de familia.', type=['csv', 'xlsx'], key=f'familias_{s.subidas}')
            if subido is not None:
                try:
                    s.familias = espacio().guardar_familias(subido.name, subido.getvalue()).name
                    invalidar('preguntas')
                except ValueError as e:
                    s.aviso = str(e)
                s.subidas += 1
                st.rerun()


def reanudar(desde: int) -> None:
    """Vuelve a hacer las preguntas a partir de la n.º `desde`. Las anteriores se conservan: el programa arranca de
    nuevo y se le entregan, comprobando cada una con su pregunta."""
    s = estado()
    pasos = [Paso.desde_dict(d) for d in s.respuestas]
    de_la_carga = sum(es_de_la_carga(p) for p in pasos)
    desde = 0 if desde < de_la_carga else desde
    if s.dialogo is not None:
        s.dialogo.cerrar()
    s.respuestas, s.guion, s.calculo, s.dialogo = s.respuestas[:desde], None, '', None
    s.vuelta += 1
    st.rerun()


def decisiones_de_la_sesion() -> DEC.Decisiones:
    s = estado()
    if s.decisiones is None:
        s.decisiones = DEC.Decisiones()
    return s.decisiones


def carpeta_de_la_revision() -> Path:
    return espacio().ruta / 'revision'


def dialogo_de_la_sesion() -> Dialogo:
    """El programa en modo de preguntas. Las de la etapa 4 las contesta la revisión agrupada con lo decidido en las
    tablas; por eso siempre arranca desde el principio: cada vez se le vuelven a pasar todas las respuestas."""
    s = estado()
    if s.dialogo is None:
        tablas = tablas_de_la_sesion()
        carga = carga_confirmada(espacio().archivos_de_ventas(), tablas, [columnas_de(t) for t in tablas], s.elecciones)
        revision = DEC.Revision(carpeta_de_la_revision(), decisiones_de_la_sesion())
        s.dialogo = Dialogo(espacio(), opciones_de_calculo(), carga, revision=revision)
        s.dialogo.empezar()
    return s.dialogo


def rehacer() -> None:
    """Vuelve a pasar al programa todas las respuestas, desde el principio, con las decisiones actuales."""
    s = estado()
    if s.dialogo is not None:
        s.dialogo.cerrar()
    s.dialogo, s.guion, s.calculo, s.resultados = None, None, '', None
    s.vuelta += 1
    st.rerun()


def respuestas_dadas(pasos: list[Paso], cambiar: bool = True) -> None:
    del_usuario = [(k, p) for k, p in enumerate(pasos) if not es_de_la_carga(p)]
    if not del_usuario:
        return
    with st.expander(f'Lo que has respondido ({len(del_usuario)})', expanded=not cambiar):
        for k, p in del_usuario:
            contexto = '\n'.join(p.contexto)
            v = vista(p.pregunta[0], contexto, pasos[:k])
            grupo = next((l.split(':')[0] for l in v.datos if l.startswith('Grupo ')), '')        # «Grupo 2 de 3»
            que = (f'{grupo} · ' if grupo else '') + (v.titulo or p.pregunta[0])
            respuesta = next((e for e, r in opciones_de_lista(p.pregunta[0], contexto, pasos[:k]) if r == p.respuesta), etiqueta_de(p))
            if not cambiar:
                st.markdown(f'{que} → **{respuesta}**')
                continue
            texto, boton = st.columns([6, 1])
            texto.markdown(f'{que} → **{respuesta}**')
            if boton.button('Cambiar', key=f'cambiar_{k}'):
                reanudar(k)
        if cambiar:
            st.caption('Si cambias una respuesta, las posteriores se vuelven a preguntar.')


def umbral_de_bloqueo(fase: str) -> float:
    return float(os.environ.get('TFG_UMBRAL_BLOQUEO') or SIN_ACTIVIDAD[fase])


def aviso_de_bloqueo(ejecucion: Ejecucion, fase: str, reintentar) -> bool:
    """Si el proceso lleva demasiado sin hacer nada, lo dice y ofrece qué hacer, en lugar de seguir esperando sin fin.
    Devuelve True si ha avisado: entonces la pantalla deja de comprobar sola, para que los botones respondan enseguida."""
    a = ejecucion.sin_actividad()
    if fase == 'calculo' and not a['con_procesador']:
        return False            # el cálculo apenas escribe: sin medir el procesador no se distingue un bloqueo
    umbral = umbral_de_bloqueo(fase if a['con_procesador'] or fase == 'calculo' else 'preguntas_sin_psutil')
    if a['segundos'] < umbral:
        return False
    st.warning(f'**El programa no avanza.** Lleva {int(a["segundos"])} s sin '
               + ('usar el procesador ni ' if a['con_procesador'] else '') + 'escribir nada. Parece bloqueado. '
               + ('Con los datos de ejemplo, cada pregunta llega en unos segundos.' if fase == 'preguntas' else ''))
    if a['etapa']:
        st.caption(f'Última etapa anotada por el programa: {a["etapa"]}')
    reintento, espera, _ = st.columns([2, 2, 3])
    if reintento.button('Volver a intentarlo', type='primary', key=f'bloqueo_reintentar_{fase}'):
        registrar(ejecucion.espacio.ruta, 'el usuario pide volver a intentarlo')
        reintentar()
    if espera.button('Seguir esperando', key=f'bloqueo_esperar_{fase}'):
        ejecucion.seguir_esperando()
        st.rerun()
    with st.expander('Información para el diagnóstico'):
        st.caption('Solo etapas del programa y líneas de código. No hay ningún dato de tus archivos.')
        st.code(ejecucion.diagnostico() or 'El programa no ha anotado nada todavía.', language=None, wrap_lines=True)
        st.caption(f'Están en `{ejecucion.espacio.ruta}`: `diagnostico.txt` y `pilas.txt`.')
    return True


ANTERIOR = {'lineas': 'series', 'columnas': 'lineas', 'completitud': 'columnas', 'productos': 'completitud'}


def pantalla_preguntas() -> None:
    """Paso 4: la revisión agrupada. Una tabla por sección; lo que no es de ninguna sección se pregunta como antes."""
    s = estado()
    if s.resumen is None:
        ir(3)
    st.header('Revisa lo que ha encontrado el programa')
    st.caption('Si no sabes algo, déjalo en «No lo sé»: lo que no se sabe no se usa. Eso puede dejar productos sin previsión.')
    if s.guion is not None:             # el programa ya ha recibido todas las respuestas: no se le vuelve a preguntar
        pantalla_antes_de_calcular([Paso.desde_dict(d) for d in s.guion])
        return
    dialogo = dialogo_de_la_sesion()
    situacion = dialogo.situacion(espera=20)
    pasos = dialogo.grabado()
    s.respuestas = [p.a_dict() for p in pasos]
    tipo = situacion['tipo']
    dec, carpeta = decisiones_de_la_sesion(), carpeta_de_la_revision()
    series = DEC.leer_datos(carpeta, 'series')
    hay_series = series is None or series['con_serie']
    if tipo == 'seccion':
        seccion = situacion['seccion']
        RV.barra(seccion, dec, carpeta, hay_series)
        anterior = ANTERIOR.get(seccion)
        if anterior == 'series' and not hay_series:
            anterior = None
        datos = DEC.datos_de(seccion, carpeta) if seccion != 'columnas' else {}
        if datos is None:
            st.warning('No tengo los datos de esta parte. Contéstala pregunta a pregunta.')
            accion = ('una_a_una',)
        elif seccion == 'series':
            accion = RV.series(datos, dec.series, s.vuelta, situacion['contexto'])
        elif seccion == 'lineas':
            accion = RV.lineas(DEC.leer_datos(carpeta, 'lineas'), dec.lineas, s.vuelta, situacion['contexto'], anterior)
        elif seccion == 'columnas':
            accion = RV.columnas(s.vuelta, situacion['contexto'], anterior)
        elif seccion == 'completitud':
            accion = RV.completitud(datos, dec.completitud, s.vuelta, situacion['contexto'], anterior)
        else:
            accion = RV.productos(datos, dec.productos, s.vuelta, situacion['contexto'], descripciones_de_la_sesion(), anterior,
                                  cambiados=dec.productos is not None and 'productos' not in dec.revisar)
        if accion is not None:
            if accion[0] == 'confirmar':
                dec.confirmar(seccion, accion[1], carpeta)
                st.rerun()
            elif accion[0] == 'atras':
                dec.revisar.add(accion[1])
                rehacer()
            elif accion[0] == 'una_a_una':
                dec.una_a_una.add(seccion)
                st.rerun()
    elif tipo == 'pregunta':
        RV.barra(DEC.seccion_de(situacion['pregunta'], pasos) or '', dec, carpeta, hay_series)
        st.caption('Esta pregunta del programa se contesta aquí, una a una.')
        mostrar_pregunta(dialogo, situacion, pasos, anotar=lambda r: dialogo.revision.anotar_suelta(situacion['pregunta'], r, pasos))
    elif tipo == 'preguntas_completas':
        s.guion = s.respuestas
        st.rerun()
    elif tipo == 'trabajando':
        if not aviso_de_bloqueo(dialogo.ejecucion, 'preguntas', rehacer):
            st.info('El programa está leyendo tus archivos y recibiendo tus respuestas...')
            time.sleep(1)
            st.rerun()
    elif tipo == 'detenida' and situacion.get('motivo') == 'respuesta_rechazada' and pasos and not es_de_la_carga(pasos[-1]):
        ultima = DEC.seccion_de(pasos[-1].pregunta[0], pasos[:-1])
        if (ultima is None or ultima in dec.una_a_una) and dec.sueltas:      # la contestó el usuario, pregunta a pregunta
            dec.sueltas.pop()
            s.aviso = f'El programa no ha aceptado tu última respuesta. {situacion.get("detalle", "")} Te la vuelvo a preguntar.'
            rehacer()
        st.error('El programa no ha aceptado una respuesta preparada con tus tablas. No se ha calculado nada. '
                 f'({situacion.get("detalle", "")})')
        if st.button('Contestar todo pregunta a pregunta', type='primary'):
            dec.una_a_una.update(DEC.SECCIONES)
            dec.sueltas.clear()
            rehacer()
    elif tipo in ('no_se_puede', 'detenida') and (tipo == 'no_se_puede' or (pasos and es_de_la_carga(pasos[-1]))):
        st.error('No he podido pasar al programa, sin dudas, las columnas que elegiste. No me invento ninguna respuesta: revisa las '
                 f'columnas y vuelve a intentarlo. Motivo: {situacion.get("detalle", "")}')
        st.info(CORREGIR_CABECERAS.replace('Vuelve al paso 1 con «← Atrás»', 'Vuelve al paso 1'))
        if st.button('Volver a las columnas'):
            invalidar('preguntas')
            ir(2)
    else:
        st.error('El programa se ha detenido y no puede seguir con estas respuestas. No hay ningún resultado y no se ha modificado '
                 'ningún archivo.')
        if st.button('Volver a pasar las respuestas'):
            rehacer()
    if st.button('← Atrás', key='p_atras'):
        ir(3)


def pantalla_antes_de_calcular(pasos: list[Paso]) -> None:
    """Resumen de lo decidido, lo que queda sin aclarar y su efecto. Corregir algo vuelve a pasar las respuestas."""
    s = estado()
    dec, carpeta = decisiones_de_la_sesion(), carpeta_de_la_revision()
    series = DEC.leer_datos(carpeta, 'series')
    RV.barra('antes', dec, carpeta, series is None or series['con_serie'])
    datos = DEC.leer_datos(carpeta, 'antes_de_calcular')
    if datos is None:
        st.error('No tengo el resumen de lo que queda pendiente. Vuelve a pasar las respuestas.')
        if st.button('Volver a pasar las respuestas'):
            rehacer()
        return
    accion = RV.antes_de_calcular(datos, dec, carpeta, s.vuelta, DEC.comprobar(dec, carpeta))
    if accion is not None:
        if accion[0] == 'calcular':
            s.calculo = 'lanzar'
            ir(5)
        elif accion[0] == 'corregir':
            dec.lineas['filas'].update(accion[1])
            rehacer()
        elif accion[0] == 'revisar':
            dec.revisar.add(accion[1])
            rehacer()
        elif accion[0] == 'una_a_una_todo':
            dec.una_a_una.update(DEC.SECCIONES)
            dec.sueltas.clear()
            rehacer()
    with st.expander('Detalles técnicos: todas las respuestas que ha recibido el programa'):
        respuestas_dadas(pasos, cambiar=False)
        if dec.sueltas and st.button('Volver a contestar las preguntas sueltas'):
            dec.sueltas.clear()
            rehacer()
    if st.button('← Atrás', key='p_atras_antes'):
        ir(3)


def mostrar_pregunta(dialogo: Dialogo, situacion: dict, pasos: list[Paso], anotar=None) -> None:
    """La pregunta, las opciones y una explicación breve. El texto completo del programa queda en un desplegable.
    `anotar(respuesta)`: se llama antes de pasar la respuesta al programa (la revisión guarda las preguntas sueltas)."""
    s = estado()
    n, pregunta = int(situacion['n']), situacion['pregunta']
    v = vista(pregunta, situacion['contexto'], pasos)
    with st.container(border=True):
        st.markdown(f'#### {v.titulo}' if v.titulo else '#### Pregunta del programa')
        if v.sobre:
            st.caption(f'Sobre lo que acabas de responder: {v.sobre}')
        if v.datos:
            with st.container(height=260) if v.es_lista and len(v.datos) > 10 else st.container():
                st.text('\n'.join(v.datos))
        if v.recortadas:
            st.caption(f'Hay {cuantos(v.recortadas, "línea más", "líneas más")} en el texto completo del programa.')
        presentacion = presentar(pregunta)
        de_lista = opciones_de_lista(pregunta, situacion['contexto'], pasos)
        opciones = presentacion.opciones + de_lista
        se_escribe = bool(presentacion.texto) and not de_lista
        clave = f'p_{s.vuelta}_{n}'
        etiquetas = [e for e, _ in opciones] + ([presentacion.texto] if se_escribe and opciones else [])
        eleccion = st.radio('Tu respuesta', etiquetas, index=0, key=clave) if etiquetas else None
        escribe = se_escribe and (not opciones or eleccion == presentacion.texto)
        texto = st.text_input(presentacion.texto, key=clave + '_texto') if escribe else None
        if v.breve:
            st.caption(v.breve)
        if not presentacion.catalogada:
            st.caption('Esta pregunta del programa no tiene opciones preparadas. Responde tal como la pide; en blanco es «no lo sé».')
        if st.button('Responder', type='primary', key=clave + '_ok'):
            if escribe and opciones and not texto.strip():
                st.warning('Escribe la respuesta o elige otra opción.')
            else:
                respuesta = texto.strip() if escribe else dict(opciones)[eleccion]
                if anotar is not None:
                    anotar(respuesta)
                dialogo.responder(n, respuesta)
                st.rerun()
        with st.expander('Texto completo del programa'):
            st.text(v.completo)
            st.caption('Es lo que escribe el programa antes de esta pregunta, tal cual, con las letras que usa en el terminal.')


NOMBRES_DE_ESTADO = (('buen_respaldo', 'Buen respaldo'), ('orientativa', 'Orientativa'), ('no_recomendada', 'No recomendada'),
                     ('datos_insuficientes', 'Datos insuficientes'))


def pantalla_calculo() -> None:
    s = estado()
    if s.guion is None:
        ir(4)
    ejecucion = Ejecucion(espacio())
    if not (s.calculo == 'lanzado' and not ejecucion.en_curso() and ejecucion.hay_resultados()):
        st.header('Cálculo de las previsiones')
    if s.calculo == 'lanzar':
        s.calculo = 'lanzado'
        ejecucion.lanzar([Paso.desde_dict(d) for d in s.guion], opciones_de_calculo())     # nunca dos a la vez, ni dos veces lo mismo
    if ejecucion.en_curso():
        segundos = int(time.time() - float(ejecucion.estado().get('inicio', time.time())))
        st.info('**Calculando.** El programa comprueba cada método con tus últimos 12 meses antes de prever el siguiente. '
                'Con muchos productos puede tardar varios minutos.')
        st.write(f'Llevo {segundos // 60} min {segundos % 60} s. El programa no informa de su avance, así que no hay barra de progreso.')
        if st.button('Cancelar el cálculo'):
            ejecucion.cancelar()
            s.calculo = 'cancelado'
            st.rerun()

        def otra_vez():
            ejecucion.cancelar()
            s.calculo = 'lanzar'
            st.rerun()
        if not aviso_de_bloqueo(ejecucion, 'calculo', otra_vez):
            time.sleep(1)
            st.rerun()
    final = ejecucion.estado()
    if s.calculo == 'lanzado' and ejecucion.hay_resultados():
        resumen_del_calculo()
    elif s.calculo in ('cancelado', ''):
        st.warning('El cálculo se canceló. No se ha guardado ningún resultado.' if s.calculo == 'cancelado' else 'Todavía no se ha calculado nada.')
        if st.button('Calcular previsiones', type='primary'):
            s.calculo = 'lanzar'
            st.rerun()
    else:
        st.error('El cálculo se ha detenido y no hay ningún resultado. No se ha modificado ningún archivo. '
                 f'({final.get("motivo") or final.get("estado") or "sin detalle"})')
        if st.button('Volver a las preguntas'):
            rehacer()
    if st.button('← Volver a la revisión'):
        ir(4)


def descripciones_de_la_sesion() -> dict[str, str]:
    """La descripción de cada producto, solo para la pantalla y solo si el usuario confirmó esa columna en todos los
    archivos y no parece un dato de personas. No se exporta."""
    s = estado()
    if s.resultado is None:
        return {}
    try:
        for t in tablas_de_la_sesion():
            elegida = columnas_de(t).get('descripcion')
            if elegida is None or C.parece_personal(t.datos, [str(c) for c in t.datos.columns].index(elegida)):
                return {}
        datos = s.resultado['limpio'].datos
        if 'descripcion' not in datos.columns:
            return {}
        d = datos[['producto', 'descripcion']].dropna()
        d = d[d.descripcion.astype(str).str.strip() != '']
        return {str(p): str(g.iloc[0]) for p, g in d.groupby(d.producto.astype(str), sort=False).descripcion}
    except Exception:                   # sin descripciones se puede seguir: solo acompañan al código
        return {}


def resultados_de_la_sesion() -> R.Resultados:
    s = estado()
    carpeta = espacio().carpeta_de_resultados()
    if s.resultados is None or s.resultados[0] != str(carpeta):
        s.resultados = (str(carpeta), R.leer(carpeta), descripciones_de_la_sesion())
    return s.resultados[1]


def revisar_respuestas() -> None:
    ir(4)


def zip_de_la_sesion() -> bytes:
    """El ZIP de los resultados. Se prepara una sola vez por cálculo, no cada vez que se dibuja la pantalla."""
    s = estado()
    if len(s.resultados) < 4:
        s.resultados = (*s.resultados, espacio().zip_de_resultados())
    return s.resultados[3]


def resumen_del_calculo() -> None:
    r = resultados_de_la_sesion()
    vista_resultados.pantalla(r, descripciones=estado().resultados[2], zip_de_resultados=zip_de_la_sesion, revisar_respuestas=revisar_respuestas)


def barra_lateral() -> None:
    s = estado()
    with st.sidebar:
        st.markdown('### Pasos')
        for n, nombre in enumerate(PASOS[1:], start=1):
            marca = '▶' if n == s.paso else ('✔' if n < s.paso else '○')
            st.markdown(f'{marca} {n}. {nombre}' if n != s.paso else f'**{marca} {n}. {nombre}**')
        if s.es_ejemplo:
            st.info('Datos de ejemplo')
        st.divider()
        st.caption('Tus archivos se quedan en este ordenador: la aplicación no los envía a ningún sitio. Trabajo con una copia temporal.')
        if st.button('Terminar y borrar mis archivos'):
            borrar_todo()
            st.rerun()
        st.caption('Para borrar ahora esa copia y los resultados, pulsa el botón. Cerrar la pestaña no los borra: en ese caso se '
                   f'borran cuando la aplicación vuelva a usarse, pasadas {HORAS_SIN_ACTIVIDAD} horas sin actividad.')
        if st.session_state.get('sin_borrar'):
            st.error('**No he podido borrar los archivos temporales.** Siguen en:\n\n' + '\n\n'.join(f'`{r}`' for r in st.session_state['sin_borrar'])
                     + '\n\nPuedes reintentarlo o borrar esa carpeta a mano.')
            if st.button('Reintentar el borrado'):
                reintentar_borrado()
                st.rerun()


def main() -> None:
    st.set_page_config(page_title='Previsión de ventas', layout='centered')
    limpiar_sesiones_antiguas()
    s = estado()
    barra_lateral()
    if s.aviso:
        st.warning(s.aviso)
        s.aviso = ''
    try:
        (pantalla_inicio, pantalla_cargar, pantalla_columnas, pantalla_revision, pantalla_preguntas, pantalla_calculo)[s.paso]()
    except EspacioNoValido as e:
        st.error(f'No puedo preparar la aplicación: {e}')
    except Exception as e:                      # el mensaje de un error inesperado puede llevar datos: no se enseña ni se anota
        st.error(f'Ha ocurrido un error inesperado ({type(e).__name__}). No se ha modificado ningún archivo. Puedes volver '
                 'atrás o terminar y empezar de nuevo.')


if __name__ == '__main__':            # así se ejecuta con `streamlit run`; importado como módulo, no dibuja nada
    main()
