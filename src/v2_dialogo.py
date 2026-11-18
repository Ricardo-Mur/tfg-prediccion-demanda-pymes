"""Versión 2, etapa 1: preguntas al usuario por terminal sobre completitud y estado de los productos.

Solo pregunta y traduce las respuestas; las reglas están en `v2_completitud.py`. Las funciones reciben
`leer` y `escribir` para poder probarse sin teclado y para que la futura interfaz web las sustituya.
La respuesta en blanco siempre es la prudente: «no lo sé».
Si no se indican `leer` y `escribir`, se usan el teclado y la pantalla en el momento de preguntar.
"""
from __future__ import annotations

from datetime import datetime
from typing import Callable

import pandas as pd

from . import v2_completitud as K

SI, NO, NO_SE = ('s', 'si', 'sí'), ('n', 'no'), ('d', '', 'no lo se', 'no lo sé', 'nose')
TEXTO_ULTIMO_MES = '¿Están ya en el archivo TODAS las ventas de {mes}?'
TEXTO_COMPLETITUD = '¿Contienen tus archivos TODAS las ventas de {inicio} a {fin}?'


def _mes(i: int) -> str:
    return f'{i % 12 + 1:02d}/{i // 12}'


def _respuesta(leer: Callable, escribir: Callable, mensaje: str, opciones: dict[str, tuple[str, ...]]) -> str:
    """Repite la pregunta hasta obtener una de las opciones. Devuelve la clave de la opción."""
    while True:
        texto = leer(mensaje).strip().casefold()
        for clave, validas in opciones.items():
            if texto in validas:
                return clave
        escribir('  No he entendido la respuesta. Escribe solo la letra de una de las opciones.')


def _leer_mes(leer: Callable, escribir: Callable, mensaje: str, minimo: int, maximo: int) -> int | None:
    """Pide un mes AAAA-MM dentro de un intervalo. En blanco: se abandona."""
    while True:
        texto = leer(mensaje).strip()
        if not texto:
            return None
        try:
            anio, mes = texto.replace('/', '-').split('-')
            if len(anio) == 2 and len(mes) == 4:      # admite también MM-AAAA
                anio, mes = mes, anio
            i = int(anio) * 12 + int(mes) - 1
            if not 1 <= int(mes) <= 12 or not minimo <= i <= maximo:
                raise ValueError
            return i
        except ValueError:
            escribir(f'  Escribe un mes entre {_mes(minimo)} y {_mes(maximo)} con el formato AAAA-MM.')


def pedir_confirmacion(primer_mes: int, ultimo_mes_archivo: int, indicio: K.IndicioUltimoMes | None = None, *,
                       leer: Callable | None = None, escribir: Callable | None = None,
                       ahora: datetime | None = None) -> K.ConfirmacionCompletitud:
    """Pregunta primero por el último mes y después por el histórico completo."""
    leer, escribir = leer or input, escribir or print
    escribir('\n¿ESTÁN COMPLETOS TUS DATOS?')
    escribir(f'El último mes de tus archivos es {_mes(ultimo_mes_archivo)}.')
    if indicio is not None:
        escribir('  ' + indicio.texto)
    pregunta_mes = TEXTO_ULTIMO_MES.format(mes=_mes(ultimo_mes_archivo))
    escribir(pregunta_mes)
    ultimo = _respuesta(leer, escribir, '[S] Sí, el mes está cerrado  [N] No, faltan ventas  [D] No lo sé: ',
                        {'si': SI, 'no': NO, 'no_lo_se': NO_SE})
    utilizable = ultimo_mes_archivo if ultimo == 'si' else ultimo_mes_archivo - 1
    if ultimo != 'si':
        escribir(f'  No usaré {_mes(ultimo_mes_archivo)}. La previsión será para ese mes.')
    if utilizable < primer_mes:
        return K.ConfirmacionCompletitud(ultimo_mes_completo=ultimo, pregunta_ultimo_mes=pregunta_mes,
                                         fecha_hora=(ahora or datetime.now()).isoformat(timespec='seconds'))
    pregunta = TEXTO_COMPLETITUD.format(inicio=_mes(primer_mes), fin=_mes(utilizable))
    escribir('\n' + pregunta)
    escribir('  Si respondes que sí, cuando un producto no tenga ventas en un mes contaré cero ventas')
    escribir('  (nunca antes de su primera venta). Si no lo sabes, esos meses seguirán como desconocidos.')
    r = _respuesta(leer, escribir, '[S] Sí, todo el periodo  [P] Solo una parte  [N] No  [D] No lo sé: ',
                   {'si': SI, 'parte': ('p',), 'no': NO, 'no_lo_se': NO_SE})
    inicio = fin = None
    if r == 'si':
        inicio, fin = primer_mes, utilizable
    elif r == 'parte':
        escribir('  Indica el periodo del que SÍ tienes todas las ventas (Intro para dejarlo sin confirmar).')
        inicio = _leer_mes(leer, escribir, '  Primer mes completo (AAAA-MM): ', primer_mes, utilizable)
        fin = None if inicio is None else _leer_mes(leer, escribir, '  Último mes completo (AAAA-MM): ',
                                                    inicio, utilizable)
        r = 'si' if inicio is not None and fin is not None else 'no_lo_se'
        if r == 'si':
            pregunta = TEXTO_COMPLETITUD.format(inicio=_mes(inicio), fin=_mes(fin))
    if r != 'si':
        escribir('  De acuerdo: no contaré ningún mes sin ventas como cero.')
    return K.ConfirmacionCompletitud(
        completo=r, periodo_inicio=K.fecha(inicio) if r == 'si' else None,
        periodo_fin=K.fecha(fin) if r == 'si' else None, ultimo_mes_completo=ultimo,
        fecha_hora=(ahora or datetime.now()).isoformat(timespec='seconds'),
        pregunta_completitud=pregunta, pregunta_ultimo_mes=pregunta_mes)


def preguntar_estado_productos(grupos: list[K.GrupoProductos], ultimo_mes: int, *,
                               ultimas: dict[str, int] | None = None, leer: Callable | None = None,
                               escribir: Callable | None = None, mostrar: int = 40) -> dict[str, dict]:
    """Pregunta por grupos si los productos han seguido a la venta, con excepciones y fechas de retirada.

    `ultimas`: último mes con operación de cada producto, para comprobar las fechas de retirada.
    Devuelve un diccionario con `respuestas`, `excepciones`, `retirada_por_grupo` y `retirada_por_producto`,
    listo para `v2_completitud.aplicar_respuestas(grupos, **resultado, ultimas=ultimas)`.
    """
    leer, escribir = leer or input, escribir or print
    ultimas = ultimas or {}
    out = {'respuestas': {}, 'excepciones': {}, 'retirada_por_grupo': {}, 'retirada_por_producto': {}}
    if not grupos:
        return out
    escribir('\nPRODUCTOS SIN VENTAS EN LOS ÚLTIMOS MESES')
    escribir('Si han seguido a la venta, contaré cero ventas en esos meses. Si no lo sabes, no usaré esos meses.')
    escribir('No daré ningún producto por retirado si tú no lo dices.')
    opciones = {'activo_confirmado': SI, 'retirado_confirmado': ('r',), 'desconocido': NO_SE, 'ver': ('v',)}
    for k, g in enumerate(grupos, 1):
        escribir(f'\nGrupo {k} de {len(grupos)}: {len(g.productos)} producto(s). {g.descripcion}.')
        escribir(f'¿Siguen a la venta hoy (hasta {_mes(ultimo_mes)})?')
        while True:
            r = _respuesta(leer, escribir,
                           '[S] Sí  [R] No, están retirados  [D] No lo sé  [V] Ver los códigos: ', opciones)
            if r != 'ver':
                break
            escribir('  ' + ', '.join(g.productos[:mostrar])
                     + (f' … y {len(g.productos) - mostrar} más' if len(g.productos) > mostrar else ''))
        out['respuestas'][g.clave] = r
        if r == 'retirado_confirmado':
            minimo = min((ultimas[p] for p in g.productos if p in ultimas), default=0) + 1
            escribir('  ¿Desde qué mes están retirados? Si todos lo están desde el mismo mes, escríbelo.')
            escribir('  Si no lo sabes o cada uno es distinto, pulsa Intro: los meses posteriores a su última')
            escribir('  venta quedarán como desconocidos, porque pudieron seguir a la venta parte de ese tiempo.')
            m = _leer_mes(leer, escribir, '  Retirados desde (AAAA-MM): ', minimo, ultimo_mes + 1)
            if m is not None:
                out['retirada_por_grupo'][g.clave] = K.fecha(m)
                tarde = [p for p in g.productos if ultimas.get(p, -1) >= m]
                if tarde:
                    escribir(f'  {len(tarde)} producto(s) del grupo tienen ventas en {_mes(m)} o después: '
                             'para ellos no aplico esa fecha.')
        texto = leer('¿Hay excepciones en este grupo? Escribe sus códigos separados por comas o pulsa Intro: ')
        for codigo in [c.strip() for c in texto.split(',') if c.strip()]:
            if codigo not in g.productos:
                escribir(f'  {codigo} no está en este grupo: no lo cambio.')
                continue
            e = _respuesta(leer, escribir, f'  {codigo}: [A] Sigue a la venta  [R] Retirado  [D] No lo sé: ',
                           {'activo_confirmado': ('a', *SI), 'retirado_confirmado': ('r',), 'desconocido': NO_SE})
            out['excepciones'][codigo] = e
            if e == 'retirado_confirmado':
                m = _leer_mes(leer, escribir, f'  {codigo}: retirado desde (AAAA-MM, Intro si no lo sabes): ',
                              ultimas.get(codigo, -1) + 1, ultimo_mes + 1)
                if m is not None:
                    out['retirada_por_producto'][codigo] = K.fecha(m)
    return out


def preparar_historico_con_usuario(objetivo, fechas=None, *, conjunto: dict | None = None,
                                   confirmaciones_guardadas=None, leer: Callable | None = None,
                                   escribir: Callable | None = None, meses_anulados=None,
                                   al_conocer_ultimo_mes: Callable | None = None,
                                   estados_de_columnas: Callable | None = None) -> tuple[K.HistoricoV2, dict]:
    """Hace las preguntas (o reutiliza unas confirmaciones guardadas) y construye el histórico.

    Devuelve el histórico y los datos que acompañan al registro auditable.
    `fechas`: fechas de las operaciones, solo para dar un indicio sobre el último mes.
    `conjunto`: identidad de los archivos (huella), para guardar con la confirmación y para comprobar que
    unas confirmaciones guardadas corresponden a estos mismos archivos.
    `al_conocer_ultimo_mes(ultimo mes completo)`: se llama en cuanto se sabe cuál es el último mes completo,
    antes de preguntar por los productos. Puede devolver (objetivo, meses_anulados) rehechos con lo que se
    conocía al cerrar ese mes; si devuelve None, se sigue con los que había.
    `estados_de_columnas(último mes con operación de cada producto, último mes completo)`: estados comerciales
    que salen de una columna del archivo cuyo significado ha confirmado el usuario (`v2_articulos`). Se aplican
    antes de preguntar por grupos, y por esos productos ya no se pregunta. Con confirmaciones guardadas no se
    llama: los estados guardados ya los incluyen.
    """
    leer, escribir = leer or input, escribir or print
    primer, ultimo_archivo = K.primer_mes_del_archivo(objetivo), K.ultimo_mes_del_archivo(objetivo)
    indicio = None
    if fechas is not None:
        try:
            indicio = K.diagnosticar_ultimo_mes(fechas)
            if K.indice(indicio.mes) != ultimo_archivo:
                indicio = None
        except ValueError:
            indicio = None
    grupos = []
    respuestas = {'respuestas': {}, 'excepciones': {}, 'retirada_por_grupo': {}, 'retirada_por_producto': {}}
    if confirmaciones_guardadas is not None:
        # Siempre se comprueba la huella: si no se conoce la de los archivos actuales, no se reutiliza nada.
        confirmacion, estados = K.cargar_registro(
            confirmaciones_guardadas, huella_conjunto=(conjunto or {}).get('huella_conjunto') or '')
        escribir(f'\nSe reutilizan las confirmaciones guardadas el {confirmacion.fecha_hora or "(sin fecha)"}.')
        utilizable = ultimo_archivo if confirmacion.ultimo_mes_completo == 'si' else ultimo_archivo - 1
        if al_conocer_ultimo_mes is not None:
            objetivo, meses_anulados = al_conocer_ultimo_mes(utilizable) or (objetivo, meses_anulados)
    else:
        confirmacion = pedir_confirmacion(primer, ultimo_archivo, indicio, leer=leer, escribir=escribir)
        estados = {}
        utilizable = ultimo_archivo if confirmacion.ultimo_mes_completo == 'si' else ultimo_archivo - 1
        if al_conocer_ultimo_mes is not None:
            objetivo, meses_anulados = al_conocer_ultimo_mes(utilizable) or (objetivo, meses_anulados)
        ultimas = K.ultimas_operaciones(objetivo, confirmacion)
        de_columnas = {}
        if estados_de_columnas is not None:
            en_el_archivo = set(objetivo.detalle['producto'])
            de_columnas = {p: e for p, e in estados_de_columnas(ultimas, utilizable).items() if p in en_el_archivo}
        if confirmacion.cubre(utilizable):
            grupos = K.proponer_grupos(objetivo, confirmacion)
            if de_columnas:
                # Por los productos cuyo estado ya sale de una columna confirmada no se vuelve a preguntar.
                grupos = [K.GrupoProductos(g.clave, g.descripcion, tuple(q for q in g.productos if q not in de_columnas))
                          for g in grupos]
                grupos = [g for g in grupos if g.productos]
                escribir(f'\nEl estado de {len(de_columnas)} producto(s) sale de la columna que has confirmado: '
                         'por ellos no pregunto.')
            respuestas = preguntar_estado_productos(grupos, utilizable, ultimas=ultimas, leer=leer, escribir=escribir)
            estados = K.aplicar_respuestas(grupos, **respuestas, ultimas=ultimas)
        estados = {**de_columnas, **estados}
    hist = K.construir_historico(objetivo, confirmacion, estados, meses_anulados=meses_anulados)
    return hist, {'conjunto': conjunto, 'grupos': grupos, 'indicio': indicio, **respuestas}


# ------------------------------------------------------------------ etapa 4: operaciones

LETRAS_TRATAMIENTO = {'a': 'anulacion', 'd': 'devolucion', 'j': 'ajuste', '': 'desconocido', 'n': 'desconocido'}
LETRAS_PAPEL = {'v': 'ventas', 'd': 'devoluciones', 'j': 'ajustes', 'm': 'mixta', '': 'desconocida', 'n': 'desconocida'}


def _preguntar_series(analisis, leer, escribir) -> list[tuple[str, str, str]]:
    """Qué es cada serie. Una serie sin confirmar no se usa: sus meses quedan pendientes."""
    from . import v2_operaciones as OP
    propuestas = OP.proponer_series(analisis)
    escribir(f'\nSERIES DEL ARCHIVO ({len(analisis.series)})')
    escribir('  Necesito saber qué contiene cada serie. Las que no confirmes no se usarán: sus meses quedarán pendientes.')
    for f in analisis.series.itertuples(index=False):
        propuesta = propuestas[f.serie][0]
        escribir(f'  Serie «{f.serie}»: {f.lineas} líneas ({f.positivas} positivas, {f.negativas} negativas, {f.cero} con cero), '
                 f'de {K.fecha(int(f.primer_mes)):%m/%Y} a {K.fecha(int(f.ultimo_mes)):%m/%Y}. '
                 + (f'Propuesta: {OP.TEXTO_PAPEL[propuesta]} ({propuestas[f.serie][1]}).' if propuesta else
                    f'Sin propuesta: {propuestas[f.serie][1]}.'))
    con_propuesta = [x for x in analisis.series.serie if propuestas[x][0]]
    aceptadas = bool(con_propuesta) and leer(
        f'  ¿Aceptas las {len(con_propuesta)} propuestas? (s = sí; n o Intro = te pregunto serie a serie): ').strip().lower()[:1] == 's'
    series = []
    if not aceptadas or len(con_propuesta) < len(analisis.series):
        escribir('    V = ventas a clientes   D = devoluciones o abonos   J = ajustes (no son ventas)')
        escribir('    M = mixta (ventas y otras operaciones)   Intro = no lo sé (no se usa)')
    for serie in analisis.series.serie:
        if aceptadas and propuestas[serie][0]:
            series.append((serie, propuestas[serie][0], 'propuesta del programa aceptada por el usuario'))
            continue
        while True:
            r = leer(f'  Serie «{serie}» (V/D/J/M/Intro): ').strip().lower()[:1]
            if r in LETRAS_PAPEL:
                break
            escribir('    Responde con una de las letras indicadas, o pulsa Intro.')
        series.append((serie, LETRAS_PAPEL[r], 'indicada por el usuario' if LETRAS_PAPEL[r] != 'desconocida' else 'sin confirmar'))
    return series


def _mostrar_lineas(lineas, todas, escribir, maximo: int = 40) -> None:
    """Líneas de un grupo, sin ningún dato de clientes."""
    for r in lineas.head(maximo).itertuples(index=False):
        texto = (f'    Fila {r.fila + 1}: {r.orden:%d/%m/%Y}, producto {r.producto}, {r.cantidad:g} unidades'
                 + (f', serie {r.serie}' if r.serie else '') + (f', documento {r.id_factura}' if r.id_factura is not None and not pd.isna(r.id_factura) else ''))
        if not pd.isna(r.fila_venta):
            v = todas.iloc[int(r.fila_venta)]
            texto += f'. Posible venta: fila {int(r.fila_venta) + 1}, {v.orden:%d/%m/%Y}, {v.cantidad:g} unidades'
        escribir(texto + '.')
    if len(lineas) > maximo:
        escribir(f'    ... y {len(lineas) - maximo} más (todas en operaciones_v2.csv).')


def _preguntar_excepciones(lineas, todas, letras: dict, opciones: str, leer, escribir) -> list[tuple[int, str]]:
    """Líneas concretas del grupo que son otra cosa. `letras`: respuesta -> tratamiento."""
    filas = {int(f) + 1 for f in lineas.fila}
    while True:
        r = leer('    ¿Alguna línea del grupo es otra cosa? (V = ver las líneas; números de fila separados por '
                 'comas; Intro = ninguna): ').strip().lower()
        if r != 'v':
            break
        _mostrar_lineas(lineas, todas, escribir)
    excepciones = []
    for trozo in filter(None, (x.strip() for x in r.split(','))):
        if not trozo.isdigit() or int(trozo) not in filas:
            escribir(f'    La fila {trozo} no está en este grupo: se ignora.')
            continue
        while True:
            t = leer(f'    Fila {trozo}: ¿qué es? ({opciones}): ').strip().lower()[:1]
            if t in letras:
                break
            escribir('    Responde con una de las letras indicadas, o pulsa Intro.')
        excepciones.append((int(trozo), letras[t]))
    return excepciones


def preguntar_operaciones(analisis, *, huella_conjunto: str = '', leer: Callable | None = None,
                          escribir: Callable | None = None):
    """Pregunta qué es cada serie y qué son las líneas negativas, por grupos y con excepciones.

    En blanco es siempre «no lo sé»: una serie sin confirmar no se usa y una línea negativa sin confirmar
    deja su mes pendiente. Si el archivo no tiene series ni líneas negativas, no pregunta nada.
    """
    from . import v2_operaciones as OP
    leer, escribir = leer or input, escribir or print
    ahora = datetime.now().isoformat(timespec='seconds')
    series = _preguntar_series(analisis, leer, escribir) if analisis.con_serie else []
    reglas, excepciones, positivas = [], [], []
    # Líneas positivas de las series mixtas o de devoluciones: la etiqueta de la serie no dice qué son.
    por_confirmar = OP.positivas_a_confirmar(analisis, OP.ReglasOperaciones(series=tuple(series)))
    if len(por_confirmar):
        escribir('\nLÍNEAS POSITIVAS DE SERIES MIXTAS O DE DEVOLUCIONES')
        escribir('  Pueden ser ventas o ajustes positivos. Si no lo sabes, quedan pendientes y sus meses no se usan.')
        letras = {'v': 'venta', 'j': 'ajuste', '': 'desconocido', 'n': 'desconocido'}
        for serie, del_grupo in por_confirmar.groupby('serie', sort=False):
            while True:
                r = leer(f'  Serie «{serie}»: {len(del_grupo)} línea(s) con cantidad positiva o cero. ¿Qué son? '
                         '(V = ventas a clientes; J = ajustes; Intro = no lo sé): ').strip().lower()[:1]
                if r in letras:
                    break
                escribir('    Responde con una de las letras indicadas, o pulsa Intro.')
            positivas.append((serie, {'venta': 'ventas', 'ajuste': 'ajustes', 'desconocido': 'desconocido'}[letras[r]]))
            sin_cero = del_grupo[del_grupo.cantidad > 0]
            if len(sin_cero):
                excepciones += _preguntar_excepciones(sin_cero, analisis.lineas, letras, 'V/J/Intro', leer, escribir)
    provisional = OP.ReglasOperaciones(series=tuple(series), positivas=tuple(positivas))
    negativas = OP.negativas_a_confirmar(analisis, provisional)
    if len(negativas):
        escribir(f'\nLÍNEAS CON CANTIDAD NEGATIVA ({len(negativas)})')
        escribir('  No se restan ni se descartan sin saber qué son. Para cada una he buscado ventas a las que')
        escribir('  PUEDA corresponder; es una posible correspondencia, no una prueba. Para cada grupo, dime qué son:')
        escribir('    A = anulaciones o correcciones: esa venta no llegó a existir. Se quita de la venta a la que anula.')
        escribir('    D = devoluciones: la venta existió y después se devolvió género. La venta se mantiene y la')
        escribir('        devolución se cuenta aparte.')
        escribir('    J = ajustes: no son ventas ni devoluciones. No cuentan.')
        escribir('    Intro = no lo sé: esos meses siguen pendientes y no se usan.')
        for g in OP.GRUPOS:
            del_grupo = negativas[negativas.grupo == g]
            if del_grupo.empty:
                continue
            sin_venta = g == 'sin_venta_anterior'
            letras = {k: v for k, v in LETRAS_TRATAMIENTO.items() if not (sin_venta and k == 'a')}
            opciones = 'D/J/Intro' if sin_venta else 'A/D/J/Intro'
            otro_mes = int((del_grupo.fila_venta.dropna().map(lambda i: analisis.lineas.mes.iat[int(i)])
                            < del_grupo.mes[del_grupo.fila_venta.notna()]).sum())
            nota = f' ({otro_mes} corresponderían a una venta de un mes anterior)' if otro_mes else ''
            while True:
                r = leer(f'  En {len(del_grupo)} línea(s) {OP.TEXTO_GRUPO[g]}{nota}. ¿Qué son? ({opciones}): ').strip().lower()[:1]
                if r in letras:
                    break
                escribir('    Responde con una de las letras indicadas, o pulsa Intro.')
            reglas.append((g, letras[r]))
            excepciones += _preguntar_excepciones(del_grupo, analisis.lineas, letras, opciones, leer, escribir)
    return OP.ReglasOperaciones(tuple(reglas), tuple(series), tuple(excepciones), tuple(positivas), ahora, huella_conjunto)


# ------------------------------------------------------------------ etapa 5: columnas del archivo de familias

def _numero_de_columna(leer, escribir, mensaje: str, n: int, excluida: int | None = None) -> int | None:
    """Número de columna entre 1 y n (distinto de `excluida`). En blanco: se abandona."""
    while True:
        r = leer(mensaje).strip()
        if not r:
            return None
        if r.isdigit() and 1 <= int(r) <= n and int(r) != excluida:
            return int(r)
        escribir(f'    Escribe un número del 1 al {n}' + (', distinto del anterior' if excluida else '') + ', o pulsa Intro.')


def preguntar_columnas_de_familias(identificacion, archivo: str, *, leer: Callable | None = None,
                                   escribir: Callable | None = None) -> tuple[str, str, str] | None:
    """Pide al usuario que confirme qué columna del archivo de familias es el producto y cuál la familia.

    `identificacion`: `v2_agregados.ColumnasDeFamilias`, cuando la identificación por cabeceras no es inequívoca.
    Devuelve (columna del producto, columna de la familia, cómo se identificaron), o None si el usuario no las
    confirma: en ese caso el archivo no se usa. En pantalla solo salen nombres de columna y recuentos.
    """
    from . import v2_agregados as AG
    leer, escribir = leer or input, escribir or print
    columnas = identificacion.columnas
    escribir(f'\nARCHIVO DE FAMILIAS «{archivo}»')
    escribir(f'  No puedo saber con seguridad qué columna identifica el producto y cuál la familia: {identificacion.motivo}.')
    escribir('  No las asigno por su posición. Columnas del archivo:')
    parece = {'producto': 'el producto', 'familia': 'la familia', 'cliente o persona': 'un dato de clientes o de personas'}
    for k, f in enumerate(columnas.itertuples(index=False), 1):
        texto = f'    {k}. «{f.columna}»: {f.valores_distintos} valores distintos'
        if f.codigos_de_las_ventas is not None and not pd.isna(f.codigos_de_las_ventas):
            texto += f'; {int(f.codigos_de_las_ventas)} son códigos de producto de tus ventas'
        escribir(texto + (f' (por el nombre, parece {parece[f.parece]})' if f.parece else '') + '.')
    sin_usar = '  No uso este archivo: sin saber qué es cada columna no agrupo los productos con él.'
    p, f = identificacion.propuesta_producto, identificacion.propuesta_familia
    if p is not None and f is not None:
        escribir(f'  Propuesta: el producto es «{p}» y la familia, «{f}».')
        r = leer('  ¿Es correcto? (s = sí; n = las indico yo; Intro = no usar este archivo): ').strip().lower()[:1]
        if r == 's':
            return p, f, AG.IDENTIFICACIONES[2]
        if r != 'n':
            escribir(sin_usar)
            return None
    nombres = list(columnas.columna)
    i = _numero_de_columna(leer, escribir, '  ¿Qué columna identifica el PRODUCTO? Escribe su número (Intro = no usar este archivo): ', len(nombres))
    j = None if i is None else _numero_de_columna(
        leer, escribir, '  ¿Qué columna identifica la FAMILIA? Escribe su número (Intro = no usar este archivo): ', len(nombres), i)
    if i is None or j is None:
        escribir(sin_usar)
        return None
    return nombres[i - 1], nombres[j - 1], AG.IDENTIFICACIONES[1]


# ------------------------------------------------------------------ etapa 4: estado, agrupaciones, importes y existencias

def _letra(leer, escribir, mensaje: str, letras: dict[str, str]) -> str:
    while True:
        r = leer(mensaje).strip().lower()[:1]
        if r in letras:
            return letras[r]
        escribir('    Responde con una de las letras indicadas, o pulsa Intro.')


def preguntar_columnas(analisis, *, huella_conjunto: str = '', leer: Callable | None = None,
                       escribir: Callable | None = None):
    """Pregunta qué son las columnas de estado, agrupación, fecha de alta, importes y existencias.

    `analisis`: `v2_articulos.ColumnasDelArticulo`. En blanco es siempre «no lo sé»: esa columna no se usa.
    Si no hay ninguna columna de este tipo, o el usuario prefiere no revisarlas, no se usa ninguna.
    """
    from . import v2_articulos as ART
    from . import v2_importes as IM
    leer, escribir = leer or input, escribir or print
    ahora = datetime.now().isoformat(timespec='seconds')
    estados = [a for a in analisis.estado if a.utilizable]
    agrupables = [a for a in analisis.atributos if a.propuesta == 'posible_agrupacion']
    importes = [a for a in analisis.importes if a.que_parece != 'no_numerico']
    if not (estados or agrupables or analisis.fechas or importes or analisis.existencias):
        return ART.ConfirmacionColumnas(fecha_hora=ahora, huella_conjunto=huella_conjunto)
    escribir('\nOTRAS COLUMNAS DEL ARCHIVO')
    partes = [(len(estados), 'de estado del artículo'), (len(agrupables), 'para agrupar productos'),
              (len(analisis.fechas), 'con una fecha fija por producto'), (len(importes), 'de importes'),
              (len(analisis.existencias), 'que podrían ser existencias')]
    escribir('  He reconocido columnas ' + ', '.join(f'{texto} ({n})' for n, texto in partes if n) + '.')
    escribir('  No uso ninguna si no me confirmas qué es, y ninguna cambia las cantidades previstas.')
    if leer('  ¿Quieres revisarlas ahora? (s = sí; Intro = no, sigo sin usarlas): ').strip().lower()[:1] != 's':
        return ART.ConfirmacionColumnas(fecha_hora=ahora, huella_conjunto=huella_conjunto)
    estado, excepciones, retirados_recientes = [], [], False
    for a in analisis.estado:
        if not a.utilizable:
            escribir(f'\n  «{a.columna}» no se puede usar como estado del artículo. {a.motivo}')
    for a in estados:
        ultimo = _mes(a.ultimo_mes_archivo)
        escribir(f'\n  Estado del artículo: columna «{a.columna}», con datos de {_mes(a.primer_mes)} a {_mes(a.ultimo_mes)}.')
        if a.sin_dato:
            escribir(f'    {a.sin_dato} producto(s) solo aparecen en archivos sin esta columna: quedan sin dato.')
        for f in a.valores.itertuples(index=False):
            escribir(f'    {ART._valor_texto(f.valor)}: {f.productos} productos; {f.vendieron_el_ultimo_mes} vendieron en {ultimo}, '
                     f'{f.vendieron_en_los_ultimos_3} en los últimos 3 meses y {f.mas_de_12_meses_sin_ventas} llevan más de 12 '
                     'meses sin ventas.')
        escribir('    El nombre de la columna no basta para saber qué indica. Dime qué significa cada valor:')
        escribir('    S = el artículo sigue a la venta   R = el artículo está retirado   Intro = no lo sé, u otra cosa (no se usa)')
        con_ventas = 0
        for f in a.valores.itertuples(index=False):
            s = _letra(leer, escribir, f'    {ART._valor_texto(f.valor)} (S/R/Intro): ',
                       {'s': 'activo', 'r': 'retirado', '': 'desconocido', 'n': 'desconocido'})
            estado.append((a.columna, f.valor, s))
            if s == 'retirado':
                con_ventas += int(f.vendieron_el_ultimo_mes)
        if con_ventas:
            escribir(f'    {con_ventas} de los productos que quedarían como retirados tienen ventas en {ultimo}.')
            retirados_recientes = retirados_recientes or leer(
                '    ¿Los doy también por retirados? (s = sí; Intro = no, los dejo sin confirmar): ').strip().lower()[:1] == 's'
    if any(s != 'desconocido' for _, _, s in estado):
        conocidos = {p for a in estados for p in a.por_producto}
        texto = leer('    ¿Algún producto es una excepción? Escribe sus códigos separados por comas o pulsa Intro: ')
        for codigo in [c.strip() for c in texto.split(',') if c.strip()]:
            if codigo not in conocidos:
                escribir(f'    {codigo} no tiene dato en esa columna: no lo cambio.')
                continue
            excepciones.append((codigo, _respuesta(leer, escribir, f'    {codigo}: [A] Sigue a la venta  [R] Retirado  [D] No lo sé: ',
                                                   {'activo_confirmado': ('a', *SI), 'retirado_confirmado': ('r',), 'desconocido': NO_SE})))
    agrupaciones, principal = [], ''
    if agrupables:
        escribir('\n  Columnas con un valor fijo por producto, que podrían servir para agrupar:')
        for a in agrupables:
            donde = '' if a.archivos_con_dato == a.archivos else (f' (solo en {a.archivos_con_dato} de los {a.archivos} archivos en '
                                                                   'que aparece; en los demás no describe al producto)')
            escribir(f'    «{a.columna}»{donde}: {a.grupos} grupos para {a.productos_con_dato} de {a.productos} productos '
                     f'(el {100 * a.parte_de_las_unidades:.0f} % de las unidades). El mayor reúne el {100 * a.mayor_grupo:.0f} % '
                     f'de esos productos; {a.grupos_de_un_producto} grupos tienen un solo producto.')
        for a in analisis.atributos:
            if a.propuesta != 'posible_agrupacion':
                escribir(f'    «{a.columna}»: poco útil para agrupar. {a.motivo}')
        escribir('    F = familia o categoría   M = marca o proveedor   O = otra agrupación que quiero conservar   Intro = no usar')
        for a in agrupables:
            tipo = _letra(leer, escribir, f'    «{a.columna}» (F/M/O/Intro): ',
                          {'f': 'familia', 'm': 'marca_o_proveedor', 'o': 'otra', '': '', 'n': ''})
            if tipo:
                agrupaciones.append((a.columna, tipo))
        if agrupaciones:
            principal = next((c for c, t in agrupaciones if t == 'familia'), agrupaciones[0][0])
            if len(agrupaciones) > 1:
                nombres = [c for c, _ in agrupaciones]
                r = leer(f'    ¿Cuál uso como agrupación principal? Escribe su nombre (Intro = «{principal}»): ').strip()
                principal = next((c for c in nombres if c.casefold() == r.casefold()), principal)
    fechas = []
    for c in analisis.fechas:
        if leer(f'\n  «{c}» es una fecha fija por producto. ¿Es la fecha de alta del artículo? (s = sí; Intro = no lo sé): '
                ).strip().lower()[:1] == 's':
            fechas.append(c)
    confirmados = []
    if importes:
        escribir('\n  Columnas de importes:')
        for a in importes:
            escribir(f'    «{a.columna}»: {IM.TEXTO_PARECE[a.que_parece]}. Datos de {_mes(a.primer_mes)} a {_mes(a.ultimo_mes)} '
                     f'({a.meses_con_dato} de los {a.meses_del_archivo} meses). Líneas vendidas con un 0 escrito: '
                     f'{a.ceros_con_venta}; sin dato: {a.sin_dato_con_venta}.' + (f' {a.motivo}' if a.motivo else ''))
        escribir('    Necesito que me digas tú cuál es el importe neto y cuál el bruto: no lo deduzco de los valores.')
        escribir('    N = importe NETO de la línea (lo que se cobra, ya con descuentos)')
        escribir('    B = importe BRUTO de la línea (antes de descuentos)')
        escribir('    U = precio neto de una unidad   T = precio de una unidad antes de descuentos (tarifa)')
        escribir('    C = coste   D = descuento   Intro = no usar')
        letras = {'n': 'importe_neto', 'b': 'importe_bruto', 'u': 'precio_neto', 't': 'precio_bruto', 'c': 'coste',
                  'd': 'descuento', '': ''}
        for a in importes:
            tipo = _letra(leer, escribir, f'    «{a.columna}» (N/B/U/T/C/D/Intro): ', letras)
            for tipos, nombre in ((ART.TIPOS_NETO, 'neto'), (ART.TIPOS_BRUTO, 'bruto')):
                ya = next((c for c, t in confirmados if t in tipos), None)
                if tipo in tipos and ya is not None:
                    escribir(f'    Ya has indicado «{ya}» como importe {nombre}: «{a.columna}» no se usa.')
                    tipo = ''
            if tipo:
                confirmados.append((a.columna, tipo))
        neto = next(((c, t) for c, t in confirmados if t in ART.TIPOS_NETO), None)
        bruto = next(((c, t) for c, t in confirmados if t in ART.TIPOS_BRUTO), None)
        if bruto and not neto:
            escribir('    Sin un importe neto no calculo precios: el bruto solo sirve para el descuento.')
        if neto and bruto and analisis.comparar_importes is not None:
            c = analisis.comparar_importes(neto, bruto)
            escribir(f'    De {c["lineas_vendidas"]} líneas vendidas, {c["con_neto"]} traen «{neto[0]}», {c["con_bruto"]} traen '
                     f'«{bruto[0]}» y {c["con_los_dos"]} traen los dos. El descuento solo se calcula con estas últimas.')
            if not c['coherente']:
                escribir(f'    Atención: «{bruto[0]}» es menor que «{neto[0]}» en el {100 * (1 - c["bruto_no_menor_que_neto"]):.0f} % '
                         'de las líneas que traen los dos. Un importe bruto no debería ser menor que el neto: puede que las '
                         'columnas no sean lo que has indicado. No calcularé ningún descuento.')
    existencias = []
    for a in analisis.existencias:
        escribir(f'\n  «{a.columna}»: {a.explicacion}')
        pregunta = ('¿Son las existencias de cada artículo al exportar el archivo?' if a.tipo == 'un_valor_por_producto' else
                    '¿Es lo que había en el almacén en el momento de cada venta?')
        if leer(f'    {pregunta} (s = sí; Intro = no lo sé): ').strip().lower()[:1] == 's':
            existencias.append((a.columna, 'existencias_al_exportar' if a.tipo == 'un_valor_por_producto' else 'existencias_en_cada_venta'))
    return ART.ConfirmacionColumnas(
        estado=tuple(estado), retirados_con_ventas_recientes=retirados_recientes, excepciones_estado=tuple(excepciones),
        agrupaciones=tuple(agrupaciones), agrupacion_principal=principal, fechas_de_alta=tuple(fechas),
        importes=tuple(confirmados), existencias=tuple(existencias), fecha_hora=ahora, huella_conjunto=huella_conjunto)
