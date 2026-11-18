"""Las preguntas del programa, llevadas a la pantalla.

La interfaz no reconstruye las preguntas: las hace el propio programa congelado, que se ejecuta en modo de
preguntas dentro del espacio de la sesión (`lanzador.py`). Cada pregunta llega aquí con su texto y con lo que el
programa había escrito antes; la respuesta vuelve al programa y queda grabada unida a esa pregunta. Cuando el
programa ya no pregunta más, se detiene sin calcular: lo grabado es el guion del cálculo.

Hay dos clases de preguntas:
  - Las de la carga (qué archivos, qué fila, qué hoja, qué columnas). El usuario ya las contestó en las pantallas
    1 y 2: `RespuestasDeLaCarga` las contesta con eso, y comprueba que el programa acaba mostrando exactamente
    las columnas elegidas. Si no puede responder algo sin ninguna duda, no responde: se detiene.
    El programa cambia las columnas de una en una y no admite una que ya esté asignada a otro dato. Cuando el
    usuario ha intercambiado dos, `siguiente_cambio` decide los pasos (apartar una en una columna libre, mover
    la otra, volver a colocar la primera) y en cada paso se comprueba lo que enseña el programa.
  - Las demás se enseñan al usuario. `presentar` solo decide con qué controles; las opciones de cada pregunta
    conocida están en `CATALOGO`. Una pregunta que no esté en el catálogo se enseña tal cual, con un cuadro de texto.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

from .carga import CAMPOS_TODOS as CAMPOS
from .carga import NOMBRES
from .ejecucion import Ejecucion
from .espacio import Espacio
from .guion import Paso, reemplazar


class NoSePuedeResponder(RuntimeError):
    """La interfaz no puede dar una respuesta sin ninguna duda. No la inventa."""


# ------------------------------------------------------------------ preguntas de la carga

@dataclass
class TablaConfirmada:
    nombre: str                         # como la nombra el programa: «archivo» o «archivo [hoja]»
    archivo: str
    hoja: str | None
    fila: int
    fila_indicada: bool                 # la fila de la cabecera la indicó el usuario
    hoja_indicada: bool                 # la hoja la eligió el usuario
    columnas: list[str]
    propuesta: dict[str, str]           # lo que propone el programa
    elegidas: dict[str, str]            # lo que confirmó el usuario
    validas: dict[str, frozenset] | None = None     # columnas que el programa admite para cada dato básico (None: todas)
    personales: frozenset = frozenset()             # columnas que parecen de clientes o de personas


OBLIGATORIOS = tuple(c for c in CAMPOS if c != 'descripcion')
MAXIMO_DE_CAMBIOS = 8                   # por tabla: cuatro datos, una columna apartada y la descripción quitada y repuesta


def siguiente_cambio(actual: dict[str, str], objetivo: dict[str, str], columnas: list[str],
                     validas: dict[str, frozenset] | None = None, personales=frozenset()) -> tuple[str, str | None] | None:
    """El siguiente cambio que hay que pedir al programa para llegar de `actual` a `objetivo`: (dato, columna), con
    columna None para quitar la descripción. None si ya coinciden.

    El programa solo deja asignar a un dato una columna que no tenga otro dato, y no deja quitar la fecha, el
    producto ni la cantidad. Por eso:
      1. Si la columna que quiere algún dato está libre, se le asigna.
      2. Si no, hay un intercambio. Si la descripción estorba, se quita (es opcional) y se repone al final.
      3. Si el intercambio es entre datos básicos, uno de ellos se aparta a una columna libre que el programa
         admita para ese dato, y vuelve a su sitio cuando queda libre.
    Si no hay ninguna columna libre que sirva, no se puede indicar al programa sin dudas: NoSePuedeResponder.
    Cada dato básico acaba en su sitio con un cambio directo y no se vuelve a mover, así que termina siempre.
    """
    usadas = [v for v in objetivo.values()]
    if len(set(usadas)) != len(usadas) or any(c not in objetivo for c in OBLIGATORIOS) or any(v not in columnas for v in usadas):
        raise NoSePuedeResponder('Las columnas confirmadas no forman una asignación válida.')
    def admite(campo: str, columna: str) -> bool:
        return campo == 'descripcion' or validas is None or columna in validas.get(campo, ())

    ocupadas = set(actual.values())
    basicos = [c for c in OBLIGATORIOS if actual.get(c) != objetivo[c]]
    if not basicos:                                             # la descripción, al final: su columna ya está libre
        if actual.get('descripcion') == objetivo.get('descripcion'):
            return None
        return 'descripcion', objetivo.get('descripcion')
    for campo in basicos:                                       # 1. lo que se puede colocar directamente
        if objetivo[campo] not in ocupadas:
            if not admite(campo, objetivo[campo]):
                raise NoSePuedeResponder(f'El programa no admite «{objetivo[campo]}» como {NOMBRES[campo].lower()}.')
            return campo, objetivo[campo]
    pedidas = {objetivo[c] for c in basicos}
    if actual.get('descripcion') in pedidas:                    # 2. la descripción estorba: se quita y se repone al final
        return 'descripcion', None
    # 3. intercambio entre datos básicos: se aparta uno cuya columna espera otro
    atascados = [c for c in basicos if actual.get(c) in pedidas]
    libres = sorted((x for x in columnas if x not in ocupadas), key=lambda x: (x in personales, columnas.index(x)))
    for campo in atascados:
        for columna in libres:
            if admite(campo, columna):
                return campo, columna
    if 'descripcion' in actual and any(admite(c, actual['descripcion']) for c in atascados):
        return 'descripcion', None                              # libera una columna; la descripción se repone al final
    raise NoSePuedeResponder('Has intercambiado dos columnas y el programa solo las cambia de una en una: haría falta otra '
                             'columna libre para apartar una de ellas mientras tanto, y en este archivo no hay ninguna que sirva.')


def plan_de_cambios(propuesta: dict[str, str], objetivo: dict[str, str], columnas: list[str],
                    validas: dict[str, frozenset] | None = None, personales=frozenset()) -> list[tuple[str, str | None]]:
    """Todos los cambios, en orden, para pasar de lo que propone el programa a lo confirmado. Sirve para saber,
    antes de arrancar el programa, si se le van a poder indicar las columnas. Los datos básicos que el programa
    no propone se los pregunta él mismo, uno a uno: aquí se dan por colocados como los colocará `responder`."""
    actual = dict(propuesta)
    for campo in OBLIGATORIOS:
        if campo not in actual:
            actual[campo] = columna_para_el_dato_que_falta(campo, actual, objetivo, columnas, validas, personales)
    cambios: list[tuple[str, str | None]] = []
    while True:
        cambio = siguiente_cambio(actual, objetivo, columnas, validas, personales)
        if cambio is None:
            return cambios
        cambios.append(cambio)
        if len(cambios) > MAXIMO_DE_CAMBIOS:
            raise NoSePuedeResponder('No encuentro una forma segura de indicar esas columnas al programa.')
        campo, columna = cambio
        actual.pop(campo, None) if columna is None else actual.__setitem__(campo, columna)


def columna_para_el_dato_que_falta(campo: str, actual: dict[str, str], objetivo: dict[str, str], columnas: list[str],
                                   validas: dict[str, frozenset] | None = None, personales=frozenset()) -> str:
    """El programa pide la columna de un dato básico que no ha identificado y no deja seguir sin ella. Se le da la
    confirmada; si ahora mismo la tiene otro dato, una libre que admita, y se corrige después."""
    elegida = objetivo.get(campo)
    if elegida is None:
        raise NoSePuedeResponder(f'Falta la columna de «{NOMBRES[campo].lower()}».')
    ocupadas = set(actual.values())
    if elegida not in ocupadas:
        return elegida
    esperadas = set(objetivo.values())                         # mejor una que no espere ningún otro dato
    libres = sorted((x for x in columnas if x not in ocupadas and (validas is None or x in validas.get(campo, ()))),
                    key=lambda x: (x in esperadas, x in personales, columnas.index(x)))
    if not libres:
        raise NoSePuedeResponder(f'El programa pide la columna de «{NOMBRES[campo].lower()}», la que elegiste la tiene asignada '
                                 'a otro dato y no hay ninguna columna libre que sirva mientras tanto.')
    return libres[0]


class RespuestasDeLaCarga:
    def __init__(self, archivos: list[str], tablas: list[TablaConfirmada]):
        self.archivos, self.tablas = list(archivos), list(tablas)
        self.modelo = [dict(t.propuesta) for t in tablas]       # lo que el programa tiene asignado en cada momento
        self.pendiente: tuple[int, str, str | None] | None = None      # (tabla, dato, columna) del cambio en curso
        self.cambios = 0

    def responder(self, pregunta: str, contexto: str) -> str | None:
        """La respuesta a una pregunta de la carga, o None si la pregunta es para el usuario."""
        lineas, p = contexto.splitlines(), pregunta.strip()
        if 'Número(s) de archivo' in p:
            n = len(self.archivos)
            if not all(f'  {k}. {a}' in lineas for k, a in enumerate(self.archivos, 1)) or any(l.startswith(f'  {n + 1}. ') for l in lineas):
                raise NoSePuedeResponder('Los archivos que ve el programa no son los que has subido.')
            return '1' if n == 1 else f'1-{n}'
        if '¿En qué fila están los nombres de las columnas?' in p:
            t = next((t for t in self.tablas if f'cabecera de {t.nombre}.' in contexto), None)
            if t is None or not t.fila_indicada:
                raise NoSePuedeResponder('El programa pide la fila de la cabecera de un archivo y no la has indicado.')
            return str(t.fila + 1)
        if 'Hoja (0 = no lo sé / cancelar)' in p:
            t = next((t for t in self.tablas if t.hoja_indicada and f'{t.archivo} contiene varias hojas' in contexto), None)
            numero = next((k for k in range(1, 200) if t is not None and f'  {k}. {t.hoja}' in lineas), None)
            if numero is None:
                raise NoSePuedeResponder('El programa pide la hoja de un Excel y no la has indicado.')
            return str(numero)
        if '[S] Sí, continuar' in p or '[S] Sí, unir ventas' in p:
            return self._confirmar(lineas, varias='unir ventas' in p)
        if '¿De qué archivo quieres cambiar una columna?' in p:
            return str(self._en_curso()[0] + 1)
        if p.startswith('Opción (0 = volver)'):
            return str(CAMPOS.index(self._en_curso()[1]) + 1)
        if p == 'Columna:':
            if self.pendiente is not None:
                (i, campo, columna), self.pendiente = self.pendiente, None
            else:                       # el programa pide una columna que no ha podido identificar
                campo = next((c for c in CAMPOS if f'¿Cuál es la columna de {NOMBRES[c].lower()}?' in contexto), None)
                i = 0 if len(self.tablas) == 1 else next((k for k, t in enumerate(self.tablas) if f'Hoja/archivo: {t.nombre}' in lineas), None)
                if campo is None or i is None or campo not in OBLIGATORIOS:
                    raise NoSePuedeResponder('El programa pide una columna y no se sabe de qué dato o de qué archivo.')
                t = self.tablas[i]
                columna = columna_para_el_dato_que_falta(campo, self.modelo[i], t.elegidas, t.columnas, t.validas, t.personales)
            t = self.tablas[i]
            if not all(f'  {k}. {nombre}' in lineas for k, nombre in enumerate(t.columnas, 1)):
                raise NoSePuedeResponder(f'Las columnas que enseña el programa no son las de {t.nombre}.')
            if columna is None:
                if campo != 'descripcion':
                    raise NoSePuedeResponder(f'Falta la columna de «{NOMBRES[campo].lower()}» en {t.nombre}.')
                self.modelo[i].pop(campo, None)
                return '0'
            if columna in {v for c, v in self.modelo[i].items() if c != campo}:
                raise NoSePuedeResponder(f'«{columna}» está asignada a otro dato en {t.nombre}.')
            self.modelo[i][campo] = columna
            return str(t.columnas.index(columna) + 1)
        return None

    def _en_curso(self) -> tuple[int, str, str | None]:
        if self.pendiente is None:
            raise NoSePuedeResponder('El programa pregunta por un cambio de columna que no se ha pedido.')
        return self.pendiente

    def _mostradas(self, asignadas: list[dict[str, str]], varias: bool) -> list[str]:
        """Las líneas con las que el programa enseña una asignación de columnas."""
        if varias:
            return [l for k, (t, a) in enumerate(zip(self.tablas, asignadas), 1) for l in (
                f'  {k}. {t.nombre}',
                '     ' + ' | '.join(f'{NOMBRES[c].split(" (")[0]}: {a.get(c) or "sin identificar"}' for c in CAMPOS))]
        return [f'  {NOMBRES[c]}: {asignadas[0].get(c) or "No identificada"}' for c in CAMPOS]

    def _confirmar(self, lineas: list[str], varias: bool) -> str:
        """«Sí» solo si el programa muestra exactamente las columnas que eligió el usuario. Si no, pide el siguiente
        cambio; antes comprueba que el programa tiene asignado lo que se espera después de los cambios anteriores."""
        if self.pendiente is not None:
            raise NoSePuedeResponder('El programa no ha completado el cambio de columna que se le pidió.')
        if all(e in lineas for e in self._mostradas([t.elegidas for t in self.tablas], varias)):
            return 's'
        if not all(e in lineas for e in self._mostradas(self.modelo, varias)):
            raise NoSePuedeResponder('El programa no muestra las columnas que se esperaban después de los cambios.')
        for i, t in enumerate(self.tablas):
            cambio = siguiente_cambio(self.modelo[i], t.elegidas, t.columnas, t.validas, t.personales)
            if cambio is not None:
                self.cambios += 1
                if self.cambios > MAXIMO_DE_CAMBIOS * len(self.tablas):
                    break
                self.pendiente = (i, *cambio)
                return 'c'
        raise NoSePuedeResponder('El programa no muestra las columnas que elegiste, y no se lo puedo indicar sin dudas.')


def columnas_que_admite_el_programa(datos) -> tuple[dict[str, frozenset], frozenset]:
    """(columnas que el programa admite para cada dato básico, columnas que parecen de clientes o de personas)."""
    from . import carga as C
    columnas = [str(c) for c in datos.columns]
    validas = {campo: frozenset(x for k, x in enumerate(columnas) if C.columna_valida(datos, k, campo)[0]) for campo in OBLIGATORIOS}
    return validas, frozenset(x for k, x in enumerate(columnas) if C.parece_personal(datos, k))


def carga_confirmada(archivos, tablas, elegidas: list[dict[str, str]], elecciones: dict | None = None) -> RespuestasDeLaCarga:
    """A partir de lo decidido en las pantallas 1 y 2. `tablas`: las de `carga.leer_archivo`, en el orden del
    programa; `elegidas`: columna confirmada para cada dato en cada tabla; `elecciones`: filas y hojas indicadas."""
    from . import carga as C
    elecciones, confirmadas = elecciones or {}, []
    for t, e in zip(tablas, elegidas):
        posiciones, _ = C.propuesta_de_columnas(t.datos)
        columnas = [str(c) for c in t.datos.columns]
        indicado = elecciones.get(t.archivo.name, {})
        validas, personales = columnas_que_admite_el_programa(t.datos)
        confirmadas.append(TablaConfirmada(t.nombre, t.archivo.name, t.hoja, int(t.fila), t.hoja in indicado.get('filas', {}),
                                           t.hoja is not None and indicado.get('hoja') == t.hoja, columnas,
                                           {c: columnas[i] for c, i in posiciones.items()}, dict(e), validas, personales))
    return RespuestasDeLaCarga([a.name for a in archivos], confirmadas)


# ------------------------------------------------------------------ cómo se enseña cada pregunta

NO_SE = 'No lo sé'
CATALOGO: list[tuple[str, list[tuple[str, str]], str]] = [
    # (trozo de la pregunta, [(etiqueta, respuesta)], etiqueta del cuadro de texto si se puede escribir)
    ('¿Es correcto? (s = sí; n = las indico yo', [('No usar este archivo', ''), ('Sí, es correcto', 's'), ('No, las indico yo', 'n')], ''),
    ('¿Qué columna identifica el PRODUCTO?', [('No usar este archivo', '')], 'Número de la columna del producto'),
    ('¿Qué columna identifica la FAMILIA?', [('No usar este archivo', '')], 'Número de la columna de la familia'),
    ('¿Aceptas las', [('No, prefiero indicarlas una a una', ''), ('Sí, acepto las propuestas', 's')], ''),
    ('(V/D/J/M/Intro)', [(NO_SE, ''), ('Ventas a clientes', 'v'), ('Devoluciones o abonos', 'd'), ('Ajustes (no son ventas)', 'j'),
                         ('Mixta: ventas y otras operaciones', 'm')], ''),
    ('(V = ventas a clientes; J = ajustes; Intro = no lo sé)', [(NO_SE, ''), ('Ventas a clientes', 'v'), ('Ajustes', 'j')], ''),
    ('¿Alguna línea del grupo es otra cosa?', [('Ninguna', ''), ('Ver las líneas', 'v')], 'Números de fila, separados por comas'),
    ('¿qué es? (A/D/J/Intro)', [(NO_SE, ''), ('Anulación', 'a'), ('Devolución', 'd'), ('Ajuste', 'j')], ''),
    ('¿qué es? (D/J/Intro)', [(NO_SE, ''), ('Devolución', 'd'), ('Ajuste', 'j')], ''),
    ('¿qué es? (V/J/Intro)', [(NO_SE, ''), ('Venta', 'v'), ('Ajuste', 'j')], ''),
    ('¿Qué son? (A/D/J/Intro)', [(NO_SE, ''), ('Anulaciones: esa venta no llegó a existir', 'a'),
                                 ('Devoluciones: la venta existió y se devolvió género', 'd'), ('Ajustes: no cuentan', 'j')], ''),
    ('¿Qué son? (D/J/Intro)', [(NO_SE, ''), ('Devoluciones: la venta existió y se devolvió género', 'd'), ('Ajustes: no cuentan', 'j')], ''),
    ('¿Quieres revisarlas ahora?', [('No, sigo sin usarlas', ''), ('Sí, revisarlas ahora', 's')], ''),
    ('(S/R/Intro)', [('No lo sé, u otra cosa', ''), ('El artículo sigue a la venta', 's'), ('El artículo está retirado', 'r')], ''),
    ('¿Los doy también por retirados?', [('No, los dejo sin confirmar', ''), ('Sí', 's')], ''),
    ('¿Algún producto es una excepción?', [('Ninguno', '')], 'Códigos, separados por comas'),
    ('[A] Sigue a la venta  [R] Retirado  [D] No lo sé', [(NO_SE, 'd'), ('Sigue a la venta', 'a'), ('Retirado', 'r')], ''),
    ('(F/M/O/Intro)', [('No usar', ''), ('Familia o categoría', 'f'), ('Marca o proveedor', 'm'), ('Otra agrupación que quiero conservar', 'o')], ''),
    ('¿Cuál uso como agrupación principal?', [('La que propone el programa', '')], 'Nombre de la columna'),
    ('(N/B/U/T/C/D/Intro)', [('No usar', ''), ('Importe neto de la línea (lo que se cobra)', 'n'), ('Importe bruto de la línea (antes de descuentos)', 'b'),
                             ('Precio neto de una unidad', 'u'), ('Precio de tarifa de una unidad', 't'), ('Coste', 'c'), ('Descuento', 'd')], ''),
    ('(s = sí; Intro = no lo sé)', [(NO_SE, ''), ('Sí', 's')], ''),
    ('[S] Sí, el mes está cerrado', [(NO_SE, 'd'), ('Sí, el mes está cerrado', 's'), ('No, faltan ventas', 'n')], ''),
    ('[S] Sí, todo el periodo', [(NO_SE, 'd'), ('Sí, todo el periodo', 's'), ('Solo una parte', 'p'), ('No', 'n')], ''),
    ('[S] Sí  [R] No, están retirados', [(NO_SE, 'd'), ('Sí, siguen a la venta', 's'), ('No, están retirados', 'r'), ('Ver los códigos', 'v')], ''),
    ('¿Hay excepciones en este grupo?', [('Ninguna', '')], 'Códigos, separados por comas'),
    ('(AAAA-MM', [('No lo sé: lo dejo sin indicar', '')], 'Mes, con el formato AAAA-MM'),
]


@dataclass
class Presentacion:
    opciones: list[tuple[str, str]]             # la primera es la prudente y viene marcada
    texto: str = ''                             # si no está vacío, también se puede escribir
    catalogada: bool = True


def presentar(pregunta: str) -> Presentacion:
    for trozo, opciones, texto in CATALOGO:
        if trozo in pregunta:
            return Presentacion(list(opciones), texto)
    return Presentacion([], 'Respuesta, tal como la pide el programa', catalogada=False)


# ------------------------------------------------------------------ la pregunta, en pocas palabras

SIN_USAR = 'Si no lo sabes, no se usa.'
AYUDAS: list[tuple[str, str, str, str]] = [
    # (trozo de la pregunta, título en claro («{nombre}» = lo que la pregunta lleva entre comillas), explicación breve,
    #  expresión para buscar en lo que el programa escribió antes la línea que describe aquello por lo que pregunta)
    ('¿Es correcto? (s = sí; n = las indico yo', '¿Son estas las columnas del archivo de familias?',
     'Sin saber cuál es el producto y cuál la familia, el archivo no se usa y no hay previsiones por familia.', ''),
    ('¿Qué columna identifica el PRODUCTO?', '¿Qué columna del archivo de familias es el producto?',
     'Si no lo sabes, el archivo de familias no se usa.', r'^\d+\. «'),
    ('¿Qué columna identifica la FAMILIA?', '¿Qué columna del archivo de familias es la familia?',
     'Si no lo sabes, el archivo de familias no se usa.', r'^\d+\. «'),
    ('¿Aceptas las', '¿Aceptas lo que propone el programa para cada serie?',
     'Una serie sin confirmar no se usa y sus meses quedan pendientes. Si prefieres, te pregunto serie a serie.', ''),
    ('(V/D/J/M/Intro)', '¿Qué contiene la serie «{nombre}»?',
     'Si no lo sabes, la serie no se usa y sus meses quedan pendientes. Hace falta al menos una serie de ventas para calcular.',
     r'^Serie «{nombre}»: '),
    ('(V = ventas a clientes; J = ajustes; Intro = no lo sé)', '',
     'Son líneas positivas de una serie mixta o de devoluciones: pueden ser ventas o ajustes. Si no lo sabes, quedan pendientes y '
     'sus meses no se usan.', ''),
    ('¿Alguna línea del grupo es otra cosa?', '¿Alguna línea de este grupo es otra cosa?',
     'Si una línea concreta no es lo que has respondido para el grupo, escribe su número de fila. «Ver las líneas» las enseña todas.', ''),
    ('¿qué es? (', '', 'Es una de las líneas que has apartado del grupo. Si no lo sabes, queda pendiente y su mes no se usa.', ''),
    ('¿Qué son? (A/D/J/Intro)', '',
     'Anulación: la venta no llegó a existir y se descuenta. Devolución: la venta existió; se mantiene y la devolución se cuenta '
     'aparte. Ajuste: no cuenta. Si no lo sabes, esos meses quedan pendientes y no se usan.', ''),
    ('¿Qué son? (D/J/Intro)', '',
     'Devolución: la venta existió; se mantiene y la devolución se cuenta aparte. Ajuste: no cuenta. Si no lo sabes, esos meses '
     'quedan pendientes y no se usan.', ''),
    ('¿Quieres revisarlas ahora?', '¿Quieres revisar ahora las otras columnas del archivo?',
     'Si no las revisas, siguen sin usarse. Ninguna cambia las cantidades previstas.', ''),
    ('(S/R/Intro)', '¿Qué indica el valor «{nombre}» en la columna de estado del artículo?',
     'El nombre de la columna no basta para saberlo. ' + SIN_USAR, r'^(?:Estado del artículo: |«{nombre}»: )'),
    ('¿Los doy también por retirados?', '¿Doy también por retirados los que tienen ventas en el último mes?',
     'Si no, se quedan sin confirmar.', ''),
    ('¿Algún producto es una excepción?', '¿Algún producto es una excepción?', 'Escribe sus códigos solo si los hay.', ''),
    ('[A] Sigue a la venta  [R] Retirado  [D] No lo sé', '', 'Si no lo sabes, sus meses sin ventas no se usan.', ''),
    ('(F/M/O/Intro)', '¿Para qué sirve la columna «{nombre}»?',
     'Solo se usa para agrupar productos: no cambia ninguna previsión. ' + SIN_USAR, r'^«{nombre}»: '),
    ('¿Cuál uso como agrupación principal?', '¿Cuál es la agrupación principal?', 'Es la que se usa para las previsiones por familia.', ''),
    ('(N/B/U/T/C/D/Intro)', '¿Qué importe es la columna «{nombre}»?',
     'El programa no lo deduce de los valores. ' + SIN_USAR + ' Los importes no cambian las cantidades previstas.', r'^«{nombre}»: '),
    ('[S] Sí, el mes está cerrado', '', 'Si faltan ventas o no lo sabes, ese mes no se usa y pasa a ser el mes que se prevé.', ''),
    ('[S] Sí, todo el periodo', '',
     'Solo con un «sí» los meses sin ventas de un producto cuentan como cero. Si no lo sabes, siguen como desconocidos y '
     'ninguna previsión pasará de «orientativa».', ''),
    ('[S] Sí  [R] No, están retirados', '',
     'Si siguen a la venta, esos meses cuentan como cero ventas. Si no lo sabes, no se usan. Ningún producto se da por '
     'retirado si no lo dices tú.', r'^Grupo \d+ de \d+: '),
    ('¿Hay excepciones en este grupo?', '¿Hay excepciones en este grupo?',
     'Escribe los códigos de los productos del grupo para los que la respuesta es otra. Te pregunto por cada uno.', r'^Grupo \d+ de \d+: '),
    ('Retirados desde (AAAA-MM', '¿Desde qué mes están retirados?',
     'Si no lo sabes o cada uno es distinto, déjalo sin indicar: los meses posteriores a su última venta quedan como desconocidos.', ''),
    ('Primer mes completo (AAAA-MM', '¿Cuál es el primer mes del que tienes todas las ventas?', 'Si no lo sabes, el periodo queda sin confirmar.', ''),
    ('Último mes completo (AAAA-MM', '¿Cuál es el último mes del que tienes todas las ventas?', 'Si no lo sabes, el periodo queda sin confirmar.', ''),
    ('(AAAA-MM', '', 'Si no lo sabes, se deja sin indicar.', ''),
]
LEYENDA = re.compile(r'^\s*(?:[A-Z]|Intro) = ')
MAXIMO_DE_LINEAS = 14


@dataclass
class Vista:
    """Lo que se enseña de una pregunta: lo imprescindible a la vista y el texto completo del programa aparte."""
    titulo: str
    datos: list[str]                    # líneas del programa que hacen falta para responder
    breve: str                          # qué pasa según lo que se responda
    completo: str                       # todo lo que escribió el programa antes de la pregunta, y la pregunta tal cual
    es_lista: bool = False              # los datos son una lista que ha pedido el usuario (líneas o códigos): se enseña entera
    sobre: str = ''                     # si la pregunta continúa la anterior: cuál era y qué se respondió
    recortadas: int = 0                 # líneas que solo están en el texto completo


def _sin_leyenda(lineas: list[str]) -> list[str]:
    """Quita las líneas que explican las letras del terminal (en pantalla son opciones con su nombre)."""
    salida, sangria = [], None
    for l in lineas:
        margen = len(l) - len(l.lstrip())
        if LEYENDA.match(l):
            sangria = margen
            continue
        if sangria is not None and margen > sangria:
            continue                    # continuación de una línea de leyenda
        sangria = None
        salida.append(l)
    return salida


def vista(pregunta: str, contexto: str, anteriores: list[Paso] = ()) -> Vista:
    p = pregunta.strip()
    trozo, titulo, breve, pista = next((a for a in AYUDAS if a[0] in p), ('', '', '', ''))
    nombre = (re.findall(r'«([^»]*)»', p) or [''])[0]
    lineas = [l.rstrip() for l in contexto.strip('\n').splitlines()]
    completo = '\n'.join(lineas + [p]).strip('\n')
    anterior = anteriores[-1] if anteriores else None
    es_lista = anterior is not None and anterior.pregunta[0].strip() == p and anterior.respuesta.strip().lower() == 'v'
    if es_lista:
        parrafo = [l for l in lineas if l.strip()]
    else:
        parrafo = []
        for l in lineas:
            parrafo = [] if not l.strip() else parrafo + [l]
        if lineas and not lineas[-1].strip():
            parrafo = []
        parrafo = _sin_leyenda(parrafo)
    titulo = titulo.replace('{nombre}', nombre) if titulo else titulo_de(p)
    if not titulo:                      # la pregunta es solo un menú: el enunciado es la última línea con interrogación
        enunciado = next((l for l in reversed(parrafo) if l.strip().endswith('?')), '')
        titulo, parrafo = enunciado.strip(), [l for l in parrafo if l is not enunciado]
    if not titulo:                      # se ha vuelto a preguntar lo mismo (después de «ver»): el enunciado está más atrás
        titulo = next((l.strip() for paso in reversed(list(anteriores)) for l in reversed(paso.contexto) if l.strip().endswith('?')), '')
    datos = [l.strip() for l in parrafo if l.strip() and l.strip() not in breve and l.strip() != titulo]
    if pista:                           # lo que describe aquello por lo que se pregunta puede estar más atrás
        patron = re.compile(pista.replace('{nombre}', re.escape(nombre)))
        if not any(patron.search(l) for l in datos):
            atras = [l.strip() for paso in anteriores for l in paso.contexto] + [l.strip() for l in lineas]
            if '«{nombre}»' in pista or 'Grupo' in pista:
                encontradas = [l for l in atras if patron.search(l)]
                if 'Grupo' in pista and encontradas:
                    encontradas = encontradas[-1:]
                elif encontradas:           # una por patrón, la más reciente de cada una
                    encontradas = list(dict.fromkeys(encontradas))[-2:]
            else:                           # una lista numerada: la última seguida
                encontradas = []
                for l in reversed(atras):
                    if patron.search(l):
                        encontradas.insert(0, l)
                    elif encontradas:
                        break
            datos = encontradas + datos
    sobre = ''
    if not datos and anterior is not None and not es_de_la_carga(anterior) and any(t in p for t in (
            '¿Alguna línea del grupo es otra cosa?', '¿Hay excepciones en este grupo?', '¿Algún producto es una excepción?')):
        sobre = f'{vista(anterior.pregunta[0], "", ()).titulo or anterior.pregunta[0]} → {etiqueta_de(anterior)}'
    recortadas = 0
    if not es_lista and len(datos) > MAXIMO_DE_LINEAS:
        recortadas, datos = len(datos) - MAXIMO_DE_LINEAS, datos[-MAXIMO_DE_LINEAS:]
    return Vista(titulo, datos, breve, completo, es_lista, sobre, recortadas)


def opciones_de_lista(pregunta: str, contexto: str, anteriores: list[Paso] = ()) -> list[tuple[str, str]]:
    """Cuando el programa pide «el número» de una columna de una lista, las columnas como opciones: (nombre, número).
    Vacío si la lista no se encuentra sin dudas; entonces se escribe el número."""
    if not any(t in pregunta for t in ('¿Qué columna identifica el PRODUCTO?', '¿Qué columna identifica la FAMILIA?')):
        return []
    lineas = [l.strip() for paso in anteriores for l in paso.contexto] + [l.strip() for l in contexto.splitlines()]
    lista: list[tuple[str, str]] = []
    for l in reversed(lineas):
        m = re.match(r'^(\d+)\. «([^»]*)»', l)
        if m:
            lista.insert(0, (f'«{m.group(2)}»', m.group(1)))
        elif lista:
            break
    return lista if [n for _, n in lista] == [str(k) for k in range(1, len(lista) + 1)] else []


def titulo_de(pregunta: str) -> str:
    """La pregunta sin el menú de letras del terminal. Si era solo un menú, nada."""
    p = pregunta.strip()
    if p.startswith('['):
        return ''
    return re.sub(r'\s*\([^()]*\)\s*:?\s*$', '', p).strip(' :')


def etiqueta_de(paso: Paso) -> str:
    """Cómo se le dice al usuario lo que respondió."""
    presentacion = presentar(paso.pregunta[0])
    return next((e for e, r in presentacion.opciones if r == paso.respuesta), paso.respuesta or NO_SE)


def es_de_la_carga(paso: Paso) -> bool:
    p = paso.pregunta[0]
    return any(t in p for t in ('Número(s) de archivo', '¿En qué fila están los nombres', 'Hoja (0 = no lo sé', '[S] Sí, continuar',
                                '[S] Sí, unir ventas', '¿De qué archivo quieres cambiar')) or p.startswith('Opción (0 = volver)') or p == 'Columna:'


def ninguna_serie_confirmada(pasos: list[Paso]) -> int | None:
    """Si el usuario ha dejado todas las series sin confirmar: posición de la primera pregunta de series. Si no, None.
    Sin ninguna serie no queda ninguna venta que usar, y el programa congelado no sabe seguir."""
    series = [k for k, p in enumerate(pasos) if '(V/D/J/M/Intro)' in p.pregunta[0]]
    aceptadas = [k for k, p in enumerate(pasos) if '¿Aceptas las' in p.pregunta[0]]
    if any(pasos[k].respuesta.strip().lower()[:1] == 's' for k in aceptadas) or not series:
        return None
    if all(pasos[k].respuesta.strip().lower()[:1] in ('', 'n') for k in series):
        return min(aceptadas + series)
    return None


# ------------------------------------------------------------------ el diálogo con el programa

class Dialogo:
    def __init__(self, espacio: Espacio, opciones: dict, carga: RespuestasDeLaCarga, revision=None):
        """`revision`: `decisiones.Revision`, si las preguntas de la etapa 4 se contestan con las tablas de la revisión
        agrupada. Sin ella, todo es como antes: cada pregunta se enseña sola."""
        self.espacio, self.opciones, self.carga, self.revision = espacio, dict(opciones), carga, revision
        if revision is not None:
            self.opciones['revision'] = True        # el lanzador instala los observadores (solo en el modo de preguntas)
        self.ejecucion = Ejecucion(espacio)
        self.automaticas: set[int] = set()

    def empezar(self, pasos=()) -> None:
        """Arranca el programa en modo de preguntas. Las respuestas ya dadas se le vuelven a entregar, comprobadas."""
        self.ejecucion.cancelar()
        self.ejecucion.lanzar(list(pasos), self.opciones | {'modo': 'preguntas'})

    def grabado(self) -> list[Paso]:
        try:
            return [Paso.desde_dict(d) for d in json.loads((self.espacio.ruta / 'grabado.json').read_text(encoding='utf-8'))]
        except (OSError, ValueError):
            peticion = self.espacio.ruta / 'peticion.json'
            return [Paso.desde_dict(d) for d in json.loads(peticion.read_text(encoding='utf-8'))['guion']] if peticion.exists() else []

    def _abierta(self) -> dict | None:
        try:
            pregunta = json.loads((self.espacio.ruta / 'pregunta.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return None
        return None if (self.espacio.ruta / f'respuesta_{pregunta["n"]}.json').exists() else pregunta

    def responder(self, n: int, respuesta: str) -> None:
        if '\n' in respuesta or '\r' in respuesta:
            raise ValueError('Una respuesta ocupa una sola línea.')
        ruta = self.espacio.ruta / f'respuesta_{n}.json'
        temporal = ruta.with_suffix('.tmp')
        temporal.write_text(json.dumps({'n': n, 'respuesta': respuesta}, ensure_ascii=False), encoding='utf-8')
        reemplazar(temporal, ruta)

    def situacion(self, espera: float = 60.0) -> dict:
        """Avanza hasta la siguiente pregunta para el usuario, contestando por el camino las de la carga.
        Devuelve {'tipo': 'pregunta', n, pregunta, contexto}, {'tipo': 'trabajando'}, {'tipo': 'no_se_puede', detalle}
        o el estado final de la ejecución ({'tipo': 'preguntas_completas' | 'detenida' | 'error' | 'sin_resultados' ...})."""
        limite = time.time() + espera
        while True:
            pregunta = self._abierta()
            if pregunta is not None:
                try:
                    automatica = self.carga.responder(pregunta['pregunta'], pregunta['contexto'])
                except NoSePuedeResponder as e:
                    self.ejecucion.cancelar()
                    return {'tipo': 'no_se_puede', 'detalle': str(e)}
                if automatica is None and self.revision is not None:
                    automatica = self.revision.responder(pregunta['pregunta'], pregunta['contexto'], self.grabado())
                    if type(automatica).__name__ == 'Necesita':
                        return {'tipo': 'seccion', 'seccion': automatica.seccion, **pregunta}
                if automatica is None:
                    return {'tipo': 'pregunta', **pregunta}
                self.automaticas.add(int(pregunta['n']))
                self.responder(pregunta['n'], automatica)
            elif not self.ejecucion.en_curso():
                if self._abierta() is not None:
                    continue
                estado = self.ejecucion.estado()
                return {**estado, 'tipo': estado.get('estado', 'error')}
            elif time.time() > limite:
                return {'tipo': 'trabajando'}
            time.sleep(0.03)

    def cerrar(self) -> None:
        self.ejecucion.cancelar()
