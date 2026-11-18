"""Revisión agrupada: lo que el usuario decide en las tablas y cómo se convierte en las respuestas del programa.

El programa congelado sigue haciendo sus preguntas, en su orden. Este módulo contesta cada una a partir de las
decisiones de las tablas, con las mismas letras que escribiría el usuario en el terminal. Lo que no sabe contestar
lo deja para la pantalla de una pregunta cada vez, como antes. Nunca inventa una respuesta:

  - Una sección sin decisiones confirmadas, o cuyos datos han cambiado desde que se confirmó, devuelve
    `Necesita(sección)`: la interfaz enseña su tabla y espera.
  - Una pregunta que no reconoce devuelve None: la interfaz la enseña sola. Lo que el usuario responda se guarda en
    `sueltas` y se repite si el programa vuelve a hacer exactamente la misma pregunta en el mismo punto.

Las decisiones guardan la misma estructura que las preguntas del programa: una decisión por grupo y, aparte, las
filas o los productos que el usuario ha marcado uno a uno (sus excepciones). No se deduce ninguna mayoría.

`comprobar()` compara lo que han devuelto los diálogos del programa (copiado por `observadores.py`) con lo decidido.
Si no coincide exactamente, no se calcula.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

SECCIONES = ('series', 'lineas', 'columnas', 'completitud', 'productos')
NOMBRE_SECCION = {'series': 'Series', 'lineas': 'Líneas por aclarar', 'columnas': 'Otras columnas',
                  'completitud': 'Datos completos', 'productos': 'Productos sin ventas recientes'}

# Letras del programa (`src/v2_dialogo.py`), de lo decidido a lo que se teclea.
LETRA_PAPEL = {'ventas': 'v', 'devoluciones': 'd', 'ajustes': 'j', 'mixta': 'm', 'desconocida': ''}
LETRA_POSITIVAS = {'ventas': 'v', 'ajustes': 'j', 'desconocido': ''}
LETRA_FILA_POSITIVA = {'venta': 'v', 'ajuste': 'j', 'desconocido': ''}
LETRA_TRATAMIENTO = {'anulacion': 'a', 'devolucion': 'd', 'ajuste': 'j', 'desconocido': ''}
LETRA_ULTIMO = {'si': 's', 'no': 'n', 'no_lo_se': 'd'}
LETRA_COMPLETO = {'si': 's', 'parte': 'p', 'no': 'n', 'no_lo_se': 'd'}
LETRA_GRUPO_PRODUCTOS = {'activo_confirmado': 's', 'retirado_confirmado': 'r', 'desconocido': 'd'}
LETRA_PRODUCTO = {'activo_confirmado': 'a', 'retirado_confirmado': 'r', 'desconocido': 'd'}
ACEPTADA = 'propuesta del programa aceptada por el usuario'


@dataclass(frozen=True)
class Necesita:
    seccion: str


def mes(i: int) -> str:
    """Índice de mes del programa (año * 12 + mes - 1) a «AAAA-MM», como lo pide el programa."""
    return f'{int(i) // 12}-{int(i) % 12 + 1:02d}'


def indice(texto: str) -> int:
    anio, m = texto.split('-')
    return int(anio) * 12 + int(m) - 1


def mes_corto(i: int) -> str:
    return f'{int(i) % 12 + 1:02d}/{int(i) // 12}'


# ------------------------------------------------------------------ datos que dejan los observadores

def leer_datos(carpeta: Path, nombre: str):
    try:
        return json.loads((Path(carpeta) / f'{nombre}.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def huella(datos) -> str:
    return hashlib.sha256(json.dumps(datos, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()[:16]


def datos_de(seccion: str, carpeta: Path):
    """Los datos que el usuario ve en la tabla de una sección. Si cambian, la sección se vuelve a enseñar."""
    if seccion == 'series':
        return leer_datos(carpeta, 'series')
    if seccion == 'lineas':
        d = leer_datos(carpeta, 'lineas')
        return None if d is None else {'positivas': d['positivas'], 'negativas': d['negativas']}
    if seccion == 'completitud':
        return leer_datos(carpeta, 'completitud')
    if seccion == 'productos':
        return leer_datos(carpeta, 'productos')
    return {}                       # «Otras columnas»: solo se decide si se revisan


# ------------------------------------------------------------------ decisiones

@dataclass
class Decisiones:
    """Lo decidido en cada sección, con la huella de los datos con que se decidió. None: sin decidir."""
    series: dict | None = None        # {'aceptar': bool, 'papel': {serie: papel}}
    lineas: dict | None = None        # {'positivas': {serie: sig}, 'negativos': {grupo: trat}, 'filas': {fila: trat}}
    columnas: dict | None = None      # {'revisar': bool}
    completitud: dict | None = None   # {'ultimo', 'completo', 'desde', 'hasta'}
    productos: dict | None = None     # {'grupos': {clave: estado}, 'desde_grupo': {clave: AAAA-MM},
                                      #  'productos': {codigo: estado}, 'desde_producto': {codigo: AAAA-MM}}
    huellas: dict = field(default_factory=dict)       # sección -> huella de sus datos al confirmar
    revisar: set = field(default_factory=set)         # secciones que el usuario quiere volver a ver
    una_a_una: set = field(default_factory=set)       # secciones que el usuario prefiere contestar pregunta a pregunta
    sueltas: list = field(default_factory=list)       # [(pregunta, respuesta)] contestadas en la pantalla de una pregunta

    def confirmar(self, seccion: str, valor: dict, carpeta: Path) -> None:
        setattr(self, seccion, valor)
        self.huellas[seccion] = huella(datos_de(seccion, carpeta))
        self.revisar.discard(seccion)

    def lista(self, seccion: str, carpeta: Path) -> bool:
        """La sección está decidida, con estos mismos datos, y nadie ha pedido volver a verla."""
        return (getattr(self, seccion) is not None and seccion not in self.revisar
                and self.huellas.get(seccion) == huella(datos_de(seccion, carpeta)))

    def a_dict(self) -> dict:
        """Copia independiente: cambiarla no cambia estas decisiones."""
        return json.loads(json.dumps(self._a_dict(), ensure_ascii=False))

    def _a_dict(self) -> dict:
        return {'series': self.series, 'lineas': _claves_texto(self.lineas), 'columnas': self.columnas,
                'completitud': self.completitud, 'productos': self.productos, 'huellas': self.huellas,
                'revisar': sorted(self.revisar), 'una_a_una': sorted(self.una_a_una), 'sueltas': [list(x) for x in self.sueltas]}

    @classmethod
    def desde_dict(cls, d: dict) -> 'Decisiones':
        lineas = d.get('lineas')
        if lineas is not None:
            lineas = dict(lineas) | {'filas': {int(k): v for k, v in lineas.get('filas', {}).items()}}
        return cls(d.get('series'), lineas, d.get('columnas'), d.get('completitud'), d.get('productos'), dict(d.get('huellas', {})),
                   set(d.get('revisar', ())), set(d.get('una_a_una', ())), [tuple(x) for x in d.get('sueltas', [])])


def _claves_texto(lineas):
    return None if lineas is None else dict(lineas) | {'filas': {str(k): v for k, v in lineas.get('filas', {}).items()}}


# ------------------------------------------------------------------ qué pregunta es cada una

R_SERIE = re.compile(r'Serie «(.*)» \(V/D/J/M/Intro\)')
R_POSITIVAS = re.compile(r'Serie «(.*)»: \d+ línea\(s\) con cantidad positiva o cero\. ¿Qué son\?')
R_NEGATIVAS = re.compile(r'En \d+ línea\(s\) (.*?)(?: \(\d+ corresponderían a una venta de un mes anterior\))?\. ¿Qué son\? \(')
R_FILA = re.compile(r'^\s*Fila (\d+): ¿qué es\? \(')
R_GRUPO = re.compile(r'Grupo (\d+) de (\d+):')
R_PRODUCTO = re.compile(r'^\s*(.+?): \[A\] Sigue a la venta  \[R\] Retirado  \[D\] No lo sé')
R_PRODUCTO_DESDE = re.compile(r'^\s*(.+?): retirado desde \(AAAA-MM, Intro si no lo sabes\)')


COMPLETITUD = '[S] Sí, el mes está cerrado'


def seccion_de(pregunta: str, anteriores=()) -> str | None:
    """A qué sección pertenece una pregunta del programa, o None si no es de la revisión agrupada.
    `anteriores`: las preguntas ya contestadas. Hacen falta porque la revisión de «Otras columnas» también pregunta
    «{código}: [A] Sigue a la venta…» por sus excepciones; esas van antes de la completitud y las de los grupos de
    productos, siempre después."""
    p = str(pregunta)
    if '¿Aceptas las' in p and 'propuestas?' in p or R_SERIE.search(p):
        return 'series'
    if (R_POSITIVAS.search(p) or R_NEGATIVAS.search(p) or '¿Alguna línea del grupo es otra cosa?' in p or R_FILA.search(p)):
        return 'lineas'
    if '¿Quieres revisarlas ahora?' in p:
        return 'columnas'
    if ('[S] Sí, el mes está cerrado' in p or '[S] Sí, todo el periodo' in p or 'Primer mes completo (AAAA-MM)' in p
            or 'Último mes completo (AAAA-MM)' in p):
        return 'completitud'
    if ('[S] Sí  [R] No, están retirados  [D] No lo sé  [V] Ver los códigos' in p or 'Retirados desde (AAAA-MM)' in p
            or '¿Hay excepciones en este grupo?' in p):
        return 'productos'
    if (R_PRODUCTO.search(p) or R_PRODUCTO_DESDE.search(p)) and any(COMPLETITUD in x.pregunta[0] for x in anteriores):
        return 'productos'
    return None


def secciones_de(pasos) -> list[str | None]:
    """La sección de cada paso contestado; None para los de la carga y las preguntas sueltas."""
    from interfaz.preguntas import es_de_la_carga
    pasos = list(pasos)
    return [None if es_de_la_carga(x) else seccion_de(x.pregunta[0], pasos[:k]) for k, x in enumerate(pasos)]


def contestadas_sueltas(pasos, una_a_una) -> int:
    """Cuántas de las preguntas contestadas se han respondido en la pantalla de una pregunta cada vez."""
    from interfaz.preguntas import es_de_la_carga
    return sum(1 for x, sec in zip(pasos, secciones_de(pasos)) if not es_de_la_carga(x) and (sec is None or sec in una_a_una))


# ------------------------------------------------------------------ de las decisiones a las respuestas

class Revision:
    """Contesta las preguntas del programa con lo decidido en las tablas. `carpeta`: `<espacio>/revision`."""

    def __init__(self, carpeta: Path, decisiones: Decisiones):
        self.carpeta, self.decisiones = Path(carpeta), decisiones

    def responder(self, pregunta: str, contexto: str, anteriores) -> str | Necesita | None:
        d = self.decisiones
        seccion = seccion_de(pregunta, anteriores)
        if seccion is None:
            return self._suelta(pregunta, anteriores)
        if seccion in d.una_a_una:
            return self._suelta(pregunta, anteriores)
        if seccion == 'columnas' and d.columnas is not None and d.columnas.get('revisar') and seccion not in d.revisar:
            return 's'                     # las preguntas de las columnas, después, una a una
        if not d.lista(seccion, self.carpeta):
            return Necesita(seccion)
        return getattr(self, '_' + seccion)(str(pregunta), str(contexto), list(anteriores))

    # Las preguntas sueltas se repiten solo si el programa hace exactamente la misma pregunta en el mismo punto.
    def _suelta(self, pregunta, anteriores):
        k = contestadas_sueltas(anteriores, self.decisiones.una_a_una)
        sueltas = self.decisiones.sueltas
        if k < len(sueltas) and sueltas[k][0] == str(pregunta).strip():
            return sueltas[k][1]
        del sueltas[k:]
        return None

    def anotar_suelta(self, pregunta: str, respuesta: str, anteriores) -> None:
        k = contestadas_sueltas(anteriores, self.decisiones.una_a_una)
        del self.decisiones.sueltas[k:]
        self.decisiones.sueltas.append((str(pregunta).strip(), respuesta))

    def _series(self, p, contexto, anteriores):
        s = self.decisiones.series
        if '¿Aceptas las' in p:
            return 's' if s['aceptar'] else ''          # Intro: «te pregunto serie a serie», la opción marcada
        serie = R_SERIE.search(p).group(1)
        return LETRA_PAPEL[s['papel'].get(serie, 'desconocida')]

    def _datos_lineas(self):
        return leer_datos(self.carpeta, 'lineas') or {'positivas': [], 'negativas': []}

    def _grupo_de_lineas(self, pregunta: str):
        """('positivas', serie) o ('negativas', grupo) de una pregunta de grupo de líneas."""
        datos = self._datos_lineas()
        m = R_POSITIVAS.search(pregunta)
        if m:
            return 'positivas', m.group(1), next(g for g in datos['positivas'] if g['serie'] == m.group(1))
        m = R_NEGATIVAS.search(pregunta)
        if m:
            g = next(g for g in datos['negativas'] if g['texto'] == m.group(1))
            return 'negativas', g['grupo'], g
        return None

    def _lineas(self, p, contexto, anteriores):
        l = self.decisiones.lineas
        grupo = self._grupo_de_lineas(p)
        if grupo is not None:
            tipo, clave, _ = grupo
            if tipo == 'positivas':
                return LETRA_POSITIVAS[l['positivas'].get(clave, 'desconocido')]
            return LETRA_TRATAMIENTO[l['negativos'].get(clave, 'desconocido')]
        if '¿Alguna línea del grupo es otra cosa?' in p:
            anterior = self._grupo_de_lineas(anteriores[-1].pregunta[0]) if anteriores else None
            if anterior is None:
                raise RuntimeError('No sé a qué grupo de líneas se refiere la pregunta de excepciones.')
            tipo, _, g = anterior
            filas = [x['fila'] for x in g['lineas'] if x['fila'] in l['filas'] and (tipo == 'negativas' or x['cantidad'] > 0)]
            return ','.join(str(f) for f in sorted(filas))
        fila = int(R_FILA.search(p).group(1))
        datos = self._datos_lineas()
        if any(x['fila'] == fila for g in datos['positivas'] for x in g['lineas']):
            return LETRA_FILA_POSITIVA[l['filas'][fila]]
        return LETRA_TRATAMIENTO[l['filas'][fila]]

    def _columnas(self, p, contexto, anteriores):
        return 's' if self.decisiones.columnas.get('revisar') else ''

    def _completitud(self, p, contexto, anteriores):
        c = self.decisiones.completitud
        if '[S] Sí, el mes está cerrado' in p:
            return LETRA_ULTIMO[c['ultimo']]
        if '[S] Sí, todo el periodo' in p:
            return LETRA_COMPLETO[c['completo']]
        if 'Primer mes completo' in p:
            return c['desde']
        return c['hasta']

    def _grupo_de_productos(self, contexto: str, anteriores):
        textos = [contexto] + ['\n'.join(x.contexto) for x in reversed(anteriores)]
        for texto in textos:
            encontrados = R_GRUPO.findall(texto)
            if encontrados:
                numero = int(encontrados[-1][0])
                datos = leer_datos(self.carpeta, 'productos')
                return next(g for g in datos['grupos'] if g['numero'] == numero)
        raise RuntimeError('No sé a qué grupo de productos se refiere la pregunta.')

    def _productos(self, p, contexto, anteriores):
        d = self.decisiones.productos
        grupo = self._grupo_de_productos(contexto, anteriores)
        if '[S] Sí  [R] No, están retirados' in p:
            return LETRA_GRUPO_PRODUCTOS[d['grupos'].get(grupo['clave'], 'desconocido')]
        if 'Retirados desde (AAAA-MM)' in p:
            return d['desde_grupo'].get(grupo['clave'], '')
        if '¿Hay excepciones en este grupo?' in p:
            return ', '.join(x['codigo'] for x in grupo['productos'] if x['codigo'] in d['productos'])
        m = R_PRODUCTO_DESDE.search(p)
        if m:
            return d['desde_producto'].get(m.group(1).strip(), '')
        return LETRA_PRODUCTO[d['productos'][R_PRODUCTO.search(p).group(1).strip()]]


# ------------------------------------------------------------------ comprobación

def esperado_de_operaciones(dec: Decisiones, carpeta: Path) -> dict | None:
    """Lo que deberían devolver los diálogos de operaciones con estas decisiones, en el formato de los observadores."""
    series = leer_datos(carpeta, 'series')
    if series is None or (series['con_serie'] and dec.series is None):
        return None
    out = {'series': [], 'positivas': [], 'negativos': [], 'excepciones': set()}
    for f in series['series'] if series['con_serie'] else []:           # sin columna de serie, el programa no pregunta por ellas
        if dec.series['aceptar'] and f['propuesta']:
            out['series'].append([f['serie'], f['propuesta'], ACEPTADA])
        else:
            papel = dec.series['papel'].get(f['serie'], 'desconocida')
            out['series'].append([f['serie'], papel, 'indicada por el usuario' if papel != 'desconocida' else 'sin confirmar'])
    lineas = leer_datos(carpeta, 'lineas')
    if lineas is not None and dec.lineas is not None:
        for g in lineas['positivas']:
            out['positivas'].append([g['serie'], dec.lineas['positivas'].get(g['serie'], 'desconocido')])
            out['excepciones'] |= {(x['fila'], dec.lineas['filas'][x['fila']]) for x in g['lineas']
                                   if x['fila'] in dec.lineas['filas'] and x['cantidad'] > 0}
        for g in lineas['negativas']:
            out['negativos'].append([g['grupo'], dec.lineas['negativos'].get(g['grupo'], 'desconocido')])
            out['excepciones'] |= {(x['fila'], dec.lineas['filas'][x['fila']]) for x in g['lineas'] if x['fila'] in dec.lineas['filas']}
    return out


def comprobar(dec: Decisiones, carpeta: Path) -> list[str]:
    """Diferencias entre lo decidido y lo que ha registrado el programa. Vacía: todo coincide exactamente.
    Las secciones contestadas una a una no se comparan: esas respuestas las ha dado el usuario tal cual."""
    carpeta = Path(carpeta)
    problemas = []
    if 'series' not in dec.una_a_una and 'lineas' not in dec.una_a_una and leer_datos(carpeta, 'series') is not None:
        registrado, esperado = leer_datos(carpeta, 'registrado_operaciones'), esperado_de_operaciones(dec, carpeta)
        if registrado is None or esperado is None:
            problemas.append('No hay copia de lo que ha registrado el programa sobre las operaciones.')
        else:
            if registrado['series'] != esperado['series']:
                problemas.append('Las series registradas por el programa no son las decididas.')
            if sorted(map(tuple, registrado['positivas'])) != sorted(map(tuple, esperado['positivas'])):
                problemas.append('Lo registrado sobre las líneas positivas no es lo decidido.')
            if sorted(map(tuple, registrado['negativos'])) != sorted(map(tuple, esperado['negativos'])):
                problemas.append('Lo registrado sobre los grupos de líneas negativas no es lo decidido.')
            if {tuple(x) for x in registrado['excepciones']} != esperado['excepciones'] or \
                    len(registrado['excepciones']) != len(esperado['excepciones']):
                problemas.append('Las líneas marcadas una a una que ha registrado el programa no son las decididas.')
        lineas, reales = leer_datos(carpeta, 'lineas'), leer_datos(carpeta, 'lineas_reales')
        if lineas is not None and reales is not None:
            pos = sorted(x['fila'] for g in lineas['positivas'] for x in g['lineas'])
            neg = sorted(x['fila'] for g in lineas['negativas'] for x in g['lineas'])
            if reales.get('positivas', pos) != pos or reales.get('negativas', neg) != neg:
                problemas.append('Las líneas que ha preguntado el programa no son las de la tabla.')
    completitud = leer_datos(carpeta, 'completitud')
    if completitud is not None and 'completitud' not in dec.una_a_una:
        r = leer_datos(carpeta, 'registrado_completitud')
        c = dec.completitud
        if r is None or c is None:
            problemas.append('No hay copia de lo que ha registrado el programa sobre si los datos están completos.')
        else:
            completo = c['completo'] if c['completo'] != 'parte' else 'si'
            if r['ultimo_mes_completo'] != c['ultimo']:
                problemas.append('Lo registrado sobre el último mes no es lo decidido.')
            utilizable = completitud['ultimo_mes_archivo'] - (0 if c['ultimo'] == 'si' else 1)
            if utilizable >= completitud['primer_mes']:
                if r['completo'] != completo:
                    problemas.append('Lo registrado sobre el periodo completo no es lo decidido.')
                elif completo == 'si':
                    desde = c['desde'] if c['completo'] == 'parte' else mes(completitud['primer_mes'])
                    hasta = c['hasta'] if c['completo'] == 'parte' else mes(utilizable)
                    if (r['periodo_inicio'], r['periodo_fin']) != (desde + '-01', hasta + '-01'):
                        problemas.append('El periodo completo registrado no es el decidido.')
    productos = leer_datos(carpeta, 'productos')
    if productos is not None and productos['grupos'] and 'productos' not in dec.una_a_una:
        r, p = leer_datos(carpeta, 'registrado_productos'), dec.productos
        if r is None or p is None:
            problemas.append('No hay copia de lo que ha registrado el programa sobre los productos.')
        else:
            claves = {g['clave'] for g in productos['grupos']}
            codigos = {x['codigo'] for g in productos['grupos'] for x in g['productos']}
            esperado_r = {k: p['grupos'].get(k, 'desconocido') for k in claves}
            esperado_e = {k: v for k, v in p['productos'].items() if k in codigos}
            esperado_g = {k: v + '-01' for k, v in p['desde_grupo'].items() if k in claves and v and esperado_r[k] == 'retirado_confirmado'}
            esperado_p = {k: v + '-01' for k, v in p['desde_producto'].items()
                          if k in esperado_e and v and esperado_e[k] == 'retirado_confirmado'}
            if r['respuestas'] != esperado_r:
                problemas.append('Lo registrado para los grupos de productos no es lo decidido.')
            if r['excepciones'] != esperado_e:
                problemas.append('Los productos marcados uno a uno que ha registrado el programa no son los decididos.')
            if r['retirada_por_grupo'] != esperado_g or r['retirada_por_producto'] != esperado_p:
                problemas.append('Las fechas de retirada registradas no son las decididas.')
    return problemas


# ------------------------------------------------------------------ desde unas respuestas guardadas

def desde_lo_registrado(reglas: dict, confirmaciones: dict, carpeta: Path, seccion: str) -> dict:
    """Lo que corresponde a una sección en una ejecución anterior (sus `reglas_operaciones_v2.json` y
    `confirmaciones_v2.json`), con los datos de esta revisión. Sirve para comprobar que, con las mismas respuestas,
    la revisión agrupada da lo mismo que el flujo de preguntas, y para rehacer una ejecución guardada."""
    reglas = reglas.get('reglas', reglas)          # `reglas_operaciones_v2.json` las guarda dentro de «reglas»
    if seccion == 'series':
        series = leer_datos(carpeta, 'series') or {'series': []}
        guardadas = {x['serie']: x for x in reglas.get('series', [])}
        con_propuesta = [f['serie'] for f in series['series'] if f['propuesta']]
        aceptar = bool(con_propuesta) and all(guardadas.get(x, {}).get('procedencia') == ACEPTADA for x in con_propuesta)
        return {'aceptar': aceptar, 'papel': {x: g['papel'] for x, g in guardadas.items()}}
    if seccion == 'lineas':
        return {'positivas': {x['serie']: x['lineas_positivas'] for x in reglas.get('series', []) if 'lineas_positivas' in x},
                'negativos': dict(reglas.get('negativos', {})),
                'filas': {int(x['fila']): x['tratamiento'] for x in reglas.get('excepciones', [])}}
    if seccion == 'columnas':
        return {'revisar': False}
    if seccion == 'completitud':
        c, datos = confirmaciones['confirmacion'], leer_datos(carpeta, 'completitud')
        if c['completo'] != 'si':
            return {'ultimo': c['ultimo_mes_completo'], 'completo': c['completo'], 'desde': '', 'hasta': ''}
        utilizable = datos['ultimo_mes_archivo'] - (0 if c['ultimo_mes_completo'] == 'si' else 1)
        todo = (c['periodo_inicio'][:7], c['periodo_fin'][:7]) == (mes(datos['primer_mes']), mes(utilizable))
        return {'ultimo': c['ultimo_mes_completo'], 'completo': 'si' if todo else 'parte',
                'desde': '' if todo else c['periodo_inicio'][:7], 'hasta': '' if todo else c['periodo_fin'][:7]}
    estado = confirmaciones.get('estado_comercial', {})
    return {'grupos': {g['clave']: g['respuesta'] for g in estado.get('grupos', [])},
            'productos': {x['producto']: x['estado'] for x in estado.get('excepciones', [])},
            'desde_grupo': {k: str(v)[:7] for k, v in estado.get('fechas_de_retirada_por_grupo', {}).items()},
            'desde_producto': {x['producto']: str(x['retirado_desde'])[:7] for x in estado.get('excepciones', []) if x.get('retirado_desde')}}
