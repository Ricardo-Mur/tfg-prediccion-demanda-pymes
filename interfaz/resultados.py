"""Lo que enseñan las pantallas de resultados (bloque 3). No calcula ninguna previsión ni decide ningún estado.

Lee los archivos que ha escrito el programa congelado en la carpeta de resultados y los prepara para la pantalla:
la tabla de productos, la ficha de un producto, de una familia o del total, los avisos y las descargas. Todo sale
de esos archivos; si un dato no está en ellos, no se enseña.

Dos reglas, aprobadas con la propuesta de interfaz:

  - Una previsión «no recomendada» o con «datos insuficientes» no lleva cantidad ni rango en la vista habitual.
    El programa ya deja vacía la columna `prediccion` en esos casos; aquí, además, las frases del programa que
    llevan una cifra (el rango) no pasan a la explicación. La cifra calculada solo aparece en el detalle
    técnico, con el aviso de que no se recomienda usarla.
  - La comparación con repetir el mes anterior es un dato del pasado. Va aparte, con su advertencia, y no entra
    ni en la tabla ni en la explicación.

Este módulo no usa Streamlit: se puede probar solo.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

PROYECTO = Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True
if str(PROYECTO) not in sys.path:
    sys.path.insert(0, str(PROYECTO))

from src.v2_diagnostico import TEXTO_PATRON as PATRON                            # noqa: E402  (las frases del programa)
from src.v2_exportacion import leer_csv                                          # noqa: E402

MESES = ('enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio', 'agosto', 'septiembre', 'octubre', 'noviembre',
         'diciembre')
ESTADOS = ('buen_respaldo', 'orientativa', 'no_recomendada', 'datos_insuficientes')
CON_CANTIDAD = ('buen_respaldo', 'orientativa')         # los únicos estados que llevan una cantidad en pantalla
ICONO = {'buen_respaldo': '●', 'orientativa': '◐', 'no_recomendada': '✕', 'datos_insuficientes': '?'}
NOMBRE = {'buen_respaldo': 'Buen respaldo', 'orientativa': 'Orientativa', 'no_recomendada': 'No recomendada',
          'datos_insuficientes': 'Datos insuficientes'}
BREVE = {'buen_respaldo': 'Ha funcionado bien con meses pasados', 'orientativa': 'Sirve de orientación, con margen',
         'no_recomendada': 'No conviene decidir una compra con ella', 'datos_insuficientes': 'Falta información para evaluarla'}
FRASE = {'buen_respaldo': 'Ha funcionado bien al comprobarla con meses pasados. No garantiza que acierte el mes que viene.',
         'orientativa': 'Úsala como orientación, con un margen de seguridad.',
         'no_recomendada': 'No conviene usar una cifra de {esto} como base para decidir una compra.',
         'datos_insuficientes': 'Todavía no se puede evaluar una previsión para {esto}.'}
MOTIVOS = {
    'error_bajo': 'En los meses comprobados, la previsión se ha desviado poco de lo vendido.',
    'error_moderado': 'En los meses comprobados, la previsión se ha desviado bastante de lo vendido.',
    'evidencia_limitada': 'El error ha sido bajo, pero todavía hay pocos meses comprobados.',
    'prevision_cero': 'La previsión es de cero unidades, y una previsión de cero no se da con buen respaldo.',
    'predicciones_recientes_empeoradas': 'En los últimos meses comprobados, la previsión ha fallado más que en los anteriores.',
    'estabilidad_no_valorable': 'No se puede comprobar si el error se ha mantenido estable.',
    'completitud_sin_confirmar': 'No está confirmado que tus archivos contengan todas las ventas.',
    'error_elevado': 'En los meses comprobados, la previsión se ha desviado demasiado de lo vendido.',
    'ventas_demasiado_esporadicas': 'Ha habido ventas en muy pocos de los meses comprobados.',
    'sin_actividad_reciente': 'No tiene ventas registradas en los últimos 12 meses.',
    'producto_retirado': 'Lo has dado por retirado.',
    'pocos_meses_comprobados': 'La previsión se ha podido comprobar en muy pocos meses.',
    'historial_corto': 'Hay muy pocos meses seguidos con ventas conocidas.',
    'historico_sin_confirmar': 'No está confirmado que tus archivos contengan todas las ventas.',
    'estado_comercial_sin_confirmar': 'No está confirmado si sigue a la venta.',
    'ultimo_mes_pendiente': 'El último mes tiene operaciones pendientes de revisar.',
    'sin_registro_en_ultimo_mes': 'No tiene ventas registradas en el último mes.',
    'sin_ventas_registradas': 'No tiene ventas que se puedan usar.',
}
METODOS = {'mes_anterior': 'repetir el mes anterior', 'anual': 'repetir el mismo mes del año anterior',
           'media_12': 'media de los últimos 12 meses', 'ses': 'suavizado exponencial simple', 'ets': 'suavizado exponencial (ETS)',
           'theta': 'método Theta', 'arima_aerolinea': 'ARIMA', 'global_xgb': 'modelo conjunto de todos los productos (XGBoost)'}
ADVERTENCIA_MES_ANTERIOR = ('Es un dato de meses ya pasados. En la evaluación de la aplicación no sirvió para saber qué acierta '
                            'más el mes siguiente: no lo uses para elegir entre esta previsión y repetir el mes anterior.')
AVISO_TECNICO = ('Esta cifra no se recomienda: se enseña solo para quien quiera revisar el cálculo. No la uses como base para '
                 'decidir una compra.')
NOTA_BUEN_RESPALDO = ('«Buen respaldo» describe cómo ha funcionado la previsión al comprobarla con meses pasados. No garantiza '
                      'que acierte el mes que viene.')
NOTA_GRAFICO_NO_RECOMENDADA = ('**Solo meses pasados. No es una recomendación para {mes}.** El gráfico compara lo que se vendió '
                               'con lo que la aplicación habría previsto en los meses comprobados, para ver por qué no se '
                               'recomienda usar la previsión.')
NOTA_RANGO = 'No es un margen garantizado: sale de los errores de los meses comprobados.'
NIVELES = ('El estado de una familia no sale de los estados de sus productos: cada nivel se comprueba por su cuenta.',
           'Los niveles no se ajustan entre sí: la suma de los productos no tiene por qué coincidir con la previsión de la familia, '
           'ni la de las familias con la del total.',
           'Los productos sin familia van aparte, con ese nombre: no son una familia.')
SIN_FAMILIA = '(sin familia)'

# Frases del programa que no pasan a la explicación de la ficha. Las del rango y la cifra se enseñan aparte, y solo
# cuando hay cantidad recomendada; las del mes anterior y el patrón anual, en su apartado.
_NUM = r'[\d.]+(?:,\d+)?'
P_CIFRA = re.compile(rf'^Previsión(?: orientativa)?: {_NUM} unidades\.\s*')
P_RANGO = re.compile(rf'\s*Rango orientativo, según los errores de esos meses: entre {_NUM} y {_NUM} unidades '
                     r'\(no es un margen garantizado\)\.(?: Es un rango amplio: la cifra es poco precisa\.)?')
P_MES_ANTERIOR = re.compile(r'\s*(?:En esos meses acertó más que repetir las ventas del mes anterior \([^)]*\)\.'
                            r'|En esos meses no acertó más que repetir las ventas del mes anterior\.'
                            r'|En estos meses, repetir las ventas del mes anterior funcionó igual o mejor\.)')
P_ANUAL = re.compile(r'\s*En los meses comprobados, tener en cuenta el mes del año mejoró las previsiones de prueba'
                     r'(?:; la combinación incluye métodos que lo tienen en cuenta\.|, pero ninguno de los métodos combinados lo tiene en cuenta\.)')
P_METODOS = re.compile(r'\s*La previsión (?:usa un único método \([^)]*\)|combina solo \d+ métodos \([^)]*\)), porque los demás no se '
                       r'pudieron comprobar en todos los meses\.')
P_NO_RECOMENDAMOS = re.compile(r'^No recomendamos una previsión(?: para este producto)?\.\s*')
P_QUEDA_UNA_CIFRA = re.compile(rf'(?:entre {_NUM} y {_NUM} unidades|Previsión(?: orientativa)?: {_NUM})')


# ------------------------------------------------------------------ formato

def unidades(x) -> str:
    """Como las escribe el programa: sin decimales desde 100, con uno por debajo."""
    x = float(x)
    if x >= 100:
        return f'{x:,.0f}'.replace(',', '.')
    return f'{x:.1f}'.replace('.', ',')


def porcentaje(x) -> str:
    return f'{round(100 * float(x))} %'


def mes_largo(fecha) -> str:
    f = pd.to_datetime(fecha)
    return f'{MESES[f.month - 1]} de {f.year}'


def mes_corto(fecha) -> str:
    f = pd.to_datetime(fecha)
    return f'{f.month:02d}/{f.year}'


def _vacio(x) -> bool:
    return x is None or (isinstance(x, float) and pd.isna(x)) or (isinstance(x, str) and not x.strip()) or x is pd.NA or x is pd.NaT


def _entero(x, defecto: int = 0) -> int:
    return defecto if _vacio(x) else int(float(x))


def _es_verdad(x) -> bool:
    return str(x).strip().lower() in ('true', '1', '1.0')


def etiqueta_de_estado(estado: str) -> str:
    return f'{ICONO.get(estado, "?")} {NOMBRE.get(estado, estado)}'


# ------------------------------------------------------------------ lectura

@dataclass
class Resultados:
    carpeta: Path
    prevision: pd.DataFrame
    agregados: pd.DataFrame
    diagnostico: pd.DataFrame
    historial: pd.DataFrame
    agregados_historial: pd.DataFrame
    agregados_meses: pd.DataFrame
    meses: pd.DataFrame
    pendientes: pd.DataFrame
    excluidos: pd.DataFrame
    columnas: pd.DataFrame
    confirmaciones: dict = field(default_factory=dict)
    registro_agregados: dict = field(default_factory=dict)
    columnas_confirmadas: dict = field(default_factory=dict)

    @property
    def mes_previsto(self) -> pd.Timestamp:
        return pd.to_datetime(self.prevision.mes_previsto.iloc[0])

    @property
    def familias(self) -> pd.DataFrame:
        return self.agregados[self.agregados.nivel != 'total'] if len(self.agregados) else self.agregados

    @property
    def total(self) -> pd.Series | None:
        filas = self.agregados[self.agregados.nivel == 'total'] if len(self.agregados) else self.agregados
        return filas.iloc[0] if len(filas) else None


def leer(carpeta: Path) -> Resultados:
    carpeta = Path(carpeta)

    def tabla(nombre: str) -> pd.DataFrame:
        ruta = carpeta / nombre
        return leer_csv(ruta) if ruta.is_file() else pd.DataFrame()

    def registro(nombre: str) -> dict:
        ruta = carpeta / nombre
        return json.loads(ruta.read_text(encoding='utf-8')) if ruta.is_file() else {}

    prevision = tabla('prevision_v2.csv')
    if not len(prevision):
        raise FileNotFoundError('No hay ninguna previsión en la carpeta de resultados.')
    return Resultados(carpeta, prevision, tabla('agregados_v2.csv'), tabla('diagnostico_v2.csv'), tabla('historial_v2.csv'),
                      tabla('agregados_historial_v2.csv'), tabla('agregados_meses_v2.csv'), tabla('meses_v2.csv'),
                      tabla('pendientes_del_total_v2.csv'), tabla('candidatos_excluidos_v2.csv'), tabla('columnas_v2.csv'),
                      registro('confirmaciones_v2.json'), registro('agregados_v2.json'), registro('columnas_confirmadas_v2.json'))


# ------------------------------------------------------------------ tabla y resumen

def recuento(tabla: pd.DataFrame) -> dict[str, int]:
    return {e: int((tabla.estado == e).sum()) for e in ESTADOS} if len(tabla) else {e: 0 for e in ESTADOS}


def _celdas(f) -> dict:
    """Las celdas comunes de una fila de la tabla. Sin cantidad ni rango si el estado no la recomienda."""
    con_cantidad = f.estado in CON_CANTIDAD and not _vacio(f.prediccion)
    return {'Estado': etiqueta_de_estado(f.estado),
            'Previsión': unidades(f.prediccion) if con_cantidad else '—',
            'Rango orientativo': (f'{unidades(f.rango_inferior)} a {unidades(f.rango_superior)}'
                                  if con_cantidad and not _vacio(f.rango_inferior) and not _vacio(f.rango_superior) else '—'),
            'Error histórico': '—' if _vacio(f.error_historico) else porcentaje(f.error_historico),
            'Meses comprobados': _entero(f.meses_comprobados)}


def tabla_de_productos(r: Resultados, descripciones: dict[str, str] | None = None) -> pd.DataFrame:
    """Una fila por producto, en el orden del programa. `descripciones`: solo para la pantalla (no se exportan)."""
    filas = []
    for f in r.prevision.itertuples(index=False):
        fila = {'Producto': str(f.producto)}
        if descripciones:
            fila['Descripción'] = descripciones.get(str(f.producto), '')
        filas.append(fila | _celdas(f) | {'_estado': f.estado})
    return pd.DataFrame(filas)


def tabla_de_grupos(grupos: pd.DataFrame) -> pd.DataFrame:
    filas = [{'Familia': str(f.grupo), 'Productos': _entero(f.productos)} | _celdas(f) | {'_estado': f.estado}
             for f in grupos.itertuples(index=False)]
    return pd.DataFrame(filas)


def filtrar(tabla: pd.DataFrame, texto: str = '', estados=()) -> pd.DataFrame:
    """Por código o descripción (sin distinguir mayúsculas ni acentos) y por estado."""
    t = tabla
    if estados:
        t = t[t['_estado'].isin(list(estados))]
    texto = _sin_acentos(texto)
    if texto:
        donde = t['Producto'].map(_sin_acentos)
        if 'Descripción' in t.columns:
            donde = donde + ' ' + t['Descripción'].map(_sin_acentos)
        t = t[donde.str.contains(texto, regex=False)]
    return t


def _sin_acentos(texto) -> str:
    return str(texto or '').strip().lower().translate(str.maketrans('áéíóúü', 'aeiouu'))


def situacion_de_completitud(r: Resultados) -> dict:
    """Qué se confirmó y cuántas previsiones se quedan en «orientativa» solo por eso."""
    p = r.prevision
    sin_confirmar = p[p.completitud_historico != 'confirmado'] if 'completitud_historico' in p.columns else p.iloc[0:0]
    rebajadas = p[(p.respaldo_estadistico == 'buen_respaldo') & (p.estado == 'orientativa') & (p.motivo == 'completitud_sin_confirmar')] \
        if 'respaldo_estadistico' in p.columns else p.iloc[0:0]
    c = r.confirmaciones.get('confirmacion', {})
    completo = str(c.get('completo', ''))
    texto = {'si': 'Has confirmado que tus archivos contienen todas las ventas del periodo.',
             'no': 'Has indicado que tus archivos no contienen todas las ventas.',
             'no_lo_se': 'No has confirmado que tus archivos contengan todas las ventas.'}.get(completo, '')
    if completo == 'si' and (c.get('periodo_inicio') or c.get('periodo_fin')):
        texto = (f'Has confirmado que tus archivos contienen todas las ventas de {mes_corto(c["periodo_inicio"])} a '
                 f'{mes_corto(c["periodo_fin"])}.' if c.get('periodo_inicio') and c.get('periodo_fin') else texto)
    ultimo = {'si': 'Has confirmado que el último mes está cerrado.', 'no': 'Has indicado que al último mes le faltan ventas: no se usa.',
              'no_lo_se': 'No has confirmado que el último mes esté cerrado: no se usa.'}.get(str(c.get('ultimo_mes_completo', '')), '')
    return {'confirmada': completo == 'si' and len(sin_confirmar) == 0, 'texto': texto, 'ultimo_mes': ultimo,
            'sin_confirmar': int(len(sin_confirmar)), 'rebajadas': int(len(rebajadas)),
            'ultimo_mes_utilizado': r.confirmaciones.get('ultimo_mes_utilizado')}


# ------------------------------------------------------------------ ficha

@dataclass
class Ficha:
    nivel: str                          # 'producto', 'familia', 'sin_familia' o 'total'
    nombre: str
    estado: str
    mes: str                            # «julio de 2025»
    frase: str                          # la frase fija del estado
    cantidad: str | None = None         # solo con cantidad recomendada
    rango: tuple[str, str] | None = None
    rango_amplio: bool = False
    motivo: str = ''                    # el motivo, en una frase
    explicacion: list[str] = field(default_factory=list)       # frases del programa, sin cifras que no se recomiendan
    avisos: list[str] = field(default_factory=list)            # sobre el historial que se ha usado
    sobre_los_datos: list[str] = field(default_factory=list)
    error: str | None = None
    meses_comprobados: int = 0
    comparacion: str | None = None      # frente a repetir el mes anterior: dato del pasado
    tecnico: list[tuple[str, str]] = field(default_factory=list)
    cifra_no_recomendada: str | None = None                    # solo para el detalle técnico
    comprobaciones: pd.DataFrame | None = None                 # mes, vendido, previsto
    ventas: pd.DataFrame | None = None                         # mes, unidades (vacío = no se conoce)

    @property
    def con_cantidad(self) -> bool:
        return self.cantidad is not None


def explicacion_del_programa(mensaje, estado: str) -> list[str]:
    """Las frases del programa que explican el estado, sin la cifra, el rango, la comparación con el mes anterior,
    el patrón anual ni el número de métodos (cada cosa va en su sitio). Si en una previsión sin cantidad
    recomendada quedara alguna cifra de unidades, no se enseña ninguna frase: es más seguro."""
    if _vacio(mensaje):
        return []
    texto = str(mensaje)
    for patron in (P_RANGO, P_MES_ANTERIOR, P_ANUAL, P_METODOS):
        texto = patron.sub('', texto)
    texto = P_NO_RECOMENDAMOS.sub('', P_CIFRA.sub('', texto.strip())).strip()
    if estado not in CON_CANTIDAD and P_QUEDA_UNA_CIFRA.search(texto):
        return []
    return [texto] if texto else []


def _comparacion(f) -> str | None:
    if _vacio(f.supera_mes_anterior) or _vacio(f.mae_mes_anterior):
        return None
    n = _entero(f.meses_comprobados)
    if _es_verdad(f.supera_mes_anterior):
        mejora = '' if _vacio(f.mejora_vs_mes_anterior) else f' ({porcentaje(f.mejora_vs_mes_anterior)} menos de error)'
        return f'En los {n} meses comprobados, esta previsión se desvió menos que repetir las ventas del mes anterior{mejora}.'
    return f'En los {n} meses comprobados, repetir las ventas del mes anterior funcionó igual o mejor que esta previsión.'


def _metodos(texto) -> str:
    if _vacio(texto):
        return ''
    return ', '.join(METODOS.get(m, m) for m in str(texto).split('+'))


def _tecnico(f, nivel: str) -> tuple[list[tuple[str, str]], str | None]:
    lineas: list[tuple[str, str]] = []
    if not _vacio(f.metodos_combinados):
        lineas.append(('Métodos combinados', _metodos(f.metodos_combinados)))
    if not _vacio(f.candidatos_excluidos):
        lineas.append(('Métodos que no entraron', '; '.join(f'{METODOS.get(t.split(":")[0].strip(), t.split(":")[0].strip())}:{t.split(":", 1)[1]}'
                                                             if ':' in t else t for t in str(f.candidatos_excluidos).split('; '))))
    estabilidad = {'estable': 'El error se ha mantenido estable entre las dos mitades de los meses comprobados.',
                   'empeorada': 'El error de los últimos meses comprobados es peor que el de los anteriores.',
                   'no_valorable': 'No se puede valorar: en alguna mitad hubo ventas en muy pocos meses.'}.get(str(f.estabilidad), '')
    if estabilidad:
        if not _vacio(f.error_mitad_antigua) and not _vacio(f.error_mitad_reciente):
            estabilidad += f' Error de la mitad antigua: {porcentaje(f.error_mitad_antigua)}; de la reciente: {porcentaje(f.error_mitad_reciente)}.'
        lineas.append(('Estabilidad del error', estabilidad))
    anual = {'ayuda': 'En los meses comprobados, tener en cuenta el mes del año mejoró las previsiones de prueba. Es informativo: no cambia la previsión.',
             'no_ayuda': 'En los meses comprobados, tener en cuenta el mes del año no mejoró las previsiones de prueba.'}.get(str(f.patron_anual), '')
    if anual:
        lineas.append(('Patrón anual', anual))
    if 'respaldo_estadistico' in f.index and not _vacio(f.respaldo_estadistico) and f.respaldo_estadistico != f.estado:
        lineas.append(('Respaldo con los registros disponibles', f'{NOMBRE.get(f.respaldo_estadistico, f.respaldo_estadistico)}. '
                       'La recomendación final es más prudente porque la completitud no está confirmada.'))
    cifra = None                        # solo en «no recomendada»; con «datos insuficientes» la cifra se queda en los archivos
    if f.estado == 'no_recomendada' and not _vacio(f.prediccion_exploratoria):
        cifra = f'Cifra calculada: {unidades(f.prediccion_exploratoria)} unidades'
        if not _vacio(f.rango_inferior) and not _vacio(f.rango_superior):
            cifra += f' (entre {unidades(f.rango_inferior)} y {unidades(f.rango_superior)})'
        cifra += '.'
    return lineas, cifra


def _sobre_los_datos(d) -> list[str]:
    """La ficha de diagnóstico del programa, en frases. Es información sobre los datos: no cambia la previsión."""
    if d is None:
        return []
    frases = []
    patron = PATRON.get(str(d.get('patron_demanda', '')).strip().lower())
    if patron:
        frases.append(patron)
    seguidos, conocidos = _entero(d.get('meses_seguidos_conocidos')), _entero(d.get('meses_conocidos'))
    if seguidos:
        frases.append(f'Se usan {seguidos} meses seguidos con ventas conocidas' + (f' (hay {conocidos} meses conocidos en total).' if conocidos > seguidos else '.'))
    else:
        frases.append('No hay un tramo final de meses seguidos con ventas conocidas.')
    if not _vacio(d.get('avisos_datos')):
        frases.append(str(d.get('avisos_datos')))
    pendientes = _entero(d.get('meses_pendientes'))
    if pendientes and _vacio(d.get('avisos_datos')):
        frases.append(f'Tiene {pendientes} mes(es) con operaciones pendientes de interpretar.')
    faltan, faltan_respaldo = _entero(d.get('meses_que_faltan_para_prever')), _entero(d.get('meses_que_faltan_para_buen_respaldo'))
    if faltan:
        frases.append(f'Faltan {faltan} meses seguidos conocidos para poder evaluar una previsión.')
    elif faltan_respaldo:
        frases.append(f'Faltan {faltan_respaldo} meses seguidos conocidos para que pueda llegar a tener buen respaldo.')
    repetidas = _entero(d.get('lineas_repetidas'))
    if repetidas and _es_verdad(d.get('repetidas_valorable')):
        frases.append(f'{repetidas} línea(s) coinciden con otra en fecha y cantidad: pueden ser duplicados o no. No se han quitado.')
    altas = _entero(d.get('cantidades_muy_altas'))
    if altas:
        frases.append(f'{altas} venta(s) muy altas para este producto' + (f', {_entero(d.get("muy_altas_en_tramo_final"))} en los meses que se usan.'
                                                                            if _entero(d.get('muy_altas_en_tramo_final')) else '.') + ' No se han corregido.')
    devueltas, anuladas = d.get('unidades_devueltas'), d.get('unidades_anuladas')
    if not _vacio(devueltas) and float(devueltas) > 0:
        frases.append(f'Unidades devueltas, contadas aparte: {unidades(devueltas)}.')
    if not _vacio(anuladas) and float(anuladas) > 0:
        frases.append(f'Unidades anuladas, descontadas de su venta: {unidades(anuladas)}.')
    return frases


def _ficha(f, nivel: str, nombre: str, esto: str) -> Ficha:
    estado = str(f.estado)
    con_cantidad = estado in CON_CANTIDAD and not _vacio(f.prediccion)
    tecnico, cifra = _tecnico(f, nivel)
    if not _vacio(f.mensaje):
        tecnico.append(('Texto completo del programa', str(f.mensaje)))
    ficha = Ficha(nivel=nivel, nombre=nombre, estado=estado, mes=mes_largo(f.mes_previsto), frase=FRASE[estado].replace('{esto}', esto),
                  motivo=MOTIVOS.get(str(f.motivo), ''), explicacion=explicacion_del_programa(f.mensaje, estado),
                  error=None if _vacio(f.error_historico) else porcentaje(f.error_historico), meses_comprobados=_entero(f.meses_comprobados),
                  comparacion=_comparacion(f), tecnico=tecnico, cifra_no_recomendada=cifra)
    if con_cantidad:
        ficha.cantidad = unidades(f.prediccion)
        if not _vacio(f.rango_inferior) and not _vacio(f.rango_superior):
            ficha.rango = (unidades(f.rango_inferior), unidades(f.rango_superior))
            ficha.rango_amplio = (float(f.rango_superior) - float(f.rango_inferior)) > float(f.prediccion)
    return ficha


def ficha_de_producto(r: Resultados, producto: str) -> Ficha:
    filas = r.prevision[r.prevision.producto.astype(str) == str(producto)]
    if not len(filas):
        raise KeyError(producto)
    f = filas.iloc[0]
    ficha = _ficha(f, 'producto', str(producto), 'este producto')
    d = r.diagnostico[r.diagnostico.producto.astype(str) == str(producto)] if len(r.diagnostico) else r.diagnostico
    ficha.sobre_los_datos = _sobre_los_datos(d.iloc[0] if len(d) else None)
    h = r.historial[r.historial.producto.astype(str) == str(producto)] if len(r.historial) else r.historial
    ficha.comprobaciones = pd.DataFrame({'mes': pd.to_datetime(h.mes), 'vendido': h.real.astype(float),
                                         'previsto': h.prevision_version2.astype(float)}) if len(h) else None
    m = r.meses[r.meses.producto.astype(str) == str(producto)] if len(r.meses) else r.meses
    ficha.ventas = pd.DataFrame({'mes': pd.to_datetime(m.mes), 'unidades': pd.to_numeric(m.unidades, errors='coerce'),
                                 'estado': m.estado.astype(str)}) if len(m) else None
    return ficha


def ficha_de_grupo(r: Resultados, nivel: str, grupo: str) -> Ficha:
    """Familia, grupo sin familia o total. Cada uno es una serie propia, con su propia comprobación."""
    filas = r.agregados[(r.agregados.nivel == nivel) & (r.agregados.grupo.astype(str) == str(grupo))]
    if not len(filas):
        raise KeyError(grupo)
    f = filas.iloc[0]
    esto = {'total': 'el total', 'familia': 'esta familia'}.get(nivel, 'este grupo')
    ficha = _ficha(f, nivel, str(grupo), esto)
    if not _vacio(f.aviso_historial):
        ficha.avisos.append(str(f.aviso_historial))
    productos, retirados = _entero(f.productos), _entero(f.productos_retirados)
    ficha.sobre_los_datos.append(f'Reúne {productos} producto(s).' if nivel != 'total' else f'Suma los {productos} productos de tus archivos.')
    conocidos, sin_total = _entero(f.meses_con_total_conocido), _entero(f.meses_sin_total_conocido)
    ficha.sobre_los_datos.append(f'Se conoce el total de {conocidos} mes(es)' + (f'; de {sin_total} no se conoce.' if sin_total else '.'))
    if retirados:
        parte = '' if _vacio(f.parte_de_productos_retirados) else f', que suman el {porcentaje(f.parte_de_productos_retirados)} de las unidades de los últimos 12 meses'
        ficha.avisos.append(f'Incluye {retirados} producto(s) que has dado por retirados{parte}. Si la retirada es reciente, a partir de '
                            'ahora puede venderse menos de lo previsto.')
    h = r.agregados_historial
    h = h[(h.nivel == nivel) & (h.grupo.astype(str) == str(grupo))] if len(h) else h
    ficha.comprobaciones = pd.DataFrame({'mes': pd.to_datetime(h.mes), 'vendido': h.real.astype(float),
                                         'previsto': h.prevision_version2.astype(float)}) if len(h) else None
    m = r.agregados_meses
    m = m[(m.nivel == nivel) & (m.grupo.astype(str) == str(grupo))] if len(m) else m
    # Un mes sin total conocido es un hueco. La suma parcial no se dibuja ni se presenta como total.
    ficha.ventas = pd.DataFrame({'mes': pd.to_datetime(m.mes), 'unidades': pd.to_numeric(m.total, errors='coerce'),
                                 'estado': ['total_conocido' if _es_verdad(c) else 'sin_total' for c in m.total_conocido]}) if len(m) else None
    return ficha


def meses_sin_total(r: Resultados, nivel: str, grupo: str) -> list[dict]:
    """Los meses cuyo total no se conoce, con su motivo. Sin la suma parcial."""
    m = r.agregados_meses
    if not len(m):
        return []
    m = m[(m.nivel == nivel) & (m.grupo.astype(str) == str(grupo)) & ~m.total_conocido.map(_es_verdad)]
    motivos = {'operaciones_pendientes_de_interpretar': 'hay operaciones pendientes de interpretar',
               'historico_sin_confirmar_para_ese_mes': 'no está confirmado que tus archivos traigan todas las ventas de ese mes'}
    return [{'Mes': mes_corto(f.mes), 'Por qué no se conoce el total': motivos.get(str(f.motivo), str(f.motivo).replace('_', ' ')),
             'Productos con operaciones pendientes': _entero(f.productos_con_operaciones_pendientes)} for f in m.itertuples(index=False)]


def serie_para_el_grafico(ficha: Ficha) -> pd.DataFrame:
    """Para dibujar: una fila por mes y por serie («Vendido» o «Lo que se habría previsto»). Solo los últimos 24 meses
    con datos y los meses comprobados. Un mes que no se conoce no tiene fila de «Vendido»: es un hueco, nunca un cero."""
    partes = []
    if ficha.ventas is not None and len(ficha.ventas):
        v = ficha.ventas.sort_values('mes').tail(24)
        partes.append(pd.DataFrame({'mes': v.mes, 'serie': 'Vendido', 'unidades': v.unidades}))
    if ficha.comprobaciones is not None and len(ficha.comprobaciones):
        c = ficha.comprobaciones.sort_values('mes')
        if not partes:
            partes.append(pd.DataFrame({'mes': c.mes, 'serie': 'Vendido', 'unidades': c.vendido}))
        if ficha.con_cantidad or ficha.estado == 'no_recomendada':
            partes.append(pd.DataFrame({'mes': c.mes, 'serie': 'Lo que se habría previsto', 'unidades': c.previsto}))
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(columns=['mes', 'serie', 'unidades'])


# ------------------------------------------------------------------ avisos y pendientes

QUE_REVISAR = {'serie_sin_confirmar': 'series', 'negativo_pendiente': 'negativas', 'positivo_sin_confirmar': 'positivas'}


def operaciones_pendientes(r: Resultados) -> pd.DataFrame:
    """Mes, producto, líneas, unidades, número de fila y qué hay que aclarar. Son las del programa, sin cambios."""
    p = r.pendientes
    if not len(p):
        return pd.DataFrame(columns=['Mes', 'Producto', 'Líneas', 'Unidades', 'Fila(s)', 'Qué hay que aclarar', 'Deja sin total'])
    return pd.DataFrame({'Mes': [mes_corto(m) for m in p.mes], 'Producto': p.producto.astype(str), 'Líneas': [_entero(x) for x in p.lineas],
                         'Unidades': [unidades(abs(float(x))) if float(x) >= 0 else '−' + unidades(abs(float(x))) for x in p.unidades],
                         'Fila(s)': p.filas.astype(str), 'Qué hay que aclarar': p.que_revisar.astype(str),
                         'Deja sin total': p.grupos_sin_total.astype(str)})


def calidad(r: Resultados) -> pd.DataFrame:
    """Productos con líneas repetidas o con cantidades muy altas. Son avisos, no errores confirmados."""
    d = r.diagnostico
    if not len(d):
        return pd.DataFrame(columns=['Producto', 'Líneas repetidas', 'Cantidades muy altas'])
    repetidas = [(_entero(x) if _es_verdad(v) else 0) for x, v in zip(d.lineas_repetidas, d.repetidas_valorable)]
    altas = [_entero(x) for x in d.cantidades_muy_altas]
    t = pd.DataFrame({'Producto': d.producto.astype(str), 'Líneas repetidas': repetidas, 'Cantidades muy altas': altas})
    return t[(t['Líneas repetidas'] > 0) | (t['Cantidades muy altas'] > 0)].reset_index(drop=True)


def productos_con_historial_cortado(r: Resultados) -> pd.DataFrame:
    d = r.diagnostico
    if not len(d) or 'mes_que_corta_el_historial' not in d.columns:
        return pd.DataFrame(columns=['Producto', 'Mes que corta el historial', 'Meses anteriores sin usar', 'Motivo'])
    d = d[~d.mes_que_corta_el_historial.map(_vacio)]
    motivos = {'operaciones_pendientes': 'operaciones pendientes de interpretar', 'sin_datos_confirmados': 'mes sin datos confirmados'}
    return pd.DataFrame({'Producto': d.producto.astype(str), 'Mes que corta el historial': [mes_corto(m) for m in d.mes_que_corta_el_historial],
                         'Meses anteriores sin usar': [_entero(x) for x in d.meses_anteriores_sin_usar],
                         'Motivo': [motivos.get(str(m), str(m).replace('_', ' ')) for m in d.motivo_del_corte]}).reset_index(drop=True)


PAPELES = {'id_linea': 'Identificador de línea', 'id_factura': 'Factura o documento', 'serie': 'Serie',
           'tipo_movimiento': 'Tipo de movimiento', 'fecha_diaria': 'Fecha con día', 'lote': 'Lote',
           'codigo_cliente': 'Código de cliente', 'dato_de_cliente': 'Dato de clientes o de personas',
           'estado_articulo': 'Estado del artículo', 'existencias': 'Existencias', 'importe': 'Importe',
           'descripcion': 'Descripción', 'atributo_del_producto': 'Posible agrupación de productos',
           'fecha_del_articulo': 'Fecha del artículo'}


def columnas_usadas(r: Resultados) -> pd.DataFrame:
    """Qué columnas se han usado, cuáles no y cuáles siguen sin confirmar. Solo nombres y recuentos."""
    c = r.columnas
    if not len(c):
        return pd.DataFrame(columns=['Columna', 'Qué parece', 'Qué se ha hecho con ella'])
    usos = {'no_usar_dato_personal': 'No se usa ni se exporta', 'pendiente_de_confirmar': 'Sin usar: no has confirmado qué es',
            'sin_datos': 'Vacía', 'solo_para_emparejar_anulaciones': 'Solo para emparejar anulaciones; no se exporta'}
    hecho = [usos.get(str(u), 'Se usa' if _es_verdad(s) else 'No se usa') for u, s in zip(c.uso, c.se_usa)]
    # Lo que el usuario confirmó en las preguntas sobre las otras columnas (el inventario se escribe antes).
    conf = r.columnas_confirmadas.get('confirmacion', {})
    confirmadas = {}
    for clave, para_que in (('estado', 'estado del artículo'), ('agrupaciones', 'agrupación de productos'), ('importes', 'importe'),
                            ('existencias', 'existencias'), ('fechas_de_alta', 'fecha de alta')):
        for elemento in conf.get(clave, []) or []:
            nombre = elemento[0] if isinstance(elemento, (list, tuple)) else elemento
            confirmadas[str(nombre)] = para_que
    hecho = [f'Confirmada por ti como {confirmadas[str(col)]}' if str(col) in confirmadas else h for col, h in zip(c.columna, hecho)]
    return pd.DataFrame({'Columna': c.columna.astype(str), 'Qué parece': [PAPELES.get(str(p), 'No reconocida') for p in c.papel],
                         'Qué se ha hecho con ella': hecho})


def familias_disponibles(r: Resultados) -> dict:
    """Si hay familias y de dónde salen, con el texto del programa."""
    reg = r.registro_agregados
    niveles = reg.get('niveles', {})
    return {'hay': int(niveles.get('familia', 0)) > 0, 'origen': str(reg.get('origen_de_las_familias') or ''),
            'sin_familia': int(niveles.get('sin_familia', 0)) > 0, 'coherencia': reg.get('coherencia_entre_niveles', {})}


# ------------------------------------------------------------------ descargas

ARCHIVOS_UTILES = (
    ('prevision_v2.csv', 'Una fila por producto: estado, cantidad si la hay, rango, error, meses comprobados y la explicación.'),
    ('agregados_v2.csv', 'Lo mismo para el total y para cada familia.'),
    ('diagnostico_v2.csv', 'Lo que conviene saber de los datos de cada producto: meses conocidos, lagunas, operaciones pendientes.'),
)
NOTA_EXCEL = ('Para abrirlos en Excel: van separados por punto y coma, con coma decimal. Los códigos que empiezan por =, +, - o @ '
              'llevan un apóstrofo delante, para que Excel no los tome por fórmulas.')
NOTA_CIFRAS = ('En los archivos, la columna `prediccion` está vacía cuando la previsión no se recomienda. `prediccion_exploratoria` '
               'guarda la cifra calculada para quien quiera revisarla: no es una cantidad aconsejada.')
