"""Versión 2, etapa 4: ventas brutas, devoluciones, anulaciones y ajustes.

El objetivo de la previsión son las UNIDADES BRUTAS VENDIDAS. Para calcularlas no basta con sumar las cantidades
positivas y descartar las negativas: una línea negativa puede ser

  una devolución   la venta existió y después se devolvió género: la venta bruta no cambia y la devolución se
                   cuenta aparte, en el mes en que se registra
  una anulación    la venta no llegó a existir (línea corregida, factura rectificada): hay que quitarla de la venta
                   a la que anula, en el mes de ESA venta
  un ajuste        ni venta ni devolución (regularización, movimiento interno): no cuenta

Nada de esto se deduce del signo, de la serie ni de un nombre de columna. El usuario confirma tres cosas:

1. Qué es cada SERIE: ventas, devoluciones, ajustes o mixta. Una serie sin confirmar no se da por buena: sus
   líneas quedan pendientes y esos meses no se usan. Si el archivo no tiene series, no hay nada que confirmar.
2. Qué son las LÍNEAS NEGATIVAS de las series de ventas o mixtas, por grupos. Para cada línea negativa el
   programa busca una venta a la que PUEDA corresponder; es una posible correspondencia, no una prueba:
     misma_factura           una venta del mismo producto en el mismo documento (serie, factura y fecha)
     venta_anterior_igual    una venta anterior del mismo producto al mismo cliente, por la misma cantidad
     venta_anterior_mayor    una venta anterior del mismo producto al mismo cliente, por una cantidad mayor
     sin_venta_anterior      ninguna venta a la que pueda corresponder
   La venta nunca puede ser posterior a la línea negativa. Un número de factura que aparece con fechas distintas
   se trata como documentos distintos (los números se reutilizan entre años).
3. Qué son las LÍNEAS POSITIVAS de una serie mixta o de devoluciones. La etiqueta «mixta» no demuestra que sean
   ventas: podrían ser ajustes positivos. Sin confirmar, quedan pendientes.
4. EXCEPCIONES: líneas concretas que no siguen la respuesta de su grupo o de su serie.

Lo que no se confirma sigue pendiente, como en la versión 1. Una anulación sin venta a la que aplicarse, también.

Dos pasos separados:
- El ANÁLISIS es exploratorio. Para cada línea negativa anota todas las ventas a las que podría corresponder, sin
  gastar ninguna: varias líneas negativas pueden señalar a la misma venta.
- La ASIGNACIÓN es definitiva y solo la hacen las anulaciones confirmadas, por orden de registro. Cada una
  toma, entre sus ventas posibles, una a la que todavía le queden unidades. Las devoluciones, los ajustes y
  las líneas sin confirmar no gastan nada. Como se asigna por orden de registro, una operación posterior nunca
  cambia lo que se asignó a una anterior.

Una venta anulada entera no se convierte por sí sola en «cero ventas ese mes». La línea deja de contar y, si
era lo único que había ese mes, el mes queda sin registros: será un cero solo si el usuario ha confirmado que
el histórico está completo (lo decide `v2_completitud`, que anota el motivo). La operación sigue en el registro.

Con las reglas se construye la tabla de ventas brutas (fecha, producto, cantidad), que vuelve a pasar por la
preparación de la versión 1 sin modificarla. De cada operación se guarda la fecha en que se registra y, si es
una anulación, el mes de la venta a la que afecta. Las anulaciones registradas en un mes posterior al de su
venta se entregan aparte como «revisiones», para que el motor no las use antes de tiempo en la comprobación
retrospectiva. Las ventas netas no se calculan: serían un objetivo distinto.

El código de cliente solo se usa aquí, en memoria, para buscar correspondencias; no se guarda en ninguna salida.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .v2_exportacion import escribir_csv

GRUPOS = ('misma_factura', 'venta_anterior_igual', 'venta_anterior_mayor', 'sin_venta_anterior')
TRATAMIENTOS = ('anulacion', 'devolucion', 'ajuste', 'desconocido')
TRATAMIENTOS_EXCEPCION = TRATAMIENTOS + ('venta',)      # una línea positiva concreta puede confirmarse como venta
SIGNIFICADOS_POSITIVAS = ('ventas', 'ajustes', 'desconocido')
PAPELES_SERIE = ('ventas', 'devoluciones', 'ajustes', 'mixta', 'desconocida')
TEXTO_GRUPO = {
    'misma_factura': 'hay una venta del mismo producto en el mismo documento',
    'venta_anterior_igual': 'hay una venta anterior del mismo producto al mismo cliente, por la misma cantidad',
    'venta_anterior_mayor': 'hay una venta anterior del mismo producto al mismo cliente, por una cantidad mayor',
    'sin_venta_anterior': 'no hay ninguna venta a la que puedan corresponder',
}
TEXTO_PAPEL = {'ventas': 'ventas a clientes', 'devoluciones': 'devoluciones o abonos', 'ajustes': 'ajustes (no son ventas)',
               'mixta': 'mixta: ventas y otras operaciones', 'desconocida': 'sin confirmar'}
SIN_SERIE = ''                               # el archivo no tiene columna de serie
MARCA_SIN_CONFIRMAR = 'sin confirmar'        # cantidad no numérica: la versión 1 deja el mes como pendiente
CLASES = ('venta', 'venta_anulada_en_parte', 'venta_anulada', 'anulacion', 'devolucion', 'ajuste',
          'negativo_pendiente', 'negativo_posterior_al_cierre', 'serie_sin_confirmar', 'positivo_sin_confirmar', 'cero',
          'no_interpretable')
COLUMNAS_REGISTRO = ('fila', 'id_linea', 'id_factura', 'serie', 'fecha_registro', 'mes', 'producto', 'cantidad',
                     'clase', 'grupo', 'origen', 'correspondencia', 'fila_de_la_venta', 'fecha_de_la_venta',
                     'mes_de_la_venta', 'unidades_brutas')
COLUMNAS_PRODUCTO = ('producto', 'lineas_de_venta', 'unidades_brutas', 'lineas_devolucion', 'unidades_devueltas',
                     'lineas_anulacion', 'unidades_anuladas', 'lineas_ajuste', 'negativos_pendientes',
                     'lineas_sin_confirmar')
COLUMNAS_REVISIONES = ('producto', 'mes_venta', 'mes_registro', 'unidades')


@dataclass(frozen=True)
class ReglasOperaciones:
    """Lo que el usuario ha confirmado. Por defecto, nada."""
    negativos: tuple[tuple[str, str], ...] = ()            # (grupo, tratamiento)
    series: tuple[tuple[str, str, str], ...] = ()          # (serie, papel, procedencia de la confirmación)
    excepciones: tuple[tuple[int, str], ...] = ()          # (fila de la línea, desde 1; tratamiento)
    positivas: tuple[tuple[str, str], ...] = ()            # (serie mixta o de devoluciones, qué son sus líneas positivas)
    fecha_hora: str = ''
    huella_conjunto: str = ''                              # archivos a los que se refiere la confirmación

    def __post_init__(self):
        for g, t in self.negativos:
            if g not in GRUPOS or t not in TRATAMIENTOS:
                raise ValueError(f'Regla no reconocida: {g} -> {t}')
        if len({g for g, _ in self.negativos}) != len(self.negativos):
            raise ValueError('Hay grupos de líneas negativas con más de una regla.')
        if dict(self.negativos).get('sin_venta_anterior') == 'anulacion':
            raise ValueError('Una anulación necesita una venta a la que aplicarse: ese grupo no tiene ninguna.')
        for serie in self.series:
            if len(serie) != 3 or serie[1] not in PAPELES_SERIE:
                raise ValueError(f'Serie no reconocida: {serie}')
        if len({s for s, _, _ in self.series}) != len(self.series):
            raise ValueError('Hay series confirmadas más de una vez.')
        for serie, significado in self.positivas:
            if significado not in SIGNIFICADOS_POSITIVAS:
                raise ValueError(f'Significado no reconocido para las líneas positivas de {serie}: {significado}')
        if len({s for s, _ in self.positivas}) != len(self.positivas):
            raise ValueError('Hay series con más de una respuesta sobre sus líneas positivas.')
        for fila, t in self.excepciones:
            if type(fila) is not int or fila < 1 or t not in TRATAMIENTOS_EXCEPCION:
                raise ValueError(f'Excepción no reconocida: {fila} -> {t}')
        if len({f for f, _ in self.excepciones}) != len(self.excepciones):
            raise ValueError('Hay líneas con más de una excepción.')

    def tratamiento(self, grupo: str) -> str:
        return dict(self.negativos).get(grupo, 'desconocido')

    def papel(self, serie: str) -> str:
        return {s: p for s, p, _ in self.series}.get(serie, 'desconocida')

    def significado_positivas(self, serie: str) -> str:
        return dict(self.positivas).get(serie, 'desconocido')

    def a_dict(self) -> dict:
        return {'series': [{'serie': s, 'papel': p, 'procedencia': o, 'lineas_positivas': self.significado_positivas(s)}
                           if p in ('mixta', 'devoluciones') else {'serie': s, 'papel': p, 'procedencia': o}
                           for s, p, o in self.series],
                'negativos': dict(self.negativos),
                'excepciones': [{'fila': f, 'tratamiento': t} for f, t in self.excepciones],
                'fecha_hora': self.fecha_hora, 'huella_conjunto': self.huella_conjunto}

    @classmethod
    def desde_dict(cls, d: dict) -> 'ReglasOperaciones':
        return cls(tuple(sorted(d.get('negativos', {}).items())),
                   tuple((x['serie'], x['papel'], x.get('procedencia', '')) for x in d.get('series', [])),
                   tuple((int(x['fila']), x['tratamiento']) for x in d.get('excepciones', [])),
                   tuple((x['serie'], x['lineas_positivas']) for x in d.get('series', []) if 'lineas_positivas' in x),
                   fecha_hora=d.get('fecha_hora', ''), huella_conjunto=d.get('huella_conjunto', ''))


@dataclass
class AnalisisOperaciones:
    """Lo que se encuentra en las líneas, antes de aplicar ninguna regla."""
    lineas: pd.DataFrame            # una fila por línea, con su grupo y su venta posible preferida (`fila_venta`)
    series: pd.DataFrame            # una fila por serie: líneas positivas, negativas y cero, y primer y último mes
    grupos: dict[str, int]          # todas las líneas negativas, por grupo
    con_serie: bool
    con_factura: bool
    con_cliente: bool
    facturas_con_varias_fechas: int = 0
    # Ventas a las que PODRÍA corresponder cada línea negativa, por orden de preferencia. Es exploratorio: no
    # gasta nada, y varias líneas negativas pueden compartir venta.
    candidatas: dict = field(default_factory=dict)


@dataclass
class ResultadoOperaciones:
    tabla_bruta: pd.DataFrame       # fecha, producto, cantidad: lo que pasa a la preparación de la versión 1
    registro: pd.DataFrame          # una fila por operación interpretada (COLUMNAS_REGISTRO), sin datos de clientes
    por_producto: pd.DataFrame      # COLUMNAS_PRODUCTO
    por_mes: pd.DataFrame           # producto, mes, unidades_devueltas, unidades_anuladas, unidades_de_ajuste
    revisiones: pd.DataFrame        # anulaciones registradas en un mes posterior al de su venta (COLUMNAS_REVISIONES)
    reglas: ReglasOperaciones
    cambia: bool                    # False: la tabla es la misma que la del archivo
    resumen: dict = field(default_factory=dict)
    # (producto, mes) en los que todo lo que había eran ventas anuladas por completo: el mes queda sin registros.
    meses_anulados: frozenset = frozenset()
    # Una fila por línea del archivo, en su orden: clase, unidades brutas que quedan y, en las anulaciones
    # aplicadas, la fila (desde 0) de la venta a la que se aplican. Para quien necesite saber qué líneas son ventas.
    lineas: pd.DataFrame | None = None

    def revisiones_para_motor(self, ultimo_mes: int, series: dict) -> dict[str, list[tuple[int, int, float, bool]]]:
        """Revisiones en el formato del motor: (mes de la venta, mes de registro, unidades, el mes quedó vacío).

        `series`: el histórico (producto -> mes -> valor). El último dato dice si el mes de la venta no tiene
        valor en el histórico PORQUE todo lo que había se anuló: antes del registro se conocía la venta y
        después el mes es desconocido.
        El mes de registro es siempre el real: nunca se adelanta. Si alguna anulación se registró después de
        `ultimo_mes` (lo normal es haber aplicado las reglas con `hasta_mes` y que no haya ninguna), el motor
        la deshace: no se conocía al cerrar ese mes.
        """
        out: dict[str, list] = {}
        for r in self.revisiones.itertuples(index=False):
            registro, venta = int(r.mes_registro), int(r.mes_venta)
            if venta < registro and venta <= ultimo_mes and r.producto in series:
                vacio = series[r.producto].get(venta) is None and (r.producto, venta) in self.meses_anulados
                out.setdefault(r.producto, []).append((venta, registro, float(r.unidades), vacio))
        return out


# ------------------------------------------------------------------ análisis

def _numero(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return np.nan
    return x if np.isfinite(x) else np.nan


def analizar_operaciones(datos: pd.DataFrame, extras: pd.DataFrame | None = None) -> AnalisisOperaciones:
    """Busca, para cada línea negativa, la venta a la que puede corresponder.

    `datos`: líneas limpias (fecha, producto, cantidad). `extras`: columnas reconocidas por `v2_columnas`,
    alineadas con las líneas; pueden faltar todas.
    El emparejamiento es prudente y repetible: recorre las líneas negativas por orden de fecha y, para cada
    una, anota las ventas que la cubren y que no son posteriores: primero las del mismo documento y después las
    anteriores al mismo cliente; dentro de cada grupo, la misma cantidad, el mismo lote y la más reciente.
    No gasta ninguna venta: eso solo lo hace `aplicar_reglas` con las anulaciones confirmadas.
    """
    n = len(datos)
    extras = pd.DataFrame(index=range(n)) if extras is None else extras.reset_index(drop=True)
    if len(extras) != n:
        raise ValueError('Las columnas adicionales no están alineadas con las líneas.')
    f = pd.to_datetime(datos['fecha'].reset_index(drop=True), errors='coerce')
    t = pd.DataFrame({'fila': range(n), 'producto': datos['producto'].reset_index(drop=True).map(
        lambda p: None if pd.isna(p) else str(p).strip()), 'cantidad': datos['cantidad'].reset_index(drop=True).map(_numero)})
    t['mes'] = f.dt.year * 12 + f.dt.month - 1
    diaria = pd.to_datetime(extras['fecha_diaria'], errors='coerce') if 'fecha_diaria' in extras else None
    t['orden'] = (f if diaria is None else diaria.where(diaria.notna(), f)).dt.normalize()
    for c in ('serie', 'id_factura', 'id_linea', 'codigo_cliente', 'lote'):
        t[c] = extras[c] if c in extras else None
    t = t.rename(columns={'codigo_cliente': 'cliente'})
    con_serie, con_factura, con_cliente = 'serie' in extras, 'id_factura' in extras, 'codigo_cliente' in extras
    t['serie'] = t.serie.where(t.serie.notna(), '(sin serie)') if con_serie else SIN_SERIE
    t['grupo'], t['fila_venta'] = None, np.nan
    interpretable = t.producto.notna() & t.cantidad.notna() & t.mes.notna()

    cant = t.cantidad.to_numpy()
    lotes = t.lote.to_numpy()
    ventas = t[interpretable & (t.cantidad > 0)]
    # Mismo documento: misma serie, mismo número y misma fecha. Con otra fecha es otro documento.
    por_documento = ventas.groupby(['serie', 'id_factura', 'orden', 'producto'], dropna=True).groups if con_factura else {}
    por_cliente = ventas.groupby(['cliente', 'producto'], dropna=True).groups if con_cliente else {}
    orden = t.orden.to_numpy()
    negativas = t[interpretable & (t.cantidad < 0)].sort_values(['orden', 'fila'])
    grupos, fila_venta, candidatas = {}, {}, {}
    for r in negativas.itertuples(index=False):
        q, cuando = -r.cantidad, np.datetime64(r.orden)
        # La venta cubre la cantidad y no es posterior. No se mira si otra línea negativa señala a la misma venta.
        vale = lambda i: cant[i] >= q - 1e-9 and orden[i] <= cuando
        doc, cli = [], []
        if con_factura and r.id_factura is not None and not pd.isna(r.id_factura):
            doc = [i for i in por_documento.get((r.serie, r.id_factura, r.orden, r.producto), []) if vale(i)]
        if con_cliente and r.cliente is not None and not pd.isna(r.cliente):
            en_doc = set(doc)
            cli = [i for i in por_cliente.get((r.cliente, r.producto), []) if vale(i) and i not in en_doc]
        mismo_lote = lambda i: r.lote is not None and not pd.isna(r.lote) and lotes[i] == r.lote
        # Preferencia: la misma cantidad antes que una mayor; el mismo lote; la venta más reciente.
        preferencia = lambda i: (abs(cant[i] - q) <= 1e-9, bool(mismo_lote(i)), orden[i], i)
        lista = sorted(doc, key=preferencia, reverse=True) + sorted(cli, key=preferencia, reverse=True)
        candidatas[r.fila] = [int(i) for i in lista]
        if doc:
            grupos[r.fila] = 'misma_factura'
        elif cli:
            grupos[r.fila] = 'venta_anterior_igual' if any(abs(cant[i] - q) <= 1e-9 for i in cli) else 'venta_anterior_mayor'
        else:
            grupos[r.fila] = 'sin_venta_anterior'
        if lista:
            fila_venta[r.fila] = lista[0]
    if grupos:
        t.loc[list(grupos), 'grupo'] = pd.Series(grupos)
    if fila_venta:
        t.loc[list(fila_venta), 'fila_venta'] = pd.Series(fila_venta, dtype=float)
    t['interpretable'] = interpretable

    g = t[interpretable].groupby('serie')
    series = pd.DataFrame({'lineas': g.size(), 'positivas': g.cantidad.agg(lambda x: int((x > 0).sum())),
                           'negativas': g.cantidad.agg(lambda x: int((x < 0).sum())),
                           'cero': g.cantidad.agg(lambda x: int((x == 0).sum())),
                           'primer_mes': g.mes.min().astype(int), 'ultimo_mes': g.mes.max().astype(int)}
                          ).reset_index().sort_values(['lineas', 'serie'], ascending=[False, True]).reset_index(drop=True)
    if not con_serie:
        series = series.iloc[0:0]
    varias = 0
    if con_factura:
        con_id = t[t.id_factura.notna() & t.orden.notna()]
        varias = int((con_id.groupby(['serie', 'id_factura']).orden.nunique() > 1).sum())
    cuenta = negativas.fila.map(grupos).value_counts().to_dict()
    return AnalisisOperaciones(t, series, {x: int(cuenta.get(x, 0)) for x in GRUPOS}, con_serie, con_factura,
                               con_cliente, varias, candidatas)


def proponer_series(analisis: AnalisisOperaciones, parte_minima: float = 0.05) -> dict[str, tuple[str | None, str]]:
    """Propuesta para cada serie, con su motivo. Es solo una propuesta: nada se aplica sin confirmar.

    Nunca propone «devoluciones» ni «ajustes»: eso solo lo puede decir el usuario.
    """
    s = analisis.series
    total = max(int(s.lineas.sum()), 1)
    out = {}
    for f in s.itertuples(index=False):
        con_signo = f.positivas + f.negativas
        if f.negativas and (f.negativas >= parte_minima * con_signo):
            out[f.serie] = ('mixta', 'tiene líneas negativas: se preguntará por ellas en grupos')
        elif f.positivas and f.lineas >= parte_minima * total:
            out[f.serie] = ('ventas', 'es una de las series principales y casi todas sus líneas son positivas')
        else:
            out[f.serie] = (None, 'tiene pocas líneas: no hay base para proponer nada')
    return out


def negativas_a_confirmar(analisis: AnalisisOperaciones, reglas: ReglasOperaciones) -> pd.DataFrame:
    """Líneas negativas de series de ventas o mixtas (o de un archivo sin series): las que se confirman por grupos."""
    t = analisis.lineas
    papel = t.serie.map(reglas.papel) if analisis.con_serie else 'ventas'
    return t[t.interpretable & (t.cantidad < 0) & pd.Series(papel, index=t.index).isin(['ventas', 'mixta'])]


def positivas_a_confirmar(analisis: AnalisisOperaciones, reglas: ReglasOperaciones) -> pd.DataFrame:
    """Líneas con cantidad positiva o cero de las series mixtas o de devoluciones: su significado hay que confirmarlo."""
    t = analisis.lineas
    if not analisis.con_serie:
        return t.iloc[0:0]
    return t[t.interpretable & (t.cantidad >= 0) & t.serie.map(reglas.papel).isin(['mixta', 'devoluciones'])]


# ------------------------------------------------------------------ aplicación de las reglas

def aplicar_reglas(datos: pd.DataFrame, analisis: AnalisisOperaciones, reglas: ReglasOperaciones | None = None,
                   *, hasta_mes: int | None = None) -> ResultadoOperaciones:
    """Tabla de ventas brutas y registro de lo que se ha hecho con cada línea que no es una venta normal.

    `hasta_mes`: último mes completo. Las líneas negativas registradas después no se conocían al cerrarlo: no
    se aplican a ninguna venta, se quedan en la tabla tal como vienen (su mes no se usa) y el registro las
    anota con su fecha real. Así el histórico es el disponible al cierre del último mes completo.

    Prioridad para una línea: su excepción; si no, lo que diga su serie; si no, la respuesta de su grupo.
    Sin respuesta, sigue pendiente. Solo las anulaciones confirmadas gastan unidades de una venta, por orden
    de registro; si a sus ventas posibles no les quedan unidades, la línea sigue pendiente.
    """
    reglas = reglas or ReglasOperaciones()
    t = analisis.lineas
    n = len(t)
    en_archivo = set(analisis.series.serie)
    desconocidas = ({s for s, _, _ in reglas.series} | {s for s, _ in reglas.positivas}) - en_archivo
    if desconocidas:
        raise ValueError(f'Series que no están en el archivo: {sorted(desconocidas)}')
    q = t.cantidad.to_numpy()
    ok = t.interpretable.to_numpy()
    papel = t.serie.map(reglas.papel).to_numpy() if analisis.con_serie else np.full(n, 'ventas', dtype=object)
    positivas = t.serie.map(reglas.significado_positivas).to_numpy() if analisis.con_serie else np.full(n, 'ventas', dtype=object)
    excepciones = {f - 1: trat for f, trat in reglas.excepciones}
    for i, trat in excepciones.items():
        if i >= n or not ok[i] or q[i] == 0:
            raise ValueError(f'La excepción de la fila {i + 1} no corresponde a una línea con cantidad.')
        if q[i] > 0 and trat in ('anulacion', 'devolucion'):
            raise ValueError(f'La fila {i + 1} tiene cantidad positiva: no puede ser una anulación ni una devolución.')
        if q[i] < 0 and trat == 'venta':
            raise ValueError(f'La fila {i + 1} tiene cantidad negativa: no puede ser una venta.')
        if trat == 'anulacion' and not analisis.candidatas.get(i):
            raise ValueError(f'La fila {i + 1} no tiene ninguna venta a la que aplicar la anulación.')

    clase = np.full(n, 'no_interpretable', dtype=object)
    origen = np.full(n, '', dtype=object)
    bruto = np.zeros(n)
    # 1. Líneas con cantidad positiva o cero: ¿son ventas?
    for i in np.flatnonzero(ok & (q >= 0)):
        if i in excepciones:
            significado, de = {'venta': 'ventas', 'ajuste': 'ajustes', 'desconocido': 'desconocido'}[excepciones[i]], 'excepción'
        elif papel[i] == 'ventas':
            significado, de = 'ventas', ''
        elif papel[i] in ('mixta', 'devoluciones'):
            significado, de = positivas[i], 'serie'
        elif papel[i] == 'ajustes':
            significado, de = 'ajustes', 'serie'
        else:
            significado, de = 'serie_sin_confirmar', 'serie'
        origen[i] = de
        if significado == 'ventas':
            clase[i], bruto[i] = ('venta', q[i]) if q[i] > 0 else ('cero', 0.0)
        elif significado == 'ajustes':
            clase[i] = 'ajuste'
        elif significado == 'serie_sin_confirmar':
            clase[i] = 'serie_sin_confirmar'
        else:
            clase[i], origen[i] = 'positivo_sin_confirmar', de if de == 'excepción' else 'sin confirmar'
    # 2. Líneas negativas: qué dice el usuario de cada una.
    tratamiento = {}
    tarde = t.mes.to_numpy() > hasta_mes if hasta_mes is not None else np.zeros(n, bool)
    for i in np.flatnonzero(ok & (q < 0)):
        if tarde[i]:
            clase[i], origen[i] = 'negativo_posterior_al_cierre', 'registrada después del último mes completo: no influye en esta previsión'
        elif i in excepciones:
            tratamiento[i] = (excepciones[i], 'excepción')
        elif papel[i] == 'devoluciones':
            tratamiento[i] = ('devolucion', 'serie')
        elif papel[i] in ('ventas', 'mixta'):
            tratamiento[i] = (reglas.tratamiento(t.grupo.iat[i]), 'grupo')
        elif papel[i] == 'ajustes':
            clase[i], origen[i] = 'ajuste', 'serie'
        else:
            clase[i], origen[i] = 'serie_sin_confirmar', 'serie'
    # 3. Asignación definitiva: solo las anulaciones confirmadas, por orden de registro.
    disponible = np.where(clase == 'venta', q, 0.0)
    anulado = np.zeros(n)
    asignada = {}
    for i in sorted(tratamiento, key=lambda i: (t.orden.iat[i], i)):
        trat, de = tratamiento[i]
        if trat == 'anulacion':
            venta = next((v for v in analisis.candidatas.get(i, []) if disponible[v] >= -q[i] - 1e-9), None)
            if venta is None:
                clase[i], origen[i] = 'negativo_pendiente', f'{de}: anulación sin una venta disponible'
                continue
            clase[i], origen[i], asignada[i] = 'anulacion', de, venta
            disponible[venta] -= -q[i]
            anulado[venta] += -q[i]
            bruto[venta] = max(q[venta] - anulado[venta], 0.0)
            clase[venta] = 'venta_anulada' if bruto[venta] <= 1e-9 else 'venta_anulada_en_parte'
        elif trat in ('devolucion', 'ajuste'):
            clase[i], origen[i] = trat, de
        else:
            clase[i], origen[i] = 'negativo_pendiente', de if de == 'excepción' else 'sin confirmar'

    # Tabla para la versión 1. Salen las anulaciones, las devoluciones, los ajustes y las ventas anuladas por
    # completo (que siguen en el registro). Las líneas de series sin confirmar y las positivas sin confirmar
    # llevan una cantidad que la versión 1 no puede leer, y deja el mes como pendiente.
    cantidad = datos['cantidad'].reset_index(drop=True).astype(object).copy()
    for i in np.flatnonzero(clase == 'venta_anulada_en_parte'):
        cantidad.iat[i] = bruto[i]
    for i in np.flatnonzero(np.isin(clase, ('serie_sin_confirmar', 'positivo_sin_confirmar'))):
        cantidad.iat[i] = MARCA_SIN_CONFIRMAR
    fuera = np.isin(clase, ('anulacion', 'devolucion', 'ajuste', 'venta_anulada'))
    tabla = pd.DataFrame({'fecha': datos['fecha'].reset_index(drop=True), 'producto': datos['producto'].reset_index(drop=True),
                          'cantidad': cantidad})[~fuera].reset_index(drop=True)
    # Meses en los que todo lo que había eran ventas anuladas por completo.
    mes_de = list(zip(t.producto, t.mes))
    quedan = {mes_de[i] for i in np.flatnonzero(ok & ~fuera)}
    meses_anulados = frozenset((p, int(m)) for p, m in {mes_de[i] for i in np.flatnonzero(clase == 'venta_anulada')} - quedan)

    interesa = ~np.isin(clase, ('venta', 'cero', 'no_interpretable', 'serie_sin_confirmar'))
    # La venta que se anota: la asignada si es una anulación; si no, la posible venta preferida.
    venta_de = [asignada.get(i, None if pd.isna(v) else int(v)) for i, v in enumerate(t.fila_venta)]
    registro = pd.DataFrame({
        'fila': t.fila + 1, 'id_linea': t.id_linea, 'id_factura': t.id_factura, 'serie': t.serie,
        'fecha_registro': t.orden.dt.date, 'mes': t.mes.map(_fecha_mes), 'producto': t.producto, 'cantidad': t.cantidad,
        'clase': clase, 'grupo': t.grupo, 'origen': origen,
        'correspondencia': ['aplicada' if i in asignada else ('posible' if v is not None else '') for i, v in enumerate(venta_de)],
        'fila_de_la_venta': [np.nan if v is None else v + 1 for v in venta_de],
        'fecha_de_la_venta': [None if v is None else t.orden.iat[v].date() for v in venta_de],
        'mes_de_la_venta': [None if v is None else _fecha_mes(t.mes.iat[v]) for v in venta_de],
        'unidades_brutas': bruto}, columns=list(COLUMNAS_REGISTRO))[interesa].reset_index(drop=True)

    # Anulaciones registradas en un mes posterior al de su venta.
    rev = pd.DataFrame([(t.producto.iat[i], int(t.mes.iat[v]), int(t.mes.iat[i]), -q[i]) for i, v in asignada.items()],
                       columns=list(COLUMNAS_REVISIONES))
    posteriores = int((rev.mes_registro > rev.mes_venta).sum())
    rev = rev[rev.mes_registro > rev.mes_venta].groupby(['producto', 'mes_venta', 'mes_registro'], as_index=False).unidades.sum()

    d = pd.DataFrame({'producto': t.producto, 'mes': t.mes, 'clase': clase, 'q': q, 'bruto': bruto, 'anulado': anulado})[ok]
    por = lambda mascara, valores=None: (mascara if valores is None else valores.where(mascara, 0.0)).groupby(d.producto).sum()
    por_producto = pd.DataFrame({
        'lineas_de_venta': por(d.clase.isin(['venta', 'venta_anulada_en_parte'])),
        'unidades_brutas': d.bruto.groupby(d.producto).sum(),
        'lineas_devolucion': por(d.clase == 'devolucion'), 'unidades_devueltas': por(d.clase == 'devolucion', -d.q),
        'lineas_anulacion': por(d.clase == 'anulacion'), 'unidades_anuladas': d.anulado.groupby(d.producto).sum(),
        'lineas_ajuste': por(d.clase == 'ajuste'), 'negativos_pendientes': por(d.clase == 'negativo_pendiente'),
        'lineas_sin_confirmar': por(d.clase.isin(['serie_sin_confirmar', 'positivo_sin_confirmar'])),
    }).fillna(0).reset_index()[list(COLUMNAS_PRODUCTO)]
    for c in COLUMNAS_PRODUCTO:
        if c.startswith('lineas') or c == 'negativos_pendientes':
            por_producto[c] = por_producto[c].astype(int)
    m = d.assign(devueltas=np.where(d.clase == 'devolucion', -d.q, 0.0), ajuste=np.where(d.clase == 'ajuste', d.q, 0.0))
    por_mes = m.groupby(['producto', 'mes'])[['devueltas', 'anulado', 'ajuste']].sum().reset_index()
    por_mes = por_mes[(por_mes[['devueltas', 'anulado', 'ajuste']] != 0).any(axis=1)].rename(
        columns={'devueltas': 'unidades_devueltas', 'anulado': 'unidades_anuladas', 'ajuste': 'unidades_de_ajuste'})
    por_mes['mes'] = por_mes.mes.map(_fecha_mes)

    cuenta = pd.Series(clase).value_counts().to_dict()
    resumen = {c: int(cuenta.get(c, 0)) for c in CLASES}
    resumen |= {'unidades_brutas': float(bruto.sum()), 'unidades_positivas_del_archivo': float(q[ok & (q > 0)].sum()),
                'unidades_devueltas': float(-q[clase == 'devolucion'].sum()), 'unidades_anuladas': float(anulado.sum()),
                'lineas_negativas': int((ok & (q < 0)).sum()), 'excepciones': len(excepciones),
                'anulaciones_de_meses_anteriores': posteriores,
                'anulaciones_sin_venta_disponible': int(sum('sin una venta disponible' in str(o) for o in origen)),
                'ultimo_mes_completo': None if hasta_mes is None else str(_fecha_mes(hasta_mes)),
                'meses_con_todas_las_ventas_anuladas': len(meses_anulados),
                'facturas_con_varias_fechas': analisis.facturas_con_varias_fechas}
    cambia = bool(fuera.any() or np.isin(clase, ('venta_anulada_en_parte', 'serie_sin_confirmar', 'positivo_sin_confirmar')).any())
    por_linea = pd.DataFrame({'clase': clase, 'unidades_brutas': bruto,
                              'fila_de_la_venta': [asignada.get(i, np.nan) for i in range(n)]})
    return ResultadoOperaciones(tabla, registro, por_producto, por_mes.reset_index(drop=True), rev.reset_index(drop=True),
                                reglas, cambia, resumen, meses_anulados, por_linea)


def _fecha_mes(i):
    return None if i is None or pd.isna(i) else pd.Timestamp(year=int(i) // 12, month=int(i) % 12 + 1, day=1).date()


# ------------------------------------------------------------------ líneas repetidas con identificadores

def repetidas_con_identificadores(analisis: AnalisisOperaciones) -> pd.DataFrame | None:
    """Líneas de venta repetidas según los identificadores, por producto. None si no hay identificadores.

    - Con identificador de línea: las que repiten documento (serie, factura y fecha), identificador de línea,
      producto y cantidad. La línea 1 de dos facturas distintas no es una repetición. Es el indicio más fuerte
      de que la misma línea está dos veces en el archivo.
    - Con identificador de factura: las que repiten producto, cantidad y lote dentro del mismo documento. Puede
      ser correcto (dos entregas iguales); es un indicio más débil.
    Se cuenta lo que sobraría: de tres líneas iguales, dos.
    """
    t = analisis.lineas
    con_linea = bool(t.id_linea.notna().any())
    if not con_linea and not analisis.con_factura:
        return None
    v = t[t.interpretable & (t.cantidad > 0)].copy()
    v['id_factura'] = v.id_factura.fillna('')
    v['lote'] = v.lote.fillna('')
    out = pd.DataFrame({'producto': sorted(v.producto.unique())}).set_index('producto')
    out['con_id_linea'], out['con_id_factura'] = con_linea, bool(analisis.con_factura)
    documento = ['serie', 'id_factura', 'orden']

    def sobran(sub, claves):
        rep = sub[sub.duplicated(claves, keep='first')]
        return rep.groupby('producto').size(), rep.groupby('producto').cantidad.sum()
    vacia = (pd.Series(dtype=float), pd.Series(dtype=float))
    lin, ulin = sobran(v[v.id_linea.notna()], documento + ['id_linea', 'producto', 'cantidad']) if con_linea else vacia
    fac, ufac = sobran(v[v.id_factura != ''], documento + ['producto', 'cantidad', 'lote']) if analisis.con_factura else vacia
    out['lineas_con_id_repetido'] = lin.reindex(out.index).fillna(0).astype(int)
    out['unidades_con_id_repetido'] = ulin.reindex(out.index).fillna(0.0)
    out['lineas_repetidas_en_factura'] = fac.reindex(out.index).fillna(0).astype(int)
    out['unidades_repetidas_en_factura'] = ufac.reindex(out.index).fillna(0.0)
    out['unidades_de_venta'] = v.groupby('producto').cantidad.sum().reindex(out.index)
    return out.reset_index()


# ------------------------------------------------------------------ registro y presentación

def exportar_operaciones(resultado: ResultadoOperaciones, analisis: AnalisisOperaciones, carpeta: Path) -> dict[str, Path]:
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    rutas = {'operaciones_v2': carpeta / 'operaciones_v2.csv', 'reglas_operaciones': carpeta / 'reglas_operaciones_v2.json'}
    escribir_csv(resultado.registro, rutas['operaciones_v2'])
    propuestas = proponer_series(analisis)
    rutas['reglas_operaciones'].write_text(json.dumps({
        'reglas': resultado.reglas.a_dict(), 'lineas_negativas_por_grupo': analisis.grupos,
        'series': [{**f, 'primer_mes': str(_fecha_mes(f['primer_mes'])), 'ultimo_mes': str(_fecha_mes(f['ultimo_mes'])),
                    'propuesta_del_programa': propuestas[f['serie']][0], 'papel_aplicado': resultado.reglas.papel(f['serie'])}
                   for f in analisis.series.to_dict('records')],
        'anulaciones_de_meses_anteriores': [{'producto': r.producto, 'mes_de_la_venta': str(_fecha_mes(r.mes_venta)),
                                             'mes_de_registro': str(_fecha_mes(r.mes_registro)), 'unidades': float(r.unidades)}
                                            for r in resultado.revisiones.itertuples(index=False)],
        'resultado': resultado.resumen,
        'nota': 'El objetivo son las unidades brutas vendidas. Las devoluciones se cuentan aparte y no se restan. '
                'Una anulación se resta de la venta a la que anula, en el mes de esa venta; si se registró en un mes '
                'posterior, la comprobación retrospectiva no la usa antes de ese mes; si se registró después del '
                'último mes completo, no influye en la previsión. Solo las anulaciones '
                'confirmadas gastan unidades de una venta. Un mes en el que todo eran ventas anuladas no es un cero '
                'si el histórico no está confirmado. Las series sin confirmar y lo no confirmado siguen pendientes. '
                'No se guarda ningún dato de clientes.'}, ensure_ascii=False, indent=2),
        encoding='utf-8')
    return rutas


def cargar_reglas(ruta: Path, huella_conjunto: str | None = None) -> ReglasOperaciones:
    """Reglas guardadas.

    `huella_conjunto`: huella de los archivos actuales. Para REUTILIZAR las reglas hay que indicarla siempre (aunque
    esté vacía): se rechazan si falta, si las reglas no traen la suya o si no coinciden. Con None no se comprueba
    nada: solo sirve para leer el archivo.
    """
    from .v2_completitud import comprobar_huellas
    reglas = ReglasOperaciones.desde_dict(json.loads(Path(ruta).read_text(encoding='utf-8'))['reglas'])
    if huella_conjunto is not None:
        comprobar_huellas(huella_conjunto, reglas.huella_conjunto, 'Las reglas de operaciones guardadas')
    return reglas


def mostrar_operaciones_corto(resultado: ResultadoOperaciones) -> None:
    r = resultado.resumen
    print('\nOPERACIONES')
    if r['serie_sin_confirmar'] or r['positivo_sin_confirmar']:
        print(f'  Líneas de series sin confirmar: {r["serie_sin_confirmar"]}. Líneas positivas sin confirmar en series '
              f'mixtas o de devoluciones: {r["positivo_sin_confirmar"]}. Sus meses quedan pendientes y no se usan.')
    if r['facturas_con_varias_fechas']:
        print(f'  {r["facturas_con_varias_fechas"]} números de factura aparecen con fechas distintas: se tratan como '
              'documentos distintos.')
    if not r['lineas_negativas'] and not r['ajuste']:
        print('  No hay líneas negativas ni ajustes.')
    else:
        print(f'  Líneas negativas: {r["lineas_negativas"]}. Devoluciones: {r["devolucion"]} '
              f'({_n(r["unidades_devueltas"])} unidades, contadas aparte). Anulaciones: {r["anulacion"]} '
              f'({_n(r["unidades_anuladas"])} unidades quitadas de su venta; {r["anulaciones_de_meses_anteriores"]} '
              f'corrigen una venta de un mes anterior). Ajustes: {r["ajuste"]} líneas. Sin confirmar: '
              f'{r["negativo_pendiente"]} (sus meses siguen pendientes).')
        if r['anulaciones_sin_venta_disponible']:
            print(f'  {r["anulaciones_sin_venta_disponible"]} anulaciones confirmadas no tienen ya una venta a la que '
                  'aplicarse (otra anulación la gastó antes): siguen pendientes.')
        if r['negativo_posterior_al_cierre']:
            print(f'  {r["negativo_posterior_al_cierre"]} líneas negativas se registraron después del último mes completo: '
                  'no se conocían al cerrarlo y no influyen en esta previsión.')
        if r['meses_con_todas_las_ventas_anuladas']:
            print(f'  En {r["meses_con_todas_las_ventas_anuladas"]} meses de producto, todo lo que había eran ventas anuladas: '
                  'serán un cero solo si confirmas que el histórico está completo.')
    print(f'  Unidades brutas vendidas: {_n(r["unidades_brutas"])} (cantidades positivas del archivo: '
          f'{_n(r["unidades_positivas_del_archivo"])}).')


def _n(x: float) -> str:
    x = 0.0 if abs(x) < 1e-9 else x
    return f'{x:,.0f}'.replace(',', '.') if abs(x - round(x)) < 1e-9 else f'{x:,.1f}'.replace(',', ' ').replace('.', ',').replace(' ', '.')
