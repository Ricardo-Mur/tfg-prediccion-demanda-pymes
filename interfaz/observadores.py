"""Observadores de solo lectura: lo que la revisión agrupada necesita para enseñar tablas en lugar de preguntas.

Se instalan solo en el proceso de cálculo, solo en el modo de preguntas (el que termina al llegar a «CALCULANDO»)
y solo en memoria: los archivos congelados no cambian. El cálculo de verdad se lanza después en otro proceso, sin
observadores, con las respuestas grabadas; por eso un observador no puede influir en ningún resultado.

Cada envoltorio llama a la función original con los mismos argumentos y devuelve lo mismo que ella; si la función
lanza una excepción, sale tal cual. Además deja en `<espacio>/revision/` una copia de lo que la función recibe o
devuelve, con solo los campos que el programa ya enseña en pantalla (fila, fecha, producto, cantidad, serie,
documento, posible venta). Nunca copia columnas de clientes. Si copiar algo falla, se anota y el programa sigue igual:
la interfaz, sin esos datos, vuelve a las preguntas una a una.

Archivos:
  series.json            antes de preguntar por las series: cada serie y la propuesta del programa
  lineas.json            después de las series: las líneas positivas y negativas que se van a preguntar, por grupos
  lineas_reales.json     las filas que devuelven de verdad `positivas_a_confirmar` y `negativas_a_confirmar`
  completitud.json       antes de preguntar si los datos están completos
  productos.json         antes de preguntar por los productos sin ventas recientes: todos, con su última venta
  registrado_*.json      lo que devuelve cada diálogo: para comprobar que coincide con lo decidido
  antes_de_calcular.json lo que queda pendiente y qué meses del total y de cada producto deja sin conocer
"""
from __future__ import annotations

import inspect
import json
import os
import time
from pathlib import Path

ESTADO: dict = {}          # referencias (no copias) a lo último que ha pasado por los observadores


def _mes_texto(i) -> str:
    return f'{int(i) % 12 + 1:02d}/{int(i) // 12}'


def _escribir(carpeta: Path, nombre: str, datos) -> None:
    temporal, ruta = carpeta / (nombre + '.tmp'), carpeta / nombre
    temporal.write_text(json.dumps(datos, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    for _ in range(99):
        try:
            os.replace(temporal, ruta)
            return
        except PermissionError:
            time.sleep(0.05)
    os.replace(temporal, ruta)


def _linea(r, todas) -> dict:
    """Los mismos datos que enseña el programa al pulsar «Ver las líneas» (`_mostrar_lineas`), sin clientes."""
    import pandas as pd
    d = {'fila': int(r.fila) + 1, 'mes': int(r.mes), 'fecha': f'{r.orden:%d/%m/%Y}' if not pd.isna(r.orden) else '',
         'producto': str(r.producto), 'cantidad': float(r.cantidad), 'serie': str(r.serie) if r.serie else '',
         'documento': str(r.id_factura) if r.id_factura is not None and not pd.isna(r.id_factura) else ''}
    if not pd.isna(r.fila_venta):
        v = todas.iloc[int(r.fila_venta)]
        d['venta'] = {'fila': int(r.fila_venta) + 1, 'fecha': f'{v.orden:%d/%m/%Y}', 'cantidad': float(v.cantidad)}
    return d


def instalar(predecir_v2, carpeta: Path, registrar=lambda texto: None) -> None:
    """Envuelve, en este proceso y en memoria, las funciones que hacen las preguntas de la etapa 4."""
    from src import v2_completitud as K
    from src import v2_dialogo as D
    from src import v2_operaciones as OP

    carpeta.mkdir(exist_ok=True)
    for viejo in carpeta.glob('*.json'):
        viejo.unlink()
    # Los observadores calculan con las funciones originales, nunca con sus envoltorios.
    positivas_a_confirmar, negativas_a_confirmar = OP.positivas_a_confirmar, OP.negativas_a_confirmar

    def envolver(modulo, nombre: str, antes=None, despues=None):
        original = getattr(modulo, nombre)
        firma = inspect.signature(original)

        def envoltorio(*args, **kwargs):
            if antes is not None:
                try:
                    antes(firma.bind(*args, **kwargs).arguments)
                except Exception as e:                       # observar nunca puede parar al programa
                    registrar(f'observador: no se ha podido copiar lo que recibe {nombre} ({type(e).__name__})')
            resultado = original(*args, **kwargs)
            if despues is not None:
                try:
                    despues(firma.bind(*args, **kwargs).arguments, resultado)
                except Exception as e:
                    registrar(f'observador: no se ha podido copiar lo que devuelve {nombre} ({type(e).__name__})')
            return resultado

        envoltorio.__wrapped__ = original
        setattr(modulo, nombre, envoltorio)

    # ---------------------------------------------------------------- operaciones
    def guardar_analisis(_, analisis):
        ESTADO['analisis'] = analisis

    def antes_de_operaciones(a):
        analisis = a['analisis']
        propuestas = OP.proponer_series(analisis)
        series = [{'serie': str(f.serie), 'lineas': int(f.lineas), 'positivas': int(f.positivas), 'negativas': int(f.negativas),
                   'cero': int(f.cero), 'desde': _mes_texto(f.primer_mes), 'hasta': _mes_texto(f.ultimo_mes),
                   'propuesta': propuestas[f.serie][0] or '', 'motivo': propuestas[f.serie][1]}
                  for f in analisis.series.itertuples(index=False)] if analisis.con_serie else []
        _escribir(carpeta, 'series.json', {'con_serie': bool(analisis.con_serie), 'series': series})
        if not analisis.con_serie:          # sin series no se pregunta por ellas: las líneas se conocen ya
            exportar_lineas(analisis, [])

    def despues_de_series(a, series):
        exportar_lineas(a['analisis'], series)

    def exportar_lineas(analisis, series):
        provisional = OP.ReglasOperaciones(series=tuple(series))
        positivas = positivas_a_confirmar(analisis, provisional)
        negativas = negativas_a_confirmar(analisis, provisional)
        todas = analisis.lineas
        grupos_positivas = [{'serie': str(serie), 'lineas': [_linea(r, todas) for r in g.itertuples(index=False)]}
                            for serie, g in positivas.groupby('serie', sort=False)]
        grupos_negativas = []
        for g in OP.GRUPOS:
            del_grupo = negativas[negativas.grupo == g]
            if not del_grupo.empty:
                grupos_negativas.append({'grupo': g, 'texto': OP.TEXTO_GRUPO[g], 'sin_anulacion': g == 'sin_venta_anterior',
                                         'lineas': [_linea(r, todas) for r in del_grupo.itertuples(index=False)]})
        _escribir(carpeta, 'lineas.json', {'series': [list(x) for x in series], 'positivas': grupos_positivas,
                                           'negativas': grupos_negativas})

    reales: dict = {}

    def filas_reales(clave):
        def anotar(_, tabla):
            reales[clave] = sorted(int(f) + 1 for f in tabla.fila)
            _escribir(carpeta, 'lineas_reales.json', reales)
        return anotar

    def despues_de_operaciones(_, reglas):
        _escribir(carpeta, 'registrado_operaciones.json', {
            'series': [list(x) for x in reglas.series], 'positivas': [list(x) for x in reglas.positivas],
            'negativos': [list(x) for x in reglas.negativos], 'excepciones': [list(x) for x in reglas.excepciones]})

    def guardar_operaciones(a, _):
        ESTADO['operaciones'] = a['resultado']

    # ---------------------------------------------------------------- completitud y productos
    def antes_de_completitud(a):
        indicio = a.get('indicio')
        _escribir(carpeta, 'completitud.json', {'primer_mes': int(a['primer_mes']), 'ultimo_mes_archivo': int(a['ultimo_mes_archivo']),
                                                'indicio': indicio.texto if indicio is not None else ''})

    def despues_de_completitud(_, confirmacion):
        _escribir(carpeta, 'registrado_completitud.json', confirmacion.a_dict())

    def antes_de_productos(a):
        grupos, ultimo, ultimas = a['grupos'], int(a['ultimo_mes']), a.get('ultimas') or {}
        _escribir(carpeta, 'productos.json', {'ultimo_mes': ultimo, 'grupos': [{
            'clave': g.clave, 'descripcion': g.descripcion, 'numero': k,
            'desde_minimo': min((ultimas[p] for p in g.productos if p in ultimas), default=0) + 1, 'desde_maximo': ultimo + 1,
            'productos': [{'codigo': p, 'ultima': ultimas.get(p), 'desde_minimo': ultimas.get(p, -1) + 1,
                           'admite_excepcion': ',' not in p and p == p.strip() and bool(p)} for p in g.productos]}
            for k, g in enumerate(grupos, 1)]})

    def despues_de_productos(_, r):
        _escribir(carpeta, 'registrado_productos.json', {
            'respuestas': r['respuestas'], 'excepciones': r['excepciones'],
            'retirada_por_grupo': {k: str(v) for k, v in r['retirada_por_grupo'].items()},
            'retirada_por_producto': {k: str(v) for k, v in r['retirada_por_producto'].items()}})

    def antes_de_calcular(a, _):
        _escribir(carpeta, 'antes_de_calcular.json', resumen_antes_de_calcular(a['hist']))

    envolver(OP, 'analizar_operaciones', despues=guardar_analisis)
    envolver(predecir_v2, 'preguntar_operaciones', antes=antes_de_operaciones, despues=despues_de_operaciones)
    envolver(D, '_preguntar_series', despues=despues_de_series)
    envolver(OP, 'positivas_a_confirmar', despues=filas_reales('positivas'))
    envolver(OP, 'negativas_a_confirmar', despues=filas_reales('negativas'))
    envolver(OP, 'mostrar_operaciones_corto', despues=guardar_operaciones)
    envolver(D, 'pedir_confirmacion', antes=antes_de_completitud, despues=despues_de_completitud)
    envolver(D, 'preguntar_estado_productos', antes=antes_de_productos, despues=despues_de_productos)
    envolver(K, 'mostrar_completitud_corta', despues=antes_de_calcular)
    registrar('observadores: instalados (solo lectura, solo en el modo de preguntas)')


def resumen_antes_de_calcular(hist) -> dict:
    """Qué queda sin aclarar y qué deja sin conocer, con las funciones del propio programa. No cambia nada."""
    import pandas as pd
    from src import v2_agregados as AG
    from src.motor_v2 import ConfiguracionV2
    from src.v2_candidatos import tramo_final

    ultimo = int(hist.ultimo_mes)
    prod = hist.productos
    datos = {'ultimo_mes': ultimo, 'ultimo_mes_archivo': int(hist.ultimo_mes_archivo), 'completo': hist.confirmacion.completo,
             'periodo': [str(hist.confirmacion.periodo_inicio or ''), str(hist.confirmacion.periodo_fin or '')],
             'evaluaciones_minimas': int(ConfiguracionV2().evaluaciones_minimas),
             'estados': {e: sorted(map(str, prod.loc[prod.estado_comercial == e, 'producto']))
                         for e in ('activo_confirmado', 'retirado_confirmado')},
             'meses_a_cero': int(len(hist.conversiones)), 'meses_desconocidos': int(prod.meses_desconocidos.sum()),
             'meses_pendientes': int(prod.meses_pendientes.sum())}
    grupos = AG.construir_grupos(hist, None)
    total = grupos[0]
    inicio, tramo = tramo_final(total.serie, ultimo + 1)
    datos['total'] = {'meses_seguidos': len(tramo), 'desde': _mes_texto(inicio) if inicio is not None else '',
                      'meses_sin_total': [{'mes': int(m), 'motivo': motivo, 'productos_pendientes': int(pend)}
                                          for m, _, conocido, motivo, _, pend in total.meses if not conocido],
                      'aviso': AG._aviso_historial(total), 'situacion': list(AG._situacion(total, ultimo) or ())}
    analisis, operaciones = ESTADO.get('analisis'), ESTADO.get('operaciones')
    pendientes, sin_confirmar, impiden = [], [], []
    if analisis is not None and operaciones is not None and operaciones.lineas is not None:
        todas = analisis.lineas
        clases = operaciones.lineas.clase.to_numpy()
        con_clase = todas[['fila', 'producto', 'mes', 'cantidad', 'serie']].assign(clase=clases)
        impiden = AG.operaciones_que_impiden_el_total(grupos, con_clase).assign(
            mes=lambda t: t.mes.astype(str)).to_dict('records')
        marcadas = todas.assign(clase=clases)
        for r in marcadas[marcadas.clase.isin(('negativo_pendiente', 'positivo_sin_confirmar', 'no_interpretable'))].itertuples(index=False):
            linea = _linea(r, todas) | {'clase': r.clase, 'en_el_periodo': int(r.mes) <= ultimo}
            pendientes.append(linea)
        sin_serie = marcadas[marcadas.clase == 'serie_sin_confirmar']
        for serie, g in sin_serie.groupby(sin_serie.serie.astype(str), sort=True):
            sin_confirmar.append({'serie': serie, 'lineas': int(len(g)), 'productos': int(g.producto.nunique()),
                                  'desde': _mes_texto(g.mes.min()), 'hasta': _mes_texto(g.mes.max())})
    datos['pendientes'], datos['series_sin_confirmar'] = pendientes, sin_confirmar
    datos['impiden_el_total'] = [{k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in f.items()} for f in impiden]
    afectados = {}
    for p in sorted({l['producto'] for l in pendientes if l['en_el_periodo']} | {f['producto'] for f in impiden}):
        serie = hist.series.get(p)
        if serie is None:
            continue
        ini, y = tramo_final(serie, ultimo + 1)
        afectados[p] = {'meses_seguidos': len(y), 'desde': _mes_texto(ini) if ini is not None else ''}
    datos['productos'] = afectados
    return datos
