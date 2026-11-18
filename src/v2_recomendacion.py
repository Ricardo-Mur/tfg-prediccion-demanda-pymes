"""Versión 2: de lo que calcula el motor a lo que se recomienda al usuario. No calcula ninguna previsión.

El motor decide un estado con los registros que hay en los archivos: es el RESPALDO ESTADÍSTICO. La RECOMENDACIÓN
FINAL añade lo que el motor no puede saber: si el usuario ha confirmado que esos archivos contienen todas las
ventas, y si en un grupo hay productos que él mismo ha dado por retirados.

Completitud (decisión del 15-11-2026, hallazgo A-2 de la auditoría de integración)
---------------------------------------------------------------------------------
Si el usuario responde «no» o «no lo sé» a si sus archivos contienen todas las ventas, ninguna previsión se
presenta con «buen respaldo»: como mucho, «orientativa». Lo mismo si solo ha confirmado una parte del periodo y
los meses que se usan para un producto no caben enteros en ella. Una previsión «no recomendada» o con «datos
insuficientes» conserva su estado.

  respaldo_estadistico    el estado que da el motor con los registros disponibles. No cambia nunca.
  estado                  la recomendación final. Solo puede diferir del anterior en un caso:
                          buen respaldo -> orientativa, con el motivo `completitud_sin_confirmar`.
  completitud_historico   confirmado | confirmado_en_parte | no_confirmado | no_se_sabe

No se toca ninguna cantidad, rango, error ni método: la limitación no entra en el cálculo. El mensaje lo explica.

Productos retirados (hallazgo M-1)
----------------------------------
El total y las familias suman todos los productos del archivo, también los que el usuario ha confirmado como
retirados. No se excluyen ni se cambia la previsión del grupo: se avisa de cuánto pesan en las unidades de los
últimos 12 meses, porque una retirada reciente puede hacer que se venda menos de lo previsto.
"""
from __future__ import annotations

import pandas as pd

from . import motor_v2 as M
from . import v2_completitud as K

COMPLETITUD = ('confirmado', 'confirmado_en_parte', 'no_confirmado', 'no_se_sabe')
MOTIVO_TOPE = 'completitud_sin_confirmar'
COLUMNAS_NUEVAS = ('respaldo_estadistico', 'completitud_historico')
COLUMNAS_RETIRADOS = ('productos_retirados', 'parte_de_productos_retirados')
MESES_RECIENTES = 12
TEXTO_COMPLETITUD = {
    'no_confirmado': 'has indicado que tus archivos no contienen todas las ventas',
    'no_se_sabe': 'no has confirmado que tus archivos contengan todas las ventas',
    'confirmado_en_parte': 'el periodo que has confirmado como completo no cubre todos los meses que se usan aquí'}
LO_QUE_SE_PREVE = 'Las cifras se refieren a las unidades registradas en los archivos, que pueden no ser todas las vendidas.'


def completitud(confirmacion: K.ConfirmacionCompletitud, ultimo_mes: int, meses_seguidos: int) -> str:
    """Situación de completitud de los meses que se usan para una serie: los `meses_seguidos` que terminan en
    `ultimo_mes`. Sin meses seguidos, se mira solo el último mes utilizable."""
    if confirmacion.completo == 'no':
        return 'no_confirmado'
    if confirmacion.completo != 'si':
        return 'no_se_sabe'
    usados = range(ultimo_mes - meses_seguidos + 1, ultimo_mes + 1) if meses_seguidos > 0 else (ultimo_mes,)
    return 'confirmado' if all(confirmacion.cubre(m) for m in usados) else 'confirmado_en_parte'


def _con_aviso(mensaje: str, situacion: str, rebajada: bool) -> str:
    porque = TEXTO_COMPLETITUD[situacion]
    if not rebajada:
        return f'{mensaje} Aviso: {porque}. {LO_QUE_SE_PREVE}'
    texto = mensaje.replace('Previsión: ', 'Previsión orientativa: ', 1) if mensaje.startswith('Previsión: ') else mensaje
    aviso = (f'Con los registros de tus archivos tendría buen respaldo, pero {porque}: por eso se presenta como '
             f'orientativa. {LO_QUE_SE_PREVE}')
    corte = texto.find(' unidades. ')
    if corte < 0:
        return f'{texto} {aviso}'
    corte += len(' unidades.')
    return f'{texto[:corte]} {aviso}{texto[corte:]}'


def aplicar_completitud(resumen: pd.DataFrame, confirmacion: K.ConfirmacionCompletitud, ultimo_mes: int) -> pd.DataFrame:
    """Añade `respaldo_estadistico` y `completitud_historico` y deja en `estado` la recomendación final.

    Vale para la tabla de productos y para la de familias y total (cualquier tabla con las columnas del motor).
    Solo cambian `estado`, `motivo` y `mensaje`, y solo en las filas que lo necesitan:
      - buen respaldo sin completitud confirmada -> orientativa, con su explicación;
      - cualquier otra fila con cifras calculadas y sin completitud confirmada -> el mismo estado y un aviso.
    """
    r = resumen.drop(columns=[c for c in COLUMNAS_NUEVAS if c in resumen.columns]).copy()
    situacion = [completitud(confirmacion, ultimo_mes, int(n) if pd.notna(n) else 0) for n in r.meses_seguidos_conocidos]
    estadistico = r.estado.tolist()
    estado, motivo, mensaje = r.estado.tolist(), r.motivo.tolist(), r.mensaje.tolist()
    for i, (s, e) in enumerate(zip(situacion, estadistico)):
        if s == 'confirmado':
            continue
        con_cifras = e in ('buen_respaldo', 'orientativa') or (e == 'no_recomendada' and pd.notna(r.rango_inferior.iloc[i]))
        if e == 'buen_respaldo':
            estado[i], motivo[i] = 'orientativa', MOTIVO_TOPE
        if con_cifras and isinstance(mensaje[i], str) and mensaje[i]:
            mensaje[i] = _con_aviso(mensaje[i], s, rebajada=e == 'buen_respaldo')
    r['estado'], r['motivo'], r['mensaje'] = estado, motivo, mensaje
    donde = list(r.columns).index('estado') + 1
    r.insert(donde, 'respaldo_estadistico', estadistico)
    r.insert(donde + 1, 'completitud_historico', situacion)
    return r


def por_estado(resumen: pd.DataFrame) -> dict[str, int]:
    return {e: int((resumen.estado == e).sum()) for e in M.ESTADOS}


def avisar_antes_de_calcular(confirmacion: K.ConfirmacionCompletitud) -> None:
    if confirmacion.completo != 'si':
        print('  Como no está confirmado que los archivos contengan todas las ventas, ninguna previsión se presentará con '
              '«buen respaldo»: como mucho, orientativa.')


def mostrar_completitud_del_resultado(resumen: pd.DataFrame) -> None:
    """En pantalla: cuántas previsiones se han quedado en orientativa por la completitud, y qué significa."""
    rebajadas = int(((resumen.respaldo_estadistico == 'buen_respaldo') & (resumen.estado == 'orientativa')
                     & (resumen.motivo == MOTIVO_TOPE)).sum())
    sin_confirmar = resumen[resumen.completitud_historico != 'confirmado']
    if sin_confirmar.empty:
        return
    print(f'  COMPLETITUD: {len(sin_confirmar)} producto(s) se han calculado con meses que no has confirmado como completos.')
    if rebajadas:
        print(f'  {rebajadas} de ellos tendrían «buen respaldo» con los registros disponibles; se presentan como orientativas. '
              'La cantidad, el rango y el error no cambian.')
    print(f'  {LO_QUE_SE_PREVE}')


# ------------------------------------------------------------------ productos retirados en las familias y en el total

def peso_de_retirados(productos_del_grupo, unidades_por_producto: dict[str, dict[int, float]], retirados: set[str],
                      ultimo_mes: int) -> tuple[int, float]:
    """(productos retirados del grupo, parte de las unidades de los últimos 12 meses que les corresponde)."""
    recientes = range(ultimo_mes - MESES_RECIENTES + 1, ultimo_mes + 1)
    suma = lambda p: float(sum(unidades_por_producto.get(p, {}).get(m, 0.0) or 0.0 for m in recientes))
    del_grupo = [p for p in productos_del_grupo if p in retirados]
    total = sum(suma(p) for p in productos_del_grupo)
    return len(del_grupo), (sum(suma(p) for p in del_grupo) / total if total > 0 else 0.0)


def avisar_retirados(resumen_grupos: pd.DataFrame, productos_por_grupo: dict[str, list[str]],
                     unidades_por_producto: dict[str, dict[int, float]], retirados: set[str], ultimo_mes: int) -> pd.DataFrame:
    """Añade a la tabla de familias y total cuántos productos retirados tiene cada grupo y cuánto pesan, y un aviso
    en el mensaje de los grupos en los que pesan algo. No cambia ninguna previsión ni ningún estado."""
    r = resumen_grupos.drop(columns=[c for c in COLUMNAS_RETIRADOS if c in resumen_grupos.columns]).copy()
    cuantos, parte, mensaje = [], [], r.mensaje.tolist()
    for i, g in enumerate(r.grupo):
        n, p = peso_de_retirados(productos_por_grupo.get(g, []), unidades_por_producto, retirados, ultimo_mes)
        cuantos.append(n)
        parte.append(p)
        if n and p > 0:
            texto = (f'Aviso: {n} producto{"" if n == 1 else "s"} de este grupo que has confirmado como '
                     f'retirado{"" if n == 1 else "s"} suma{"" if n == 1 else "n"} el {_porcentaje(p)} de sus unidades de los '
                     f'últimos {MESES_RECIENTES} meses. La previsión cuenta con esas ventas: si la retirada es reciente, lo que '
                     'se venda puede ser menos, tanto más cuanto más pesaban.')
            mensaje[i] = f'{mensaje[i]} {texto}' if isinstance(mensaje[i], str) and mensaje[i] else texto
    r['mensaje'] = mensaje
    r['productos_retirados'], r['parte_de_productos_retirados'] = cuantos, parte
    return r


def _porcentaje(x: float) -> str:
    v = 100 * x
    if 0 < v < 1:
        return 'menos del 1 %'
    return f'{v:.0f} %'


def mostrar_retirados(resumen_grupos: pd.DataFrame) -> None:
    con = resumen_grupos[(resumen_grupos.productos_retirados > 0) & (resumen_grupos.parte_de_productos_retirados > 0)]
    if con.empty:
        return
    total = con[con.nivel == 'total']
    if len(total):
        f = total.iloc[0]
        print(f'  PRODUCTOS RETIRADOS: el total incluye {int(f.productos_retirados)} producto(s) confirmado(s) como retirado(s), que '
              f'suman el {_porcentaje(f.parte_de_productos_retirados)} de las unidades de los últimos {MESES_RECIENTES} meses.')
    print('  No se han quitado de los totales ni se ha cambiado ninguna previsión. Si se han retirado hace poco, la venta '
          'futura puede ser menor que la prevista, sobre todo en los grupos donde más pesaban.')
