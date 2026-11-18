"""Versión 2: previsión del mes siguiente con aviso de fiabilidad.

Reutiliza la importación, la confirmación de columnas y la preparación de datos de la versión 1
(sin modificarlas). Después pregunta si el histórico y el último mes están completos y si los productos
sin ventas recientes han seguido a la venta, y calcula las previsiones con el motor de la versión 2.
Antes revisa las columnas adicionales del archivo y pregunta qué son las líneas negativas, para calcular las
unidades brutas vendidas (las devoluciones se cuentan aparte; las anulaciones se quitan de su venta).
También pregunta qué son las columnas de estado del artículo, las que sirven para agrupar productos, los
importes y las existencias: sin confirmar no se usa ninguna, y ninguna cambia las cantidades previstas.
Al final añade el diagnóstico de los datos de cada producto (lagunas, operaciones pendientes, líneas
repetidas, ventas muy altas, patrón de ventas, suficiencia y métodos), que no cambia ninguna previsión.
Además de la previsión de cada producto calcula la del total de los productos del archivo y, si hay familias
(una columna confirmada como agrupación o un archivo con la familia de cada producto), la de cada familia. Cada
nivel tiene su propia comprobación retrospectiva y su propio estado (etapa 5).
Los resultados y las confirmaciones se guardan en salidas_v2/, separados de los de la versión 1.

Uso:  python predecir_v2.py [--procesos 4] [--confirmaciones salidas_v2/<fecha>/confirmaciones_v2.json]
                            [--familias familias.csv [--columna-producto X --columna-familia Y]]
Las columnas del archivo de familias se reconocen por su cabecera. Si no queda claro cuál es el producto y cuál
la familia, el programa pide que se confirmen; también se pueden indicar con --columna-producto y --columna-familia.
Sin confirmación, el archivo no se usa.
"""
import argparse
import contextlib
import hashlib
import io
from pathlib import Path

from src import v2_agregados as AG
from src import v2_articulos as ART
from src import v2_columnas as COL
from src import v2_completitud as K
from src import v2_diagnostico as DG
from src import v2_exportacion as EXP
from src import v2_importes as IM
from src import v2_operaciones as OP
from src import v2_recomendacion as REC
from src.flujo_sencillo import construir_resumen, ejecutar_terminal
from src.gestion_salidas import identificar_conjunto
from src.motor_v2 import ConfiguracionV2, ejecutar_v2, mostrar_v2_corto
from src.objetivo_mensual import mostrar_objetivo_mensual_corto, preparar_objetivo_mensual
from src.v2_dialogo import (preguntar_columnas, preguntar_columnas_de_familias, preguntar_operaciones,
                            preparar_historico_con_usuario)

RAIZ = Path(__file__).resolve().parent
CARPETA = RAIZ / 'datos_prueba'
SALIDAS = RAIZ / 'salidas_v2'
COLUMNAS_PRODUCTO = ['producto', 'estado_comercial', 'retirado_desde', 'ultima_venta', 'meses_sin_ventas',
                     'sin_ventas_recientes', 'ceros_confirmados',
                     'meses_desconocidos', 'meses_pendientes']


def identidad_del_conjunto(resultado: dict) -> dict | None:
    """Huella de los archivos y columnas confirmados (función de la versión 1, solo lectura)."""
    try:
        identificador, manifiesto = identificar_conjunto(resultado)
    except (ValueError, FileNotFoundError):
        return None
    return {'identificador': identificador, **manifiesto}


def confirmaciones_reutilizables(ruta: Path | None, huella: str) -> Path | None:
    """La ruta de unas confirmaciones guardadas, solo si TODAS son de estos mismos archivos y columnas.

    Se comprueban las de completitud, las reglas de operaciones y las de columnas. Si falta la huella de los
    archivos actuales, falta la de alguna confirmación o no coinciden, no se reutiliza ninguna: se dice con
    claridad y se devuelve None, para que el programa vuelva a preguntar. Lo mismo si no se pueden leer.
    """
    if ruta is None:
        return None
    ruta = Path(ruta)
    try:
        if not ruta.exists():
            raise K.ConfirmacionNoReutilizable(f'No encuentro el archivo de confirmaciones «{ruta.name}».')
        K.cargar_registro(ruta, huella_conjunto=huella)
        for nombre, cargar in (('reglas_operaciones_v2.json', OP.cargar_reglas), ('columnas_confirmadas_v2.json', ART.cargar_confirmacion)):
            if ruta.with_name(nombre).exists():
                cargar(ruta.with_name(nombre), huella_conjunto=huella)
    except K.ConfirmacionNoReutilizable as e:
        motivo = str(e)
    except (OSError, KeyError, TypeError, ValueError):
        motivo = 'Las confirmaciones guardadas no se pueden leer.'
    else:
        return ruta
    print('\nCONFIRMACIONES GUARDADAS')
    print(f'  {motivo}')
    print('  Unas confirmaciones solo valen para los mismos archivos y las mismas columnas con los que se hicieron: no reutilizo '
          'ninguna.')
    print('  Te hago las preguntas de nuevo. Al terminar, las respuestas se guardan con los resultados y podrás reutilizarlas.')
    return None


def familias_del_archivo(ruta: Path, tabla, productos_de_las_ventas, columna_producto: str | None, columna_familia: str | None,
                         confirmaciones: Path | None):
    """(producto -> familia y su registro, '', columnas), o (None, motivo, columnas) si no se sabe qué es cada
    columna y nadie lo confirma.

    Por orden: las columnas indicadas en la orden; las cabeceras, si identifican el producto y la familia sin
    ninguna duda; las que el usuario confirmó en una ejecución anterior para este mismo archivo; y, si no,
    se pregunta. Nunca se asignan por su posición.
    `columnas` son los nombres de las columnas del archivo. Si no tiene fila de cabecera son «Columna 1»,
    «Columna 2»...: su primera fila son datos y no se anota en ningún sitio.
    """
    if columna_producto is not None:
        return AG.correspondencia(tabla, columna_producto, columna_familia, ruta, AG.IDENTIFICACIONES[1]), '', list(tabla.columns)
    ident = AG.identificar_columnas(tabla, productos_de_las_ventas)
    columnas = [str(c) for c in ident.tabla.columns]
    if ident.inequivoca:
        return AG.correspondencia(ident.tabla, ident.producto, ident.familia, ruta, AG.IDENTIFICACIONES[0]), '', columnas
    if confirmaciones is not None:
        guardadas = AG.columnas_guardadas(Path(confirmaciones).with_name('agregados_v2.json'),
                                          hashlib.sha256(Path(ruta).read_bytes()).hexdigest())
        if guardadas is not None and all(c in ident.tabla.columns for c in guardadas):
            print(f'\nSe reutilizan las columnas del archivo de familias que confirmaste: producto «{guardadas[0]}» y familia '
                  f'«{guardadas[1]}».')
            return AG.correspondencia(ident.tabla, *guardadas, ruta, AG.IDENTIFICACIONES[1]), '', columnas
    elegidas = preguntar_columnas_de_familias(ident, Path(ruta).name)
    if elegidas is None:
        return None, ident.motivo, columnas
    return AG.correspondencia(ident.tabla, elegidas[0], elegidas[1], ruta, elegidas[2]), '', columnas


def main(procesos: int = 1, confirmaciones: Path | None = None, familias: Path | None = None,
         columna_producto: str | None = None, columna_familia: str | None = None):
    # El archivo de familias se abre antes de nada: si no vale, se dice al principio y no después de calcular.
    if (columna_producto is None) != (columna_familia is None):
        raise ValueError('Hay que indicar las dos columnas: --columna-producto y --columna-familia.')
    if familias is None and columna_producto is not None:
        raise ValueError('--columna-producto y --columna-familia solo tienen sentido con --familias.')
    tabla_familias = AG.leer_tabla_de_familias(familias) if familias is not None else None
    for c in (columna_producto, columna_familia):
        if c is not None and c not in tabla_familias.columns:
            raise ValueError(f'El archivo de familias no tiene ninguna columna «{c}». Tiene: '
                             + ', '.join(f'«{x}»' for x in tabla_familias.columns))
    resultado = ejecutar_terminal(CARPETA)  # elegir archivos y confirmar columnas, igual que en la versión 1
    if resultado is None:
        return None
    # Etapa 4: columnas adicionales y operaciones. Sin reglas confirmadas, todo queda como en la versión 1.
    datos = resultado['limpio'].datos
    # Las confirmaciones guardadas solo se reutilizan si son de estos mismos archivos. Se decide aquí, antes de usar
    # ninguna: si no valen, no vale ninguna (tampoco las columnas del archivo de familias) y se pregunta todo.
    conjunto = identidad_del_conjunto(resultado)
    huella = (conjunto or {}).get('huella_conjunto', '')
    confirmaciones = confirmaciones_reutilizables(confirmaciones, huella)
    # Qué columna del archivo de familias es el producto y cuál la familia: se resuelve ya, con los códigos de las
    # ventas delante y antes de calcular nada. Si no es inequívoco y no se confirma, el archivo no se usa.
    familias_importadas, familias_sin_usar, columnas_de_familias = None, '', []
    if tabla_familias is not None:
        codigos = {str(x).strip() for x in datos['producto'].dropna()}
        familias_importadas, familias_sin_usar, columnas_de_familias = familias_del_archivo(
            familias, tabla_familias, codigos, columna_producto, columna_familia, confirmaciones)
    inventario = COL.inventariar_resultado(resultado)
    COL.mostrar_inventario_corto(inventario)
    analisis = OP.analizar_operaciones(datos, inventario.extras)
    guardadas = Path(confirmaciones).with_name('reglas_operaciones_v2.json') if confirmaciones else None
    if guardadas is not None and guardadas.exists():
        reglas = OP.cargar_reglas(guardadas, huella_conjunto=huella)     # deben ser de estos mismos archivos
        print(f'\nSe reutilizan las reglas de operaciones guardadas el {reglas.fecha_hora or "(sin fecha)"}.')
    else:
        reglas = preguntar_operaciones(analisis, huella_conjunto=huella)
    operaciones = OP.aplicar_reglas(datos, analisis, reglas)
    OP.mostrar_operaciones_corto(operaciones)
    # Estado del artículo, agrupaciones, fecha de alta, importes y existencias: solo lo que el usuario confirme.
    columnas = ART.analizar_columnas(inventario, datos)
    guardadas_columnas = Path(confirmaciones).with_name('columnas_confirmadas_v2.json') if confirmaciones else None
    if guardadas_columnas is not None and guardadas_columnas.exists():
        conf_columnas = ART.cargar_confirmacion(guardadas_columnas, huella_conjunto=huella)
        print(f'Se reutilizan las confirmaciones de columnas guardadas el {conf_columnas.fecha_hora or "(sin fecha)"}.')
        if conf_columnas.importes_sin_identificar:
            print('  Los importes confirmados entonces no decían si eran netos o brutos: no se usan. Para usarlos, vuelve a '
                  'ejecutar el programa sin --confirmaciones e identifícalos.')
    else:
        conf_columnas = preguntar_columnas(columnas, huella_conjunto=huella)
    resumen_estado: dict = {}

    def estados_de_columnas(ultimas: dict, ultimo_completo: int) -> dict:
        nonlocal resumen_estado
        estados, resumen_estado = ART.estados_desde_columnas(columnas.estado, conf_columnas, ultimas, ultimo_completo)
        return estados

    con_estado = any(s != 'desconocido' for _, _, s in conf_columnas.estado)
    series, limpio = resultado['series'], resultado['limpio']
    if operaciones.cambia:
        # La tabla de ventas brutas vuelve a pasar por la preparación de la versión 1, sin modificarla.
        with contextlib.redirect_stdout(io.StringIO()):
            bruto = construir_resumen(operaciones.tabla_bruta, {'fecha': 0, 'producto': 1, 'cantidad': 2})
        if len(bruto['limpio'].datos) != len(operaciones.tabla_bruta):
            raise RuntimeError('La tabla de ventas brutas no se ha preparado completa.')
        series, limpio = bruto['series'], bruto['limpio']
    objetivo = preparar_objetivo_mensual(series, limpieza=limpio)
    mostrar_objetivo_mensual_corto(objetivo)
    # Con una fecha diaria reconocida, el indicio sobre el último mes se calcula con ella.
    fechas = inventario.extras['fecha_diaria'] if inventario.tiene('fecha_diaria') else datos['fecha']
    def al_cierre(ultimo_completo: int):
        """Si alguna línea negativa se registró después del último mes completo, las operaciones se vuelven a
        aplicar sin ella: el histórico es el disponible al cerrar ese mes."""
        nonlocal operaciones
        lineas_neg = analisis.lineas[analisis.lineas.interpretable & (analisis.lineas.cantidad < 0)]
        if not (lineas_neg.mes > ultimo_completo).any():
            return None
        operaciones = OP.aplicar_reglas(datos, analisis, reglas, hasta_mes=ultimo_completo)
        OP.mostrar_operaciones_corto(operaciones)
        if not operaciones.cambia:
            return preparar_objetivo_mensual(resultado['series'], limpieza=resultado['limpio']), operaciones.meses_anulados
        with contextlib.redirect_stdout(io.StringIO()):
            rehecho = construir_resumen(operaciones.tabla_bruta, {'fecha': 0, 'producto': 1, 'cantidad': 2})
        if len(rehecho['limpio'].datos) != len(operaciones.tabla_bruta):
            raise RuntimeError('La tabla de ventas brutas no se ha preparado completa.')
        return preparar_objetivo_mensual(rehecho['series'], limpieza=rehecho['limpio']), operaciones.meses_anulados

    hist, datos_registro = preparar_historico_con_usuario(
        objetivo, fechas, conjunto=conjunto, confirmaciones_guardadas=confirmaciones,
        meses_anulados=operaciones.meses_anulados, al_conocer_ultimo_mes=al_cierre,
        estados_de_columnas=estados_de_columnas if con_estado else None)
    K.mostrar_completitud_corta(hist)
    REC.avisar_antes_de_calcular(hist.confirmacion)
    print('\nCALCULANDO LA VERSIÓN 2 (puede tardar unos minutos con muchos productos)...')
    cfg = ConfiguracionV2(procesos=procesos)
    # Anulaciones registradas después del mes de su venta: la comprobación retrospectiva no las usa antes de tiempo.
    revisiones = operaciones.revisiones_para_motor(hist.ultimo_mes, hist.series)
    informe = ejecutar_v2(hist.series, cfg, ultimo_mes=hist.ultimo_mes, situaciones=hist.situaciones,
                          revisiones=revisiones)
    # Del respaldo estadístico (lo que da el motor con los registros que hay) a la recomendación final: sin la
    # completitud confirmada, ninguna previsión pasa de «orientativa». No cambia cantidades, rangos ni errores.
    informe.resumen = REC.aplicar_completitud(informe.resumen, hist.confirmacion, hist.ultimo_mes)
    informe.por_estado = REC.por_estado(informe.resumen)
    # Diagnóstico de los datos de cada producto: solo con fecha, producto y cantidad. No cambia el resultado.
    lineas = datos.assign(fecha=fechas) if inventario.tiene('fecha_diaria') else datos
    diagnostico = DG.diagnosticar_productos(hist, informe.resumen, lineas, cfg, operaciones=operaciones.por_producto,
                                            repetidas=OP.repetidas_con_identificadores(analisis))
    informe.resumen = informe.resumen.merge(hist.productos[COLUMNAS_PRODUCTO], on='producto', how='left')
    informe.resumen = DG.integrar_en_resumen(informe.resumen, diagnostico)
    # Lo que aportan las columnas confirmadas: grupo de cada producto, precios y existencias. No cambia la previsión.
    productos = sorted(hist.series)
    if con_estado and not resumen_estado:      # confirmaciones reutilizadas: el recuento se rehace solo para informar
        ultimas = {f.producto: K.indice(f.ultima_operacion_valida) for f in hist.productos.itertuples(index=False)
                   if f.ultima_operacion_valida is not None}
        resumen_estado = ART.estados_desde_columnas(columnas.estado, conf_columnas, ultimas, hist.ultimo_mes)[1]
    # Los precios se calculan con las ventas que quedan después de las anulaciones confirmadas.
    precios = IM.resumen_de_precios(inventario, datos, conf_columnas, hist.ultimo_mes, operaciones=operaciones.lineas) \
        if conf_columnas.importes else None
    existencias = IM.existencias_por_producto(columnas.existencias, conf_columnas)
    # Familias, por este orden:
    #   1. el archivo producto-familia, si sus columnas están identificadas y da familia a algún producto de las ventas;
    #   2. si no (columnas sin confirmar o ninguna correspondencia utilizable), la agrupación confirmada en la etapa 4;
    #   3. si tampoco la hay, se prevé por producto y el total.
    familia_de, tipo_agrupacion, origen_familias, detalle_familias = None, '', '', {}
    if familias_importadas is not None:
        mapa, info = familias_importadas
        del_archivo, cuenta = AG.asignar_familias(mapa, productos)
        print(f'\nFAMILIAS DEL ARCHIVO «{info["archivo"]}»: {info["familias"]} familias para {cuenta["con_familia"]} de los '
              f'{cuenta["productos_del_archivo_de_ventas"]} productos; {cuenta["sin_familia"]} quedan sin familia.')
        if info['productos_con_dos_familias_distintas']:
            print(f'  {info["productos_con_dos_familias_distintas"]} producto(s) aparecen con dos familias distintas: quedan sin familia.')
        if cuenta['codigos_de_la_correspondencia_que_no_estan_en_las_ventas']:
            print(f'  {cuenta["codigos_de_la_correspondencia_que_no_estan_en_las_ventas"]} código(s) del archivo de familias no '
                  'están en las ventas.')
        if del_archivo:
            familia_de = del_archivo
            detalle_familias = {'archivo_de_familias': info, 'asignacion': cuenta}
            tipo_agrupacion, origen_familias = 'archivo', f'archivo «{info["archivo"]}» con la familia de cada producto'
            if conf_columnas.agrupacion_principal:
                print(f'  Hay también una agrupación confirmada («{conf_columnas.agrupacion_principal}»): para los niveles se usa el archivo.')
        else:
            # El archivo se ha podido leer, pero no da familia a ningún producto de las ventas: no sirve para agrupar.
            familias_sin_usar = AG.MOTIVO_SIN_CORRESPONDENCIAS
            detalle_familias = {'archivo_de_familias_sin_usar': {
                'archivo': info['archivo'], 'motivo': familias_sin_usar, 'huella_sha256': info['huella_sha256'],
                'columnas_del_archivo': columnas_de_familias, 'columna_del_producto': info['columna_del_producto'],
                'columna_de_la_familia': info['columna_de_la_familia'],
                'identificacion_de_las_columnas': info['identificacion_de_las_columnas'], 'productos': info['productos'],
                'productos_con_familia': info['productos_con_familia'], 'asignacion': cuenta}}
    elif tabla_familias is not None:
        detalle_familias = {'archivo_de_familias_sin_usar': {'archivo': Path(familias).name, 'motivo': familias_sin_usar,
                                                             'columnas_del_archivo': columnas_de_familias}}
    if familia_de is None and tabla_familias is not None:
        porque = (AG.MOTIVO_SIN_CORRESPONDENCIAS if familias_importadas is not None
                  else 'sus columnas no se han confirmado')
        sigue = (f'Se usa la agrupación confirmada «{conf_columnas.agrupacion_principal}».' if conf_columnas.agrupacion_principal
                 else 'Se prevé por producto y el total.')
        print(f'{"  " if familias_importadas is not None else chr(10)}El archivo de familias «{Path(familias).name}» no se usa: '
              f'{porque}. {sigue}')
    if familia_de is None and conf_columnas.agrupacion_principal:
        principal = conf_columnas.agrupacion_principal
        de_la_columna = ART.grupos_de_productos(columnas.atributos, principal, productos)
        familia_de = {q: g for q, g in de_la_columna.items() if g != ART.SIN_GRUPO}
        tipo_agrupacion = dict(conf_columnas.agrupaciones)[principal]
        origen_familias = f'columna «{principal}», confirmada como {ART.TEXTO_AGRUPACION[tipo_agrupacion]}'
    resumen_familias = None
    if familia_de:
        grupo_de = {q: familia_de.get(q, AG.SIN_FAMILIA) for q in productos}
        informe.resumen['grupo'] = informe.resumen.producto.map(grupo_de)
        resumen_familias = ART.resumen_por_grupo(grupo_de, informe.resumen, hist.series, hist.ultimo_mes)
    # Etapa 5: el total y cada familia, con su propia serie, su comprobación retrospectiva y su estado.
    # Las líneas con su clase solo sirven para decir qué operaciones impiden conocer algún total. Sin datos de clientes.
    lineas_con_clase = analisis.lineas[['fila', 'producto', 'mes', 'cantidad', 'serie']].assign(clase=operaciones.lineas.clase.to_numpy())
    grupos = AG.construir_grupos(hist, familia_de)
    agregados = AG.prever_grupos(grupos, cfg, hist.ultimo_mes, revisiones=operaciones.revisiones,
                                 tipo_de_agrupacion=tipo_agrupacion, origen_familias=origen_familias, lineas=lineas_con_clase)
    agregados.registro |= detalle_familias
    # Mismas columnas que en los productos (un total solo existe con meses confirmados, así que aquí nunca se rebaja)
    # y aviso de cuánto pesan los productos confirmados como retirados. No cambia ninguna previsión.
    agregados.resumen = REC.aplicar_completitud(agregados.resumen, hist.confirmacion, hist.ultimo_mes)
    retirados = set(hist.productos.loc[hist.productos.estado_comercial == 'retirado_confirmado', 'producto'])
    agregados.resumen = REC.avisar_retirados(agregados.resumen, {g.nombre: list(g.productos) for g in grupos},
                                             AG.registrado_por_producto(hist), retirados, hist.ultimo_mes)
    con_retirados = agregados.resumen[agregados.resumen.productos_retirados > 0]
    if len(con_retirados):
        agregados.registro |= {'productos_retirados_por_grupo': {
            f.grupo: {'productos': int(f.productos_retirados), 'parte_de_las_unidades_de_los_ultimos_12_meses': float(f.parte_de_productos_retirados)}
            for f in con_retirados.itertuples(index=False)}}
    articulos = ART.tabla_de_articulos(columnas, conf_columnas, inventario, datos, productos, hist.estados_comerciales,
                                       precios.por_producto if precios else None, existencias)
    mostrar_v2_corto(informe)
    REC.mostrar_completitud_del_resultado(informe.resumen)
    DG.mostrar_diagnostico_corto(diagnostico)
    ART.mostrar_articulos_corto(columnas, conf_columnas, resumen_estado, resumen_familias)
    IM.mostrar_importes_corto(precios, existencias)
    AG.mostrar_agregados_corto(agregados)
    REC.mostrar_retirados(agregados.resumen)
    carpeta = EXP.carpeta_nueva(SALIDAS)          # nunca reutiliza una carpeta que ya exista
    rutas = (EXP.exportar_motor(informe, carpeta) | K.guardar_registro(hist, carpeta, **datos_registro)
             | DG.exportar_diagnostico(diagnostico, carpeta) | COL.exportar_inventario(inventario, carpeta)
             | OP.exportar_operaciones(operaciones, analisis, carpeta)
             | ART.exportar_articulos(articulos, conf_columnas, columnas, carpeta, familias=resumen_familias,
                                      precios_por_mes=precios.por_mes if precios else None, resumen_estado=resumen_estado)
             | AG.exportar_agregados(agregados, carpeta) | EXP.guardar_entorno(carpeta))
    print(f'\nResultados guardados en: {carpeta}')
    for nombre, ruta in rutas.items():
        print(f'  {nombre}: {ruta.name}')
    return informe


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TFG: previsión con la versión 2 del motor')
    parser.add_argument('--procesos', type=int, default=1,
                        help='procesos en paralelo para calcular los candidatos (por defecto 1)')
    parser.add_argument('--confirmaciones', type=Path, default=None,
                        help='reutilizar un confirmaciones_v2.json guardado para estos mismos archivos')
    parser.add_argument('--familias', type=Path, default=None,
                        help='archivo (CSV o Excel) con una columna de producto y otra con su familia')
    parser.add_argument('--columna-producto', default=None, help='cabecera de la columna del producto en el archivo de familias')
    parser.add_argument('--columna-familia', default=None, help='cabecera de la columna de la familia en el archivo de familias')
    a = parser.parse_args()
    main(a.procesos, a.confirmaciones, a.familias, a.columna_producto, a.columna_familia)
