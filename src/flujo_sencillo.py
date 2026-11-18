"""Flujo breve para el prototipo de importación y calidad del TFG.

No aplica reglas comerciales que no conoce. Se conserva el archivo fuente,
solo hay una confirmación conjunta de columnas y las incidencias se agrupan.
La lógica de carga, diagnóstico, auditoría y limpieza sigue siendo reutilizable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .carga import (
    listar_hojas, proponer_cabeceras, cargar_tabla, proponer_columnas,
    seleccionar_campos, evaluar_campo_obligatorio,
)
from .calidad import analizar_calidad
from .limpieza import limpiar_datos
from .auditoria_avanzada import auditar_calidad_avanzada
from .auditoria_operaciones import auditar_operaciones
from .configuracion_operaciones import candidatas_operacion
from .preparacion_historico import analizar_preparacion, mostrar_preparacion_corta
from .series_mensuales import construir_series_mensuales, mostrar_diagnostico_temporal_corto
from .cobertura_productos import analizar_cobertura_productos, mostrar_cobertura_corta
from .patrones_demanda import analizar_patrones_preliminares, mostrar_patrones_corto
from .consolidacion import (
    FuenteVentas, consolidar_fuentes, analizar_consolidacion, mostrar_consolidacion_corta,
)

CAMPOS = ("fecha", "producto", "cantidad")
NOMBRES = {
    "fecha": "Fecha de la venta",
    "producto": "Código o nombre del producto",
    "cantidad": "Unidades o cantidad vendida",
    "descripcion": "Descripción (opcional)",
}


class CargaNoIdentificable(ValueError):
    """No debe usarse la tabla sin confirmación de columnas esenciales."""


@dataclass(frozen=True)
class Sugerencia:
    posiciones: dict[str, int]
    por_revisar: tuple[str, ...]


def sugerir_columnas(tabla: pd.DataFrame) -> Sugerencia:
    """Sugiere posiciones sin adjudicar un campo si falta evidencia mínima.

    Nunca considera una descripción/importe/stock como cantidad. Si dos
    candidatas se parecen, se muestra una propuesta conjunta al usuario.
    """
    candidatos = proponer_columnas(tabla)
    usadas: set[int] = set()
    elegidas: dict[str, int] = {}
    dudas: list[str] = []
    for campo in CAMPOS:
        opciones = [p for p in candidatos[campo]
                    if p['indice'] not in usadas and
                    evaluar_campo_obligatorio(tabla, p['indice'], campo)[0]]
        if not opciones:
            dudas.append(campo)
            continue
        opcion = opciones[0]
        # Evidencia fuerte por cabecera (>=90), pero la propuesta siempre se
        # confirma en conjunto. Valores dudosos no obligan a elegir a ciegas.
        if opcion['puntos'] < 90:
            dudas.append(campo)
            continue
        elegidas[campo] = opcion['indice']
        usadas.add(opcion['indice'])
        if len(opciones) > 1 and opciones[1]['puntos'] >= opcion['puntos'] - 15:
            dudas.append(campo)
    descripcion = [p for p in candidatos['descripcion']
                   if p['indice'] not in usadas and p['puntos'] >= 85]
    if descripcion:
        elegidas['descripcion'] = descripcion[0]['indice']
    return Sugerencia(elegidas, tuple(dudas))


def elegir_cabecera(archivo: Path, hoja: str | None = None) -> int | None:
    """Elige la mejor cabecera solo si reconoce los tres campos sin empate."""
    propuestas = proponer_cabeceras(archivo, hoja)
    if not propuestas or len(propuestas[0]['campos']) != 3:
        return None
    siguientes = [p for p in propuestas[1:] if len(p['campos']) == 3]
    if siguientes and siguientes[0]['puntos'] >= propuestas[0]['puntos'] - 250:
        return None
    return int(propuestas[0]['fila'])


def _pedir_numero(mensaje: str, minimo: int, maximo: int) -> int:
    while True:
        valor = input(mensaje).strip()
        if valor.isdigit() and minimo <= int(valor) <= maximo:
            return int(valor)
        print(f"Escribe un número entre {minimo} y {maximo}.")


def _elegir_campo_manual(tabla: pd.DataFrame, campo: str, actuales: dict[str, int]) -> bool:
    print(f"\n¿Cuál es la columna de {NOMBRES[campo].lower()}?")
    for i, nombre in enumerate(tabla.columns, start=1):
        print(f"  {i}. {nombre}")
    print("  0. No lo sé / cancelar sin inventar una asignación")
    while True:
        numero = _pedir_numero("Columna: ", 0, len(tabla.columns))
        if numero == 0:
            if campo == 'descripcion':
                actuales.pop(campo, None)
                print("Descripción omitida; es opcional.")
                return True
            return False
        indice = numero - 1
        if indice in {v for k, v in actuales.items() if k != campo}:
            print("Esa columna ya corresponde a otro dato. Prueba otra o elige 0.")
            continue
        if campo in CAMPOS:
            bien, explicacion = evaluar_campo_obligatorio(tabla, indice, campo)
            if not bien:
                print(f"No puedo usarla: {explicacion}. Prueba otra o elige 0.")
                continue
        actuales[campo] = indice
        return True


def confirmar_columnas(tabla: pd.DataFrame, propuesta: Sugerencia) -> dict[str, int] | None:
    """Una sola pregunta para aceptar el mapeo habitual.

    Si faltan campos, permite cambiar SOLO los necesarios; no fuerza
    confirmar una descripción opcional ni inventar significados.
    """
    indices = dict(propuesta.posiciones)
    while True:
        print("\nHE ENCONTRADO ESTOS DATOS:")
        for campo in (*CAMPOS, 'descripcion'):
            nombre = str(tabla.columns[indices[campo]]) if campo in indices else "No identificada"
            print(f"  {NOMBRES[campo]}: {nombre}")
        if propuesta.por_revisar:
            sospechosos = [NOMBRES[c] for c in propuesta.por_revisar if c in indices]
            if sospechosos:
                print("  Revisa especialmente: " + ", ".join(sospechosos) + ".")
        faltan = [c for c in CAMPOS if c not in indices]
        if faltan:
            print("\nNecesito identificar los tres datos básicos antes de continuar.")
            print("No pasa nada si no sabes cuál es una columna: se puede cancelar.")
            for campo in faltan:
                if not _elegir_campo_manual(tabla, campo, indices):
                    print("Importación detenida sin modificar tu archivo.")
                    return None
            continue
        print("\n¿Son correctas estas columnas? La descripción es opcional.")
        resp = input("[S] Sí, continuar  [C] Cambiar alguna  [N] No lo sé: ").strip().casefold()
        if resp in ('s','si','sí'):
            return indices
        if resp in ('n','no','no lo se','no lo sé',''):
            print("Sin confirmación no se analizarán columnas que podrían ser incorrectas.")
            print("Puedes abrir el archivo para comprobarlo y volver a intentarlo.")
            return None
        if resp == 'c':
            print("¿Qué quieres cambiar? 1 Fecha / 2 Producto / 3 Cantidad / 4 Descripción")
            numero = _pedir_numero("Opción (0 = volver): ", 0, 4)
            if numero == 0:
                continue
            campo = (*CAMPOS, 'descripcion')[numero - 1]
            _elegir_campo_manual(tabla, campo, indices)
        else:
            print("Puedes responder S, C o N.")


def columnas_operacion_opcionales(tabla: pd.DataFrame, usadas: set[int]) -> tuple[int | None, int | None]:
    """Reconoce metadatos opcionales por nombre; no infiere su significado.

    Solo se utilizan si la correspondencia por cabecera es única.
    """
    candidatas = candidatas_operacion(tabla, usadas)
    def unica(tipo: str) -> int | None:
        opciones = candidatas[tipo]
        return opciones[0] if len(opciones) == 1 else None
    indice_id = unica('identificador')
    indice_tipo = unica('tipo_movimiento')
    if indice_id == indice_tipo:
        indice_tipo = None
    return indice_id, indice_tipo


def construir_resumen(
    tabla: pd.DataFrame, indices: dict[str, int], *,
    originales: pd.DataFrame | None = None,
    identificadores: pd.Series | None = None,
    estados_movimiento: pd.Series | None = None,
    columnas_operacion_externas: bool = False,
) -> dict:
    """Prepara una tabla individual o un histórico combinado sin conjeturas."""
    seleccion = seleccionar_campos(tabla, indices)
    raw = tabla if originales is None else originales
    if len(raw) != len(seleccion):
        raise ValueError('Las columnas de origen y las ventas no están alineadas.')
    antes = analizar_calidad(seleccion, originales=raw)
    limpio = limpiar_datos(seleccion)   # SIN conjeturar siglo, orden o separador
    despues = analizar_calidad(limpio.datos, originales=raw)
    auditoria = auditar_calidad_avanzada(limpio.datos, raw)
    if not columnas_operacion_externas:
        indice_id, indice_tipo = columnas_operacion_opcionales(tabla, set(indices.values()))
        identificadores = tabla.iloc[:, indice_id] if indice_id is not None else None
        estados_movimiento = tabla.iloc[:, indice_tipo] if indice_tipo is not None else None
    operaciones = None
    if identificadores is not None or estados_movimiento is not None:
        operaciones = auditar_operaciones(
            limpio.datos,
            identificadores=identificadores,
            tipos_movimiento=estados_movimiento,
        )
    preparacion = analizar_preparacion(limpio)
    series = construir_series_mensuales(limpio, preparacion)
    cobertura_productos = analizar_cobertura_productos(series)
    patrones = analizar_patrones_preliminares(series, cobertura_productos)
    return dict(antes=antes, limpio=limpio, despues=despues,
                auditoria=auditoria, operaciones=operaciones,
                seleccion=seleccion, preparacion=preparacion, series=series,
                cobertura_productos=cobertura_productos, patrones=patrones)


def mostrar_resumen_corto(resultado: dict, nombre_archivo: str) -> None:
    antes, limpio, despues, audit = (resultado[k] for k in ('antes','limpio','despues','auditoria'))
    tipos = limpio.cambios['tipo_cambio'].value_counts().to_dict() if not limpio.cambios.empty else {}
    pendientes = len(limpio.pendientes)
    print("\n" + "="*58)
    print("RESUMEN DE TUS VENTAS")
    print("="*58)
    print(f"Archivo: {nombre_archivo}")
    print(f"Registros conservados: {len(limpio.datos):,}".replace(',', '.'))
    print(f"Ajustes seguros de formato: {tipos.get('normalizacion_formato',0)}")
    print(f"Conversiones internas de fecha/número: {tipos.get('conversion_tipo',0)}")
    if pendientes:
        afectados = limpio.pendientes['fila_tabla'].nunique()
        print(f"DATOS SIN RESOLVER: {pendientes} campo(s) en {afectados} registro(s)")
        por = limpio.pendientes['campo'].value_counts().to_dict()
        print("  " + "; ".join(f"{c}: {n}" for c,n in por.items()))
        print("  No los he rellenado ni he supuesto un valor. Pueden limitar las predicciones.")
    else:
        print("Datos básicos interpretables: sí (no equivale a ventas verificadas).")
    print("\nOTROS AVISOS (no se han corregido automáticamente)")
    print(f"  Filas idénticas: {audit.filas_identicas} (posibles duplicados)")
    print(f"  Productos con nombres diferentes: {audit.productos_descripciones_inconsistentes}")
    print(f"  Cantidades negativas: {despues.conteos.get('cantidad_negativa',0)}")
    print(f"  Cantidades cero: {despues.conteos.get('cantidad_cero',0)}")
    print(f"  Cantidades muy altas para su producto: {audit.cantidades_extremas}")
    if audit.filas_completamente_vacias:
        print(f"  Filas totalmente vacías: {audit.filas_completamente_vacias}")
    ops = resultado['operaciones']
    if ops and ops.hay_identificador:
        print(f"  Identificadores de pedido/factura encontrados: {ops.identificadores_distintos}")
        print("  Varias líneas en una factura/pedido no son duplicados por sí mismas.")
    if ops and ops.hay_tipo_movimiento:
        print("  Hay una columna de estado/tipo, pero no he supuesto su significado.")
    print("\nLos archivos originales no han cambiado." if resultado.get('consolidacion') is not None else "\nEl archivo original no ha cambiado.")
    print("Para examinar casos concretos existe el modo avanzado, opcional.")
    if resultado.get('consolidacion') is not None:
        mostrar_consolidacion_corta(resultado['consolidacion'])
    mostrar_preparacion_corta(resultado['preparacion'])
    mostrar_diagnostico_temporal_corto(resultado['series'])
    mostrar_cobertura_corta(resultado['cobertura_productos'])
    mostrar_patrones_corto(resultado['patrones'])


def _seleccionar_archivos(archivos: list[Path]) -> list[Path]:
    """Una pregunta para uno o muchos archivos; admite listas y rangos."""
    import re
    print("\n1. ELIGE TUS ARCHIVOS DE VENTAS")
    for n, archivo in enumerate(archivos, 1):
        print(f"  {n}. {archivo.name}")
    print("Puedes elegir varios a la vez (por ejemplo: 3,4,5 o 3-5).")
    while True:
        respuesta = input("Número(s) de archivo: ").strip()
        partes = [p.strip() for p in re.split(r'[,;]', respuesta) if p.strip()]
        elegidos: list[int] = []
        valido = bool(partes)
        for parte in partes:
            if re.fullmatch(r'\d+', parte):
                numeros = [int(parte)]
            elif re.fullmatch(r'\d+\s*-\s*\d+', parte):
                a,b = map(int, re.split(r'\s*-\s*',parte))
                numeros = list(range(a,b+1)) if 1 <= a <= b <= len(archivos) else []
            else:
                numeros = []
            if not numeros or any(n<1 or n>len(archivos) for n in numeros):
                valido = False
                break
            for n in numeros:
                if n not in elegidos:
                    elegidos.append(n)
        if valido and elegidos:
            return [archivos[n-1] for n in elegidos]
        print('Elige números de la lista, separados por comas, o un rango como 3-5.')


def _elegir_hoja(archivo: Path) -> str | None:
    hojas = listar_hojas(archivo)
    if archivo.suffix.lower() == '.csv':
        return None
    if len(hojas) == 1:
        return hojas[0]
    # Solo elegir si UNA hoja presenta inequívocamente los tres campos.
    candidatas = []
    for nombre in hojas:
        propuestas = proponer_cabeceras(archivo, nombre)
        fila = elegir_cabecera(archivo, nombre)
        if fila is not None and propuestas:
            candidatas.append((propuestas[0]['puntos'], nombre))
    candidatas.sort(reverse=True)
    if len(candidatas) == 1 or (len(candidatas)>1 and
                               candidatas[0][0]-candidatas[1][0] >= 250):
        print(f'  Hoja reconocida en {archivo.name}: {candidatas[0][1]}')
        return candidatas[0][1]
    print(f'\n{archivo.name} contiene varias hojas. ¿Cuál tiene las ventas?')
    for i,nombre in enumerate(hojas,1):
        print(f'  {i}. {nombre}')
    numero = _pedir_numero('Hoja (0 = no lo sé / cancelar): ',0,len(hojas))
    if numero == 0:
        raise CargaNoIdentificable('No se ha confirmado la hoja de ventas.')
    return hojas[numero-1]


def _cargar_una_tabla(archivo: Path, hoja: str | None, fila: int | None = None) -> FuenteVentas:
    """Carga una hoja concreta y conserva su identidad para auditoría."""
    if fila is None:
        fila = elegir_cabecera(archivo, hoja)
    if fila is None:
        nombre = f'{archivo.name} [{hoja}]' if hoja else archivo.name
        print(f'\nNo puedo localizar con seguridad la cabecera de {nombre}.')
        propuestas = proponer_cabeceras(archivo, hoja)
        for p in propuestas[:3]:
            print(f"  Posible fila {p['fila']+1}")
        numero = _pedir_numero('¿En qué fila están los nombres de las columnas? (0 = no lo sé): ',0,100)
        if numero == 0:
            raise CargaNoIdentificable('Cabecera sin identificar.')
        fila = numero-1
    tabla = cargar_tabla(archivo, fila, hoja)
    if len(tabla) == 0:
        raise ValueError(f'No hay registros en {archivo.name}' + (f' [{hoja}]' if hoja else '') + '.')
    detalle = f'{archivo.name} [{hoja}]' if hoja else archivo.name
    print(f'  {detalle}: {len(tabla):,} filas leídas.'.replace(',', '.'))
    return FuenteVentas(archivo=archivo, hoja=hoja, fila_cabecera=fila,
                        datos_originales=tabla, columnas={})


def _cargar_fuentes_de_archivo(archivo: Path) -> list[FuenteVentas]:
    """Un Excel con varias pestañas se analiza como varias tablas de ventas.

    Solo se incorporan automáticamente hojas cuya cabecera de ventas sea
    reconocible con seguridad. Hojas no identificadas se anuncian como no
    incluidas; nunca se presentan como datos consolidados ni como meses vacíos.
    """
    if archivo.suffix.lower() == '.csv':
        return [_cargar_una_tabla(archivo, None)]
    hojas = listar_hojas(archivo)
    if len(hojas) == 1:
        return [_cargar_una_tabla(archivo, hojas[0])]
    reconocidas: list[tuple[str, int]] = []
    otras: list[str] = []
    for hoja in hojas:
        fila = elegir_cabecera(archivo, hoja)
        if fila is None:
            otras.append(hoja)
            continue
        reconocidas.append((hoja, fila))
    if not reconocidas:
        # Caso realmente ambiguo: no se inventa que todas las hojas sean ventas.
        return [_cargar_una_tabla(archivo, _elegir_hoja(archivo))]
    print(f'\nEn {archivo.name} he reconocido {len(reconocidas)} hoja(s) de ventas: '
          + ', '.join(nombre for nombre,_ in reconocidas) + '.')
    if otras:
        print('  Hojas NO incorporadas (no identificadas como tablas de ventas): '
              + ', '.join(otras) + '.')
        print('  Si contienen ventas, deberán revisarse antes de considerar completo el histórico.')
    return [_cargar_una_tabla(archivo, hoja, fila) for hoja, fila in reconocidas]


def _confirmar_columnas_multiples(fuentes: list[FuenteVentas]) -> bool:
    """Pantalla conjunta. Solo se entra al editor si el usuario lo necesita."""
    propuestas = [sugerir_columnas(f.datos_originales) for f in fuentes]
    for fuente, propuesta in zip(fuentes, propuestas):
        fuente.columnas = dict(propuesta.posiciones)
    while True:
        print('\nHE ENCONTRADO ESTOS DATOS EN TUS ARCHIVOS:')
        for n, fuente in enumerate(fuentes,1):
            t=fuente.datos_originales
            m=fuente.columnas
            detalle = (f'{fuente.archivo.name} [{fuente.hoja}]' if fuente.hoja else fuente.archivo.name)
            print(f'  {n}. {detalle}')
            print('     ' + ' | '.join(
                f'{NOMBRES[c].split(" (")[0]}: {t.columns[m[c]] if c in m else "sin identificar"}'
                for c in (*CAMPOS, 'descripcion')
            ))
        incompletos = [(i,c) for i,f in enumerate(fuentes)
                       for c in CAMPOS if c not in f.columnas]
        if incompletos:
            print('\nAlgunos archivos tienen columnas que no he podido identificar.')
            print('Solo te preguntaré por esas columnas. Si no lo sabes, puedes cancelar.')
            for i,campo in incompletos:
                f = fuentes[i]
                print(f'Hoja/archivo: {f.archivo.name}' + (f' [{f.hoja}]' if f.hoja else '') )
                if not _elegir_campo_manual(fuentes[i].datos_originales,campo,fuentes[i].columnas):
                    return False
            continue
        print('\n¿Son correctas las columnas de todos los archivos?')
        resp = input('[S] Sí, unir ventas  [C] Cambiar alguna  [N] No lo sé: ').strip().casefold()
        if resp in ('s','si','sí'):
            return True
        if resp in ('n','no','no lo se','no lo sé',''):
            print('No uniré archivos sin confirmar qué datos representa cada columna.')
            return False
        if resp == 'c':
            indice = _pedir_numero('¿De qué archivo quieres cambiar una columna? (0 = volver): ', 0, len(fuentes))
            if indice:
                f=fuentes[indice-1]
                print('¿Qué quieres cambiar? 1 Fecha / 2 Producto / 3 Cantidad / 4 Descripción')
                numero = _pedir_numero('Opción (0 = volver): ',0,4)
                if numero:
                    _elegir_campo_manual(f.datos_originales,(*CAMPOS,'descripcion')[numero-1],f.columnas)
        else:
            print('Puedes responder S, C o N.')


def ejecutar_terminal(carpeta: Path) -> dict | None:
    archivos = sorted(p for p in carpeta.iterdir()
                      if p.is_file() and p.suffix.lower() in ('.csv','.xlsx')) if carpeta.exists() else []
    if not archivos:
        print('Guarda tus archivos Excel o CSV en datos_prueba y vuelve a ejecutar el programa.')
        return None
    elegidos = _seleccionar_archivos(archivos)
    try:
        fuentes = [fuente for archivo in elegidos for fuente in _cargar_fuentes_de_archivo(archivo)]
    except (CargaNoIdentificable, ValueError) as exc:
        print(f'No se ha unido ningún archivo: {exc}')
        return None
    if len(fuentes) == 1:
        fuente=fuentes[0]
        indices=confirmar_columnas(fuente.datos_originales,sugerir_columnas(fuente.datos_originales))
        if indices is None:
            return None
        print('\n2. PREPARANDO DATOS (sin modificar el archivo original)...')
        resultado=construir_resumen(fuente.datos_originales,indices)
        fuente.columnas = dict(indices)
        resultado['fuentes'] = [fuente]
        mostrar_resumen_corto(resultado, fuente.archivo.name + (f" [{fuente.hoja}]" if len(listar_hojas(fuente.archivo)) > 1 and fuente.hoja else ""))
        return resultado

    if not _confirmar_columnas_multiples(fuentes):
        print('Se han conservado los archivos originales sin combinarlos.')
        return None
    print('\n2. UNIENDO Y PREPARANDO TUS VENTAS (sin modificar los archivos originales)...')
    for fuente in fuentes:
        id_indice, tipo_indice = columnas_operacion_opcionales(
            fuente.datos_originales, set(fuente.columnas.values())
        )
        fuente.identificador_indice = id_indice
        fuente.estado_indice = tipo_indice
    unido = consolidar_fuentes(fuentes)
    indices={'fecha':0,'producto':1,'cantidad':2,'descripcion':3}
    resultado = construir_resumen(
        unido.tabla, indices, originales=unido.originales,
        identificadores=unido.identificadores,
        estados_movimiento=unido.estados_movimiento,
        columnas_operacion_externas=True,
    )
    resultado['consolidacion'] = analizar_consolidacion(unido,resultado['limpio'])
    resultado['trazabilidad'] = unido.trazabilidad
    resultado['fuentes'] = fuentes
    mostrar_resumen_corto(resultado, f'{len(elegidos)} archivo(s), {len(fuentes)} hoja(s)/tabla(s) combinadas')
    return resultado
