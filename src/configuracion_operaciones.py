"""Configuración guiada de columnas y reglas de operaciones (FASE 2.5).

Esta capa pregunta al usuario por las convenciones de SU archivo. No
contiene valores de ningún dataset en particular ni deduce reglas por el signo.
"""
from __future__ import annotations

from collections import Counter
import unicodedata

import pandas as pd

from .auditoria_operaciones import ReglaPrefijo, ReglaValor, ReglasMovimiento, TIPOS
from .calidad import _sin_valor


CABECERAS_IDENTIFICADOR = frozenset({
    'invoice', 'invoiceno', 'invoicenumber', 'numerofactura',
    'numfactura', 'factura', 'numeropedido', 'pedido', 'orderno',
    'orderid', 'ticket', 'idtransaccion', 'transactionid',
    'numerodocumento', 'iddocumento', 'idmovimiento', 'pedidoid',
    'idpedido', 'nfactura', 'numpedido', 'idfactura',
})
CABECERAS_TIPO = frozenset({
    'tipomovimiento', 'tipooperacion', 'tipotransaccion',
    'clasemovimiento', 'claseoperacion', 'estadomovimiento',
    'estadooperacion', 'documenttype', 'movementtype',
    'transactiontype', 'tipodocumento', 'tiporegistro',
    'tipodeoperacion', 'tipodemovimiento', 'motivooperacion',
    'motivo', 'estado', 'tipo',
})


def _cabecera_normalizada(valor: object) -> str:
    texto = unicodedata.normalize('NFKD', str(valor).casefold())
    return ''.join(car for car in texto if car.isalnum() and not unicodedata.combining(car))


def candidatas_operacion(datos: pd.DataFrame, ocupadas: set[int]) -> dict[str, list[int]]:
    candidatos = {'identificador': [], 'tipo_movimiento': []}
    for i, nombre in enumerate(datos.columns):
        if i in ocupadas:
            continue
        clave = _cabecera_normalizada(nombre)
        if clave in CABECERAS_IDENTIFICADOR:
            candidatos['identificador'].append(i)
        if clave in CABECERAS_TIPO:
            candidatos['tipo_movimiento'].append(i)
    return candidatos


def _pedir_numero(mensaje: str, inferior: int, superior: int) -> int:
    while True:
        entrada = input(mensaje).strip()
        if entrada.isdigit() and inferior <= int(entrada) <= superior:
            return int(entrada)
        print(f'Introduce un número entre {inferior} y {superior}.')


def _si(mensaje: str) -> bool:
    return input(mensaje).strip().casefold() in ('s', 'si', 'sí')


def _elegir_columna(datos: pd.DataFrame, reservadas: set[int], nombre: str, candidatos: list[int]) -> int | None:
    print(f'\nCANDIDATAS A {nombre.upper()} (opcional)')
    if candidatos:
        for indice in candidatos[:8]:
            print(f'  Columna {indice + 1}: {datos.columns[indice]}')
    else:
        print('  No se reconoce una columna clara; puedes elegirla manualmente.')
    print('  0 = no utilizar este campo. No es obligatorio para el análisis de ventas.')
    while True:
        numero = _pedir_numero(f'{nombre} (0 = omitir): ', 0, len(datos.columns))
        if numero == 0:
            return None
        indice = numero - 1
        if indice in reservadas:
            print('Esta columna ya ha sido asignada. Selecciona otra o elige 0.')
            continue
        print(f'  Campo seleccionado: {datos.columns[indice]}')
        reservadas.add(indice)
        return indice


def _tipo_por_terminal() -> str | None:
    print('  0 = cancelar regla; 1 = venta; 2 = devolución; 3 = cancelación; 4 = ajuste')
    numero = _pedir_numero('  Clase de movimiento: ', 0, 4)
    return None if numero == 0 else TIPOS[numero - 1]


def configurar_auditoria_por_terminal(
    datos: pd.DataFrame,
    indices_ocupados: set[int],
) -> tuple[int | None, int | None, ReglasMovimiento]:
    """Elige campos opcionales y convenciones explícitas de ESTE archivo.

    Devuelve las posiciones de las columnas y las reglas; no altera `datos`
    ni `indices_ocupados`. Ninguna regla se activa sin confirmación.
    """
    reservadas = set(indices_ocupados)
    candidatos = candidatas_operacion(datos, reservadas)
    indice_id = _elegir_columna(datos, reservadas, 'Identificador de factura/pedido/operación',
                               candidatos['identificador'])
    # Recalcular por si se seleccionó la columna de tipo manualmente como ID.
    candidatos = candidatas_operacion(datos, reservadas)
    indice_tipo = _elegir_columna(datos, reservadas, 'Tipo o estado de movimiento',
                                 candidatos['tipo_movimiento'])

    prefijos: list[ReglaPrefijo] = []
    valores: list[ReglaValor] = []

    if indice_id is not None:
        print('\nREGLAS OPCIONALES SOBRE IDENTIFICADORES')
        print('Ejemplo: un prefijo confirmado por la empresa puede identificar devoluciones.')
        print('No hay prefijos predeterminados. El prefijo distingue el comienzo del documento.')
        while len(prefijos) < 10 and _si('¿Añadir regla de prefijo? (s/N): '):
            prefijo = input('  Prefijo literal (ej. DEV-): ').strip()
            if not prefijo:
                print('  Prefijo vacío: no se crea ninguna regla.')
                continue
            if any(p.prefijo.casefold() == prefijo.casefold() for p in prefijos):
                print('  Ese prefijo ya tiene una regla.')
                continue
            tipo = _tipo_por_terminal()
            if tipo is None:
                continue
            if _si(f'  ¿Confirmas que {prefijo!r} significa {tipo} EN ESTE ARCHIVO? (s/N): '):
                prefijos.append(ReglaPrefijo(prefijo, tipo))
                print('  Regla añadida. No modifica ni reclasifica transacciones originales.')

    if indice_tipo is not None:
        print('\nREGLAS OPCIONALES SOBRE TIPO / ESTADO')
        bruto = datos.iloc[:, indice_tipo].tolist()
        observados = Counter(str(v).strip() for v in bruto if not _sin_valor(v))
        print('  Valores literales observados (muestra, sin asignarles significado):')
        for valor, n in observados.most_common(12):
            print(f'    {valor!r}: {n} fila(s)')
        print('  Configura solo valores cuyo significado conozcas.\n'
              '  Se compararán como valor COMPLETO, sin importar mayúsculas o espacios externos.')
        while len(valores) < 20 and _si('¿Añadir regla de valor exacto? (s/N): '):
            valor = input('  Valor literal observado: ').strip()
            if not valor:
                print('  Valor vacío: se mantiene sin clasificar.')
                continue
            normalizado = ' '.join(valor.split()).casefold()
            if not any(' '.join(v.split()).casefold() == normalizado for v in observados):
                print('  Ese valor no aparece en esta columna. No se crea la regla.')
                continue
            if any(' '.join(r.valor.split()).casefold() == normalizado for r in valores):
                print('  Ese valor ya tiene una regla.')
                continue
            tipo = _tipo_por_terminal()
            if tipo is None:
                continue
            if _si(f'  ¿Confirmas que el valor {valor!r} significa {tipo} EN ESTE ARCHIVO? (s/N): '):
                valores.append(ReglaValor(valor, tipo))
                print('  Regla añadida; el valor original se conserva.')

    reglas = ReglasMovimiento(prefijos=tuple(prefijos), valores_tipo=tuple(valores))
    print('\nConfiguración de auditoría: '
          f'identificador={"sí" if indice_id is not None else "no"}; '
          f'tipo/estado={"sí" if indice_tipo is not None else "no"}; '
          f'prefijos={len(prefijos)}; valores exactos={len(valores)}')
    return indice_id, indice_tipo, reglas
