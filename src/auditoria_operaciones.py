"""Fase 2.5: auditoría configurable de documentos y tipos de movimiento.

No requiere número de factura, ni impone convenciones del origen de datos.
Una regla definida por el usuario sirve solo como INDICIO documentado: no
reconcilia devoluciones, no anula ventas y no modifica las tablas originales.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

from .calidad import _cantidad_estado, _sin_valor

TIPOS = ('venta', 'devolucion', 'cancelacion', 'ajuste')


def _texto(valor: object) -> str | None:
    if _sin_valor(valor):
        return None
    return str(valor).strip()


def _clave(valor: str) -> str:
    # Normalización SOLO para comparar etiquetas; nunca reescribe los datos.
    return ' '.join(valor.split()).casefold()


@dataclass(frozen=True)
class ReglaPrefijo:
    prefijo: str
    tipo: str


@dataclass(frozen=True)
class ReglaValor:
    valor: str
    tipo: str


@dataclass(frozen=True)
class ReglasMovimiento:
    """Reglas explícitas y opcionales; no hay valores especiales precargados.

    - prefijos: prefijo del identificador de operación -> clase de movimiento.
    - valores_tipo: valor exacto de la columna tipo/estado -> clase.
    - tipos: venta, devolucion, cancelacion, ajuste.
    """
    prefijos: tuple[ReglaPrefijo, ...] = ()
    valores_tipo: tuple[ReglaValor, ...] = ()

    def __post_init__(self) -> None:
        vistos_prefijos: set[str] = set()
        vistos_valores: set[str] = set()
        for regla in self.prefijos:
            if not isinstance(regla, ReglaPrefijo) or not isinstance(regla.prefijo, str) or not regla.prefijo.strip():
                raise ValueError('El prefijo de operación debe ser texto no vacío')
            if regla.tipo not in TIPOS:
                raise ValueError(f'Tipo de movimiento desconocido: {regla.tipo!r}')
            clave = regla.prefijo.strip().casefold()
            if clave in vistos_prefijos:
                raise ValueError('El mismo prefijo no puede configurarse dos veces')
            vistos_prefijos.add(clave)
        for regla in self.valores_tipo:
            if not isinstance(regla, ReglaValor) or not isinstance(regla.valor, str) or not regla.valor.strip():
                raise ValueError('La marca de tipo debe ser texto no vacío')
            if regla.tipo not in TIPOS:
                raise ValueError(f'Tipo de movimiento desconocido: {regla.tipo!r}')
            clave = _clave(regla.valor)
            if clave in vistos_valores:
                raise ValueError('El mismo valor de tipo no puede configurarse dos veces')
            vistos_valores.add(clave)


@dataclass(frozen=True)
class AuditoriaOperaciones:
    total_filas: int
    hay_identificador: bool
    filas_con_identificador: int | None
    filas_sin_identificador: int | None
    identificadores_distintos: int | None
    identificadores_multilinea: int | None
    filas_en_operaciones_multilinea: int | None
    hay_tipo_movimiento: bool
    prefijos_configurados: int
    valores_configurados: int
    movimientos_por_tipo: dict[str, int]
    filas_clasificadas: int
    filas_sin_clasificar: int
    filas_conflicto: int
    negativas_total: int
    negativas_sin_clasificar: int
    negativas_clasificadas_venta: int
    negativas_clasificadas_otros_tipos: int
    no_negativas_marcadas_devolucion_o_cancelacion: int
    cantidades_no_interpretables: int
    valores_tipo_no_clasificados: dict[str, int]
    muestra_conflictos: list[int] = field(default_factory=list)
    muestra_negativas_sin_clasificar: list[int] = field(default_factory=list)
    muestra_negativas_venta: list[int] = field(default_factory=list)
    muestra_no_negativas_devolucion_cancelacion: list[int] = field(default_factory=list)


def auditar_operaciones(
    seleccion: pd.DataFrame,
    identificadores: pd.Series | None = None,
    *,
    tipos_movimiento: pd.Series | None = None,
    reglas: ReglasMovimiento | None = None,
    max_ejemplos: int = 5,
) -> AuditoriaOperaciones:
    """Cruza filas por POSICIÓN, sin interpretar el signo como clase de operación.

    Las reglas son evidencia aportada por el usuario, NO garantías sobre el
    significado comercial. Si identificador y tipo se contradicen, se avisa
    de conflicto y la fila se deja sin clasificar.
    """
    if 'cantidad' not in seleccion.columns:
        raise ValueError('Falta el campo cantidad')
    if max_ejemplos < 0:
        raise ValueError('max_ejemplos no puede ser negativo')
    if identificadores is not None and len(identificadores) != len(seleccion):
        raise ValueError('Los identificadores deben estar alineados en longitud')
    if tipos_movimiento is not None and len(tipos_movimiento) != len(seleccion):
        raise ValueError('Los tipos de movimiento deben estar alineados en longitud')
    reglas = reglas if reglas is not None else ReglasMovimiento()
    if not isinstance(reglas, ReglasMovimiento):
        raise TypeError('reglas debe ser una instancia de ReglasMovimiento')
    if reglas.prefijos and identificadores is None:
        raise ValueError('Se configuraron prefijos sin una columna de identificador')
    if reglas.valores_tipo and tipos_movimiento is None:
        raise ValueError('Se configuraron marcas sin columna de tipo/movimiento')

    ids = ([_texto(x) for x in identificadores.tolist()]
           if identificadores is not None else [None] * len(seleccion))
    estados = ([_texto(x) for x in tipos_movimiento.tolist()]
               if tipos_movimiento is not None else [None] * len(seleccion))
    frecuencias = Counter(x for x in ids if x is not None)
    diccionario_marcas = {_clave(r.valor): r.tipo for r in reglas.valores_tipo}

    cuenta_tipos = dict.fromkeys(TIPOS, 0)
    sin_clasificar = conflictos = negativas = neg_sin = neg_venta = neg_otros = no_neg_marcadas = sin_cantidad = 0
    muestra_conflictos: list[int] = []
    muestra_neg_sin: list[int] = []
    muestra_neg_venta: list[int] = []
    muestra_no_neg: list[int] = []
    desconocidos: Counter[str] = Counter()

    for fila, (identificador, etiqueta, cantidad) in enumerate(
            zip(ids, estados, seleccion['cantidad'].tolist()), start=1):
        coincidencias: set[str] = set()
        if identificador is not None:
            clave_id = identificador.casefold()
            for regla in reglas.prefijos:
                if clave_id.startswith(regla.prefijo.strip().casefold()):
                    coincidencias.add(regla.tipo)
        if etiqueta is not None:
            tipo_por_etiqueta = diccionario_marcas.get(_clave(etiqueta))
            if tipo_por_etiqueta is not None:
                coincidencias.add(tipo_por_etiqueta)
            else:
                desconocidos[etiqueta] += 1

        conflicto = len(coincidencias) > 1
        tipo = next(iter(coincidencias)) if len(coincidencias) == 1 else None
        if conflicto:
            conflictos += 1
            if len(muestra_conflictos) < max_ejemplos:
                muestra_conflictos.append(fila)
        if tipo is None:
            sin_clasificar += 1
        else:
            cuenta_tipos[tipo] += 1

        estado_cantidad, numero = _cantidad_estado(cantidad)
        if estado_cantidad not in ('valida', 'normalizable') or numero is None:
            sin_cantidad += 1
            continue
        if numero < 0:
            negativas += 1
            if tipo is None:
                neg_sin += 1
                if len(muestra_neg_sin) < max_ejemplos:
                    muestra_neg_sin.append(fila)
            elif tipo == 'venta':
                neg_venta += 1
                if len(muestra_neg_venta) < max_ejemplos:
                    muestra_neg_venta.append(fila)
            else:
                neg_otros += 1
        elif tipo in ('cancelacion', 'devolucion'):
            no_neg_marcadas += 1
            if len(muestra_no_neg) < max_ejemplos:
                muestra_no_neg.append(fila)

    return AuditoriaOperaciones(
        total_filas=len(seleccion),
        hay_identificador=identificadores is not None,
        filas_con_identificador=sum(x is not None for x in ids) if identificadores is not None else None,
        filas_sin_identificador=sum(x is None for x in ids) if identificadores is not None else None,
        identificadores_distintos=len(frecuencias) if identificadores is not None else None,
        identificadores_multilinea=sum(n > 1 for n in frecuencias.values()) if identificadores is not None else None,
        filas_en_operaciones_multilinea=sum(n for n in frecuencias.values() if n > 1) if identificadores is not None else None,
        hay_tipo_movimiento=tipos_movimiento is not None,
        prefijos_configurados=len(reglas.prefijos),
        valores_configurados=len(reglas.valores_tipo),
        movimientos_por_tipo=cuenta_tipos,
        filas_clasificadas=sum(cuenta_tipos.values()),
        filas_sin_clasificar=sin_clasificar,
        filas_conflicto=conflictos,
        negativas_total=negativas,
        negativas_sin_clasificar=neg_sin,
        negativas_clasificadas_venta=neg_venta,
        negativas_clasificadas_otros_tipos=neg_otros,
        no_negativas_marcadas_devolucion_o_cancelacion=no_neg_marcadas,
        cantidades_no_interpretables=sin_cantidad,
        valores_tipo_no_clasificados=dict(desconocidos.most_common(max_ejemplos)) if tipos_movimiento is not None else {},
        muestra_conflictos=muestra_conflictos,
        muestra_negativas_sin_clasificar=muestra_neg_sin,
        muestra_negativas_venta=muestra_neg_venta,
        muestra_no_negativas_devolucion_cancelacion=muestra_no_neg,
    )


def mostrar_auditoria_operaciones(informe: AuditoriaOperaciones) -> None:
    print('\n=== FASE 2.5: AUDITORÍA GENÉRICA DE OPERACIONES ===')
    print(f'Registros examinados: {informe.total_filas}')
    if informe.hay_identificador:
        print(f'Con identificador de operación: {informe.filas_con_identificador}')
        print(f'Sin identificador de operación: {informe.filas_sin_identificador}')
        print(f'Identificadores distintos: {informe.identificadores_distintos}')
        print(f'Identificadores con varias líneas: {informe.identificadores_multilinea} '
              f'(abarcan {informe.filas_en_operaciones_multilinea} filas)')
        print('Varias líneas en una operación NO implican duplicados.')
    else:
        print('Sin columna de identificador; la auditoría funciona igualmente.')
    if informe.hay_tipo_movimiento:
        print('Se ha seleccionado una columna opcional de tipo/estado de movimiento.')
    print(f'Reglas configuradas: {informe.prefijos_configurados} prefijo(s), '
          f'{informe.valores_configurados} valor(es) exacto(s).')
    print(f'Filas clasificadas por reglas: {informe.filas_clasificadas}')
    for tipo, n in informe.movimientos_por_tipo.items():
        if n:
            print(f'  {tipo}: {n}')
    print(f'Filas sin clase determinada: {informe.filas_sin_clasificar} (no son errores por defecto)')
    print(f'Conflictos entre reglas de clasificación: {informe.filas_conflicto}')
    if informe.muestra_conflictos:
        print('  Ejemplos de filas con conflicto:', informe.muestra_conflictos)
    print(f'Cantidades negativas totales: {informe.negativas_total}')
    print(f'  Negativas con clase sin determinar: {informe.negativas_sin_clasificar}')
    print(f'  Negativas clasificadas como venta: {informe.negativas_clasificadas_venta}')
    print(f'  Negativas clasificadas como devolución/cancelación/ajuste: '
          f'{informe.negativas_clasificadas_otros_tipos}')
    print(f'Cantidades no negativas clasificadas como devolución o cancelación: '
          f'{informe.no_negativas_marcadas_devolucion_o_cancelacion}')
    if informe.muestra_negativas_sin_clasificar:
        print('  Ejemplos de negativas sin clasificar:', informe.muestra_negativas_sin_clasificar)
    if informe.muestra_negativas_venta:
        print('  Ejemplos de negativas clasificadas venta:', informe.muestra_negativas_venta)
    if informe.muestra_no_negativas_devolucion_cancelacion:
        print('  Ejemplos de devoluciones/cancelaciones no negativas:', informe.muestra_no_negativas_devolucion_cancelacion)
    if informe.cantidades_no_interpretables:
        print(f'Cantidades no interpretables para el cruce: {informe.cantidades_no_interpretables}')
    if informe.valores_tipo_no_clasificados:
        print('Valores observados en tipo/estado SIN regla configurada:', informe.valores_tipo_no_clasificados)
    print('AVISO: una regla describe la convención declarada, no demuestra el motivo real de la operación.')
    print('El signo de la cantidad por sí solo NO determina devolución, venta o cancelación.')
    print('Ningún registro se excluye, fusiona, descuenta, elimina ni modifica automáticamente.')
