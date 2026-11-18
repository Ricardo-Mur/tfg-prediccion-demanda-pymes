"""Lo que necesitan las pantallas de carga, columnas y revisión.

Sigue los mismos pasos que `flujo_sencillo.ejecutar_terminal` (versión 1) y usa sus mismas funciones, pero sin
preguntar por teclado: devuelve lo que hay que enseñar y lo que queda por decidir. No decide nada por su cuenta
y no modifica nada del programa.

Todo lo que el programa escribiría en pantalla se descarta: en el terminal de la aplicación no queda ningún
dato de los archivos.
"""
from __future__ import annotations

import contextlib
import io
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

PROYECTO = Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True                      # leer el código del proyecto no deja nada escrito en él
if str(PROYECTO) not in sys.path:
    sys.path.insert(0, str(PROYECTO))

from src import v2_columnas as COL                                                          # noqa: E402
from src.carga import cargar_tabla, evaluar_campo_obligatorio, listar_hojas, proponer_cabeceras   # noqa: E402
from src.configuracion_operaciones import _cabecera_normalizada                             # noqa: E402
from src.consolidacion import FuenteVentas, analizar_consolidacion, consolidar_fuentes      # noqa: E402
from src.flujo_sencillo import (CAMPOS, NOMBRES, columnas_operacion_opcionales, construir_resumen,   # noqa: E402
                                elegir_cabecera, sugerir_columnas)

CAMPOS_TODOS = (*CAMPOS, 'descripcion')


@contextlib.contextmanager
def _en_silencio():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ------------------------------------------------------------------ paso 1: leer los archivos

@dataclass
class Tabla:
    """Una hoja de un Excel o un CSV."""
    archivo: Path
    hoja: str | None
    fila: int | None = None                 # fila de la cabecera (desde 0)
    datos: pd.DataFrame | None = None
    falta: str = ''                         # '' (lista), 'cabecera' o 'vacia'
    filas_posibles: list[int] = field(default_factory=list)

    @property
    def nombre(self) -> str:
        return f'{self.archivo.name} [{self.hoja}]' if self.hoja else self.archivo.name


@dataclass
class ArchivoLeido:
    archivo: Path
    tablas: list[Tabla] = field(default_factory=list)
    hojas_sin_incorporar: list[str] = field(default_factory=list)
    hojas_para_elegir: list[str] = field(default_factory=list)      # ninguna hoja se reconoce: tiene que elegir el usuario
    error: str = ''

    @property
    def listo(self) -> bool:
        return not self.error and not self.hojas_para_elegir and bool(self.tablas) and all(not t.falta for t in self.tablas)


def _tabla(archivo: Path, hoja: str | None, fila_indicada: int | None) -> Tabla:
    fila = fila_indicada if fila_indicada is not None else elegir_cabecera(archivo, hoja)
    if fila is None:
        return Tabla(archivo, hoja, falta='cabecera', filas_posibles=[int(p['fila']) for p in proponer_cabeceras(archivo, hoja)[:3]])
    datos = cargar_tabla(archivo, int(fila), hoja)
    return Tabla(archivo, hoja, int(fila), datos, falta='' if len(datos) else 'vacia')


def leer_archivo(archivo: Path, hoja_elegida: str | None = None, filas: dict | None = None) -> ArchivoLeido:
    """Como `_cargar_fuentes_de_archivo`: un Excel con varias hojas se lee como varias tablas, pero solo las que
    se reconocen con seguridad. `filas`: fila de cabecera que ha indicado el usuario para una hoja (None en un CSV).
    """
    archivo, filas = Path(archivo), dict(filas or {})
    leido = ArchivoLeido(archivo)
    try:
        with _en_silencio():
            if archivo.suffix.lower() == '.csv':
                leido.tablas = [_tabla(archivo, None, filas.get(None))]
                return leido
            hojas = listar_hojas(archivo)
            if len(hojas) == 1:
                leido.tablas = [_tabla(archivo, hojas[0], filas.get(hojas[0]))]
                return leido
            reconocidas = [(h, elegir_cabecera(archivo, h)) for h in hojas]
            if any(f is not None for _, f in reconocidas):
                leido.tablas = [_tabla(archivo, h, f) for h, f in reconocidas if f is not None]
                leido.hojas_sin_incorporar = [h for h, f in reconocidas if f is None]
            elif hoja_elegida in hojas:
                leido.tablas = [_tabla(archivo, hoja_elegida, filas.get(hoja_elegida))]
                leido.hojas_sin_incorporar = [h for h in hojas if h != hoja_elegida]
            else:
                leido.hojas_para_elegir = list(hojas)
    except Exception as e:                  # el mensaje no lleva nunca contenido del archivo
        leido.error = f'No he podido leer este archivo ({type(e).__name__}). Comprueba que es un Excel o un CSV de ventas.'
    return leido


# ------------------------------------------------------------------ paso 2: columnas

def propuesta_de_columnas(datos: pd.DataFrame) -> tuple[dict[str, int], tuple[str, ...]]:
    """(columna propuesta para cada dato, datos que conviene revisar). Sin propuesta, el dato no aparece."""
    with _en_silencio():
        s = sugerir_columnas(datos)
    return dict(s.posiciones), tuple(s.por_revisar)


def columna_valida(datos: pd.DataFrame, indice: int, campo: str) -> tuple[bool, str]:
    if campo not in CAMPOS:
        return True, ''
    with _en_silencio():
        bien, explicacion = evaluar_campo_obligatorio(datos, int(indice), campo)
    return bool(bien), str(explicacion)


def parece_personal(datos: pd.DataFrame, indice: int) -> bool:
    """Con las mismas reglas del programa: por el nombre de la columna o por su contenido. Ante la duda, sí."""
    papel = COL._papel_por_cabecera(_cabecera_normalizada(str(datos.columns[int(indice)])))
    return papel in ('codigo_cliente', 'dato_de_cliente') or COL._parece_personal(datos.iloc[:, int(indice)])


def vista_previa(datos: pd.DataFrame, columnas: dict[str, int], filas: int = 5) -> tuple[pd.DataFrame, list[str]]:
    """Primeras filas de las columnas elegidas, y solo de esas. De una columna que parece un dato de personas no se
    enseña ningún valor: se devuelve aparte su nombre."""
    visibles, ocultas = {}, []
    for campo in CAMPOS_TODOS:
        if campo not in columnas:
            continue
        nombre = str(datos.columns[columnas[campo]])
        if parece_personal(datos, columnas[campo]):
            ocultas.append(nombre)
        else:
            visibles[nombre] = datos.iloc[:filas, columnas[campo]].astype(str).to_list()
    return pd.DataFrame(visibles), ocultas


# ------------------------------------------------------------------ paso 3: revisión

def preparar(tablas: list[Tabla], columnas: list[dict[str, int]]) -> dict:
    """Lo mismo que hace `ejecutar_terminal` cuando el usuario ha confirmado las columnas, más el inventario de
    las demás columnas (`v2_columnas`)."""
    fuentes = [FuenteVentas(archivo=t.archivo, hoja=t.hoja, fila_cabecera=t.fila, datos_originales=t.datos, columnas=dict(c))
               for t, c in zip(tablas, columnas)]
    with _en_silencio():
        if len(fuentes) == 1:
            resultado = construir_resumen(fuentes[0].datos_originales, fuentes[0].columnas)
        else:
            for f in fuentes:
                f.identificador_indice, f.estado_indice = columnas_operacion_opcionales(f.datos_originales, set(f.columnas.values()))
            unido = consolidar_fuentes(fuentes)
            resultado = construir_resumen(unido.tabla, {'fecha': 0, 'producto': 1, 'cantidad': 2, 'descripcion': 3},
                                          originales=unido.originales, identificadores=unido.identificadores,
                                          estados_movimiento=unido.estados_movimiento, columnas_operacion_externas=True)
            resultado['consolidacion'] = analizar_consolidacion(unido, resultado['limpio'])
            resultado['trazabilidad'] = unido.trazabilidad
        resultado['fuentes'] = fuentes
        resultado['inventario'] = COL.inventariar_resultado(resultado)
    return resultado


PAPELES = {'id_linea': 'Identificador de línea', 'id_factura': 'Factura o documento', 'serie': 'Serie',
           'tipo_movimiento': 'Tipo de movimiento', 'fecha_diaria': 'Fecha con día', 'lote': 'Lote',
           'codigo_cliente': 'Código de cliente', 'dato_de_cliente': 'Dato de clientes o de personas',
           'estado_articulo': 'Estado del artículo', 'existencias': 'Existencias', 'importe': 'Importe',
           'descripcion': 'Descripción', 'atributo_del_producto': 'Posible agrupación de productos',
           'fecha_del_articulo': 'Fecha del artículo'}
USOS = {'no_usar_dato_personal': 'No se usa ni se exporta', 'pendiente_de_confirmar': 'Sin usar hasta que confirmes qué es',
        'sin_datos': 'Vacía'}


def resumen(resultado: dict) -> dict:
    """Las cifras de la pantalla de revisión, tal como las da la versión 1. Solo recuentos: ningún valor del archivo."""
    limpio, despues, auditoria = resultado['limpio'], resultado['despues'], resultado['auditoria']
    preparacion, cobertura, consolidacion = resultado['preparacion'], resultado['cobertura_productos'], resultado.get('consolidacion')
    pendientes = limpio.pendientes
    inventario = resultado['inventario'].tabla
    columnas = [{'Columna': str(f.columna), 'Qué parece': PAPELES.get(f.papel, 'No reconocida'),
                 'Qué se hace con ella': ('Se usa' + (' solo para emparejar anulaciones; no se exporta' if f.papel == 'codigo_cliente' else '')
                                          if f.se_usa else USOS.get(f.uso, 'No se usa')),
                 'Líneas con dato': f'{100 * float(f.cobertura):.0f} %' if pd.notna(f.cobertura) else ''}
                for f in inventario.itertuples(index=False)]
    return {
        'registros': int(len(limpio.datos)), 'productos': int(preparacion.productos_identificados),
        'primer_dia': preparacion.primer_dia, 'ultimo_dia': preparacion.ultimo_dia,
        'meses_con_registros': int(preparacion.meses_con_registros), 'meses_del_intervalo': int(preparacion.meses_en_intervalo),
        'campos_sin_resolver': int(len(pendientes)),
        'registros_sin_resolver': int(pendientes['fila_tabla'].nunique()) if len(pendientes) else 0,
        'negativas': int(despues.conteos.get('cantidad_negativa', 0)), 'ceros': int(despues.conteos.get('cantidad_cero', 0)),
        'filas_identicas': int(auditoria.filas_identicas), 'muy_altas': int(auditoria.cantidades_extremas),
        'nombres_distintos': int(auditoria.productos_descripciones_inconsistentes),
        'en_un_mes': int(cobertura.en_un_mes), 'en_dos_o_tres': int(cobertura.en_dos_o_tres_meses),
        'en_cuatro_a_once': int(cobertura.en_cuatro_o_mas_meses - cobertura.en_doce_o_mas_meses),
        'en_doce_o_mas': int(cobertura.en_doce_o_mas_meses), 'con_huecos': int(cobertura.con_huecos_interiores),
        'archivos': int(consolidacion.archivos) if consolidacion else 1, 'tablas': int(consolidacion.tablas) if consolidacion else 1,
        'solapes': len(consolidacion.periodos_solapados) if consolidacion else 0,
        'coincidencias_entre_archivos': int(consolidacion.coincidencias_entre_archivos) if consolidacion else 0,
        'columnas': columnas, 'aviso_columnas': str(resultado['inventario'].aviso or ''),
    }
