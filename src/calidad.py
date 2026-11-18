"""Fase 2.1: diagnóstico NO destructivo de históricos de ventas.

Recibe el resultado estandarizado de `seleccionar_campos` (fase 1).
No corrige, elimina, agrega ni rellena datos. Las incidencias son avisos
para la posterior limpieza/confirmación y NO equivalen a errores probados.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import math
import numbers
import re

import pandas as pd


CAMPOS_OBLIGATORIOS = ("fecha", "producto", "cantidad")
# "fila_tabla" indica la posición 1-based en la tabla ya importada.
# No equivale necesariamente al número físico de fila del Excel/CSV.
COLUMNAS_INCIDENCIAS = [
    "fila_tabla", "campo", "valor_original", "tipo", "nivel",
    "accion_propuesta", "detalle",
]

# Son propuestas de tratamiento, NUNCA correcciones realizadas.
ACCIONES_PROPUESTAS = {
    "fecha_ausente": "Solicitar la fecha; no inventar una.",
    "fecha_ambigua": "Confirmar si el archivo utiliza día/mes o mes/día.",
    "fecha_revisar": "Revisar formato; normalizar solo si no hay ambigüedad.",
    "fecha_invalida": "Solicitar corrección manual de la fecha imposible.",
    "producto_ausente": "Solicitar identificador; no asignarlo por suposición.",
    "cantidad_ausente": "Solicitar la cantidad; no rellenar con cero.",
    "cantidad_ambigua": "Confirmar separador decimal y de millares.",
    "cantidad_normalizable": "Proponer normalización segura para la fase 2.2.",
    "cantidad_revisar": "Revisar representación numérica con el usuario.",
    "cantidad_invalida": "Solicitar corrección; no inferir una cantidad.",
    "cantidad_negativa": "Conservar; revisar si es devolución o ajuste.",
    "cantidad_cero": "Conservar; un cero de ventas no prueba demanda cero.",
    "filas_identicas_origen": "Revisar identificadores de operación; no eliminar automáticamente.",
    "descripcion_inconsistente": "Revisar descripciones; conservar el código del producto.",
}


def _representar_valor(valor: object) -> str:
    """Representa el valor sin recortarlo: la auditoría conserva todo el texto."""
    if _sin_valor(valor):
        return "<vacío>"
    return str(valor)



@dataclass
class InformeCalidad:
    total_registros: int
    conteos: dict[str, int]
    incidencias: pd.DataFrame
    filas_con_incidentes: int
    duplicados_exactos_origen: int | None
    productos_con_descripciones_distintas: int | None


def _sin_valor(valor: object) -> bool:
    if valor is None:
        return True
    try:
        if bool(pd.isna(valor)):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(valor, str) and not valor.strip()


def _fecha_estado(valor: object) -> str:
    """'valida', 'ambigua', 'revisar', 'invalida' o 'ausente'.

    No interpreta automáticamente fechas ambiguas dd/mm vs mm/dd.
    """
    if _sin_valor(valor):
        return "ausente"
    if isinstance(valor, (pd.Timestamp, datetime, date)):
        return "valida"
    if isinstance(valor, (int, float)) and not isinstance(valor, bool):
        # Fechas seriales Excel: sin contexto del formato, requieren revisión.
        return "revisar"
    if not isinstance(valor, str):
        return "revisar"
    texto = valor.strip()
    if not texto:
        return "ausente"

    # ISO sin ambigüedad: 2025-03-15 y opcionalmente hora.
    if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}(?:[T ]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?", texto):
        try:
            datetime.fromisoformat(texto)
            return "valida"
        except ValueError:
            return "invalida"
    if re.fullmatch(r"\d{4}/\d{1,2}/\d{1,2}", texto):
        try:
            anio, mes, dia = map(int, texto.split("/"))
            date(anio, mes, dia)
            return "valida"
        except ValueError:
            return "invalida"

    # En formato de 4 cifras al final, la posición de día y mes puede cambiar.
    coincidencia = re.fullmatch(r"(\d{1,2})([/.-])(\d{1,2})\2(\d{4})", texto)
    if coincidencia:
        primero, _, segundo, anio = coincidencia.groups()
        a, b, y = int(primero), int(segundo), int(anio)
        fechas_posibles = []
        for d, m in ((a, b), (b, a)):
            try:
                fechas_posibles.append(date(y, m, d))
            except ValueError:
                pass
        if not fechas_posibles:
            return "invalida"
        if len(set(fechas_posibles)) > 1:
            return "ambigua"
        return "valida"

    # Fechas con año de dos cifras: distinguir ambigüedad día/mes aunque
    # el siglo siga pendiente de confirmación. Ej.: 04/05/25.
    coincidencia_corta = re.fullmatch(r"(\d{1,2})([/.-])(\d{1,2})\2(\d{2})", texto)
    if coincidencia_corta:
        a, _, b, yy = coincidencia_corta.groups()
        posibles = set()
        # La validez depende del año (29/02/24 sí es posible). No se
        # presupone el siglo: basta que alguna interpretación de 19xx/20xx
        # sea una fecha válida para no declararla imposible prematuramente.
        for d, m in ((int(a), int(b)), (int(b), int(a))):
            for anio in (1900 + int(yy), 2000 + int(yy)):
                try:
                    date(anio, m, d)
                    posibles.add((d, m))
                except ValueError:
                    pass
        if len(posibles) > 1:
            return "ambigua"
        if len(posibles) == 0:
            return "invalida"
        return "revisar"  # Falta confirmar el siglo de dos cifras.

    # Ej. 15//03/2025: no se declara fecha inválida hasta revisar el formato.
    if any(char.isdigit() for char in texto):
        return "revisar"
    return "invalida"


def _cantidad_estado(valor: object) -> tuple[str, Decimal | None]:
    """Estado y valor únicamente cuando no tiene interpretaciones alternativas.

    No cambia la celda, solo lee para diagnosticar. Evita interpretar
    silenciosamente '1,234' como 1.234 o 1234.
    """
    if _sin_valor(valor):
        return "ausente", None
    if isinstance(valor, bool):
        return "invalida", None
    if isinstance(valor, (numbers.Number, Decimal)):
        try:
            numero = Decimal(str(valor))
            return ("valida", numero) if numero.is_finite() else ("invalida", None)
        except (InvalidOperation, ValueError):
            return "invalida", None
    if not isinstance(valor, str):
        return "revisar", None

    texto = valor.strip()
    # Valores simples positivos o negativos, enteros o decimal con punto.
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", texto):
        # 1.234 puede ser decimal o millar: requiere confirmar la convención.
        if re.fullmatch(r"[+-]?\d{1,3}\.\d{3}", texto):
            return "ambigua", None
        return "valida", Decimal(texto)
    # Decimal con coma de 1-2 cifras, sin confusión con miles.
    if re.fullmatch(r"[+-]?\d+,\d{1,2}", texto):
        return "normalizable", Decimal(texto.replace(",", "."))
    if re.fullmatch(r"[+-]?\d{1,3},\d{3}", texto):
        return "ambigua", None
    # Caso inequívoco con dos tipos de separador: 1.234,50 / 1,234.50.
    if re.fullmatch(r"[+-]?\d{1,3}(?:\.\d{3})+,\d{1,2}", texto):
        return "normalizable", Decimal(texto.replace(".", "").replace(",", "."))
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+\.\d{1,2}", texto):
        return "normalizable", Decimal(texto.replace(",", ""))
    if any(char.isdigit() for char in texto):
        return "revisar", None
    return "invalida", None


def analizar_calidad(
    seleccion: pd.DataFrame, originales: pd.DataFrame | None = None,
) -> InformeCalidad:
    """Genera incidencias sobre datos ya mapeados, sin alterar las entradas.

    `originales` es opcional: permite detectar filas idénticas en la tabla
    completa, sin confundir varias transacciones iguales del mismo producto.
    """
    faltantes = [c for c in CAMPOS_OBLIGATORIOS if c not in seleccion.columns]
    if faltantes:
        raise ValueError(f"Faltan los campos confirmados: {faltantes}")
    if originales is not None and len(originales) != len(seleccion):
        raise ValueError("Los originales y los datos seleccionados deben tener igual número de filas")

    cuenta: Counter[str] = Counter()
    eventos: list[dict] = []
    filas_con_incidencia: set[int] = set()

    def incluir(fila: int, campo: str, tipo: str, nivel: str, detalle: str) -> None:
        cuenta[tipo] += 1
        filas_con_incidencia.add(fila)
        if campo in seleccion.columns:
            valor_original = _representar_valor(seleccion.iloc[fila][campo])
        else:
            # No mostramos la fila completa de origen: puede contener clientes,
            # direcciones u otra información ajena al diagnóstico.
            valor_original = "<registro completo no mostrado>"
        eventos.append({
            "fila_tabla": fila + 1,
            "campo": campo,
            "valor_original": valor_original,
            "tipo": tipo,
            "nivel": nivel,
            "accion_propuesta": ACCIONES_PROPUESTAS.get(tipo, "Revisar."),
            "detalle": detalle,
        })

    for i, (fecha, producto, cantidad) in enumerate(
        seleccion.loc[:, CAMPOS_OBLIGATORIOS].itertuples(index=False, name=None)
    ):
        estado_fecha = _fecha_estado(fecha)
        if estado_fecha != "valida":
            incluir(i, "fecha", f"fecha_{estado_fecha}", "revisar",
                    "Fecha sin valor, ambigua, no reconocida o inválida; revisar antes de utilizar.")
        if _sin_valor(producto):
            incluir(i, "producto", "producto_ausente", "revisar",
                    "Falta el identificador del producto o categoría.")

        estado_cantidad, valor = _cantidad_estado(cantidad)
        if estado_cantidad != "valida":
            incluir(i, "cantidad", f"cantidad_{estado_cantidad}", "revisar",
                    "Cantidad que requiere normalización, confirmación o revisión.")
        if valor is not None:
            if valor < 0:
                incluir(i, "cantidad", "cantidad_negativa", "informativo",
                        "Posible devolución, cancelación o ajuste: NO eliminar automáticamente.")
            elif valor == 0:
                incluir(i, "cantidad", "cantidad_cero", "informativo",
                        "Venta registrada como cero; no indica necesariamente demanda nula.")

    duplicados = None
    if originales is not None:
        # Cuenta FILAS participantes, no el número de grupos duplicados.
        es_duplicado = originales.duplicated(keep=False).to_numpy()
        duplicados = int(es_duplicado.sum())
        for i, coincide in enumerate(es_duplicado):
            if coincide:
                incluir(i, "registro", "filas_identicas_origen", "revisar",
                        "Fila idéntica a otra del archivo original; puede ser otra venta legítima.")

    divergencias = None
    if "descripcion" in seleccion.columns:
        descripciones: dict[str, set[str]] = defaultdict(set)
        for codigo, desc in seleccion.loc[:, ["producto", "descripcion"]].itertuples(index=False, name=None):
            if not _sin_valor(codigo) and not _sin_valor(desc):
                clave = str(codigo).strip()
                etiqueta = " ".join(str(desc).casefold().split())
                descripciones[clave].add(etiqueta)
        codigos_divergentes = {codigo for codigo, etiquetas in descripciones.items()
                              if len(etiquetas) > 1}
        divergencias = len(codigos_divergentes)
        cuenta["productos_con_descripciones_distintas"] = divergencias
        # Alerta de nivel PRODUCTO, no una incidencia repetida en cada
        # transacción. Registrar una incidencia por fila haría que una sola
        # variación de etiqueta activase miles de falsos "campos pendientes".
        # Los códigos y las variantes se presentan en la auditoría avanzada.

    # Claves de salida estables, incluso cuando el resultado es cero.
    categorias = (
        "fecha_ausente", "fecha_ambigua", "fecha_revisar", "fecha_invalida",
        "producto_ausente", "cantidad_ausente", "cantidad_ambigua",
        "cantidad_normalizable", "cantidad_revisar", "cantidad_invalida",
        "cantidad_negativa", "cantidad_cero",
    )
    for categoria in categorias:
        cuenta.setdefault(categoria, 0)

    incidencias = pd.DataFrame(eventos, columns=COLUMNAS_INCIDENCIAS)
    return InformeCalidad(
        total_registros=len(seleccion),
        conteos=dict(cuenta),
        incidencias=incidencias,
        filas_con_incidentes=len(filas_con_incidencia),
        duplicados_exactos_origen=duplicados,
        productos_con_descripciones_distintas=divergencias,
    )


def mostrar_informe(informe: InformeCalidad) -> None:
    """Imprime SOLO cantidades agregadas (no enseña clientes, productos ni IP)."""
    print("\n=== FASE 2.1: DIAGNÓSTICO INICIAL DE CALIDAD ===")
    print(f"Registros examinados: {informe.total_registros}")
    print(f"Registros con avisos individuales: {informe.filas_con_incidentes}")
    etiquetas = {
        "fecha_ausente": "Fechas ausentes",
        "fecha_ambigua": "Fechas ambiguas (día/mes)",
        "fecha_revisar": "Fechas con formato por revisar",
        "fecha_invalida": "Fechas no válidas",
        "producto_ausente": "Productos sin identificar",
        "cantidad_ausente": "Cantidades ausentes",
        "cantidad_ambigua": "Cantidades con separador ambiguo",
        "cantidad_normalizable": "Cantidades que parecen normalizables",
        "cantidad_revisar": "Cantidades con formato por revisar",
        "cantidad_invalida": "Cantidades no interpretables",
        "cantidad_negativa": "Cantidades negativas (no son errores automáticamente)",
        "cantidad_cero": "Cantidades cero (no son errores automáticamente)",
    }
    for clave, etiqueta in etiquetas.items():
        print(f"  {etiqueta}: {informe.conteos[clave]}")
    if informe.duplicados_exactos_origen is not None:
        print(f"  Filas idénticas en archivo original (a revisar): {informe.duplicados_exactos_origen}")
    if informe.productos_con_descripciones_distintas is not None:
        print(f"  Productos con varias descripciones (revisión agrupada): {informe.productos_con_descripciones_distintas}")
        if informe.productos_con_descripciones_distintas:
            print("  NOTA: las etiquetas diferentes se cuentan por producto, no por cada venta.")
    print("NOTA: estos avisos no corrigen, eliminan ni completan ningún dato.")
    print("La revisión de periodos sin registros requiere confirmar cobertura y frecuencia temporal.")


def mostrar_detalle(informe: InformeCalidad, max_incidencias: int = 20) -> None:
    """Muestra fila de la tabla, valor, diagnóstico y acción propuesta.

    Solo se enseñan los campos seleccionados; no se imprime la fila original
    completa (podría incluir clientes, direcciones, etc.).
    """
    print("\n=== REGISTRO DETALLADO DE INCIDENCIAS (FASE 2.1) ===")
    print("'Fila' = posición dentro de los datos cargados, NO fila física del Excel/CSV.")
    if informe.incidencias.empty:
        print("No se han detectado incidencias individuales en los campos analizados.")
        if informe.productos_con_descripciones_distintas:
            print("Hay discrepancias agrupadas por producto; consulta la auditoría avanzada.")
        return
    for _, e in informe.incidencias.head(max_incidencias).iterrows():
        original = str(e["valor_original"])
        vista = original[:87] + "..." if len(original) > 90 else original
        print(f"\nFila {int(e['fila_tabla'])} | {e['campo']} = {vista!r}")
        print(f"  Diagnóstico: {e['tipo']} ({e['nivel']})")
        print(f"  Acción propuesta: {e['accion_propuesta']}")
    restantes = len(informe.incidencias) - min(max_incidencias, len(informe.incidencias))
    if restantes:
        print(f"\n... y {restantes} incidencias más (se ha limitado la vista).")
    print("\nNOTA: ninguna acción propuesta se ha ejecutado; no se han corregido datos.")
