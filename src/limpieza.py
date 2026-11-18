"""Fase 2.2: limpieza conservadora de un histórico de ventas ya mapeado.

No edita los archivos originales. Solo normaliza datos inequívocos y aplica
reglas EXPLÍCITAMENTE confirmadas por el usuario. Deja los demás valores
intactos y los recoge en `pendientes`.

No completa periodos sin ventas: eso requiere confirmar cobertura y frecuencia.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
import numbers
import re

import pandas as pd

from .calidad import _cantidad_estado, _fecha_estado, _sin_valor

COLUMNAS_CAMBIOS = (
    "fila_tabla", "campo", "valor_original", "valor_nuevo", "motivo", "origen", "tipo_cambio"
)
COLUMNAS_PENDIENTES = (
    "fila_tabla", "campo", "valor_original", "tipo", "motivo"
)


@dataclass
class ResultadoLimpieza:
    datos: pd.DataFrame
    cambios: pd.DataFrame
    pendientes: pd.DataFrame


def _fecha_normalizada(
    valor: object,
    convencion_fecha: str | None,
    siglo_dos_digitos: int | None,
) -> tuple[object | None, str | None, str | None, str]:
    """Devuelve (valor, incidencia_pendiente, motivo, origen).

    `convencion_fecha`: None, 'dia_mes', 'mes_dia'. Solo es aplicable si el
    usuario ha confirmado que el archivo sigue uniformemente ese criterio.
    `siglo_dos_digitos`: None, 1900, 2000 (confirmación expresa del usuario).
    """
    if _sin_valor(valor):
        return None, "fecha_ausente", "Falta la fecha; no se inventa.", ""

    if isinstance(valor, (pd.Timestamp, datetime, date)):
        try:
            return pd.Timestamp(valor), None, None, "sin_cambio"
        except (ValueError, TypeError):
            return None, "fecha_invalida", "Fecha fuera del intervalo válido.", ""

    if not isinstance(valor, str):
        return None, "fecha_revisar", "Formato de fecha no reconocido; necesita revisión.", ""
    texto = valor.strip()

    # ISO, incluido el caso YYYY/MM/DD. Estos formatos no invierten día y mes.
    if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}(?:[T ]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?", texto):
        try:
            return pd.Timestamp(datetime.fromisoformat(texto)), None, None, "automatica"
        except (ValueError, TypeError):
            return None, "fecha_invalida", "Fecha imposible en formato ISO.", ""
    if re.fullmatch(r"\d{4}/\d{1,2}/\d{1,2}", texto):
        anio, mes, dia = map(int, texto.split("/"))
        try:
            return pd.Timestamp(date(anio, mes, dia)), None, None, "automatica"
        except ValueError:
            return None, "fecha_invalida", "Fecha imposible.", ""

    # Aceptamos separadores repetidos SOLAMENTE cuando todas las partes
    # conservan el mismo separador. Ej.: 16//03/25 y 15//03/2025.
    m = re.fullmatch(r"(\d{1,2})([/.\-]{1,3})(\d{1,2})([/.\-]{1,3})(\d{2}|\d{4})", texto)
    if not m:
        return None, "fecha_revisar", "Formato de fecha no reconocido.", ""
    a_txt, s1, b_txt, s2, y_txt = m.groups()
    if len(set(s1 + s2)) != 1:
        return None, "fecha_revisar", "Separadores mezclados: requiere revisión.", ""

    a, b = int(a_txt), int(b_txt)
    if len(y_txt) == 2:
        if siglo_dos_digitos is None:
            # Aunque el siglo sea desconocido, podemos comprobar si la fecha
            # sería imposible en ambas convenciones y ambos siglos.
            ordenes_viables = set()
            for siglo_candidato in (1900, 2000):
                for clave, dia, mes in (("dia_mes", a, b), ("mes_dia", b, a)):
                    try:
                        date(siglo_candidato + int(y_txt), mes, dia)
                        ordenes_viables.add(clave)
                    except ValueError:
                        pass
            if not ordenes_viables:
                return None, "fecha_invalida", "Fecha imposible con ambas interpretaciones.", ""
            explicacion = "Año de dos cifras: confirmar si es 19xx o 20xx."
            if len(ordenes_viables) == 2 and a != b:
                explicacion += " También hay que confirmar si es día/mes o mes/día."
            return None, "fecha_siglo_sin_confirmar", explicacion, ""
        anio = siglo_dos_digitos + int(y_txt)
    else:
        anio = int(y_txt)

    posibles: dict[str, date] = {}
    for clave, dia, mes in (("dia_mes", a, b), ("mes_dia", b, a)):
        try:
            posibles[clave] = date(anio, mes, dia)
        except ValueError:
            pass

    if not posibles:
        return None, "fecha_invalida", "Fecha imposible; no se inventa una sustitución.", ""

    if convencion_fecha is not None:
        if convencion_fecha not in posibles:
            return None, "fecha_contradice_convencion", (
                "No coincide con el formato uniforme confirmado; puede haber formatos mezclados."
            ), ""
        fecha = posibles[convencion_fecha]
        # Si hubo interpretación de orden o siglo, es una decisión del usuario.
        origen = "confirmada_usuario" if (len(posibles) == 2 and posibles["dia_mes"] != posibles["mes_dia"]) or len(y_txt) == 2 else "automatica"
        return pd.Timestamp(fecha), None, None, origen

    if len(set(posibles.values())) > 1:
        return None, "fecha_ambigua", "Confirmar si el archivo usa día/mes o mes/día.", ""

    fecha = next(iter(posibles.values()))
    origen = "confirmada_usuario" if len(y_txt) == 2 else "automatica"
    return pd.Timestamp(fecha), None, None, origen


def _cantidad_normalizada(
    valor: object,
    separador_ambiguo: str | None,
) -> tuple[Decimal | None, str | None, str | None, str]:
    estado, numero = _cantidad_estado(valor)
    if estado == "ausente":
        return None, "cantidad_ausente", "Falta la cantidad; no se sustituye por cero.", ""
    if estado == "invalida":
        return None, "cantidad_invalida", "Cantidad no numérica; solicitar corrección.", ""
    if estado == "revisar":
        return None, "cantidad_revisar", "Formato numérico no reconocido.", ""
    if estado == "ambigua":
        if separador_ambiguo is None:
            return None, "cantidad_ambigua", "Confirmar si el separador indica miles o decimales.", ""
        texto = str(valor).strip()
        if separador_ambiguo == "miles":
            numero = Decimal(texto.replace(".", "").replace(",", ""))
        else:
            numero = Decimal(texto.replace(",", "."))
        return numero, None, None, "confirmada_usuario"
    # Números ya reconocibles, incluidos negativos y cero: conservar el valor.
    if isinstance(valor, (numbers.Number, Decimal)) and not isinstance(valor, bool):
        return numero, None, None, "sin_cambio"
    return numero, None, None, "automatica"


def _tipo_de_cambio(campo: str, original: object, origen: str) -> str:
    """Clasifica la transformación, sin presentarla como un error corregido."""
    if origen == "confirmada_usuario":
        return "interpretacion_confirmada"
    if campo in ("producto", "descripcion"):
        return "normalizacion_formato"  # Espacios exteriores sobrantes.
    if campo == "cantidad":
        if isinstance(original, str):
            texto = original.strip()
            # Número en notación inequívoca que ya podía leerse: cambio de tipo.
            if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", texto):
                return "conversion_tipo"
        return "normalizacion_formato"
    if campo == "fecha":
        if isinstance(original, str) and re.fullmatch(
            r"\d{4}-\d{1,2}-\d{1,2}(?:[T ]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?",
            original.strip(),
        ):
            return "conversion_tipo"
        return "normalizacion_formato"
    return "conversion_tipo"


def limpiar_datos(
    seleccion: pd.DataFrame,
    *,
    convencion_fecha: str | None = None,
    siglo_dos_digitos: int | None = None,
    separador_ambiguo: str | None = None,
) -> ResultadoLimpieza:
    """Normaliza sin descartar filas, manteniendo `seleccion` intacto.

    - convencion_fecha: None | 'dia_mes' | 'mes_dia'.
    - siglo_dos_digitos: None | 1900 | 2000.
    - separador_ambiguo: None | 'miles' | 'decimal'.

    Las reglas NO se deducen automáticamente de otros registros: deben haber
    sido confirmadas expresamente para el archivo, sin formatos mezclados.
    """
    if convencion_fecha not in (None, "dia_mes", "mes_dia"):
        raise ValueError("convencion_fecha inválida")
    if siglo_dos_digitos not in (None, 1900, 2000):
        raise ValueError("siglo_dos_digitos inválido")
    if separador_ambiguo not in (None, "miles", "decimal"):
        raise ValueError("separador_ambiguo inválido")
    for campo in ("fecha", "producto", "cantidad"):
        if campo not in seleccion.columns:
            raise ValueError(f"Falta la columna obligatoria {campo!r}")

    datos = seleccion.copy(deep=True)

    # Pandas 3 puede importar columnas de texto con dtype 'str' (StringDtype).
    # Ese tipo NO admite escribir directamente Timestamp o Decimal. Como
    # necesitamos conservar temporalmente fechas originales sin resolver y
    # fechas normalizadas en la misma columna, usamos dtype object SOLO en
    # la copia de trabajo. La tabla original y sus tipos no se modifican.
    for campo in ("fecha", "producto", "cantidad", "descripcion"):
        if campo in datos.columns:
            datos[campo] = datos[campo].astype(object)

    cambios: list[dict] = []
    pendientes: list[dict] = []

    for indice in range(len(seleccion)):
        for campo in ("fecha", "producto", "cantidad", "descripcion"):
            if campo not in seleccion.columns:
                continue
            original = seleccion.iloc[indice][campo]
            if campo == "fecha":
                nuevo, error, motivo, origen = _fecha_normalizada(
                    original, convencion_fecha, siglo_dos_digitos
                )
            elif campo == "cantidad":
                nuevo, error, motivo, origen = _cantidad_normalizada(
                    original, separador_ambiguo
                )
            else:
                if _sin_valor(original):
                    if campo == "producto":
                        nuevo, error, motivo, origen = (
                            None, "producto_ausente", "Falta el identificador; no se infiere.", ""
                        )
                    else:
                        # La descripción es opcional.
                        continue
                elif isinstance(original, str):
                    nuevo = original.strip()
                    error = motivo = None
                    origen = "automatica" if nuevo != original else "sin_cambio"
                else:
                    # Un identificador puede ser numérico; no lo reinterpretamos.
                    continue

            if error is not None:
                pendientes.append({
                    "fila_tabla": indice + 1, "campo": campo,
                    "valor_original": original, "tipo": error, "motivo": motivo,
                })
                continue
            if origen == "sin_cambio":
                continue
            # Actualizamos únicamente la COPIA. El original no se altera.
            datos.iat[indice, datos.columns.get_loc(campo)] = nuevo
            tipo_cambio = _tipo_de_cambio(campo, original, origen)
            cambios.append({
                "fila_tabla": indice + 1, "campo": campo,
                "valor_original": original, "valor_nuevo": nuevo,
                "motivo": "Eliminación de espacios exteriores" if campo in ("producto", "descripcion") else "Preparación de dato conforme a reglas verificadas",
                "origen": origen, "tipo_cambio": tipo_cambio,
            })

    return ResultadoLimpieza(
        datos=datos,
        cambios=pd.DataFrame(cambios, columns=COLUMNAS_CAMBIOS),
        pendientes=pd.DataFrame(pendientes, columns=COLUMNAS_PENDIENTES),
    )


def mostrar_limpieza(resultado: ResultadoLimpieza, max_filas: int = 12) -> None:
    print("\n=== FASE 2.2: LIMPIEZA CONTROLADA ===")
    print(f"Registros conservados: {len(resultado.datos)}")
    print(f"Transformaciones técnicas registradas: {len(resultado.cambios)}")
    categorias = (
        ("conversion_tipo", "Conversiones de tipo"),
        ("normalizacion_formato", "Normalizaciones de formato"),
        ("interpretacion_confirmada", "Interpretaciones confirmadas por el usuario"),
    )
    for clave, etiqueta in categorias:
        n = int((resultado.cambios["tipo_cambio"] == clave).sum()) if not resultado.cambios.empty else 0
        print(f"  {etiqueta}: {n}")
    print(f"Campos pendientes de revisión: {len(resultado.pendientes)}")
    if not resultado.cambios.empty:
        print("\nPRIMERAS MODIFICACIONES (en memoria, no en el archivo original)")
        for _, c in resultado.cambios.head(max_filas).iterrows():
            print(f"  Fila {int(c.fila_tabla)}, {c.campo}: {str(c.valor_original)!r} -> {str(c.valor_nuevo)!r} [{c.origen}; {c.tipo_cambio}]")
    if not resultado.pendientes.empty:
        print("\nPENDIENTES (no se han corregido por suposición)")
        for _, p in resultado.pendientes.head(max_filas).iterrows():
            print(f"  Fila {int(p.fila_tabla)}, {p.campo}: {str(p.valor_original)!r} | {p.motivo}")
        restantes = len(resultado.pendientes) - max_filas
        if restantes > 0:
            print(f"  ... y {restantes} pendientes más")
    print("\nEl archivo original sigue intacto. Las correcciones están solo en una copia en memoria.")
