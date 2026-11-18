"""Fase 1: carga y propuesta de cabeceras/columnas para ficheros Excel/CSV.

Todas las decisiones sobre hoja, fila de cabecera y columnas necesitan
confirmación del usuario. Las puntuaciones son heurísticas de ordenación,
NO probabilidades ni garantías semánticas. No se modifica el archivo fuente.
"""

from datetime import date, datetime
from pathlib import Path
import re
import unicodedata

import pandas as pd


# Alias generales: ningún nombre de columna es específico de una empresa.
SINONIMOS = {
    "fecha": {
        "fecha", "fechaventa", "fechaoperacion", "fechafactura",
        "fechapedido", "date", "salesdate", "orderdate",
        "invoicedate", "transactiondate", "timestamp", "dia",
        "diaoperacion", "diaventa", "fechatransaccion",
    },
    "producto": {
        "producto", "articulo", "codart", "codigoarticulo", "codigoproducto",
        "sku", "referencia", "item", "product", "productid", "stockcode",
        "categoria", "familia", "descripcion", "codigoitem",
        "refarticulo", "refproducto", "itemcode", "productcode",
        "nombreproducto", "nombrearticulo", "productname", "itemname",
    },
    "cantidad": {
        "cantidad", "unidades", "uds", "qty", "quantity", "units",
        "cantidadvendida", "unidadesvendidas", "quantitysold", "salesqty",
        "udsvendidas", "udvendidas", "cantidadventas", "unidadesventa",
    },
    # Campo opcional: etiqueta legible, NO reemplaza el identificador.
    "descripcion": {
        "descripcion", "description", "descart", "descart1",
        "descripcionarticulo", "descripcionproducto", "productdescription",
        "itemdescription", "productname", "itemname", "nombreproducto",
        "nombrearticulo", "nombrecomercial", "nomarticulo",
        "denominacion", "nombre", "descrip",
    },
}

# Términos de advertencia según función. Una columna descartada de la lista
# de propuestas sigue pudiendo elegirse manualmente si el usuario lo confirma.
CONTEXTOS_INCOMPATIBLES = {
    "fecha": (
        "recepcion", "caduc", "vencim", "expiry", "expiration", "expir",
        "alta", "creacion", "modificacion", "manufact", "fabricacion",
        "nacimiento", "deliverydate", "shippingdate",
    ),
    "producto": (
        "activo", "estado", "estatus", "disponible", "habilitado",
        "productstatus", "articlestatus", "preciounitario", "importe",
    ),
    "cantidad": (
        "stock", "existencia", "inventario", "almacen", "disponible",
        "reservad", "saldoes", "saldo", "prevista", "prevision",
    ),
    "descripcion": (
        "cliente", "proveedor", "transport", "vendedor", "empleado",
        "responsable", "codig", "codigo", "referencia", "sku",
        "stockcode", "productid", "itemcode", "estado", "activo",
        "preci", "importe", "ticket",
    ),
}

# Algunas palabras aumentan la pertinencia del nombre pero NO autorizan a
# seleccionar la columna sin que el usuario confirme qué representa.
CONTEXTOS_FAVORABLES = {
    "fecha": ("venta", "operacion", "transaccion", "invoice", "order"),
    "producto": ("codigo", "cod", "referencia", "sku", "productid", "itemcode"),
    "cantidad": ("vendid", "venta", "sold", "salesqty"),
    "descripcion": ("descripcion", "description", "nombre", "descart"),
}


def _normalizar(texto: object) -> str:
    """Quita tildes, puntuación y espacios, sin alterar el dato fuente."""
    if texto is None or pd.isna(texto):
        return ""
    texto = unicodedata.normalize("NFKD", str(texto).casefold())
    texto = "".join(letra for letra in texto if not unicodedata.combining(letra))
    return re.sub(r"[^a-z0-9]", "", texto)


def _puntuacion_nombre(nombre: object, tipo: str) -> int:
    """Coincidencia por nombre con penalización contextual: no es confianza."""
    nombre = _normalizar(nombre)
    if not nombre:
        return 0

    # 'StockCode' es un alias válido de identificador de producto, y no debe
    # descartarse por contener 'stock' (el contexto se evalúa por función).
    if any(palabra in nombre for palabra in CONTEXTOS_INCOMPATIBLES[tipo]):
        return 0

    sinonimos = SINONIMOS[tipo]
    if nombre in sinonimos:
        puntos = 100
    elif any(nombre.startswith(s) for s in sinonimos if len(s) >= 4):
        puntos = 65
    elif any(s in nombre for s in sinonimos if len(s) >= 5):
        puntos = 40
    else:
        return 0

    if any(palabra in nombre for palabra in CONTEXTOS_FAVORABLES[tipo]):
        puntos += 10
    return puntos


def _config_csv(archivo: Path, cabecera: int | None) -> tuple[str, str]:
    """Obtiene codificación y delimitador preservando filas en blanco.

    La cabecera elegida prima sobre la primera línea, que puede ser un título
    de informe. No interpreta columnas ni sus valores.
    """
    import csv
    for codificacion in ("utf-8-sig", "cp1252"):
        try:
            with archivo.open("r", encoding=codificacion, newline="") as fh:
                lineas = [line for _,line in zip(range(max(40,(cabecera or 0)+5)),fh)]
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("No se ha podido leer la codificación del CSV")
    candidatos = (",", ";", "\t", "|")
    evaluaciones = []
    for delimitador in candidatos:
        filas = []
        for linea in lineas:
            try:
                filas.append(next(csv.reader([linea],delimiter=delimitador)))
            except csv.Error:
                filas.append([linea])
        if cabecera is not None and cabecera < len(filas):
            indice = cabecera
            valores = filas[indice]
            puntos = sum(max((_puntuacion_nombre(v,t) for v in valores), default=0) >= 65 for t in ("fecha","producto","cantidad"))
            score = (puntos*1000 + len(valores)*10, len(valores))
        else:
            puntajes = []
            for valores in filas[:30]:
                roles = sum(max((_puntuacion_nombre(v,t) for v in valores), default=0) >= 65 for t in ("fecha","producto","cantidad"))
                puntajes.append((roles*1000+len(valores)*10, len(valores)))
            score = max(puntajes, default=(0,0))
        evaluaciones.append((score,delimitador))
    _, delimitador = max(evaluaciones,key=lambda v:v[0])
    return codificacion, delimitador


def _leer(
    archivo: str | Path,
    *,
    hoja: str | None = None,
    cabecera: int | None = None,
    filas: int | None = None,
) -> pd.DataFrame:
    archivo = Path(archivo)
    if archivo.suffix.lower() == ".xlsx":
        return pd.read_excel(
            archivo, sheet_name=hoja if hoja is not None else 0,
            header=cabecera, nrows=filas, dtype=object, engine="openpyxl",
        )
    if archivo.suffix.lower() == ".csv":
        import csv
        from itertools import islice
        codificacion, delimitador = _config_csv(archivo, cabecera)
        if cabecera is None:
            # Vista preliminar para detectar encabezados incluso cuando las
            # primeras líneas tienen una cantidad distinta de columnas.
            with archivo.open('r', encoding=codificacion, newline='') as fh:
                lector = csv.reader(fh, delimiter=delimitador)
                registros = list(islice(lector, filas)) if filas is not None else list(lector)
            ancho = max(map(len, registros), default=0)
            return pd.DataFrame([r+[None]*(ancho-len(r)) for r in registros],dtype=object)
        # skiprows evita que una cabecera desplazada se interprete según la
        # longitud de una línea de metadatos previa. Las filas vacías intermedias
        # se conservan; 'NA' y 'NULL' pueden ser códigos comerciales legítimos.
        return pd.read_csv(
            archivo, sep=delimitador, encoding=codificacion, skiprows=cabecera,
            header=0, nrows=filas, dtype=object, keep_default_na=False,
            skip_blank_lines=False, engine="python",
        )
    raise ValueError("Formato no admitido: utiliza .xlsx o .csv")


def listar_hojas(archivo: str | Path) -> list[str]:
    archivo = Path(archivo)
    if archivo.suffix.lower() == ".csv":
        return ["Archivo CSV"]
    if archivo.suffix.lower() != ".xlsx":
        raise ValueError("Solo se admiten .xlsx o .csv en este prototipo")
    with pd.ExcelFile(archivo, engine="openpyxl") as libro:
        return libro.sheet_names


def proponer_cabeceras(
    archivo: str | Path, hoja: str | None = None, max_filas: int = 30
) -> list[dict]:
    """Propone filas candidatas (índice desde 0); nunca elige automáticamente."""
    vista = _leer(archivo, hoja=hoja, cabecera=None, filas=max_filas)
    propuestas = []
    for fila, valores in vista.iterrows():
        celdas = [valor for valor in valores if isinstance(valor, str) and valor.strip()]
        if len(celdas) < 2:
            continue
        puntuaciones = {
            tipo: max((_puntuacion_nombre(v, tipo) for v in celdas), default=0)
            for tipo in ("fecha", "producto", "cantidad")
        }
        roles = [tipo for tipo, puntos in puntuaciones.items() if puntos >= 65]
        if not roles:
            continue
        puntos = len(roles) * 1000 + sum(puntuaciones.values())
        propuestas.append({"fila": int(fila), "puntos": puntos, "campos": roles})
    return sorted(propuestas, key=lambda p: (-p["puntos"], p["fila"]))[:5]


def cargar_tabla(
    archivo: str | Path, fila_cabecera: int, hoja: str | None = None
) -> pd.DataFrame:
    """Lee la tabla tras la confirmación de cabecera, sin limpiar registros."""
    if fila_cabecera < 0:
        raise ValueError("La fila de cabecera no puede ser negativa")
    return _leer(archivo, hoja=hoja, cabecera=fila_cabecera)


def _evidencia_contenido(serie: pd.Series, tipo: str) -> tuple[int, str]:
    """Ajuste pequeño por muestra de contenido; no valida los datos.

    Solo se usa en columnas candidatas por su nombre. Un identificador de
    producto puede ser numérico; nunca se le exige tener letras.
    """
    muestra = serie.dropna().head(200)
    if muestra.empty:
        return -10, "muestra sin valores"

    if tipo == "fecha":
        patron_fecha = re.compile(
            r"^\s*(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})(?:\s.*)?$"
        )
        compatibles = sum(
            isinstance(v, (date, datetime, pd.Timestamp))
            or (isinstance(v, str) and bool(patron_fecha.match(v)))
            for v in muestra
        )
        proporcion = compatibles / len(muestra)
        if proporcion >= 0.8:
            return 10, "valores con apariencia de fecha"
        if proporcion < 0.2:
            return -10, "pocos valores con apariencia de fecha"
        return 0, "formato de fecha por revisar"

    if tipo == "cantidad":
        validos = pd.to_numeric(muestra, errors="coerce").notna().mean()
        if validos >= 0.8:
            return 10, "predominan valores numéricos"
        if validos < 0.2:
            return -10, "pocos valores numéricos reconocibles"
        return 0, "formato numérico por revisar"

    if tipo == "descripcion":
        return 0, "texto descriptivo por confirmar"

    # No penalizamos los códigos de producto numéricos, frecuentes en ERP.
    return 0, "identificadores por confirmar"


def proponer_columnas(datos: pd.DataFrame) -> dict[str, list[dict]]:
    """Ordena candidatos por nombre, contexto y pequeña evidencia de muestra.

    No devuelve una elección automática. Todas las columnas pueden seguir
    seleccionándose manualmente, incluso las excluidas de las sugerencias.
    """
    resultado = {}
    for tipo in SINONIMOS:
        opciones = []
        for indice, nombre in enumerate(datos.columns):
            puntos_nombre = _puntuacion_nombre(nombre, tipo)
            if not puntos_nombre:
                continue
            ajuste, observacion = _evidencia_contenido(datos.iloc[:, indice], tipo)
            opciones.append({
                "indice": indice,
                "nombre": str(nombre),
                "puntos": puntos_nombre + ajuste,
                "observacion": observacion,
            })
        resultado[tipo] = sorted(
            opciones, key=lambda p: (-p["puntos"], p["indice"])
        )
    return resultado


# Estos nombres indican variables económicas o de inventario, no unidades
# vendidas. Es una protección contra asignaciones manuales claramente erróneas.
# No representa una comprobación de que exista una columna de ventas.
NOMBRES_INCOMPATIBLES_CANTIDAD = (
    "precio", "price", "coste", "costo", "pvp", "importe",
    "facturacion", "revenue", "unitprice", "netvalue",
    "amount", "subtotal", "totalprice", "valorventa",
    "stock", "existencias", "inventario", "almacen",
)


def evaluar_campo_obligatorio(
    datos: pd.DataFrame, indice: int, tipo: str
) -> tuple[bool, str]:
    """Comprueba incompatibilidades evidentes, no la semántica real.

    Devuelve (admisible, explicación). Si una cabecera no es reconocida,
    puede confirmarla el usuario; no se descarta solo por el nombre.
    """
    if tipo not in ("fecha", "producto", "cantidad"):
        raise ValueError(f"Campo obligatorio desconocido: {tipo}")
    if not isinstance(indice, int) or not 0 <= indice < len(datos.columns):
        return False, "La posición de columna no existe."
    nombre = str(datos.columns[indice])
    normalizado = _normalizar(nombre)
    if tipo == "cantidad" and any(
        termino in normalizado for termino in NOMBRES_INCOMPATIBLES_CANTIDAD
    ):
        return (False, f"'{nombre}' parece indicar precios, importes o "
                       "existencias, no unidades vendidas.")
    return True, "La asignación debe ser confirmada por el usuario."


def seleccionar_campos(datos: pd.DataFrame, indices: dict[str, int]) -> pd.DataFrame:
    """Crea una copia con campos estándar, sin modificar los datos de entrada.

    Los tres campos básicos son obligatorios; `descripcion` es opcional.
    Se selecciona por posición para admitir cabeceras repetidas en origen.
    Las transformaciones/validaciones semánticas vendrán en la fase 2.
    """
    obligatorios = {"fecha", "producto", "cantidad"}
    permitidos = obligatorios | {"descripcion"}
    faltantes = obligatorios - indices.keys()
    desconocidos = indices.keys() - permitidos
    if faltantes or desconocidos:
        raise ValueError(
            f"Asignación inválida. Faltan: {sorted(faltantes)}; "
            f"campos desconocidos: {sorted(desconocidos)}"
        )
    posiciones = list(indices.values())
    if len(set(posiciones)) != len(posiciones):
        raise ValueError("Una misma columna no puede asignarse a varios campos")
    if any(not isinstance(i, int) or not 0 <= i < len(datos.columns) for i in posiciones):
        raise ValueError("Hay posiciones de columnas fuera del archivo")
    for campo in obligatorios:
        admisible, explicacion = evaluar_campo_obligatorio(datos, indices[campo], campo)
        if not admisible:
            raise ValueError(f"Asignación inválida para '{campo}': {explicacion}")
    return pd.DataFrame({
        campo: datos.iloc[:, indice].to_numpy(copy=True)
        for campo, indice in indices.items()
    })
