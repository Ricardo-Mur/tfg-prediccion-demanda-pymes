"""Datos de ejemplo: archivos de ventas inventados. Ningún dato es real.

Sirven para probar la aplicación sin archivos propios y para las pruebas de la interfaz.

Una tienda con 10 productos, de 01/2023 a 06/2025, repartida en tres archivos (dos CSV y un Excel). Tiene lo
que hace que el programa pregunte cosas distintas según el caso:
  - tres series: «A» (ventas), «R» (rectificativas, con líneas negativas y una positiva) y «X» (ajustes);
  - líneas negativas de los cuatro tipos que distingue el programa;
  - productos que dejan de venderse 2, 8 y 15 meses antes del final, y uno que empieza 6 meses antes;
  - columnas de clientes con valores reconocibles («CANARIO»), para comprobar que no salen en ningún sitio;
  - columnas del artículo: familia, estado e importe.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MESES = 30                       # k = 0 es 01/2023
PRODUCTOS = {                    # código: (unidades al mes, variación, primer mes, último mes, familia, descripción)
    'P01': (210, 6, 0, 29, 'Bebidas', 'Agua mineral 1,5 L'),
    'P02': (120, 5, 0, 29, 'Bebidas', 'Zumo de naranja 1 L'),
    'P03': (60, 14, 0, 29, 'Limpieza', 'Lavavajillas 750 ml'),
    'P04': (30, 22, 0, 29, 'Limpieza', 'Estropajo pack 3'),
    'P05': (24, 3, 0, 29, 'Hogar', 'Paraguas plegable'),          # solo se vende algunos meses
    'P06': (50, 4, 24, 29, 'Hogar', 'Funda de sofá'),             # nuevo
    'P07': (45, 4, 0, 27, 'Hogar', 'Bombilla LED'),               # sin ventas los 2 últimos meses
    'P08': (40, 4, 0, 21, 'Limpieza', 'Recambio de fregona'),     # sin ventas los 8 últimos meses
    'P09': (35, 4, 0, 14, '', 'Vela aromática'),                  # sin ventas los 15 últimos meses
    'P10': (80, 5, 0, 29, 'Bebidas', 'Refresco de cola 2 L'),
}
COLUMNAS = ['Fecha', 'CodArt', 'Descripcion', 'Cantidad', 'Serie', 'Factura', 'CodCliente', 'NomCli', 'Familia', 'Activo',
            'Importe']


def _fila(fecha, producto, cantidad, serie, factura, cliente):
    _, _, _, _, familia, descripcion = PRODUCTOS[producto]
    precio = 1.5 + int(producto[1:]) * 0.75
    return (fecha, producto, descripcion, cantidad, serie, factura, f'CLI-CANARIO-{cliente:02d}', f'CANARIO_NOMBRE_{cliente:02d}',
            familia, 'N' if producto == 'P09' else 'S', round(cantidad * precio, 2))


def tabla(semilla: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    filas, n = [], 1000
    for k in range(MESES):
        anio, mes = 2023 + k // 12, k % 12 + 1
        for j, (p, (base, variacion, primero, ultimo, *_)) in enumerate(PRODUCTOS.items()):
            if not primero <= k <= ultimo or (p == 'P05' and mes not in (3, 10)):
                continue
            nivel = base * (1 + (0.25 * np.sin(2 * np.pi * (mes - 1) / 12) if p == 'P02' else 0))
            for _ in range(3):
                n += 1
                q = max(int(round(nivel / 3 + rng.normal(0, variacion))), 1)
                filas.append(_fila(f'{anio}-{mes:02d}-{int(rng.integers(1, 21)):02d}', p, q, 'A', f'A{n}', int(rng.integers(0, 5))))
    t = pd.DataFrame(filas, columns=COLUMNAS)

    def venta(producto, mes):
        return t[(t.CodArt == producto) & t.Fecha.str.startswith(mes)].iloc[0]

    def cliente(v):
        return int(v.CodCliente[-2:])

    v1, v2, v3, v4 = venta('P01', '2024-03'), venta('P02', '2024-05'), venta('P03', '2024-07'), venta('P01', '2024-09')
    extras = [
        _fila(v1.Fecha, 'P01', -2, 'A', v1.Factura, cliente(v1)),               # en el mismo documento que su venta
        _fila('2024-05-25', 'P02', -int(v2.Cantidad), 'R', 'R1', cliente(v2)),  # venta anterior, misma cantidad
        _fila('2024-07-26', 'P03', -int(v3.Cantidad), 'R', 'R2', cliente(v3)),  # venta anterior, misma cantidad
        _fila('2024-09-27', 'P01', -3, 'R', 'R3', cliente(v4)),                 # venta anterior, cantidad mayor
        _fila('2024-10-15', 'P04', -5, 'R', 'R4', 99),                          # ninguna venta a la que corresponda
        _fila('2024-11-10', 'P02', 4, 'R', 'R5', 0),                            # línea positiva en la serie de rectificativas
        _fila('2024-02-10', 'P10', 7, 'X', 'X1', 1),                            # ajustes
        _fila('2024-08-12', 'P10', -7, 'X', 'X2', 1),
    ]
    t = pd.concat([t, pd.DataFrame(extras, columns=COLUMNAS)], ignore_index=True)
    return t.sort_values(['Fecha', 'Factura', 'Cantidad'], kind='stable').reset_index(drop=True)


def tres_archivos(carpeta: Path, semilla: int = 11) -> list[Path]:
    """ventas_2023.csv, ventas_2024.csv y ventas_2025.xlsx."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    t = tabla(semilla)
    rutas = []
    for anio in (2023, 2024):
        ruta = carpeta / f'ventas_{anio}.csv'
        t[t.Fecha.str.startswith(str(anio))].to_csv(ruta, index=False)
        rutas.append(ruta)
    ruta = carpeta / 'ventas_2025.xlsx'
    t[t.Fecha.str.startswith('2025')].to_excel(ruta, index=False, sheet_name='Ventas')
    return rutas + [ruta]


def archivo_de_familias(carpeta: Path) -> Path:
    """Producto, familia y subfamilia: dos columnas podrían ser la familia, así que el programa pregunta."""
    ruta = Path(carpeta) / 'secciones.csv'
    pd.DataFrame([(p, f or 'Varios', f'{f or "Varios"} {1 + i % 2}') for i, (p, (*_, f, _d)) in enumerate(PRODUCTOS.items())],
                 columns=['Referencia', 'Familia', 'Subfamilia']).to_csv(ruta, index=False)
    return ruta
