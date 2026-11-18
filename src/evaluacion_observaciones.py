"""Fase 3.11: evaluación retrospectiva de referencias con estados mensuales.

Las cantidades positivas y los ceros explícitos son observaciones PRELIMINARES.
NUNCA se transforma un mes sin registros o pendiente en cero. Las referencias
usan exclusivamente valores de meses anteriores y el horizonte es un mes.
Un cero explícito NO demuestra completitud de ventas ni ausencia de demanda.

Esta etapa NO entrena modelos complejos ni valida la demanda real de la pyme.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

import pandas as pd

from .objetivo_mensual import InformeObjetivoMensual
from .validacion_referencias import _errores

ESTADOS_CON_VALOR = frozenset({'positivo_registrado', 'cero_explicito'})
ESTADOS_PENDIENTES = frozenset({
    'sin_registro', 'pendiente_negativos', 'pendiente_dato',
    'pendiente_producto_desconocido',
})


@dataclass
class InformeEvaluacionObservaciones:
    detalle: pd.DataFrame
    evaluaciones: pd.DataFrame
    productos: int
    con_tres_cortes_ingenuos: int
    con_tres_cortes_anuales: int
    comparables: int
    ingenua_mejor: int
    anual_mejor: int
    empate: int
    sin_comparacion: int
    cortes_ingenuos: int
    cortes_anuales: int
    cortes_comunes: int
    cortes_con_cero_como_objetivo: int
    cortes_con_cero_como_predictor: int
    meses_sin_valor: int
    ceros_inferidos: int = 0
    modelos_complejos_entrenados: int = 0


def _indice(m: object) -> int:
    if type(m) is not date or m.day != 1:
        raise ValueError('Cada mes debe ser una fecha date del primer día de mes.')
    return m.year * 12 + m.month - 1


def _cantidad(v: object) -> Decimal | None:
    if v is None or v is pd.NA:
        return None
    if isinstance(v, bool):
        raise ValueError('No se admiten valores booleanos como cantidades.')
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError) as e:
        raise ValueError('Cantidad mensual inválida.') from e
    try:
        x = v if isinstance(v, Decimal) else Decimal(str(v))
    except Exception as e:
        raise ValueError('Cantidad mensual inválida.') from e
    if not x.is_finite() or x < 0:
        raise ValueError('Cantidad mensual negativa o no finita.')
    return x


def evaluar_observaciones_mensuales(
    objetivo: InformeObjetivoMensual,
    *,
    meses_contexto: int = 3,
    cortes_minimos: int = 3,
    meses_anual: int = 12,
) -> InformeEvaluacionObservaciones:
    """Evalúa mes previo y año previo, sin sustituir desconocidos por ceros.

    - Mes anterior: necesita los 3 (configurable) meses inmediatamente previos
      con valores conocidos; predice con el inmediatamente anterior.
    - Anual: predice con el mismo mes del año anterior si ambos valores existen.
    - Compara los MAE exclusivamente en meses comunes, RMSE desempata.
    """
    for etiqueta, v in [('meses de contexto', meses_contexto),
                        ('cortes mínimos', cortes_minimos), ('periodo anual', meses_anual)]:
        if type(v) is not int or v < 1:
            raise ValueError(f'Parámetro incorrecto: {etiqueta}.')

    t = objetivo.detalle
    requeridas = {'producto','mes','estado','unidades_observadas'}
    if not requeridas.issubset(t.columns):
        raise ValueError('Faltan columnas del objetivo mensual.')
    if t[['producto','mes','estado']].isna().any().any():
        raise ValueError('Existen filas sin producto, mes o estado.')
    if t.duplicated(['producto','mes']).any():
        raise ValueError('Hay más de una fila para el mismo producto y mes.')
    if t['producto'].map(lambda p: not isinstance(p, str) or not p.strip()).any():
        raise ValueError('Producto no identificable.')

    cols_d = [
        'producto','meses_con_valor','meses_con_cero_explicito','cortes_ingenua',
        'cortes_anual','cortes_comunes','mae_ingenua','rmse_ingenua',
        'mae_anual','rmse_anual','mae_ingenua_comunes','rmse_ingenua_comunes',
        'mae_anual_comunes','rmse_anual_comunes','referencia_mejor_comunes',
    ]
    cols_e = [
        'producto','mes','observado','estado_observado','prediccion_ingenua',
        'prediccion_anual','error_absoluto_ingenua','error_absoluto_anual',
        'predictor_ingenua_cero','predictor_anual_cero',
    ]
    detalles = []
    evaluaciones = []
    sin_valor = 0
    ceros_objetivo_evaluado = 0
    ceros_predictor_evaluado = 0

    for producto, grupo in t.groupby('producto', sort=False):
        datos: dict[int, Decimal] = {}
        estados: dict[int, str] = {}
        index_fechas = set()
        for r in grupo.itertuples(index=False):
            i = _indice(r.mes)
            if i in index_fechas:
                raise ValueError('Fecha mensual duplicada.')
            index_fechas.add(i)
            estado = r.estado
            valor = _cantidad(r.unidades_observadas)
            if estado not in ESTADOS_CON_VALOR | ESTADOS_PENDIENTES:
                raise ValueError(f'Estado mensual desconocido: {estado!r}.')
            if estado in ESTADOS_CON_VALOR:
                if valor is None or (estado == 'cero_explicito' and valor != 0) or (
                        estado == 'positivo_registrado' and valor <= 0):
                    raise ValueError('Estado mensual y cantidad no concuerdan.')
                datos[i] = valor
                estados[i] = estado
            else:
                if valor is not None:
                    raise ValueError('Una laguna o dato pendiente no debe tener unidades.')
                sin_valor += 1
        # El orden de las filas del DataFrame no condiciona las comparaciones.
        naive: list[tuple[Decimal,Decimal]] = []
        anual: list[tuple[Decimal,Decimal]] = []
        naive_comunes: list[tuple[Decimal,Decimal]] = []
        anual_comunes: list[tuple[Decimal,Decimal]] = []
        for i in sorted(datos):
            real = datos[i]
            p_ingenua = (datos[i - 1] if all((i - k) in datos for k in range(1, meses_contexto + 1))
                         else None)
            p_anual = datos.get(i - meses_anual)
            if p_ingenua is None and p_anual is None:
                continue
            # Cero explícito válido tanto como observación como predictor.
            if real == 0:
                ceros_objetivo_evaluado += 1
            if p_ingenua == 0 or p_anual == 0:
                ceros_predictor_evaluado += 1
            evaluaciones.append({
                'producto':producto,'mes':r_mes(i),'observado':real,
                'estado_observado':estados[i],
                'prediccion_ingenua':p_ingenua, 'prediccion_anual':p_anual,
                'error_absoluto_ingenua':abs(real-p_ingenua) if p_ingenua is not None else None,
                'error_absoluto_anual':abs(real-p_anual) if p_anual is not None else None,
                'predictor_ingenua_cero':p_ingenua == 0 if p_ingenua is not None else False,
                'predictor_anual_cero':p_anual == 0 if p_anual is not None else False,
            })
            if p_ingenua is not None:
                naive.append((real,p_ingenua))
            if p_anual is not None:
                anual.append((real,p_anual))
            if p_ingenua is not None and p_anual is not None:
                naive_comunes.append((real,p_ingenua))
                anual_comunes.append((real,p_anual))
        mn,rn = _errores(naive)
        ma,ra = _errores(anual)
        mcn,rcn = _errores(naive_comunes)
        mca,rca = _errores(anual_comunes)
        ganador = None
        if len(naive_comunes) >= cortes_minimos:
            if (mcn,rcn) < (mca,rca):
                ganador = 'mes_anterior'
            elif (mca,rca) < (mcn,rcn):
                ganador = 'mismo_mes_ano_anterior'
            else:
                ganador = 'empate'
        detalles.append({
            'producto':producto,'meses_con_valor':len(datos),
            'meses_con_cero_explicito':sum(x == 0 for x in datos.values()),
            'cortes_ingenua':len(naive),'cortes_anual':len(anual),
            'cortes_comunes':len(naive_comunes),
            'mae_ingenua':mn,'rmse_ingenua':rn,'mae_anual':ma,'rmse_anual':ra,
            'mae_ingenua_comunes':mcn,'rmse_ingenua_comunes':rcn,
            'mae_anual_comunes':mca,'rmse_anual_comunes':rca,
            'referencia_mejor_comunes':ganador,
        })
    d = pd.DataFrame(detalles, columns=cols_d).sort_values('producto').reset_index(drop=True)
    e = pd.DataFrame(evaluaciones, columns=cols_e).sort_values(['producto','mes']).reset_index(drop=True)
    resultados = d['referencia_mejor_comunes'].value_counts().to_dict() if not d.empty else {}
    cont = lambda k: int(resultados.get(k, 0))
    n_ingenua = int((d['cortes_ingenua'] >= cortes_minimos).sum()) if not d.empty else 0
    n_anual = int((d['cortes_anual'] >= cortes_minimos).sum()) if not d.empty else 0
    comp = cont('mes_anterior') + cont('mismo_mes_ano_anterior') + cont('empate')
    return InformeEvaluacionObservaciones(
        detalle=d, evaluaciones=e, productos=len(d),
        con_tres_cortes_ingenuos=n_ingenua, con_tres_cortes_anuales=n_anual,
        comparables=comp, ingenua_mejor=cont('mes_anterior'),
        anual_mejor=cont('mismo_mes_ano_anterior'), empate=cont('empate'),
        sin_comparacion=len(d)-comp,
        cortes_ingenuos=int(d['cortes_ingenua'].sum()) if not d.empty else 0,
        cortes_anuales=int(d['cortes_anual'].sum()) if not d.empty else 0,
        cortes_comunes=int(d['cortes_comunes'].sum()) if not d.empty else 0,
        cortes_con_cero_como_objetivo=ceros_objetivo_evaluado,
        cortes_con_cero_como_predictor=ceros_predictor_evaluado,
        meses_sin_valor=sin_valor,
    )


def r_mes(i: int) -> date:
    year, month0 = divmod(i, 12)
    if year < 1 or year > 9999:
        raise ValueError('Mes fuera del calendario admitido.')
    return date(year, month0 + 1, 1)


def mostrar_evaluacion_observaciones_corta(r: InformeEvaluacionObservaciones) -> None:
    fmt = lambda v: f'{v:,}'.replace(',', '.')
    print('\n12. ¿CÓMO CAMBIA LA EVALUACIÓN AL CONSERVAR LOS CEROS EXPLÍCITOS?')
    print(f'Productos revisados: {fmt(r.productos)}.')
    print(f'Con al menos tres predicciones evaluadas frente al mes anterior: {fmt(r.con_tres_cortes_ingenuos)}.')
    print(f'Con al menos tres predicciones evaluadas frente al año anterior: {fmt(r.con_tres_cortes_anuales)}.')
    print(f'Con al menos tres fechas comunes: {fmt(r.comparables)}.')
    print(f'  Menor MAE (RMSE para desempate) en mes anterior: {fmt(r.ingenua_mejor)}.')
    print(f'  Menor MAE (RMSE para desempate) en año anterior: {fmt(r.anual_mejor)}.')
    print(f'  Empates: {fmt(r.empate)}.')
    print(f'Productos sin comparación entre ambas referencias: {fmt(r.sin_comparacion)}.')
    print(f'Cortes evaluados: mes anterior {fmt(r.cortes_ingenuos)}; anual {fmt(r.cortes_anuales)}.')
    print(f'Fechas de evaluación con cero explícito como valor real: {fmt(r.cortes_con_cero_como_objetivo)}.')
    print(f'Fechas con cero explícito usado en alguna predicción: {fmt(r.cortes_con_cero_como_predictor)}.')
    print(f'Meses-producto desconocidos o pendientes no usados como valores: {fmt(r.meses_sin_valor)}.')
    print('No se han inventado ventas cero: una laguna sigue siendo desconocida.')
    print('Los resultados evalúan cantidades observadas PROVISIONALES, no demanda real ni ventas netas.')
    print('Solo se comparan MAE y RMSE calculados sobre las mismas fechas del producto.')
    print('Los métodos complejos siguen pendientes de entrenamiento y validación.')
