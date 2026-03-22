"""
kpis.py
=======
Módulo de cálculo de KPIs bancarios — datos CMF Chile
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# BANCOS OBJETIVO
# ─────────────────────────────────────────────────────────────────────────────
BANCOS = {
    "012": "BancoEstado",
    "001": "Banco de Chile",
    "037": "Santander",
    "016": "BCI",
}
CODIGO_SISTEMA = "999"
NOMBRE_SISTEMA = "Sistema"
BANCOS_CON_SISTEMA = {**BANCOS, CODIGO_SISTEMA: NOMBRE_SISTEMA}

COLORES_BANCO = {
    "BancoEstado":    "#F5A623",
    "Banco de Chile": "#1A5FA8",
    "Santander":      "#CC0000",
    "BCI":            "#2E7D32",
    "Sistema":        "#888888",
}

# ─────────────────────────────────────────────────────────────────────────────
# CUENTAS CLAVE (verificadas contra datos reales)
# ─────────────────────────────────────────────────────────────────────────────
# B1 — Balance
CTA_TOTAL_ACTIVOS      = "100000000"
CTA_COLOCACIONES       = "505000000"
CTA_COLOCACIONES_COMER = "145000000"
CTA_COLOCACIONES_VIVI  = "146000000"
CTA_COLOCACIONES_CONS  = "148000000"
CTA_TOTAL_PASIVOS      = "200000000"
CTA_DEPOSITOS          = "241000000"
CTA_PATRIMONIO         = "300000000"

# R1 — Resultados (flujo mensual — NO acumulado)
CTA_ING_INTERESES      = "411000000"
CTA_GTO_INTERESES      = "412000000"
CTA_MARGEN_INTERESES   = "520000000"
CTA_MARGEN_REAJUSTES   = "525000000"
CTA_COMISIONES_NETAS   = "530000000"
CTA_ROF                = "540000000"
CTA_GASTO_PROVISION    = "470000000"
CTA_GASTO_PERSONAL     = "462000000"
CTA_GASTO_ADMIN        = "464000000"
CTA_RESULTADO_EJERCICIO= "590000000"


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────
def cargar_datos(data_dir: Path, moneda: str = "total") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga B1 y R1. Convierte a MMM$.
    moneda: 'total' | 'clp' | 'uf' | 'mx'
    """
    output_dir = data_dir / "output"

    def _leer_b1() -> pd.DataFrame:
        ruta = output_dir / "B1_con_ifrs.csv"
        if not ruta.exists():
            ruta = data_dir / "consolidado" / "B1_historico.csv"
        if not ruta.exists():
            raise FileNotFoundError(f"No se encontró B1_con_ifrs.csv")

        df = pd.read_csv(ruta, dtype=str, low_memory=False)
        df["banco_codigo"] = (
            df["banco_codigo"].astype(str).str.strip()
            .apply(lambda x: str(int(x)).zfill(3) if x.isdigit() else x)
        )
        df = df[df["banco_codigo"].isin(BANCOS_CON_SISTEMA.keys())].copy()
        df["banco_nombre"] = df["banco_codigo"].map(BANCOS_CON_SISTEMA)
        df["cuenta"] = df["cuenta"].astype(str).str.strip()
        df["periodo"] = pd.to_datetime(df["periodo"], format="%Y-%m")

        # Seleccionar columna según moneda
        col_map = {
            "total": "saldo_total",
            "clp":   "saldo_clp",
            "uf":    "saldo_uf",
            "mx":    "saldo_mx",
        }
        col_usar = col_map.get(moneda, "saldo_total")

        # Si no existe la columna (archivo viejo), intentar fallback
        if col_usar not in df.columns:
            # Intentar calcular desde columnas disponibles
            for c in ["saldo_clp", "saldo_uf", "saldo_mx1", "saldo_mx2", "saldo_mes_actual"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(
                        df[c].astype(str).str.replace(",", ".", regex=False),
                        errors="coerce"
                    ).fillna(0)

            if "saldo_mx1" in df.columns and "saldo_mx2" in df.columns:
                df["saldo_mx"] = df["saldo_mx1"] + df["saldo_mx2"]

            cols_sum = [c for c in ["saldo_clp","saldo_uf","saldo_mx"] if c in df.columns]
            if cols_sum:
                df["saldo_total"] = df[cols_sum].sum(axis=1)
            elif "saldo_mes_actual" in df.columns:
                df["saldo_total"] = pd.to_numeric(df["saldo_mes_actual"], errors="coerce").fillna(0)

            col_usar = col_map.get(moneda, "saldo_total")
            if col_usar not in df.columns:
                col_usar = "saldo_total"

        df[col_usar] = pd.to_numeric(df[col_usar], errors="coerce").fillna(0)
        df[col_usar] = df[col_usar] / 1_000_000_000  # → MMM$
        df = df.rename(columns={col_usar: "saldo_mes_actual"})
        return df

    def _leer_r1() -> pd.DataFrame:
        ruta = output_dir / "R1_con_ifrs.csv"
        if not ruta.exists():
            ruta = data_dir / "consolidado" / "R1_historico.csv"
        if not ruta.exists():
            raise FileNotFoundError(f"No se encontró R1_con_ifrs.csv")

        df = pd.read_csv(ruta, dtype=str, low_memory=False)
        df["banco_codigo"] = (
            df["banco_codigo"].astype(str).str.strip()
            .apply(lambda x: str(int(x)).zfill(3) if x.isdigit() else x)
        )
        df = df[df["banco_codigo"].isin(BANCOS_CON_SISTEMA.keys())].copy()
        df["banco_nombre"] = df["banco_codigo"].map(BANCOS_CON_SISTEMA)
        df["cuenta"] = df["cuenta"].astype(str).str.strip()
        df["periodo"] = pd.to_datetime(df["periodo"], format="%Y-%m")

        col = "flujo_mes_actual" if "flujo_mes_actual" in df.columns else "col_1"
        df[col] = (
            df[col].astype(str).str.replace(",", ".", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df[col] = df[col] / 1_000_000_000  # → MMM$
        if col != "flujo_mes_actual":
            df = df.rename(columns={col: "flujo_mes_actual"})
        return df

    df_b1 = _leer_b1()
    df_r1_raw = _leer_r1()
    df_r1 = _desacumular_r1(df_r1_raw)

    return df_b1, df_r1


def _desacumular_r1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte flujos acumulados del año a flujos mensuales.
    En enero (mes=1) el valor ya es mensual.
    Para los demás meses: flujo_mes = acumulado_mes - acumulado_mes_anterior.
    """
    df = df.sort_values(["banco_codigo", "cuenta", "periodo"]).copy()

    def _desacum_grupo(g):
        g = g.sort_values("periodo").copy()
        meses = g["periodo"].dt.month
        # Calcular diferencia
        diff = g["flujo_mes_actual"].diff()
        # En enero (mes=1) mantener el valor original
        es_enero = meses == 1
        g["flujo_mes_actual"] = np.where(es_enero, g["flujo_mes_actual"], diff)
        return g

    df = df.groupby(["banco_codigo", "cuenta"], group_keys=False).apply(_desacum_grupo)
    return df.dropna(subset=["flujo_mes_actual"])


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _saldo(df: pd.DataFrame, cuenta: str, col: str = "saldo_mes_actual") -> pd.DataFrame:
    return (
        df[df["cuenta"] == cuenta][["periodo", "banco_nombre", col]]
        .rename(columns={col: "valor"}).copy()
    )

def _flujo(df: pd.DataFrame, cuenta: str, col: str = "flujo_mes_actual") -> pd.DataFrame:
    return (
        df[df["cuenta"] == cuenta][["periodo", "banco_nombre", col]]
        .rename(columns={col: "valor"}).copy()
    )

def _pivot(df: pd.DataFrame, col_val: str = "valor") -> pd.DataFrame:
    return df.pivot_table(
        index="periodo", columns="banco_nombre", values=col_val, aggfunc="sum"
    )

def _solo_bancos(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["banco_nombre"] != NOMBRE_SISTEMA]

def _solo_sistema(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["banco_nombre"] == NOMBRE_SISTEMA]


# ─────────────────────────────────────────────────────────────────────────────
# KPI 1: COLOCACIONES
# ─────────────────────────────────────────────────────────────────────────────
def colocaciones_totales(df_b1: pd.DataFrame, incluir_sistema: bool = True) -> pd.DataFrame:
    df = df_b1 if incluir_sistema else _solo_bancos(df_b1)
    return _pivot(_saldo(df, CTA_COLOCACIONES))

def colocaciones_por_segmento(df_b1: pd.DataFrame, periodo: pd.Timestamp) -> pd.DataFrame:
    segmentos = {
        "Comercial": CTA_COLOCACIONES_COMER,
        "Vivienda":  CTA_COLOCACIONES_VIVI,
        "Consumo":   CTA_COLOCACIONES_CONS,
    }
    df_b = _solo_bancos(df_b1)
    filas = []
    for seg, cta in segmentos.items():
        df_seg = _saldo(df_b, cta)
        for _, row in df_seg[df_seg["periodo"] == periodo].iterrows():
            filas.append({"tipo": seg, "banco": row["banco_nombre"], "valor": row["valor"]})
    return pd.DataFrame(filas)


# ─────────────────────────────────────────────────────────────────────────────
# KPI 2: BALANCE
# ─────────────────────────────────────────────────────────────────────────────
def estructura_balance(df_b1: pd.DataFrame, periodo: pd.Timestamp) -> pd.DataFrame:
    df_b = _solo_bancos(df_b1)
    cuentas = {
        "Total Activos": CTA_TOTAL_ACTIVOS,
        "Colocaciones":  CTA_COLOCACIONES,
        "Depósitos":     CTA_DEPOSITOS,
        "Patrimonio":    CTA_PATRIMONIO,
    }
    filas = []
    for nombre, cta in cuentas.items():
        sub = df_b[(df_b["cuenta"] == cta) & (df_b["periodo"] == periodo)][["banco_nombre", "saldo_mes_actual"]].copy()
        sub = sub.rename(columns={"saldo_mes_actual": "valor"})
        sub["partida"] = nombre
        filas.append(sub)
    return pd.concat(filas, ignore_index=True) if filas else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# KPI 3: INGRESOS
# ─────────────────────────────────────────────────────────────────────────────
def composicion_ingresos(df_r1: pd.DataFrame, periodo: pd.Timestamp) -> pd.DataFrame:
    df_b = _solo_bancos(df_r1)
    componentes = {
        "Margen Intereses": CTA_MARGEN_INTERESES,
        "Margen Reajustes": CTA_MARGEN_REAJUSTES,
        "Comisiones Netas": CTA_COMISIONES_NETAS,
        "ROF":              CTA_ROF,
    }
    filas = []
    for nombre, cta in componentes.items():
        sub = df_b[(df_b["cuenta"] == cta) & (df_b["periodo"] == periodo)][["banco_nombre", "flujo_mes_actual"]].copy()
        sub = sub.rename(columns={"flujo_mes_actual": "valor"})
        sub["componente"] = nombre
        filas.append(sub)
    return pd.concat(filas, ignore_index=True) if filas else pd.DataFrame()

def evolucion_ingresos(df_r1: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df_b = _solo_bancos(df_r1)
    componentes = {
        "Margen Intereses": CTA_MARGEN_INTERESES,
        "Margen Reajustes": CTA_MARGEN_REAJUSTES,
        "Comisiones Netas": CTA_COMISIONES_NETAS,
        "ROF":              CTA_ROF,
    }
    return {nombre: _pivot(_flujo(df_b, cta)) for nombre, cta in componentes.items()}


# ─────────────────────────────────────────────────────────────────────────────
# KPI 4: RENTABILIDAD
# ─────────────────────────────────────────────────────────────────────────────
def calcular_nim(df_b1: pd.DataFrame, df_r1: pd.DataFrame) -> pd.DataFrame:
    """NIM = Margen Intereses mensual × 12 / Activos. En %."""
    df_b = _solo_bancos(df_b1)
    df_r = _solo_bancos(df_r1)
    margen  = _pivot(_flujo(df_r, CTA_MARGEN_INTERESES))
    activos = _pivot(_saldo(df_b, CTA_TOTAL_ACTIVOS))
    return (margen * 12 / activos * 100).round(2)

def calcular_roa(df_b1: pd.DataFrame, df_r1: pd.DataFrame) -> pd.DataFrame:
    """ROA = Resultado mensual × 12 / Activos. En %."""
    df_b = _solo_bancos(df_b1)
    df_r = _solo_bancos(df_r1)
    resultado = _pivot(_flujo(df_r, CTA_RESULTADO_EJERCICIO))
    activos   = _pivot(_saldo(df_b, CTA_TOTAL_ACTIVOS))
    return (resultado * 12 / activos * 100).round(2)

def calcular_roe(df_b1: pd.DataFrame, df_r1: pd.DataFrame) -> pd.DataFrame:
    """ROE = Resultado mensual × 12 / Patrimonio. En %."""
    df_b = _solo_bancos(df_b1)
    df_r = _solo_bancos(df_r1)
    resultado  = _pivot(_flujo(df_r, CTA_RESULTADO_EJERCICIO))
    patrimonio = _pivot(_saldo(df_b, CTA_PATRIMONIO))
    return (resultado * 12 / patrimonio * 100).round(2)

def calcular_eficiencia(df_r1: pd.DataFrame) -> pd.DataFrame:
    """Eficiencia = Gastos operacionales / Ingreso bruto. En %. Menor = mejor."""
    df_b = _solo_bancos(df_r1)
    personal = _pivot(_flujo(df_b, CTA_GASTO_PERSONAL))
    admin    = _pivot(_flujo(df_b, CTA_GASTO_ADMIN))
    gastos   = personal.add(admin, fill_value=0).abs()

    margen = _pivot(_flujo(df_b, CTA_MARGEN_INTERESES))
    reaj   = _pivot(_flujo(df_b, CTA_MARGEN_REAJUSTES))
    comis  = _pivot(_flujo(df_b, CTA_COMISIONES_NETAS))
    rof    = _pivot(_flujo(df_b, CTA_ROF))
    ingreso = (margen.add(reaj, fill_value=0)
                     .add(comis, fill_value=0)
                     .add(rof,   fill_value=0)).abs()

    return (gastos / ingreso * 100).round(2)


# ─────────────────────────────────────────────────────────────────────────────
# KPI 5: TABLA DE RESULTADOS (estilo presentación — con Δ 12m)
# ─────────────────────────────────────────────────────────────────────────────
def tabla_resultados(df_r1: pd.DataFrame, periodo: pd.Timestamp) -> pd.DataFrame:
    periodo_12m = periodo - pd.DateOffset(months=12)
    bancos_orden = list(BANCOS.values()) + [NOMBRE_SISTEMA]

    lineas = [
        # (nombre, cuentas, nivel, es_total)
        ("Margen Intereses",       CTA_MARGEN_INTERESES,    2, False),
        ("Margen Reajustes",       CTA_MARGEN_REAJUSTES,    2, False),
        ("Comisiones Netas",       CTA_COMISIONES_NETAS,    2, False),
        ("ROF",                    CTA_ROF,                 2, False),
        ("Total Ing. Operacional", None,                    0, True),
        ("Gasto Provisiones",      CTA_GASTO_PROVISION,     2, False),
        ("Gastos Personal",        CTA_GASTO_PERSONAL,      2, False),
        ("Gastos Admin",           CTA_GASTO_ADMIN,         2, False),
        ("Resultado Ejercicio",    CTA_RESULTADO_EJERCICIO, 0, True),
    ]

    CTAS_ING = [CTA_MARGEN_INTERESES, CTA_MARGEN_REAJUSTES,
                CTA_COMISIONES_NETAS, CTA_ROF]

    resultados = []
    for nombre_linea, cta, nivel, es_total in lineas:
        fila = {"Línea": nombre_linea, "nivel": nivel, "es_total": es_total}
        for banco in bancos_orden:
            def _v(p, cuentas):
                if isinstance(cuentas, list):
                    return sum(
                        df_r1[(df_r1["cuenta"] == c) &
                              (df_r1["banco_nombre"] == banco) &
                              (df_r1["periodo"] == p)]["flujo_mes_actual"].sum()
                        for c in cuentas
                    )
                sub = df_r1[(df_r1["cuenta"] == cuentas) &
                            (df_r1["banco_nombre"] == banco) &
                            (df_r1["periodo"] == p)]["flujo_mes_actual"]
                return sub.sum() if not sub.empty else np.nan

            cuentas_usar = CTAS_ING if cta is None else cta
            val  = _v(periodo,    cuentas_usar)
            v12m = _v(periodo_12m, cuentas_usar)

            fila[f"{banco}_val"]  = round(val,  1) if not np.isnan(val)  else np.nan
            fila[f"{banco}_d12m"] = round(val - v12m, 1) if (
                not np.isnan(val) and not np.isnan(v12m)
            ) else np.nan
        resultados.append(fila)

    return pd.DataFrame(resultados)


ORDEN_BANCOS = ["BancoEstado", "Banco de Chile", "Santander", "BCI", "Sistema"]

# ─────────────────────────────────────────────────────────────────────────────
# KPI 6: VARIACIONES
# ─────────────────────────────────────────────────────────────────────────────
def tabla_variaciones(df_b1: pd.DataFrame, df_r1: pd.DataFrame,
                      periodo: pd.Timestamp) -> pd.DataFrame:
    metricas = {
        "Colocaciones (MMM$)":     _pivot(_saldo(df_b1, CTA_COLOCACIONES)),
        "Total Activos (MMM$)":    _pivot(_saldo(df_b1, CTA_TOTAL_ACTIVOS)),
        "Depósitos Vista (MMM$)":  _pivot(_saldo(df_b1, CTA_DEPOSITOS)),
        "Margen Intereses (MMM$)": _pivot(_flujo(df_r1, CTA_MARGEN_INTERESES)),  # incluye sistema
    }

    filas = []
    for nombre_kpi, tabla in metricas.items():
        if periodo not in tabla.index:
            continue
        for banco in tabla.columns:
            serie = tabla[banco].dropna()
            if periodo not in serie.index:
                continue
            valor_actual = serie.loc[periodo]
            idx = serie.index.get_loc(periodo)
            valor_mom = serie.iloc[idx - 1] if idx > 0 else np.nan
            var_mom_abs = valor_actual - valor_mom if not np.isnan(valor_mom) else np.nan
            var_mom_pct = (var_mom_abs / abs(valor_mom) * 100) if (
                valor_mom and not np.isnan(valor_mom)
            ) else np.nan
            valor_yoy = serie.iloc[idx - 12] if idx >= 12 else np.nan
            var_yoy_pct = ((valor_actual - valor_yoy) / abs(valor_yoy) * 100) if (
                not np.isnan(valor_yoy)
            ) else np.nan

            filas.append({
                "KPI":            nombre_kpi,
                "Banco":          banco,
                "Valor":          round(valor_actual, 0),
                "Var MoM (MMM$)": round(var_mom_abs, 0) if not np.isnan(var_mom_abs) else None,
                "Var MoM (%)":    round(var_mom_pct, 1) if not np.isnan(var_mom_pct) else None,
                "Var YoY (%)":    round(var_yoy_pct, 1) if not np.isnan(var_yoy_pct) else None,
            })
    return pd.DataFrame(filas)


# ─────────────────────────────────────────────────────────────────────────────
# KPI 7: RANKING
# ─────────────────────────────────────────────────────────────────────────────
def ranking_kpis(df_b1: pd.DataFrame, df_r1: pd.DataFrame,
                 periodo: pd.Timestamp) -> pd.DataFrame:
    nim = calcular_nim(df_b1, df_r1)
    roa = calcular_roa(df_b1, df_r1)
    roe = calcular_roe(df_b1, df_r1)
    ef  = calcular_eficiencia(df_r1)
    col = _pivot(_saldo(_solo_bancos(df_b1), CTA_COLOCACIONES))

    filas = []
    for tabla, nombre in [(nim, "NIM (%)"), (roa, "ROA (%)"), (roe, "ROE (%)"),
                          (ef, "Eficiencia (%)"), (col, "Colocaciones (MMM$)")]:
        if periodo not in tabla.index:
            continue
        for banco, val in tabla.loc[periodo].items():
            filas.append({"Banco": banco, "KPI": nombre, "Valor": val})

    if not filas:
        return pd.DataFrame()
    return pd.DataFrame(filas).pivot_table(
        index="Banco", columns="KPI", values="Valor", aggfunc="first"
    ).round(2)


# ─────────────────────────────────────────────────────────────────────────────
# TABLA BALANCE ACTIVOS (estilo presentación interna)
# ─────────────────────────────────────────────────────────────────────────────

# Estructura jerárquica de activos con códigos verificados
ESTRUCTURA_ACTIVOS = [
    # (nombre, cuentas, nivel, es_total)
    # cuentas puede ser: str (una cuenta), list (suma), None (calculado)
    ("Activos Totales",       "100000000",                          0, True),
    ("Créditos",              "144000000",                          1, True),
    ("  Comerciales",         "145000000",                          2, False),
    ("  Hipotecarios",        "146000000",                          2, False),
    ("  Consumo",             "148000000",                          2, False),
    ("  Provisiones",         "149000000",                          2, False),
    ("Activos Financieros",   ["110000000","120000000","130000000",
                               "140000000"],                        1, True),
    ("  Cartera Inversiones", "120000000",                          2, False),
    ("  Pactos Retroventa",   "141000000",                          2, False),
    ("  Derivados",           "110000000",                          2, False),
    ("  Coberturas",          "130000000",                          2, False),
    ("Otros Activos",         "190000000",                          1, True),
]

ESTRUCTURA_PASIVOS = [
    ("Pasivos Totales",       "200000000",                          0, True),
    ("Depósitos",             ["241000000","242000000"],            1, True),
    ("  Vista",               "241000000",                          2, False),
    ("  Plazo",               "242000000",                          2, False),
    ("Fondeo Mercados",       ["243000000","244000000","245000000"],1, True),
    ("  Pactos Retrocompra",  "243000000",                          2, False),
    ("  Bonos y Letras",      "245000000",                          2, False),
    ("  Oblig. con Bancos",   "244000000",                          2, False),
    ("Pasivos Permanentes",   ["260000000","290000000"],            1, True),
    ("  Patrimonio",          "300000000",                          2, False),
    ("  Provisiones",         "260000000",                          2, False),
    ("  Otros Pasivos",       "290000000",                          2, False),
]


def _sum_cuentas(df_b1, cuentas, banco, periodo):
    """Suma saldos de una o varias cuentas para un banco y período."""
    if isinstance(cuentas, str):
        cuentas = [cuentas]
    total = 0
    for c in cuentas:
        sub = df_b1[
            (df_b1["cuenta"] == c) &
            (df_b1["banco_nombre"] == banco) &
            (df_b1["periodo"] == periodo)
        ]["saldo_mes_actual"]
        total += sub.sum() if not sub.empty else 0
    return total


def _sum_cuentas_ant(df_b1, cuentas, banco, periodo):
    """Suma saldos del período anterior."""
    periodo_ant = periodo - pd.DateOffset(months=1)
    return _sum_cuentas(df_b1, cuentas, banco, periodo_ant)


def tabla_balance_activos(df_b1: pd.DataFrame, periodo: pd.Timestamp,
                          periodo_ref: pd.Timestamp = None) -> pd.DataFrame:
    """Tabla de activos con saldos + variación respecto a periodo_ref."""
    if periodo_ref is None:
        periodo_ref = periodo - pd.DateOffset(months=1)
    bancos = list(BANCOS.values()) + [NOMBRE_SISTEMA]
    filas = []

    for nombre, cuentas, nivel, es_total in ESTRUCTURA_ACTIVOS:
        fila = {"Línea": nombre, "nivel": nivel, "es_total": es_total}
        for banco in bancos:
            val     = _sum_cuentas(df_b1, cuentas, banco, periodo)
            val_ref = _sum_cuentas(df_b1, cuentas, banco, periodo_ref)
            fila[f"{banco}_saldo"] = round(val, 0)
            fila[f"{banco}_mtd"]   = round(val - val_ref, 0)
        filas.append(fila)

    return pd.DataFrame(filas)


def tabla_balance_pasivos(df_b1: pd.DataFrame, periodo: pd.Timestamp,
                          periodo_ref: pd.Timestamp = None) -> pd.DataFrame:
    """Tabla de pasivos con saldos + variación respecto a periodo_ref."""
    if periodo_ref is None:
        periodo_ref = periodo - pd.DateOffset(months=1)
    bancos = list(BANCOS.values()) + [NOMBRE_SISTEMA]
    filas = []

    for nombre, cuentas, nivel, es_total in ESTRUCTURA_PASIVOS:
        fila = {"Línea": nombre, "nivel": nivel, "es_total": es_total}
        for banco in bancos:
            val     = _sum_cuentas(df_b1, cuentas, banco, periodo)
            val_ref = _sum_cuentas(df_b1, cuentas, banco, periodo_ref)
            fila[f"{banco}_saldo"] = round(val, 0)
            fila[f"{banco}_mtd"]   = round(val - val_ref, 0)
        filas.append(fila)

    return pd.DataFrame(filas)


# ─────────────────────────────────────────────────────────────────────────────
# TABLA DE RESULTADOS (estilo presentación — con Δ 12m)
# ─────────────────────────────────────────────────────────────────────────────

ESTRUCTURA_RESULTADOS = [
    # (nombre, cuentas, nivel, es_total)
    ("Ingreso Neto Int. y Reajustes", ["520000000","525000000"], 1, True),
    ("  Margen de Intereses",         "520000000",              2, False),
    ("  Margen de Reajustes",         "525000000",              2, False),
    ("Ingreso Neto por Comisiones",   "530000000",              1, False),
    ("Utilidad Neta Op. Financieras", "540000000",              1, True),
    ("  Cartera de Inversiones",      ["431000000","431800000"],2, False),
    ("  Cambios y Derivados",         "433000000",              2, False),
    ("Otros Ingresos Operacionales",  ["455000000","440000000"],1, False),
    ("Total Ingresos Operacionales",  "570000000",              0, True),
    ("Provisiones Riesgo Crédito",    "470000000",              1, False),
    ("Gastos Personal",               "462000000",              2, False),
    ("Gastos Administración",         "464000000",              2, False),
    ("Depreciación y Amortización",   "466000000",              2, False),
    ("Otros Gastos Operacionales",    "469000000",              2, False),
    ("Gastos Operacionales",         ["462000000","464000000",
                                       "466000000","469000000"],1, True),
    ("RAI",                           "585000000",              0, True),
]


def _flujo_r1(df_r1, cuentas, banco, periodo):
    """Suma flujos de una o varias cuentas R1 para un banco y período."""
    if isinstance(cuentas, str):
        cuentas = [cuentas]
    total = 0
    for c in cuentas:
        sub = df_r1[
            (df_r1["cuenta"] == c) &
            (df_r1["banco_nombre"] == banco) &
            (df_r1["periodo"] == periodo)
        ]["flujo_mes_actual"]
        total += sub.sum() if not sub.empty else 0
    return total


def tabla_resultados_full(df_r1: pd.DataFrame, periodo: pd.Timestamp,
                          periodo_ref: pd.Timestamp = None) -> pd.DataFrame:
    """Tabla de resultados con flujo mensual + variación respecto a periodo_ref."""
    if periodo_ref is None:
        periodo_ref = periodo - pd.DateOffset(months=12)
    bancos = list(BANCOS.values()) + [NOMBRE_SISTEMA]
    filas = []

    for nombre, cuentas, nivel, es_total in ESTRUCTURA_RESULTADOS:
        fila = {"Línea": nombre, "nivel": nivel, "es_total": es_total}
        for banco in bancos:
            val     = _flujo_r1(df_r1, cuentas, banco, periodo)
            val_ref = _flujo_r1(df_r1, cuentas, banco, periodo_ref)
            fila[f"{banco}_val"]  = round(val, 0)
            fila[f"{banco}_d12m"] = round(val - val_ref, 0) if val_ref != 0 else np.nan
        filas.append(fila)

    return pd.DataFrame(filas)


# ─────────────────────────────────────────────────────────────────────────────
# TABLA ROF — Apertura detallada
# ─────────────────────────────────────────────────────────────────────────────
ESTRUCTURA_ROF = [
    # (nombre, cuentas_r1, nivel, es_total)
    # Inst. Negociación = neto derivados activos + pasivos + instrumentos deuda intereses/reajustes
    ("Inst. Negociación",        ["431150100","431650100",
                                   "431180105","431180106"], 1, True),
    ("  Derivados (neto)",       ["431150100","431650100"], 2, False),
    ("  Int. y Reaj. Inst.",     ["431180105","431180106"], 2, False),
    # Ajuste VR = valorización instrumentos deuda (activos) neto
    ("Ajuste Valor Razonable",   ["431180101","431180102"], 1, True),
    ("  Utilidad Valorización",   "431180101",  2, False),
    ("  Pérdida Valorización",    "431180102",  2, False),
    # Utilidad neta venta instrumentos deuda
    ("Utilidad Neta por Venta",  ["431180103","431180104"], 1, True),
    ("  Utilidad por Venta",      "431180103",  2, False),
    ("  Pérdida por Venta",       "431180104",  2, False),
    # Otros inst. financieros (fondos mutuos + patrimonio)
    ("Otros Inst. Financieros",   "431250000",  1, True),
    # Utilidad venta DPV
    ("Utilidad Venta DPV",        "432400000",  1, True),
    # ROF Total = cuenta resumen CMF
    ("ROF Total",                 "540000000",  0, True),
]

def tabla_rof(df_r1: pd.DataFrame, periodo: pd.Timestamp,
              periodo_ref: pd.Timestamp = None) -> pd.DataFrame:
    """Tabla ROF con apertura detallada por tipo de resultado."""
    if periodo_ref is None:
        periodo_ref = periodo - pd.DateOffset(months=12)
    bancos = list(BANCOS.values()) + [NOMBRE_SISTEMA]
    filas = []

    for nombre, cuentas, nivel, es_total in ESTRUCTURA_ROF:
        fila = {"Línea": nombre, "nivel": nivel, "es_total": es_total}
        for banco in bancos:
            val     = _flujo_r1(df_r1, cuentas, banco, periodo)
            val_ref = _flujo_r1(df_r1, cuentas, banco, periodo_ref)
            fila[f"{banco}_val"]  = round(val, 0)
            fila[f"{banco}_d12m"] = round(val - val_ref, 0) if val_ref != 0 else np.nan
        filas.append(fila)

    return pd.DataFrame(filas)


# ─────────────────────────────────────────────────────────────────────────────
# CARTERA DE INVERSIONES — evolución por tipo (g3: Negociación, DPV, VCTO)
# ─────────────────────────────────────────────────────────────────────────────
CUENTAS_NEGOCIACION = [
    "112000101","112000102","112000109","112000201","112000202","112000209",
    "112000301","112000302","112000303","112000304","112000309",
]
CUENTAS_DPV = [
    "122000101","122000102","122000109","122000201","122000202","122000209",
    "122000301","122000302","122000303","122000304","122000309",
    "123000301","123000302","123000303","123000304","123000400",
]
CUENTAS_VCTO = [
    "141500101","141500102","141500109","141500201","141500202","141500209",
    "141500301","141500302","141500303","141500304","141500309",
]

def evolucion_cartera_inversiones(df_b1: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Evolución de cartera de inversiones por tipo (Negociación, DPV, VCTO)."""
    df_b = _solo_bancos(df_b1)
    tipos = {
        "Negociación": CUENTAS_NEGOCIACION,
        "DPV":         CUENTAS_DPV,
        "VCTO":        CUENTAS_VCTO,
    }
    resultado = {}
    for nombre, cuentas in tipos.items():
        filas = []
        for periodo in sorted(df_b["periodo"].unique()):
            for banco in BANCOS.values():
                val = _sum_cuentas(df_b, cuentas, banco, periodo)
                filas.append({"periodo": periodo, "banco_nombre": banco, "valor": val})
        if filas:
            df_tipo = pd.DataFrame(filas)
            resultado[nombre] = df_tipo.pivot_table(
                index="periodo", columns="banco_nombre", values="valor", aggfunc="sum"
            )
    return resultado
