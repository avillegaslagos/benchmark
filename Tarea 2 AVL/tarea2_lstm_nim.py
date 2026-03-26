"""
=============================================================================
TAREA 2 — DIPLOMADO EN CIENCIA DE DATOS PARA LAS FINANZAS
=============================================================================
Modelo:   Red Neuronal Recurrente (LSTM) — Versión Multivariada
Problema: Predicción del NIM (Net Interest Margin) mensual por banco
Datos:    CMF Chile (R1, B1) + Banco Central de Chile (TPM, IPC)
Autor:    [Angélica Villegas]
=============================================================================

DESCRIPCIÓN
-----------
El NIM (Margen Neto de Intereses) mide la eficiencia con que un banco genera
ingresos a partir de sus activos productivos:

    NIM = (Ingresos por Intereses − Gastos por Intereses) / Activos Totales

Esta versión mejora el modelo univariado incorporando variables exógenas que
explican el comportamiento del NIM:

  • TPM  (Tasa de Política Monetaria, BCCh) — principal driver del NIM
  • IPC  (Inflación mensual, INE via BCCh)  — afecta márgenes en UF
  • Spread Tasas (activa − pasiva del sistema, CMF) — proxy directo del margen

ESTRUCTURA
----------
1.  Carga y cálculo del NIM desde archivos CMF
2.  Descarga automática de TPM e IPC desde mindicador.cl (sin credenciales)
3.  Cálculo del spread de tasas desde R1 del sistema (código 999)
4.  Consolidación y alineación temporal de todas las variables
5.  Preprocesamiento: normalización, ventana deslizante multivariada
6.  Arquitectura LSTM multivariada
7.  Entrenamiento con EarlyStopping
8.  Evaluación: MAE, RMSE, MAPE
9.  Visualización de resultados y variables exógenas
10. Predicción del próximo período
"""

# =============================================================================
# 1. IMPORTACIONES
# =============================================================================
import os
import warnings
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    print("✓ TensorFlow/Keras cargado correctamente")
except ImportError:
    raise ImportError(
        "\n[ERROR] TensorFlow no instalado.\n"
        "Ejecuta: pip install tensorflow\n"
    )

# =============================================================================
# 2. CONFIGURACIÓN
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "output")
R1_PATH  = os.path.join(DATA_DIR, "R1_con_ifrs.csv")
B1_PATH  = os.path.join(DATA_DIR, "B1_con_ifrs.csv")

BANCOS = {
    "012": "BancoEstado",
    "001": "Banco de Chile",
    "037": "Santander",
    "016": "BCI",
}

CTA_ING_INTERESES = "411000000"
CTA_GTO_INTERESES = "412000000"
CTA_TOTAL_ACTIVOS = "100000000"
CODIGO_SISTEMA    = "999"

VENTANA_TEMPORAL = 6
EPOCAS           = 150
BATCH_SIZE       = 16
UNIDADES_LSTM    = 64
DROPOUT_RATE     = 0.2
TEST_SIZE        = 0.2
SEMILLA          = 42
np.random.seed(SEMILLA)

# =============================================================================
# 3. CARGA DE DATOS CMF
# =============================================================================

def detectar_col(df, candidatas):
    return next(c for c in candidatas if c in df.columns)


def desacumular(df_grupo, col):
    """
    R1 reporta flujos ACUMULADOS desde enero de cada año.
    Convierte a flujo mensual puro usando diff() dentro de cada año.
    Estrategia: groupby año + diff, luego parchamos enero con el valor original.
    """
    df_grupo = df_grupo.copy().sort_values("periodo").reset_index(drop=True)
    df_grupo["_anio"] = df_grupo["periodo"].str[:4]
    # diff dentro de cada año respeta el índice original → alineación correcta
    df_grupo["_flujo_m"] = df_grupo.groupby("_anio")[col].diff()
    # enero (primer mes de cada año) queda NaN tras el diff → usar valor directo
    mask_enero = df_grupo.groupby("_anio")[col].transform("first") == df_grupo[col]
    # más robusto: NaN después del diff = primer mes del año
    mask_nan = df_grupo["_flujo_m"].isna()
    df_grupo.loc[mask_nan, "_flujo_m"] = df_grupo.loc[mask_nan, col]
    return df_grupo["_flujo_m"].reset_index(drop=True)


def cargar_nim_bancos():
    """
    Calcula el NIM mensual PURO para los 4 bancos desde R1 y B1.
    Desacumula los flujos de R1 para eliminar el patron de diente de sierra
    causado por los acumulados anuales de la CMF.
    """
    r1 = pd.read_csv(R1_PATH, dtype=str)
    r1 = r1[r1["banco_codigo"].isin(BANCOS.keys())]
    r1 = r1[r1["cuenta"].isin([CTA_ING_INTERESES, CTA_GTO_INTERESES])]

    col = detectar_col(r1, ["flujo_mes_actual", "col_1", "saldo_mes_actual"])
    r1["saldo"] = pd.to_numeric(r1[col], errors="coerce") / 1e9

    # Desacumular por banco y por cuenta antes del pivot
    partes = []
    for (banco, cuenta), grupo in r1.groupby(["banco_codigo", "cuenta"]):
        grupo = grupo.sort_values("periodo").reset_index(drop=True)
        grupo["saldo_mensual"] = desacumular(grupo, "saldo")
        partes.append(grupo[["periodo", "banco_codigo", "cuenta", "saldo_mensual"]])
    r1_desac = pd.concat(partes, ignore_index=True)

    r1p = r1_desac.pivot_table(
        index=["periodo", "banco_codigo"], columns="cuenta",
        values="saldo_mensual", aggfunc="sum"
    ).reset_index()
    r1p.columns.name = None
    r1p = r1p.rename(columns={CTA_ING_INTERESES: "ing", CTA_GTO_INTERESES: "gto"})
    for c in ["ing", "gto"]:
        if c not in r1p.columns:
            r1p[c] = np.nan

    b1 = pd.read_csv(B1_PATH, dtype=str)
    b1 = b1[b1["banco_codigo"].isin(BANCOS.keys())]
    b1 = b1[b1["cuenta"] == CTA_TOTAL_ACTIVOS]
    col_b = detectar_col(b1, ["saldo_total", "saldo_mes_actual", "col_1", "flujo_mes_actual"])
    b1["activos"] = pd.to_numeric(b1[col_b], errors="coerce") / 1e9

    df = r1p.merge(b1[["periodo", "banco_codigo", "activos"]], on=["periodo", "banco_codigo"], how="inner")
    # NIM mensual anualizado: flujo_mensual / activos * 12 * 100
    # Multiplicar x12 normaliza la estacionalidad del flujo acumulado CMF
    df["nim_pct"] = ((df["ing"] - df["gto"]) / df["activos"]) * 12 * 100
    df["banco_nombre"] = df["banco_codigo"].map(BANCOS)
    df = df.sort_values(["banco_codigo", "periodo"]).dropna(subset=["nim_pct"])
    df = df[df["nim_pct"].between(0.1, 15)]  # filtrar solo outliers extremos
    # Diagnóstico: mostrar primeros períodos de un banco para verificar desacumulación
    muestra = df[df["banco_codigo"]=="001"][["periodo","nim_pct"]].head(6)
    print(f"   [DIAG] NIM Banco de Chile primeros meses:\n{muestra.to_string(index=False)}")
    return df


def cargar_spread_sistema():
    """Spread NIM del total sistema (código 999) como variable exógena."""
    r1 = pd.read_csv(R1_PATH, dtype=str)
    r1s = r1[r1["banco_codigo"] == CODIGO_SISTEMA].copy()
    r1s = r1s[r1s["cuenta"].isin([CTA_ING_INTERESES, CTA_GTO_INTERESES])]
    if r1s.empty:
        return pd.DataFrame(columns=["periodo", "spread_sistema"])

    col = detectar_col(r1s, ["flujo_mes_actual", "col_1", "saldo_mes_actual"])
    r1s["saldo"] = pd.to_numeric(r1s[col], errors="coerce") / 1e9

    piv = r1s.pivot_table(index="periodo", columns="cuenta",
                           values="saldo", aggfunc="sum").reset_index()
    piv.columns.name = None
    piv = piv.rename(columns={CTA_ING_INTERESES: "ing", CTA_GTO_INTERESES: "gto"})

    b1 = pd.read_csv(B1_PATH, dtype=str)
    b1s = b1[b1["banco_codigo"] == CODIGO_SISTEMA].copy()
    b1s = b1s[b1s["cuenta"] == CTA_TOTAL_ACTIVOS]
    col_b = detectar_col(b1s, ["saldo_total", "saldo_mes_actual", "col_1", "flujo_mes_actual"])
    b1s["activos"] = pd.to_numeric(b1s[col_b], errors="coerce") / 1e9

    merged = piv.merge(b1s[["periodo", "activos"]], on="periodo", how="inner")
    if "ing" in merged.columns and "gto" in merged.columns:
        # Desacumular tambien el spread del sistema
        merged = merged.sort_values("periodo").reset_index(drop=True)
        merged["ing_m"] = desacumular(merged, "ing")
        merged["gto_m"] = desacumular(merged, "gto")
        merged["spread_sistema"] = ((merged["ing_m"] - merged["gto_m"]) / merged["activos"]) * 12 * 100
    else:
        merged["spread_sistema"] = np.nan

    return merged[["periodo", "spread_sistema"]].dropna()


# =============================================================================
# 4. DESCARGA VARIABLES MACROECONÓMICAS (mindicador.cl — sin credenciales)
# =============================================================================

def descargar_mindicador(indicador, desde_año, hasta_año):
    """Descarga una serie anual desde mindicador.cl y retorna DataFrame mensual."""
    registros = []
    for año in range(desde_año, hasta_año + 1):
        url = f"https://mindicador.cl/api/{indicador}/{año}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                for obs in resp.json().get("serie", []):
                    registros.append({
                        "periodo": obs["fecha"][:7],
                        indicador: float(obs["valor"])
                    })
        except Exception:
            pass

    if not registros:
        print(f"   ⚠  No se pudo descargar '{indicador}'")
        return pd.DataFrame(columns=["periodo", indicador])

    df = (pd.DataFrame(registros)
            .groupby("periodo")[indicador].mean()
            .reset_index()
            .sort_values("periodo"))
    return df


def obtener_variables_macro(periodos):
    """Descarga TPM e IPC y los alinea con los períodos CMF."""
    año_min = int(min(periodos)[:4])
    año_max = int(max(periodos)[:4]) + 1

    print("   → Descargando TPM desde mindicador.cl ...")
    df_tpm = descargar_mindicador("tpm", año_min, año_max)

    print("   → Descargando IPC desde mindicador.cl ...")
    df_ipc = descargar_mindicador("ipc", año_min, año_max)

    macro = pd.DataFrame({"periodo": periodos})
    macro = macro.merge(df_tpm, on="periodo", how="left") if not df_tpm.empty else macro.assign(tpm=np.nan)
    macro = macro.merge(df_ipc, on="periodo", how="left") if not df_ipc.empty else macro.assign(ipc=np.nan)
    macro["tpm"] = macro["tpm"].interpolate().ffill().bfill()
    macro["ipc"] = macro["ipc"].interpolate().ffill().bfill()
    return macro


# =============================================================================
# 5. PREPARACIÓN SECUENCIAS MULTIVARIADAS
# =============================================================================

def crear_secuencias(data, target_idx, ventana):
    """Crea pares (X, y) para LSTM. data: (T, n_features)."""
    X, y = [], []
    for i in range(len(data) - ventana):
        X.append(data[i : i + ventana, :])
        y.append(data[i + ventana, target_idx])
    return np.array(X), np.array(y)


# =============================================================================
# 6. MODELO
# =============================================================================

def construir_modelo(ventana, n_features, unidades, dropout):
    """
    Arquitectura LSTM multivariada:
      - LSTM(64, return_sequences=True): captura patrones de corto plazo
      - Dropout(0.2): regularización
      - LSTM(32): abstracción de alto nivel
      - Dropout(0.2)
      - Dense(16, relu) + Dense(1): salida escalar (NIM del próximo mes)
    """
    modelo = Sequential([
        LSTM(unidades, return_sequences=True, input_shape=(ventana, n_features)),
        Dropout(dropout),
        LSTM(unidades // 2, return_sequences=False),
        Dropout(dropout),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    modelo.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return modelo


# =============================================================================
# 7. EVALUACIÓN
# =============================================================================

def evaluar(y_real, y_pred, nombre):
    mae  = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mape = np.mean(np.abs((y_real - y_pred) / (y_real + 1e-8))) * 100
    print(f"\n   📊 {nombre}")
    print(f"      MAE  = {mae:.4f}%")
    print(f"      RMSE = {rmse:.4f}%")
    print(f"      MAPE = {mape:.2f}%")
    return {"banco": nombre, "MAE": mae, "RMSE": rmse, "MAPE": mape}


# =============================================================================
# 8. VISUALIZACIONES
# =============================================================================

def graficar_predicciones(resultados):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    fig.suptitle("LSTM Multivariado — Predicción NIM (CMF Chile)",
                 fontsize=13, fontweight="bold")
    for i, r in enumerate(resultados):
        ax = axes[i]
        ax.plot(r["periodos"], r["nim_hist"], color="#90CAF9", linewidth=1.2,
                alpha=0.6, label="Histórico")
        ax.plot(r["periodos_test"], r["y_real"], color="#43A047", linewidth=2,
                marker="o", markersize=4, label="Real (test)")
        ax.plot(r["periodos_test"], r["y_pred"], color="#E53935", linewidth=2,
                linestyle="--", marker="s", markersize=4, label="Predicho")
        ax.scatter([r["prox_periodo"]], [r["prox_nim"]], color="purple",
                   zorder=6, s=90, label=f"Próx: {r['prox_nim']:.2f}%")
        ax.axvline(x=r["periodos_test"].iloc[0], color="gray",
                   linestyle=":", linewidth=1, alpha=0.6)
        ax.set_title(r["banco"], fontsize=11, fontweight="bold")
        ax.set_ylabel("NIM (%)")
        ax.tick_params(axis="x", rotation=40, labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    plt.tight_layout()
    path = os.path.join(BASE_DIR, "tarea2_predicciones.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n   ✓ Guardado: tarea2_predicciones.png")
    plt.show()


def graficar_exogenas(df_macro, df_spread):
    periodos = pd.to_datetime(df_macro["periodo"])
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Variables Exógenas del Modelo", fontsize=13, fontweight="bold")

    axes[0].plot(periodos, df_macro["tpm"], color="#1565C0", linewidth=2)
    axes[0].fill_between(periodos, df_macro["tpm"], alpha=0.15, color="#1565C0")
    axes[0].set_ylabel("TPM (%)"); axes[0].set_title("Tasa de Política Monetaria (BCCh)")
    axes[0].grid(True, alpha=0.25)

    axes[1].bar(periodos, df_macro["ipc"], color="#2E7D32", alpha=0.7, width=20)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Var. mensual (%)"); axes[1].set_title("IPC — Variación Mensual")
    axes[1].grid(True, alpha=0.25)

    if not df_spread.empty:
        per_sp = pd.to_datetime(df_spread["periodo"])
        axes[2].plot(per_sp, df_spread["spread_sistema"], color="#6A1B9A", linewidth=2)
        axes[2].fill_between(per_sp, df_spread["spread_sistema"], alpha=0.15, color="#6A1B9A")
        axes[2].set_title("Spread Sistema Financiero CMF (NIM proxy)")
    else:
        axes[2].text(0.5, 0.5, "No disponible", ha="center", va="center",
                     transform=axes[2].transAxes)
    axes[2].set_ylabel("Spread (%)"); axes[2].grid(True, alpha=0.25)

    plt.tight_layout()
    path = os.path.join(BASE_DIR, "tarea2_variables_exogenas.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"   ✓ Guardado: tarea2_variables_exogenas.png")
    plt.show()


def graficar_metricas(df_m):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Métricas — LSTM Multivariado", fontsize=12, fontweight="bold")
    cols = ["#1E88E5", "#43A047", "#FB8C00", "#8E24AA"]
    for j, met in enumerate(["MAE", "RMSE", "MAPE"]):
        ax = axes[j]
        bars = ax.bar(df_m["banco"], df_m[met], color=cols, width=0.5)
        ax.set_title(met); ax.set_ylabel("%")
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h * 1.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(BASE_DIR, "tarea2_metricas.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"   ✓ Guardado: tarea2_metricas.png")
    plt.show()


# =============================================================================
# 9. PIPELINE PRINCIPAL
# =============================================================================

def main():
    print("=" * 68)
    print("  TAREA 2 — LSTM Multivariado: Predicción del NIM Bancario")
    print("  Variables: NIM + TPM (BCCh) + IPC + Spread Sistema (CMF)")
    print("=" * 68)

    # Paso 1: CMF
    print("\n[1/5] Cargando datos CMF...")
    df_nim    = cargar_nim_bancos()
    df_spread = cargar_spread_sistema()
    periodos  = sorted(df_nim["periodo"].unique())
    print(f"   ✓ {len(periodos)} períodos | {df_nim['banco_codigo'].nunique()} bancos")
    print(f"   ✓ Rango: {periodos[0]} → {periodos[-1]}")
    print(f"   ✓ Spread sistema: {len(df_spread)} períodos")

    # Paso 2: Macro
    print("\n[2/5] Descargando variables macroeconómicas...")
    df_macro = obtener_variables_macro(periodos)
    print(f"   ✓ TPM: {df_macro['tpm'].notna().sum()}/{len(df_macro)} meses")
    print(f"   ✓ IPC: {df_macro['ipc'].notna().sum()}/{len(df_macro)} meses")
    graficar_exogenas(df_macro, df_spread)

    # Paso 3: Entrenamiento
    print("\n[3/5] Entrenando modelos LSTM por banco...")
    metricas   = []
    resultados = []
    FEATURES   = ["nim_pct", "tpm", "ipc", "spread_sistema"]
    TARGET_IDX = 0

    for codigo, nombre in BANCOS.items():
        print(f"\n   → {nombre} ({codigo})")
        db = (df_nim[df_nim["banco_codigo"] == codigo]
              .sort_values("periodo").reset_index(drop=True))
        if len(db) < VENTANA_TEMPORAL + 8:
            print("   ⚠  Datos insuficientes, se omite."); continue

        db = db.merge(df_macro, on="periodo", how="left")
        db = db.merge(df_spread, on="periodo", how="left") if not df_spread.empty else db.assign(spread_sistema=0.0)
        for f in ["tpm", "ipc", "spread_sistema"]:
            db[f] = db[f].interpolate().ffill().bfill()

        data_raw  = db[FEATURES].values.astype(float)
        periodos_ = pd.to_datetime(db["periodo"])

        scaler    = MinMaxScaler(feature_range=(0, 1))
        data_norm = scaler.fit_transform(data_raw)

        X, y  = crear_secuencias(data_norm, TARGET_IDX, VENTANA_TEMPORAL)
        split = int(len(X) * (1 - TEST_SIZE))
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]

        modelo = construir_modelo(VENTANA_TEMPORAL, len(FEATURES),
                                  UNIDADES_LSTM, DROPOUT_RATE)
        cb = EarlyStopping(monitor="val_loss", patience=20,
                           restore_best_weights=True, verbose=0)
        hist = modelo.fit(X_tr, y_tr, epochs=EPOCAS, batch_size=BATCH_SIZE,
                          validation_split=0.15, callbacks=[cb], verbose=0)
        print(f"   ✓ {len(hist.history['loss'])} épocas | features: {FEATURES}")

        # Desnormalizar solo columna NIM
        nim_min = scaler.data_min_[TARGET_IDX]
        nim_rng = scaler.data_range_[TARGET_IDX]
        desnorm = lambda arr: arr * nim_rng + nim_min

        y_pred = desnorm(modelo.predict(X_te, verbose=0).flatten())
        y_real = desnorm(y_te)
        metricas.append(evaluar(y_real, y_pred, nombre))

        # Predicción siguiente mes
        X_next   = data_norm[-VENTANA_TEMPORAL:].reshape(1, VENTANA_TEMPORAL, len(FEATURES))
        prox_nim = desnorm(modelo.predict(X_next, verbose=0)[0, 0])
        prox_per = periodos_.max() + pd.DateOffset(months=1)
        print(f"   📅 Predicción {prox_per.strftime('%Y-%m')}: NIM = {prox_nim:.3f}%")

        # Índices test para gráfico
        n_seq   = len(data_raw) - VENTANA_TEMPORAL
        sp_seq  = int(n_seq * (1 - TEST_SIZE))
        ini_tst = sp_seq + VENTANA_TEMPORAL
        per_tst = periodos_.iloc[ini_tst : ini_tst + len(y_real)].reset_index(drop=True)

        resultados.append({
            "banco":        nombre,
            "periodos":     periodos_,
            "nim_hist":     data_raw[:, TARGET_IDX],
            "periodos_test": per_tst,
            "y_real":       y_real,
            "y_pred":       y_pred,
            "prox_periodo": prox_per,
            "prox_nim":     prox_nim,
        })

    # Paso 4: Resultados
    print("\n[4/5] Métricas finales")
    print("-" * 52)
    df_m = pd.DataFrame(metricas)
    print(df_m.to_string(index=False, float_format="{:.4f}".format))
    df_m.to_csv(os.path.join(BASE_DIR, "tarea2_metricas.csv"), index=False)
    print("\n   ✓ tarea2_metricas.csv guardado")
    graficar_predicciones(resultados)
    graficar_metricas(df_m)

    # Paso 5: Conclusiones
    mejor    = df_m.loc[df_m["MAE"].idxmin(), "banco"]
    peor     = df_m.loc[df_m["MAE"].idxmax(), "banco"]
    mae_prom = df_m["MAE"].mean()

    print(f"""
[5/5] Conclusiones
{"=" * 68}

  MODELO:       LSTM Multivariado (capas: {UNIDADES_LSTM} + {UNIDADES_LSTM//2} unidades)
  VENTANA:      {VENTANA_TEMPORAL} meses → predicción t+1
  FEATURES:     NIM propio · TPM · IPC mensual · Spread Sistema
  MAE promedio: {mae_prom:.4f}%

  VARIABLES EXÓGENAS Y SU ROL:
  • TPM  → Principal driver del NIM. Las alzas de tasas del BCCh
           expanden el margen porque las colocaciones se reprecian
           antes que los depósitos.
  • IPC  → La inflación afecta los márgenes en cartera UF. Mayor IPC
           implica mayor ingreso financiero en activos reajustables.
  • Spread Sistema → Captura el ciclo crediticio general de la
           industria, complementando la información del banco propio.

  RESULTADOS:
  • Mejor predicción en: {mejor}
  • Más difícil de predecir: {peor}

  LIMITACIONES:
  • Serie corta (49 períodos) limita la generalización del modelo.
  • NIM acumulado en el año, no marginal mensual.
  • Modelo no incorpora expectativas de TPM (BEI, swap rates).

  EXTENSIONES POSIBLES:
  • Agregar swap TPM a 1 año (expectativas de mercado).
  • Predicción multi-horizonte (3 y 6 meses).
  • Comparar LSTM vs GRU vs Transformer.
    """)

    print("=" * 68)
    print("  ✅ Script finalizado correctamente")
    print("=" * 68)


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()
