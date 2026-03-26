"""
nim_predictor.py — Módulo LSTM para predicción del NIM bancario
===============================================================
Parte del proyecto: Dashboard Benchmark Bancario Chile

Uso:
    # 1. Entrenar y guardar modelos (correr una vez, o cuando lleguen datos nuevos)
    python nim_predictor.py

    # 2. Desde app.py, cargar resultados ya entrenados
    import nim_predictor as NP
    resultados = NP.cargar_resultados()
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import requests
import pickle

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
except ImportError:
    raise ImportError("Ejecuta: pip install tensorflow")

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data" / "output"
MODEL_DIR = BASE_DIR / "data" / "nim_models"   # carpeta donde se guardan los modelos

R1_PATH = DATA_DIR / "R1_con_ifrs.csv"
B1_PATH = DATA_DIR / "B1_con_ifrs.csv"

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
FEATURES         = ["nim_pct", "tpm", "ipc", "spread_sistema"]
TARGET_IDX       = 0

np.random.seed(SEMILLA)


# =============================================================================
# HELPERS DE CARGA
# =============================================================================

def _detectar_col(df, candidatas):
    return next(c for c in candidatas if c in df.columns)


def _desacumular(df_grupo, col):
    """Convierte flujos acumulados CMF a flujo mensual puro."""
    df_grupo = df_grupo.copy().sort_values("periodo").reset_index(drop=True)
    df_grupo["_anio"] = df_grupo["periodo"].str[:4]
    df_grupo["_flujo_m"] = df_grupo.groupby("_anio")[col].diff()
    mask_nan = df_grupo["_flujo_m"].isna()
    df_grupo.loc[mask_nan, "_flujo_m"] = df_grupo.loc[mask_nan, col]
    return df_grupo["_flujo_m"].reset_index(drop=True)


def _cargar_nim_bancos():
    r1 = pd.read_csv(R1_PATH, dtype=str)
    r1 = r1[r1["banco_codigo"].isin(BANCOS.keys())]
    r1 = r1[r1["cuenta"].isin([CTA_ING_INTERESES, CTA_GTO_INTERESES])]
    col = _detectar_col(r1, ["flujo_mes_actual", "col_1", "saldo_mes_actual"])
    r1["saldo"] = pd.to_numeric(r1[col], errors="coerce") / 1e9

    partes = []
    for (banco, cuenta), grupo in r1.groupby(["banco_codigo", "cuenta"]):
        grupo = grupo.sort_values("periodo").reset_index(drop=True)
        grupo["saldo_mensual"] = _desacumular(grupo, "saldo")
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
    col_b = _detectar_col(b1, ["saldo_total", "saldo_mes_actual", "col_1", "flujo_mes_actual"])
    b1["activos"] = pd.to_numeric(b1[col_b], errors="coerce") / 1e9

    df = r1p.merge(b1[["periodo", "banco_codigo", "activos"]], on=["periodo", "banco_codigo"], how="inner")
    df["nim_pct"] = ((df["ing"] - df["gto"]) / df["activos"]) * 12 * 100
    df["banco_nombre"] = df["banco_codigo"].map(BANCOS)
    df = df.sort_values(["banco_codigo", "periodo"]).dropna(subset=["nim_pct"])
    df = df[df["nim_pct"].between(0.1, 15)]
    return df


def _cargar_spread_sistema():
    r1 = pd.read_csv(R1_PATH, dtype=str)
    r1s = r1[r1["banco_codigo"] == CODIGO_SISTEMA].copy()
    r1s = r1s[r1s["cuenta"].isin([CTA_ING_INTERESES, CTA_GTO_INTERESES])]
    if r1s.empty:
        return pd.DataFrame(columns=["periodo", "spread_sistema"])

    col = _detectar_col(r1s, ["flujo_mes_actual", "col_1", "saldo_mes_actual"])
    r1s["saldo"] = pd.to_numeric(r1s[col], errors="coerce") / 1e9
    piv = r1s.pivot_table(index="periodo", columns="cuenta",
                           values="saldo", aggfunc="sum").reset_index()
    piv.columns.name = None
    piv = piv.rename(columns={CTA_ING_INTERESES: "ing", CTA_GTO_INTERESES: "gto"})

    b1 = pd.read_csv(B1_PATH, dtype=str)
    b1s = b1[b1["banco_codigo"] == CODIGO_SISTEMA].copy()
    b1s = b1s[b1s["cuenta"] == CTA_TOTAL_ACTIVOS]
    col_b = _detectar_col(b1s, ["saldo_total", "saldo_mes_actual", "col_1", "flujo_mes_actual"])
    b1s["activos"] = pd.to_numeric(b1s[col_b], errors="coerce") / 1e9

    merged = piv.merge(b1s[["periodo", "activos"]], on="periodo", how="inner")
    if "ing" in merged.columns and "gto" in merged.columns:
        merged = merged.sort_values("periodo").reset_index(drop=True)
        merged["ing_m"] = _desacumular(merged, "ing")
        merged["gto_m"] = _desacumular(merged, "gto")
        merged["spread_sistema"] = ((merged["ing_m"] - merged["gto_m"]) / merged["activos"]) * 12 * 100
    else:
        merged["spread_sistema"] = np.nan
    return merged[["periodo", "spread_sistema"]].dropna()


def _descargar_mindicador(indicador, desde_año, hasta_año):
    registros = []
    for año in range(desde_año, hasta_año + 1):
        url = f"https://mindicador.cl/api/{indicador}/{año}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                for obs in resp.json().get("serie", []):
                    registros.append({"periodo": obs["fecha"][:7], indicador: float(obs["valor"])})
        except Exception:
            pass
    if not registros:
        return pd.DataFrame(columns=["periodo", indicador])
    return (pd.DataFrame(registros)
              .groupby("periodo")[indicador].mean()
              .reset_index()
              .sort_values("periodo"))


def _obtener_macro(periodos):
    año_min = int(min(periodos)[:4])
    año_max = int(max(periodos)[:4]) + 1
    df_tpm = _descargar_mindicador("tpm", año_min, año_max)
    df_ipc = _descargar_mindicador("ipc", año_min, año_max)
    macro = pd.DataFrame({"periodo": periodos})
    macro = macro.merge(df_tpm, on="periodo", how="left") if not df_tpm.empty else macro.assign(tpm=np.nan)
    macro = macro.merge(df_ipc, on="periodo", how="left") if not df_ipc.empty else macro.assign(ipc=np.nan)
    macro["tpm"] = macro["tpm"].interpolate().ffill().bfill()
    macro["ipc"] = macro["ipc"].interpolate().ffill().bfill()
    return macro


def _crear_secuencias(data, target_idx, ventana):
    X, y = [], []
    for i in range(len(data) - ventana):
        X.append(data[i : i + ventana, :])
        y.append(data[i + ventana, target_idx])
    return np.array(X), np.array(y)


def _construir_modelo(ventana, n_features, unidades, dropout):
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
# ENTRENAMIENTO Y GUARDADO
# =============================================================================

def entrenar_y_guardar():
    """
    Entrena un modelo LSTM por banco y guarda en disco:
        data/nim_models/{banco}_model/     ← modelo Keras
        data/nim_models/{banco}_scaler.pkl ← scaler MinMax
        data/nim_models/resultados.json    ← métricas + series para graficar
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("\n[nim_predictor] Iniciando entrenamiento...")

    df_nim    = _cargar_nim_bancos()
    df_spread = _cargar_spread_sistema()
    periodos  = sorted(df_nim["periodo"].unique())
    df_macro  = _obtener_macro(periodos)

    resultados_json = {}

    for codigo, nombre in BANCOS.items():
        print(f"  → Entrenando {nombre}...")
        db = df_nim[df_nim["banco_codigo"] == codigo].sort_values("periodo").reset_index(drop=True)
        if len(db) < VENTANA_TEMPORAL + 8:
            print(f"  ⚠  Datos insuficientes para {nombre}")
            continue

        db = db.merge(df_macro, on="periodo", how="left")
        db = db.merge(df_spread, on="periodo", how="left") if not df_spread.empty else db.assign(spread_sistema=0.0)
        for f in ["tpm", "ipc", "spread_sistema"]:
            db[f] = db[f].interpolate().ffill().bfill()

        data_raw  = db[FEATURES].values.astype(float)
        periodos_ = db["periodo"].tolist()

        scaler    = MinMaxScaler(feature_range=(0, 1))
        data_norm = scaler.fit_transform(data_raw)

        X, y  = _crear_secuencias(data_norm, TARGET_IDX, VENTANA_TEMPORAL)
        split = int(len(X) * (1 - TEST_SIZE))
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]

        modelo = _construir_modelo(VENTANA_TEMPORAL, len(FEATURES), UNIDADES_LSTM, DROPOUT_RATE)
        cb = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=0)
        modelo.fit(X_tr, y_tr, epochs=EPOCAS, batch_size=BATCH_SIZE,
                   validation_split=0.15, callbacks=[cb], verbose=0)

        # Guardar modelo y scaler
        modelo.save(MODEL_DIR / f"{codigo}_model.keras")
        with open(MODEL_DIR / f"{codigo}_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        # Desnormalizar
        nim_min = scaler.data_min_[TARGET_IDX]
        nim_rng = scaler.data_range_[TARGET_IDX]
        desnorm = lambda arr: arr * nim_rng + nim_min

        y_pred = desnorm(modelo.predict(X_te, verbose=0).flatten())
        y_real = desnorm(y_te)

        mae  = float(mean_absolute_error(y_real, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_real, y_pred)))
        mape = float(np.mean(np.abs((y_real - y_pred) / (y_real + 1e-8))) * 100)

        # Predicción próximo mes
        X_next   = data_norm[-VENTANA_TEMPORAL:].reshape(1, VENTANA_TEMPORAL, len(FEATURES))
        prox_nim = float(desnorm(modelo.predict(X_next, verbose=0)[0, 0]))
        ultimo_periodo = pd.Timestamp(periodos_[-1])
        prox_periodo   = (ultimo_periodo + pd.DateOffset(months=1)).strftime("%Y-%m")

        # Índices del test para graficar
        n_seq   = len(data_raw) - VENTANA_TEMPORAL
        sp_seq  = int(n_seq * (1 - TEST_SIZE))
        ini_tst = sp_seq + VENTANA_TEMPORAL
        per_tst = periodos_[ini_tst : ini_tst + len(y_real)]

        resultados_json[nombre] = {
            "codigo":          codigo,
            "mae":             mae,
            "rmse":            rmse,
            "mape":            mape,
            "prox_periodo":    prox_periodo,
            "prox_nim":        prox_nim,
            "periodos_hist":   periodos_,
            "nim_hist":        data_raw[:, TARGET_IDX].tolist(),
            "periodos_test":   per_tst,
            "y_real":          y_real.tolist(),
            "y_pred":          y_pred.tolist(),
        }
        print(f"    ✓ MAE={mae:.3f}% | Predicción {prox_periodo}: {prox_nim:.2f}%")

    # Guardar resultados consolidados
    with open(MODEL_DIR / "resultados.json", "w") as f:
        json.dump(resultados_json, f)

    print(f"\n[nim_predictor] Modelos guardados en {MODEL_DIR}")
    return resultados_json


# =============================================================================
# CARGA DE RESULTADOS (para app.py)
# =============================================================================

def modelos_disponibles() -> bool:
    """Retorna True si los modelos ya fueron entrenados y guardados."""
    return (MODEL_DIR / "resultados.json").exists()


def cargar_resultados() -> dict:
    """
    Carga los resultados pre-entrenados desde disco.
    Retorna dict con la información de cada banco lista para graficar en Streamlit.
    """
    if not modelos_disponibles():
        return {}
    with open(MODEL_DIR / "resultados.json", "r") as f:
        return json.load(f)


# =============================================================================
# ENTRY POINT (entrenar desde línea de comandos)
# =============================================================================

if __name__ == "__main__":
    entrenar_y_guardar()
    print("\n✅ Listo. Ahora puedes abrir el dashboard y ver la tab 'Predicción NIM'.")
