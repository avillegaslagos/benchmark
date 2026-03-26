# Benchmark Bancario Chile

Dashboard de análisis comparativo de la industria bancaria chilena usando datos públicos de la CMF.

**Diplomado en Ciencia de Datos para las Finanzas · FEN UChile · 2026**
Angélica Villegas Lagos

---

## 📂 Módulos

| # | Archivo | Descripción |
|---|---------|-------------|
| 01 | `cmf_downloader.py` | Scraping CMF · Descarga ZIP · Parseo TXT ancho fijo · Consolidación histórica |
| 02 | `cmf_mapeo_ifrs.py` | Join con Excel IFRS · Jerarquía contable G0–G7 · Normalización CLP/UF/MX |
| 03 | `kpis.py` | Desacumulación R1 · Cálculo NIM/ROA/ROE · Tablas de presentación |
| 04 | `app.py` | Dashboard Streamlit · 8 pestañas temáticas · Filtro Δ global |
| 05 | `nim_predictor.py` | Modelo LSTM multivariado · Predicción NIM por banco · TensorFlow/Keras |

---

## Dashboard — 8 pestañas

| # | Pestaña | Contenido |
|---|---------|-----------|
| 1 | **Resumen** | KPIs y tarjetas por banco |
| 2 | **Activos** | Composición y evolución jerárquica de activos |
| 3 | **Pasivos** | Composición y evolución jerárquica de pasivos |
| 4 | **Balance** | Activos y pasivos con Δ variación |
| 5 | **Resultados** | EERR completo con Δ variación |
| 6 | **Ingresos** | Composición del ingreso operacional |
| 7 | **Rentabilidad** | NIM · ROA · ROE · Eficiencia |
| 8 | **Predicción NIM** | Modelo LSTM · Forecast por banco · Métricas MAE/RMSE/MAPE |

> ✦ **Filtro global de variación:** Δ Mes · Δ 12m · Δ Año — aplicado simultáneamente en todas las pestañas.

---

## Módulo de Predicción NIM (LSTM)

El módulo `nim_predictor.py` entrena un modelo LSTM multivariado independiente por banco.

**Variables de entrada:**
- NIM propio del banco (desacumulado, anualizado)
- TPM — Tasa de Política Monetaria (BCCh vía mindicador.cl)
- IPC — Variación mensual (INE vía mindicador.cl)
- Spread del sistema financiero (CMF, código 999)

**Arquitectura:** 2 capas LSTM (64 + 32 unidades) + Dropout(0.2) + Dense(1)
**Ventana temporal:** 6 meses → predice el mes siguiente
**Entrenamiento:** EarlyStopping sobre val_loss (patience=20)

```bash
# Entrenar modelos (correr una vez, o al llegar datos nuevos)
python nim_predictor.py
```

---

## ⚙️ Stack tecnológico

`Python 3.11` · `Streamlit` · `Plotly` · `Pandas` · `NumPy` · `TensorFlow/Keras` · `scikit-learn` · `BeautifulSoup4`

---

## Instalación y uso

```bash
# 1. Clonar el repositorio
git clone https://github.com/avillegaslagos/benchmark
cd benchmark

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. (Opcional) Entrenar modelos LSTM
python nim_predictor.py

# 4. Lanzar el dashboard
streamlit run app.py
```

---

## Fuente de datos

- **CMF Chile** · Portal público · Archivos ZIP mensuales (B1, R1, C1)
- **mindicador.cl** · TPM e IPC (Banco Central / INE)
- Actualización: mensual · Uso: académico

---

## Bancos analizados

| Código CMF | Banco |
|-----------|-------|
| 012 | BancoEstado |
| 001 | Banco de Chile |
| 037 | Santander |
| 016 | BCI |

> Referencia de sistema consolidado: código CMF 999.
