# Benchmark Bancario Chile 🏦

Dashboard de análisis comparativo de la industria bancaria chilena usando datos públicos de la CMF.

**Diplomado en Ciencia de Datos para las Finanzas · FEN UChile · 2026**  
Angélica Villegas Lagos

## 📂 Módulos

| # | Archivo | Descripción |
|---|---------|-------------|
| 01 | `cmf_downloader.py` | Scraping CMF · Descarga ZIP · Parseo TXT ancho fijo |
| 02 | `cmf_mapeo_ifrs.py` | Join con Excel IFRS · Jerarquía G0–G7 · Normalización CLP/UF/MX |
| 03 | `kpis.py` | Desacumulación R1 · Cálculo NIM/ROA/ROE · Tablas de presentación |
| 04 | `app.py` | Dashboard Streamlit · 8 pestañas temáticas · Filtro Δ global |
| 05 | `nim_predictor.py` | LSTM · Predicción NIM por banco · TensorFlow/Keras |

## 🖥️ Dashboard — 8 pestañas

1. **Resumen** — KPIs y tarjetas por banco
2. **Activos** — Composición y evolución jerárquica
3. **Balance** — Activos y pasivos jerárquicos
4. **Resultados** — EERR completo con Δ variación
5. **Ingresos** — Composición ingreso operacional
6. **Rentabilidad** — NIM · ROA · ROE · Eficiencia
7. **Ranking** — Barras horizontales + radar
8. **Predicción NIM** — Modelo LSTM por banco

## ⚙️ Stack tecnológico

`Python 3.11` · `Streamlit` · `Plotly` · `Pandas` · `TensorFlow/Keras` · `BeautifulSoup4`

## 🚀 Instalación
```bash
git clone https://github.com/avillegaslagos/benchmark
cd benchmark
pip install -r requirements.txt
streamlit run app.py
```

## 📊 Fuente de datos

CMF Chile · Portal público · Archivos ZIP mensuales (B1, R1, C1)
