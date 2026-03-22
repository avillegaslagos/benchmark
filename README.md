# 🏦 Benchmark Bancario Chile

Dashboard interactivo para el análisis comparativo de la industria bancaria chilena, desarrollado con datos públicos de la **Comisión para el Mercado Financiero (CMF)**.

## 📊 Descripción

Permite comparar los principales indicadores financieros de los cuatro grandes bancos del sistema chileno: **BancoEstado**, **Banco de Chile**, **Santander** y **BCI**, incluyendo cifras del sistema total.

## 🏗️ Arquitectura

El proyecto se compone de 4 módulos Python en cadena:

| Módulo | Descripción |
|--------|-------------|
| `cmf_downloader.py` | Descarga y consolida datos mensuales desde el portal CMF |
| `cmf_mapeo_ifrs.py` | Enriquece los datos con jerarquía contable IFRS |
| `kpis.py` | Cálculo de KPIs y funciones analíticas |
| `app.py` | Dashboard interactivo con Streamlit |

## 📁 Estructura del proyecto
```
benchmark/
├── app.py                  # Dashboard principal
├── kpis.py                 # Módulo de KPIs
├── cmf_downloader.py       # Descarga de datos CMF
├── cmf_mapeo_ifrs.py       # Mapeo jerarquía IFRS
├── requirements.txt        # Dependencias
└── data/
    └── IFRS_mapeo.xlsx     # Archivo de mapeo contable (no incluido)
```

## 🚀 Instalación y uso

### 1. Clonar el repositorio
```bash
git clone https://github.com/avillegaslagos/benchmark.git
cd benchmark
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Descargar datos históricos CMF
```bash
python cmf_downloader.py --full
```

### 4. Generar archivos con mapeo IFRS
```bash
python cmf_mapeo_ifrs.py
```

### 5. Iniciar el dashboard
```bash
streamlit run app.py
```

## 📈 KPIs calculados

- **NIM** — Margen Neto de Intereses (anualizado)
- **ROA** — Retorno sobre Activos (anualizado)
- **ROE** — Retorno sobre Patrimonio (anualizado)
- **Índice de Eficiencia** — Gastos operacionales / Ingreso bruto

## 🗂️ Pestañas del dashboard

1. **Resumen** — KPIs por banco y variaciones
2. **Colocaciones** — Evolución y mix por segmento
3. **Balance** — Activos y pasivos con variación
4. **Resultados** — Estado de resultados completo
5. **Ingresos** — Composición del ingreso operacional
6. **Rentabilidad** — Evolución NIM, ROA, ROE, Eficiencia
7. **Ranking** — Comparativa multidimensional

## 🔧 Tecnologías

- Python 3.11
- Streamlit
- Plotly
- Pandas
- BeautifulSoup4

## 📌 Notas importantes

- Los datos de la carpeta `data/` no se incluyen en el repositorio por su tamaño.
- Se requiere el archivo `IFRS_mapeo.xlsx` en `data/` para ejecutar el mapeo contable.
- Los datos históricos se descargan directamente desde el portal público de la CMF.

## 👩‍💻 Autora

**Angélica Villegas Lagos**  
Diplomado en Ciencia de Datos para las Finanzas — FEN UCHILE

---
*Fuente: CMF Chile · Datos públicos · Uso académico* 
