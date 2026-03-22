"""
app.py — Dashboard Benchmark Bancario Chile
Ejecutar: streamlit run app.py
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import kpis as K

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Benchmark Bancario Chile", page_icon="🏦",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Open Sans', sans-serif; background:#f4f4f4; }

[data-testid="stSidebar"] { background:#1a1a2e; }
[data-testid="stSidebar"] * { color:#e8edf2 !important; }
[data-testid="stSidebar"] .stSelectbox label {
    color:#F5A623 !important; font-size:0.72rem;
    text-transform:uppercase; letter-spacing:0.08em; font-weight:700;
}
.main .block-container { padding-top:0.8rem; padding-bottom:2rem; max-width:1440px; }

.dash-header {
    background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
    border-radius:8px; padding:1.1rem 1.8rem; margin-bottom:1rem;
    border-left:6px solid #F5A623;
}
.dash-header h1 { color:#fff; font-size:1.4rem; font-weight:700; margin:0; }
.dash-header p  { color:#94a9be; font-size:0.8rem; margin:0.2rem 0 0 0; }

.section-title {
    font-size:0.82rem; font-weight:700; color:#F5A623;
    text-transform:uppercase; letter-spacing:0.05em;
    padding:0.35rem 0; margin-bottom:0.7rem;
    border-bottom:2px solid #F5A623;
}

/* Tabla de resultados */
.res-table { border-collapse:collapse; width:100%; font-size:0.8rem; }
.res-table th { padding:0.35rem 0.5rem; text-align:center; font-weight:700;
                color:#fff; font-size:0.72rem; white-space:nowrap; }
.res-table td { padding:0.28rem 0.5rem; text-align:right;
                border-bottom:1px solid #e8edf2; }
.res-table td:first-child { text-align:left; font-weight:600; white-space:nowrap; }
.res-table tr.total-row td { font-weight:700; background:#f0f0f0;
                              border-top:2px solid #ccc; }
.res-table tr.total-row td:first-child { color:#1a1a2e; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FORMATO NUMÉRICO
# ─────────────────────────────────────────────────────────────────────────────
def cl(val, dec=0):
    """Número con formato chileno (puntos miles, coma decimal)."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    s = f"{val:,.{dec}f}".replace(",","X").replace(".",",").replace("X",".")
    return s

def mmm(val):
    """MMM$ sin decimales."""
    if val is None or (isinstance(val, float) and np.isnan(val)): return "—"
    return f"MMM$ {cl(val, 0)}"

def pct(val, dec=2):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "—"
    return f"{cl(val, dec)}%"

def delta_html(val, es_pct=False):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return '<span style="color:#888">—</span>'
    color = "#2E7D32" if val > 0 else ("#CC0000" if val < 0 else "#888")
    arrow = "▲" if val > 0 else ("▼" if val < 0 else "●")
    txt = f"{cl(abs(val),1)}%" if es_pct else cl(val, 0)
    return f'<span style="color:{color};font-weight:700">{arrow} {txt}</span>'

# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICOS — helpers
# ─────────────────────────────────────────────────────────────────────────────
COLORES = K.COLORES_BANCO
FILL    = {
    "BancoEstado":    "rgba(245,166,35,0.10)",
    "Banco de Chile": "rgba(26,95,168,0.10)",
    "Santander":      "rgba(204,0,0,0.10)",
    "BCI":            "rgba(46,125,50,0.10)",
    "Sistema":        "rgba(136,136,136,0.10)",
}
PLOT_CFG = dict(displayModeBar=False, responsive=True)

MESES_ES = {
    "Jan":"Ene","Feb":"Feb","Mar":"Mar","Apr":"Abr","May":"May","Jun":"Jun",
    "Jul":"Jul","Aug":"Ago","Sep":"Sep","Oct":"Oct","Nov":"Nov","Dec":"Dic"
}

def tickvals_es(periodos):
    """Genera tickvals y ticktext con meses en español para un índice de fechas."""
    if len(periodos) == 0:
        return [], []
    tickvals = list(periodos)
    ticktext = [f"{MESES_ES.get(p.strftime('%b'), p.strftime('%b'))} {p.strftime('%y')}" for p in periodos]
    return tickvals, ticktext

def layout_base(title="", h=340, yaxis_visible=False, yaxis_suffix="", periodos=None, r=60):
    tv, tt = (tickvals_es(periodos) if periodos is not None else (None, None))
    xaxis = dict(showgrid=False, zeroline=False, linecolor="#ddd",
                 tickfont=dict(size=11, color="#333"))
    if tv:
        xaxis["tickvals"] = tv
        xaxis["ticktext"] = tt
    else:
        xaxis["tickformat"] = "%b %Y"

    return dict(
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Open Sans", size=11, color="#1a1a2e"),
        height=h,
        margin=dict(l=10, r=r, t=45, b=40),
        title=dict(text=f"<b>{title}</b>", font=dict(size=13,color="#1a1a2e"), x=0),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(size=12, color="#1a1a2e"),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        xaxis=xaxis,
        yaxis=dict(visible=yaxis_visible, showgrid=False, zeroline=False,
                   ticksuffix=yaxis_suffix if yaxis_visible else ""),
    )

# Orden fijo de bancos
ORDEN_BANCOS_DISPLAY = K.ORDEN_BANCOS  # ["BancoEstado","Banco de Chile","Santander","BCI","Sistema"]

# Colores para tipos de colocación (distintos de colores banco)
COLORES_TIPO_COL = {
    "Comercial": "#2196F3",   # azul material
    "Vivienda":  "#9C27B0",   # morado
    "Consumo":   "#FF5722",   # naranja-rojo
}

POSICION_ETIQUETA = {
    "BancoEstado":    "top center",
    "Banco de Chile": "bottom center",
    "Santander":      "top center",
    "BCI":            "bottom center",
    "Sistema":        "top center",
}

def agregar_linea(fig, serie, banco, dash="solid", width=2.5, size=5, mostrar_etiquetas=True):
    """Agrega traza con etiquetas en primer y último punto."""
    color = COLORES.get(banco, "#aaa")
    pos   = POSICION_ETIQUETA.get(banco, "top center")
    texts = [""] * len(serie)
    if mostrar_etiquetas and len(serie) > 0:
        texts[0]  = f"{cl(serie.iloc[0], 0)}"
        texts[-1] = f"{cl(serie.iloc[-1], 0)}"

    fig.add_trace(go.Scatter(
        x=serie.index, y=serie.values, name=banco,
        mode="lines+markers+text",
        line=dict(color=color, width=width, dash=dash),
        marker=dict(size=size, color=color),
        text=texts,
        textposition=pos,
        textfont=dict(size=12, color=color),
        hovertemplate=f"<b>{banco}</b><br>%{{x|%b %Y}}: %{{y:,.1f}}<extra></extra>",
    ))

# ─────────────────────────────────────────────────────────────────────────────
def _bancos_ordenados(cols):
    orden = ["BancoEstado", "Banco de Chile", "Santander", "BCI", "Sistema"]
    return [b for b in orden if b in cols] + [b for b in cols if b not in orden]

# SIDEBAR Y CARGA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando datos CMF…")
def cargar(data_dir_str, moneda): return K.cargar_datos(Path(data_dir_str), moneda)

with st.sidebar:
    st.markdown("### 🏦 Benchmark Bancario")
    st.markdown("---")
    data_dir = st.text_input("Ruta de datos", value="data")

    st.markdown("**Moneda**")
    moneda_sel = st.radio("", ["Total","CLP","UF","MX"],
                          horizontal=True, label_visibility="collapsed")
    moneda_key = moneda_sel.lower()

    st.markdown("---")
    try:
        df_b1, df_r1 = cargar(data_dir, moneda_key)
        periodos = sorted(df_b1["periodo"].unique())
        DATA_OK = True
    except Exception as e:
        st.error(f"⚠️ {e}")
        DATA_OK = False
        periodos = []

    if DATA_OK:
        periodo_str = st.selectbox("Período",
            [p.strftime("%Y-%m") for p in periodos[::-1]], index=0)
        periodo_sel = pd.Timestamp(periodo_str)
        anio_desde  = st.selectbox("Desde año", [2025, 2024, 2023, 2022], index=0)
        fecha_desde = pd.Timestamp(f"{anio_desde}-01-01")
        df_b1_f = df_b1[df_b1["periodo"] >= fecha_desde]
        df_r1_f = df_r1[df_r1["periodo"] >= fecha_desde]

        st.markdown("---")
        st.markdown(
            '<span style="color:#F5A623;font-size:0.72rem;font-weight:700;'
            'text-transform:uppercase;letter-spacing:0.08em">Variación comparativa</span>',
            unsafe_allow_html=True)
        delta_modo = st.radio(
            "", ["Δ Mes", "Δ 12m", "Δ Año"],
            horizontal=False, label_visibility="collapsed",
            help="Δ Mes = vs mes anterior · Δ 12m = mismo mes año anterior · Δ Año = vs diciembre año anterior")

        st.markdown("---")
        st.markdown(f"<small style='color:#94a9be'>CMF Chile · {len(periodos)} períodos<br>"
                    f"Último: {periodos[-1].strftime('%b %Y')}<br>"
                    f"Moneda: {moneda_sel}</small>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
mes_lbl = pd.Timestamp(periodo_str).strftime("%B %Y").capitalize() if DATA_OK else "—"
st.markdown(f"""
<div class="dash-header">
  <h1>🏦 Análisis Industria Bancaria — Chile</h1>
  <p>4 Grandes Bancos · Datos CMF · {mes_lbl} · Cifras en MMM$ · Moneda: {moneda_sel if DATA_OK else "—"}</p>
</div>""", unsafe_allow_html=True)

if not DATA_OK:
    st.warning("Configura la ruta de datos en el panel lateral.")
    st.stop()

# ── Período de referencia según filtro global de delta ────────────────────────
# Δ Mes  → mes anterior
# Δ 12m  → mismo mes año anterior
# Δ Año  → diciembre del año anterior (YTD)
def _periodo_ref(periodo: pd.Timestamp, modo: str) -> pd.Timestamp:
    if modo == "Δ Mes":
        return periodo - pd.DateOffset(months=1)
    elif modo == "Δ 12m":
        return periodo - pd.DateOffset(months=12)
    else:  # Δ Año
        return pd.Timestamp(f"{periodo.year - 1}-12-01")

periodo_ref  = _periodo_ref(periodo_sel, delta_modo)
delta_lbl    = delta_modo  # etiqueta para mostrar en columnas

tab1,tab2,tab3,tab3b,tab4,tab5,tab6 = st.tabs([
    "📊 Resumen","📈 Colocaciones","⚖️ Balance",
    "📋 Resultados","💰 Ingresos","📉 Rentabilidad","🏆 Ranking"])

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: renderizar tabla de balance estilo presentación
# ─────────────────────────────────────────────────────────────────────────────
def _render_tabla_balance(df_tab: pd.DataFrame, delta_lbl: str = "MTD"):
    """Renderiza tabla de balance con colores por banco y variaciones MTD."""
    bancos = list(K.BANCOS.values()) + [K.NOMBRE_SISTEMA]
    color_hdrs = {
        "BancoEstado":    "#F5A623",
        "Banco de Chile": "#1A5FA8",
        "Santander":      "#CC0000",
        "BCI":            "#2E7D32",
        "Sistema":        "#444444",
    }

    # Encabezados
    hdr_saldos = "".join(
        f'<th colspan="2" style="background:{color_hdrs.get(b,"#555")};'
        f'padding:0.3rem 0.4rem;color:#fff;font-size:0.72rem;text-align:center">{b}</th>'
        for b in bancos
    )
    sub_hdr = "".join(
        f'<th style="background:#e8e8e8;color:#555;font-size:0.68rem;'
        f'padding:0.2rem 0.4rem;text-align:right">Saldo</th>'
        f'<th style="background:#e8e8e8;color:#555;font-size:0.68rem;'
        f'padding:0.2rem 0.4rem;text-align:right">{delta_lbl}</th>'
        for _ in bancos
    )

    filas_html = ""
    for _, row in df_tab.iterrows():
        es_total = row.get("es_total", False)
        nivel    = row.get("nivel", 1)
        linea    = row["Línea"]

        # Estilo según nivel
        if nivel == 0:
            bg    = "#1a1a2e"
            color = "#fff"
            fw    = "700"
            fs    = "0.82rem"
        elif es_total:
            bg    = "#2a2a4a"
            color = "#fff"
            fw    = "700"
            fs    = "0.8rem"
        else:
            bg    = "#fff"
            color = "#333"
            fw    = "400"
            fs    = "0.78rem"

        fila_html = (
            f'<tr style="background:{bg}">'
            f'<td style="padding:0.25rem 0.5rem;color:{color};font-weight:{fw};'
            f'font-size:{fs};white-space:nowrap;border-bottom:1px solid #eee">'
            f'{linea}</td>'
        )

        for banco in bancos:
            saldo = row.get(f"{banco}_saldo", np.nan)
            mtd   = row.get(f"{banco}_mtd",   np.nan)
            s_str = cl(saldo, 0) if pd.notna(saldo) else "—"
            m_str = cl(mtd,   0) if pd.notna(mtd)   else "—"
            m_color = "#2E7D32" if (pd.notna(mtd) and mtd > 0) else (
                      "#CC0000" if (pd.notna(mtd) and mtd < 0) else "#888")
            # Sistema: gris oscuro en totales, claro en detalle
            if banco == K.NOMBRE_SISTEMA:
                if nivel == 0:
                    bg_cel = "background:#3a3a3a;"
                    txt_cel = "#fff"
                elif es_total:
                    bg_cel = "background:#4a4a4a;"
                    txt_cel = "#fff"
                else:
                    bg_cel = "background:#f0f4ff;"
                    txt_cel = "#1a1a2e"
            else:
                bg_cel = ""
                txt_cel = color

            fila_html += (
                f'<td style="padding:0.25rem 0.5rem;text-align:right;'
                f'color:{txt_cel};font-weight:{fw};font-size:{fs};'
                f'border-bottom:1px solid #eee;{bg_cel}">{s_str}</td>'
                f'<td style="padding:0.25rem 0.5rem;text-align:right;'
                f'color:{m_color};font-weight:700;font-size:{fs};'
                f'border-bottom:1px solid #eee;{bg_cel}">{m_str}</td>'
            )

        fila_html += "</tr>"
        filas_html += fila_html

    tabla_html = f"""
    <div style="overflow-x:auto;margin-bottom:0.5rem">
    <table style="border-collapse:collapse;width:100%;font-size:0.8rem">
      <thead>
        <tr>
          <th style="background:#1a1a2e;color:#fff;padding:0.3rem 0.5rem;
                     text-align:left;font-size:0.75rem;min-width:180px">Línea</th>
          {hdr_saldos}
        </tr>
        <tr>
          <th style="background:#1a1a2e;padding:0.2rem 0.5rem"></th>
          {sub_hdr}
        </tr>
      </thead>
      <tbody>{filas_html}</tbody>
    </table>
    </div>
    <small style="color:#888">Cifras en MMM$ · {delta_lbl} = variación respecto al período de referencia</small>
    """
    st.markdown(tabla_html, unsafe_allow_html=True)


def _render_tabla_resultados(df_tab: pd.DataFrame, delta_lbl: str = "Δ"):
    """Renderiza tabla de resultados con 3 niveles de jerarquía."""
    bancos = list(K.BANCOS.values()) + [K.NOMBRE_SISTEMA]
    color_hdrs = {
        "BancoEstado":    "#F5A623",
        "Banco de Chile": "#1A5FA8",
        "Santander":      "#CC0000",
        "BCI":            "#2E7D32",
        "Sistema":        "#444444",
    }
    hdr = "".join(
        f'<th colspan="2" style="background:{color_hdrs.get(b,"#555")};'
        f'padding:0.3rem 0.5rem;color:#fff;font-size:0.72rem;text-align:center">{b}</th>'
        for b in bancos)
    sub = "".join(
        f'<th style="background:#e8e8e8;color:#555;font-size:0.68rem;padding:0.2rem 0.4rem;text-align:right">MMM$</th>'
        f'<th style="background:#e8e8e8;color:#555;font-size:0.68rem;padding:0.2rem 0.4rem;text-align:right">{delta_lbl}</th>'
        for _ in bancos)

    filas_html = ""
    for _, row in df_tab.iterrows():
        nivel    = row.get("nivel", 2)
        es_total = row.get("es_total", False)
        linea    = row["Línea"]

        if nivel == 0:
            bg = "#1a1a2e"; fw = "700"; fs = "0.82rem"; color = "#fff"
        elif nivel == 1 or es_total:
            bg = "#2a3a5a"; fw = "700"; fs = "0.80rem"; color = "#fff"
        else:
            bg = "#fff";    fw = "400"; fs = "0.78rem"; color = "#333"

        filas_html += (
            f'<tr style="background:{bg}">'
            f'<td style="padding:0.25rem 0.6rem;color:{color};font-weight:{fw};'
            f'font-size:{fs};white-space:nowrap;border-bottom:1px solid #eee">{linea}</td>'
        )
        for banco in bancos:
            val  = row.get(f"{banco}_val",  np.nan)
            dval = row.get(f"{banco}_d12m", np.nan)
            vs   = cl(val,  0) if pd.notna(val)  and val  != 0 else "—"
            ds   = cl(dval, 0) if pd.notna(dval) and dval != 0 else "—"
            dc   = "#2E7D32" if (pd.notna(dval) and dval > 0) else (
                   "#CC0000" if (pd.notna(dval) and dval < 0) else "#aaa")
            if banco == K.NOMBRE_SISTEMA:
                if nivel == 0:
                    bg_cel = "background:#3a3a3a;"; txt_cel = "#fff"
                elif nivel == 1 or es_total:
                    bg_cel = "background:#4a4a4a;"; txt_cel = "#fff"
                else:
                    bg_cel = "background:#f0f4ff;"; txt_cel = "#1a1a2e"
            else:
                bg_cel = ""; txt_cel = color
            filas_html += (
                f'<td style="padding:0.25rem 0.5rem;text-align:right;color:{txt_cel};'
                f'font-weight:{fw};font-size:{fs};border-bottom:1px solid #eee;{bg_cel}">{vs}</td>'
                f'<td style="padding:0.25rem 0.5rem;text-align:right;color:{dc};'
                f'font-weight:700;font-size:{fs};border-bottom:1px solid #eee;{bg_cel}">{ds}</td>'
            )
        filas_html += "</tr>"

    _NOTAS = {
        "Δ Mes": "variación vs mes anterior",
        "Δ 12m": "variación vs mismo mes año anterior",
        "Δ Año": "variación vs diciembre año anterior",
    }
    st.markdown(f"""
    <div style="overflow-x:auto;margin-bottom:0.5rem">
    <table style="border-collapse:collapse;width:100%;font-size:0.8rem">
      <thead>
        <tr>
          <th style="background:#1a1a2e;color:#fff;padding:0.3rem 0.6rem;
                     text-align:left;font-size:0.75rem;min-width:200px">Línea</th>
          {hdr}
        </tr>
        <tr>
          <th style="background:#1a1a2e;padding:0.2rem 0.5rem"></th>
          {sub}
        </tr>
      </thead>
      <tbody>{filas_html}</tbody>
    </table></div>
    <small style="color:#888">Cifras en MMM$ · {delta_lbl} = {_NOTAS.get(delta_lbl, "variación comparativa")}</small>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — RESUMEN
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    col_t = K._pivot(K._saldo(K._solo_bancos(df_b1_f), K.CTA_COLOCACIONES))
    act_t = K._pivot(K._saldo(K._solo_bancos(df_b1_f), K.CTA_TOTAL_ACTIVOS))
    nim_t = K.calcular_nim(df_b1_f, df_r1_f)
    roe_t = K.calcular_roe(df_b1_f, df_r1_f)

    def _g(tabla, banco):
        try: return tabla.loc[periodo_sel, banco] if (periodo_sel in tabla.index and banco in tabla.columns) else np.nan
        except: return np.nan

    def _mom(tabla, banco):
        try:
            idx = tabla.index.get_loc(periodo_sel)
            return tabla.loc[periodo_sel, banco] - tabla.iloc[idx-1][banco] if idx>0 else np.nan
        except: return np.nan

    # Sistema
    df_sis = df_b1_f[df_b1_f["banco_nombre"] == K.NOMBRE_SISTEMA]
    def _sv(cta):
        s = df_sis[(df_sis["cuenta"]==cta)&(df_sis["periodo"]==periodo_sel)]["saldo_mes_actual"]
        return s.sum() if not s.empty else np.nan

    st.markdown('<div class="section-title">Indicadores del Sistema — período seleccionado</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    def kcard(col, label, val, delta=""):
        with col:
            st.markdown(f"""
            <div style="background:#fff;border-radius:8px;padding:0.9rem 1.1rem;
                        box-shadow:0 2px 6px rgba(0,0,0,0.07);border-top:4px solid #F5A623">
                <div style="font-size:0.68rem;color:#888;text-transform:uppercase;
                            letter-spacing:0.05em;font-weight:600">{label}</div>
                <div style="font-size:1.3rem;font-weight:700;color:#1a1a2e;
                            font-family:'Courier New',monospace">{val}</div>
                <div style="font-size:0.8rem;margin-top:0.2rem">{delta}</div>
            </div>""", unsafe_allow_html=True)

    kcard(c1, "Colocaciones Sistema",   mmm(_sv(K.CTA_COLOCACIONES)))
    kcard(c2, "Total Activos Sistema",  mmm(_sv(K.CTA_TOTAL_ACTIVOS)))
    kcard(c3, "Patrimonio Sistema",     mmm(_sv(K.CTA_PATRIMONIO)))
    kcard(c4, "NIM Promedio (4 bancos)",pct(float(np.nanmean([_g(nim_t,b) for b in K.BANCOS.values()]))))

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">KPIs por banco</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for i,(cod,banco) in enumerate(K.BANCOS.items()):
        color = COLORES.get(banco,"#aaa")
        col_val  = _g(col_t, banco)
        nim_val  = _g(nim_t, banco)
        roe_val  = _g(roe_t, banco)

        # Variación según filtro global
        def _var_color(v):
            if v is None or (isinstance(v, float) and np.isnan(v)): return "#888"
            return "#2E7D32" if v > 0 else ("#CC0000" if v < 0 else "#888")
        def _var_arrow(v):
            if v is None or (isinstance(v, float) and np.isnan(v)): return "●"
            return "▲" if v > 0 else ("▼" if v < 0 else "●")

        try:
            val_ref = col_t.loc[periodo_ref, banco] if periodo_ref in col_t.index else np.nan
            delta_abs = col_val - val_ref if pd.notna(col_val) and pd.notna(val_ref) else np.nan
            delta_pct = (delta_abs / abs(val_ref) * 100) if (pd.notna(delta_abs) and val_ref != 0) else np.nan
        except:
            delta_abs = np.nan
            delta_pct = np.nan

        d_color = _var_color(delta_abs)
        d_abs_s = f"{_var_arrow(delta_abs)} {cl(abs(delta_abs),0)}" if pd.notna(delta_abs) else "—"
        d_pct_s = f"({cl(delta_pct,1)}%)" if pd.notna(delta_pct) else ""

        with cols[i]:
            st.markdown(f"""
            <div style="background:#ffffff;border-radius:8px;padding:1rem 1.1rem;
                        box-shadow:0 2px 6px rgba(0,0,0,0.10);border-left:5px solid {color};
                        border:1px solid #e0e0e0">
              <div style="font-size:0.9rem;font-weight:700;color:{color};
                          text-transform:uppercase;margin-bottom:0.7rem">{banco}</div>
              <div style="margin-bottom:0.5rem">
                <div style="font-size:0.65rem;color:#666;text-transform:uppercase;font-weight:600">Colocaciones</div>
                <div style="font-size:1.15rem;font-weight:700;color:#1a1a2e;
                            font-family:'Courier New',monospace">{mmm(col_val)}</div>
              </div>
              <div style="border-top:1px solid #eee;padding-top:0.45rem;margin-bottom:0.5rem">
                <div style="font-size:0.62rem;color:#888;text-transform:uppercase;font-weight:600;margin-bottom:0.1rem">{delta_lbl}</div>
                <div style="font-size:0.82rem;font-weight:700;color:{d_color}">{d_abs_s}</div>
                <div style="font-size:0.75rem;color:{d_color}">{d_pct_s}</div>
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.3rem;
                          border-top:1px solid #eee;padding-top:0.45rem">
                <div>
                  <div style="font-size:0.65rem;color:#666;text-transform:uppercase;font-weight:600">NIM anual.</div>
                  <div style="font-size:1rem;font-weight:700;color:#1a1a2e">{pct(nim_val)}</div>
                </div>
                <div>
                  <div style="font-size:0.65rem;color:#666;text-transform:uppercase;font-weight:600">ROE anual.</div>
                  <div style="font-size:1rem;font-weight:700;color:#1a1a2e">{pct(roe_val)}</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Variaciones período seleccionado</div>', unsafe_allow_html=True)

    df_var = K.tabla_variaciones(df_b1_f, df_r1_f, periodo_sel)
    if not df_var.empty:
        # Excluir Colocaciones — ya se muestra en las tarjetas KPI
        df_var = df_var[df_var["KPI"] != "Colocaciones (MMM$)"]
        for kpi_nombre in df_var["KPI"].unique():
            st.markdown(f"**{kpi_nombre}**")
            df_sub = df_var[df_var["KPI"]==kpi_nombre].copy()

            # Sistema primero, luego orden fijo
            orden_var = [K.NOMBRE_SISTEMA] + [b for b in K.ORDEN_BANCOS if b != K.NOMBRE_SISTEMA]
            df_sub["_ord"] = df_sub["Banco"].apply(
                lambda x: orden_var.index(x) if x in orden_var else 99)
            df_sub = df_sub.sort_values("_ord").drop(columns="_ord")

            filas_html = ""
            for _, row in df_sub.iterrows():
                es_sis = row["Banco"] == K.NOMBRE_SISTEMA
                bg  = "#f0f4ff" if es_sis else "#fff"
                fw  = "700"    if es_sis else "400"
                txt = "#1a1a2e"

                v    = cl(row["Valor"], 0)
                mom  = row["Var MoM (MMM$)"]
                momv = f'{cl(mom,0)}' if pd.notna(mom) else "—"
                momc = "#2E7D32" if (pd.notna(mom) and mom>0) else ("#CC0000" if pd.notna(mom) else "#888")
                momp = row["Var MoM (%)"]
                mompv= f'{cl(momp,1)}%' if pd.notna(momp) else "—"
                yoy  = row["Var YoY (%)"]
                yoyv = f'{cl(yoy,1)}%' if pd.notna(yoy) else "—"
                yoyc = "#2E7D32" if (pd.notna(yoy) and yoy>0) else ("#CC0000" if pd.notna(yoy) else "#888")

                filas_html += f"""<tr style="background:{bg}">
                    <td style="padding:0.3rem 0.6rem;text-align:left;border-bottom:1px solid #eee;
                               color:{txt};font-weight:{fw}">{row['Banco']}</td>
                    <td style="padding:0.3rem 0.6rem;text-align:right;border-bottom:1px solid #eee;
                               color:{txt};font-weight:{fw}">{v}</td>
                    <td style="padding:0.3rem 0.6rem;text-align:right;border-bottom:1px solid #eee;
                               color:{momc};font-weight:700">{momv}</td>
                    <td style="padding:0.3rem 0.6rem;text-align:right;border-bottom:1px solid #eee;
                               color:{momc};font-weight:700">{mompv}</td>
                    <td style="padding:0.3rem 0.6rem;text-align:right;border-bottom:1px solid #eee;
                               color:{yoyc};font-weight:700">{yoyv}</td>
                </tr>"""

            tabla_html = f"""
            <div style="overflow-x:auto;margin-bottom:1rem">
            <table style="border-collapse:collapse;width:100%;font-size:0.82rem">
              <thead><tr style="background:#1a1a2e;color:#fff">
                <th style="padding:0.4rem 0.6rem;text-align:left">Banco</th>
                <th style="padding:0.4rem 0.6rem;text-align:right">Valor</th>
                <th style="padding:0.4rem 0.6rem;text-align:right">Var MoM</th>
                <th style="padding:0.4rem 0.6rem;text-align:right">Var MoM (%)</th>
                <th style="padding:0.4rem 0.6rem;text-align:right">Var YoY (%)</th>
              </tr></thead>
              <tbody>{filas_html}</tbody>
            </table></div>"""
            st.markdown(tabla_html, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — COLOCACIONES
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    col_all   = K.colocaciones_totales(df_b1_f, incluir_sistema=True)
    col_bancos = K.colocaciones_totales(df_b1_f, incluir_sistema=False)

    # Gráfico Sistema solo
    st.markdown('<div class="section-title">Evolución Colocaciones — Sistema (MMM$)</div>', unsafe_allow_html=True)
    fig_sis = go.Figure()
    if K.NOMBRE_SISTEMA in col_all.columns:
        serie_sis = col_all[K.NOMBRE_SISTEMA].dropna()
        agregar_linea(fig_sis, serie_sis, K.NOMBRE_SISTEMA, dash="dot", width=2, size=4)
    periodos_sis = list(col_all.index) if not col_all.empty else []
    fig_sis.update_layout(**layout_base("Sistema Financiero (MMM$)", h=260, periodos=periodos_sis))
    st.plotly_chart(fig_sis, use_container_width=True, config=PLOT_CFG)

    # Gráfico 4 bancos
    st.markdown('<div class="section-title">Evolución Colocaciones — 4 Grandes Bancos (MMM$)</div>', unsafe_allow_html=True)
    fig_b = go.Figure()
    for banco in col_bancos.columns:
        serie = col_bancos[banco].dropna()
        agregar_linea(fig_b, serie, banco)
    periodos_b = list(col_bancos.index) if not col_bancos.empty else []
    fig_b.update_layout(**layout_base("Bancos (MMM$)", h=320, periodos=periodos_b))
    st.plotly_chart(fig_b, use_container_width=True, config=PLOT_CFG)

    # Mix por tipo
    st.markdown('<div class="section-title">Mix por Tipo de Colocación — período seleccionado (MMM$)</div>', unsafe_allow_html=True)
    df_seg = K.colocaciones_por_segmento(df_b1_f, periodo_sel)
    if not df_seg.empty:
        df_seg["texto"] = df_seg["valor"].apply(lambda x: cl(x,0))
        fig_seg = px.bar(df_seg, x="banco", y="valor", color="tipo",
            barmode="stack",
            color_discrete_map={"Comercial":"#2196F3","Vivienda":"#9C27B0","Consumo":"#FF5722"},
            labels={"valor":"MMM$","banco":"","tipo":""},
            text="texto")
        fig_seg.update_layout(**{**layout_base("Composición por tipo de colocación (MMM$)", h=360),
            "showlegend":True,
            "xaxis": dict(tickfont=dict(size=13, color="#1a1a2e", family="Open Sans"),
                          linecolor="#ddd", gridcolor="#f0f0f0"),
        })
        fig_seg.update_traces(textfont_size=13, textposition="outside", cliponaxis=False)
        st.plotly_chart(fig_seg, use_container_width=True, config=PLOT_CFG)

    # Variación mensual
    st.markdown('<div class="section-title">Variación Mensual Colocaciones (MMM$)</div>', unsafe_allow_html=True)
    if not col_bancos.empty:
        var_mom = col_bancos.diff(1).dropna()
        fig_var = go.Figure()
        for banco in var_mom.columns:
            serie = var_mom[banco].dropna()
            fig_var.add_trace(go.Bar(
                x=serie.index, y=serie.values, name=banco,
                marker_color=COLORES.get(banco,"#aaa"),
                text=[cl(v,0) for v in serie.values],
                textposition="outside",
                textfont=dict(size=10),
                hovertemplate=f"<b>{banco}</b><br>%{{x|%b %Y}}: %{{y:,.1f}}<extra></extra>",
            ))
        fig_var.add_hline(y=0, line_color="#aaa", line_width=1)
        periodos_var = list(var_mom.index) if not var_mom.empty else []
        fig_var.update_layout(**{**layout_base("Variación mensual (MMM$)", h=320, periodos=periodos_var), "barmode":"group", "showlegend":True})
        st.plotly_chart(fig_var, use_container_width=True, config=PLOT_CFG)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — BALANCE
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Estructura de Balance — período seleccionado (MMM$)</div>', unsafe_allow_html=True)
    df_bal = K.estructura_balance(df_b1_f, periodo_sel)

    if not df_bal.empty:
        df_bal["texto"] = df_bal["valor"].apply(lambda x: cl(x,0))
        ca, cb = st.columns(2)

        with ca:
            df_col = df_bal[df_bal["partida"]=="Colocaciones"]
            fig_ca = px.bar(df_col, x="banco_nombre", y="valor", color_discrete_sequence=["#1A5FA8"],
                labels={"valor":"MMM$","banco_nombre":""}, text="texto")
            fig_ca.update_layout(**{**layout_base("Colocaciones por banco (MMM$)", h=320),
                "showlegend":False,
                "xaxis": dict(tickfont=dict(size=13,color="#1a1a2e"), linecolor="#ddd", gridcolor="#f0f0f0"),
            })
            fig_ca.update_traces(textfont_size=13, textposition="outside",
                                 marker_color=[COLORES.get(b,"#1A5FA8") for b in df_col["banco_nombre"]])
            st.plotly_chart(fig_ca, use_container_width=True, config=PLOT_CFG)

        with cb:
            df_dp = df_bal[df_bal["partida"].isin(["Depósitos","Patrimonio"])]
            fig_cb = px.bar(df_dp, x="banco_nombre", y="valor", color="partida",
                barmode="group",
                color_discrete_map={"Depósitos":"#CC0000","Patrimonio":"#2E7D32"},
                labels={"valor":"MMM$","banco_nombre":"","partida":""},
                text="texto")
            fig_cb.update_layout(**{**layout_base("Depósitos y Patrimonio (MMM$)", h=320),
                "showlegend":True,
                "xaxis": dict(tickfont=dict(size=13,color="#1a1a2e"), linecolor="#ddd", gridcolor="#f0f0f0"),
            })
            fig_cb.update_traces(textfont_size=13, textposition="outside")
            st.plotly_chart(fig_cb, use_container_width=True, config=PLOT_CFG)

    # ── Tabla Activos estilo presentación ────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">Balance Activos — Saldos y Variación {delta_lbl} (MMM$)</div>', unsafe_allow_html=True)

    df_act_tab = K.tabla_balance_activos(df_b1_f, periodo_sel, periodo_ref)
    _render_tabla_balance(df_act_tab, delta_lbl)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">Balance Pasivos — Saldos y Variación {delta_lbl} (MMM$)</div>', unsafe_allow_html=True)

    df_pas_tab = K.tabla_balance_pasivos(df_b1_f, periodo_sel, periodo_ref)
    _render_tabla_balance(df_pas_tab, delta_lbl)

    st.markdown("<br>", unsafe_allow_html=True)

    # Evolución activos — Sistema separado
    st.markdown('<div class="section-title">Evolución Activos Totales — Sistema (MMM$)</div>', unsafe_allow_html=True)
    act_all = K._pivot(K._saldo(df_b1_f, K.CTA_TOTAL_ACTIVOS))
    fig_as = go.Figure()
    if K.NOMBRE_SISTEMA in act_all.columns:
        agregar_linea(fig_as, act_all[K.NOMBRE_SISTEMA].dropna(), K.NOMBRE_SISTEMA, dash="dot", width=2, size=4)
    periodos_as = list(act_all.index) if not act_all.empty else []
    fig_as.update_layout(**layout_base("Activos Totales — Sistema (MMM$)", h=250, periodos=periodos_as, r=80))
    st.plotly_chart(fig_as, use_container_width=True, config=PLOT_CFG)

    st.markdown('<div class="section-title">Evolución Activos Totales — 4 Grandes Bancos (MMM$)</div>', unsafe_allow_html=True)
    fig_ab = go.Figure()
    for banco in [b for b in act_all.columns if b != K.NOMBRE_SISTEMA]:
        agregar_linea(fig_ab, act_all[banco].dropna(), banco)
    periodos_ab = list(act_all.index) if not act_all.empty else []
    fig_ab.update_layout(**layout_base("Activos Totales — Bancos (MMM$)", h=320, periodos=periodos_ab))
    st.plotly_chart(fig_ab, use_container_width=True, config=PLOT_CFG)

    # Colocaciones / Activo
    st.markdown('<div class="section-title">Colocaciones como % del Activo</div>', unsafe_allow_html=True)
    col_b1 = K._pivot(K._saldo(K._solo_bancos(df_b1_f), K.CTA_COLOCACIONES))
    act_b1 = K._pivot(K._saldo(K._solo_bancos(df_b1_f), K.CTA_TOTAL_ACTIVOS))
    pct_col = (col_b1 / act_b1 * 100).round(2)
    fig_pct = go.Figure()
    for banco in pct_col.columns:
        serie = pct_col[banco].dropna()
        texts = [""]*len(serie)
        if len(serie)>0:
            texts[0]  = f"{cl(serie.iloc[0],1)}%"
            texts[-1] = f"{cl(serie.iloc[-1],1)}%"
        fig_pct.add_trace(go.Scatter(
            x=serie.index, y=serie.values, name=banco,
            mode="lines+markers+text",
            line=dict(color=COLORES.get(banco,"#aaa"), width=2.5),
            marker=dict(size=5),
            text=texts, textposition="top center",
            textfont=dict(size=10, color=COLORES.get(banco,"#aaa")),
            hovertemplate=f"<b>{banco}</b><br>%{{x|%b %Y}}: %{{y:.1f}}%<extra></extra>",
        ))
    periodos_pct = list(pct_col.index) if not pct_col.empty else []
    fig_pct.update_layout(**layout_base("Colocaciones / Activos (%)", h=320, yaxis_visible=True, yaxis_suffix="%", periodos=periodos_pct))
    st.plotly_chart(fig_pct, use_container_width=True, config=PLOT_CFG)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3B — RESULTADOS
# ═════════════════════════════════════════════════════════════════════════════
with tab3b:
    st.markdown('<div class="section-title">Estado de Resultados — Saldos Mensuales y Variación ' + delta_lbl + ' (MMM$)</div>', unsafe_allow_html=True)

    df_res_tab = K.tabla_resultados_full(df_r1_f, periodo_sel, periodo_ref)

    if not df_res_tab.empty:
        _render_tabla_resultados(df_res_tab, delta_lbl)

    # ── Tabla ROF ─────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Detalle ROF — período seleccionado (MMM$)</div>', unsafe_allow_html=True)

    df_rof_tab = K.tabla_rof(df_r1_f, periodo_sel, periodo_ref)
    if not df_rof_tab.empty:
        _render_tabla_resultados(df_rof_tab, delta_lbl)

    # ── Gráficos Cartera de Inversiones ──────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Evolución Cartera de Inversiones por Tipo (MMM$)</div>', unsafe_allow_html=True)

    evo_cartera = K.evolucion_cartera_inversiones(df_b1_f)
    # Orden: DPV primero, luego Negociación, luego VCTO
    nombres_cartera = {
        "DPV":        "Cartera Disponible para la Venta (MMM$)",
        "Negociación":"Cartera Negociación (MMM$)",
        "VCTO":       "Cartera Vencimiento (MMM$)",
    }
    for tipo in ["DPV", "Negociación", "VCTO"]:
        tabla = evo_cartera.get(tipo)
        if tabla is None or tabla.empty: continue
        fig_ci = go.Figure()
        periodos_ci = list(tabla.index)
        for banco in _bancos_ordenados(tabla.columns):
            if banco == K.NOMBRE_SISTEMA: continue
            serie = tabla[banco].dropna()
            agregar_linea(fig_ci, serie, banco, size=4)
        fig_ci.update_layout(**layout_base(nombres_cartera[tipo], h=300, periodos=periodos_ci))
        st.plotly_chart(fig_ci, use_container_width=True, config=PLOT_CFG)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — INGRESOS
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    # Tabla de resultados
    st.markdown('<div class="section-title">Resultados Industria — período seleccionado (MMM$)</div>', unsafe_allow_html=True)
    df_res = K.tabla_resultados(df_r1_f, periodo_sel)

    if not df_res.empty:
        _render_tabla_resultados(df_res, delta_lbl)

    st.markdown("<br>", unsafe_allow_html=True)

    # Composición ingresos
    st.markdown('<div class="section-title">Composición Ingreso Operacional (MMM$)</div>', unsafe_allow_html=True)
    df_ing = K.composicion_ingresos(df_r1_f, periodo_sel)
    if not df_ing.empty:
        df_ing["texto"] = df_ing["valor"].apply(lambda x: cl(x,0))
        # Orden fijo de bancos
        orden_ing = [b for b in K.ORDEN_BANCOS if b in df_ing["banco_nombre"].unique()]
        df_ing["banco_nombre"] = pd.Categorical(df_ing["banco_nombre"], categories=orden_ing, ordered=True)
        df_ing = df_ing.sort_values("banco_nombre")
        fig_ing = px.bar(df_ing, x="banco_nombre", y="valor", color="componente",
            barmode="stack",
            color_discrete_map={
                "Margen Intereses":"#1A5FA8","Margen Reajustes":"#F5A623",
                "Comisiones Netas":"#888888","ROF":"#BBBBBB"},
            labels={"valor":"MMM$","banco_nombre":"","componente":""},
            text="texto")
        fig_ing.update_layout(**{**layout_base("Ingreso operacional por componente (MMM$)", h=380),
            "showlegend":True,
            "xaxis": dict(showgrid=False, zeroline=False, linecolor="#ddd",
                          tickfont=dict(size=13,color="#1a1a2e")),
        })
        fig_ing.update_traces(textfont_size=12, textposition="inside", insidetextanchor="middle")
        st.plotly_chart(fig_ing, use_container_width=True, config=PLOT_CFG)

    # Evolución ingresos mensuales
    st.markdown('<div class="section-title">Evolución Mensual de Ingresos (MMM$)</div>', unsafe_allow_html=True)
    evo_ing = K.evolucion_ingresos(df_r1_f)
    ci1, ci2 = st.columns(2)
    for idx, (nombre, tabla) in enumerate(evo_ing.items()):
        if tabla.empty: continue
        fig_c = go.Figure()
        for banco in tabla.columns:
            agregar_linea(fig_c, tabla[banco].dropna(), banco, size=4)
        periodos_inc = list(tabla.index) if not tabla.empty else []
        fig_c.update_layout(**layout_base(f"{nombre} (MMM$)", h=300, periodos=periodos_inc))
        (ci1 if idx%2==0 else ci2).plotly_chart(fig_c, use_container_width=True, config=PLOT_CFG)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — RENTABILIDAD
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    nim = K.calcular_nim(df_b1_f, df_r1_f)
    roa = K.calcular_roa(df_b1_f, df_r1_f)
    roe = K.calcular_roe(df_b1_f, df_r1_f)
    ef  = K.calcular_eficiencia(df_r1_f)

    # KPIs resumen
    st.markdown('<div class="section-title">KPIs de rentabilidad — período seleccionado</div>', unsafe_allow_html=True)
    r1c, r2c, r3c, r4c = st.columns(4)
    for col_obj, tabla, label in [(r1c,nim,"NIM"),(r2c,roa,"ROA"),(r3c,roe,"ROE"),(r4c,ef,"Eficiencia")]:
        with col_obj:
            st.markdown(f"**{label}**")
            for banco in list(K.BANCOS.values()):
                val = tabla.loc[periodo_sel, banco] if (
                    periodo_sel in tabla.index and banco in tabla.columns) else np.nan
                color = COLORES.get(banco,"#aaa")
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:0.25rem 0;border-bottom:1px solid #f0f0f0">'
                    f'<span style="color:{color};font-weight:700;font-size:0.82rem">{banco}</span>'
                    f'<span style="font-weight:700;font-size:0.85rem">{pct(val)}</span>'
                    f'</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Evolución indicadores de rentabilidad</div>', unsafe_allow_html=True)

    metricas_r = [
        ("NIM — Margen Neto de Intereses (%)", nim, "%"),
        ("ROA — Retorno sobre Activos (%)",    roa, "%"),
        ("ROE — Retorno sobre Patrimonio (%)", roe, "%"),
        ("Índice de Eficiencia (%) — menor = mejor", ef, "%"),
    ]
    cr1, cr2 = st.columns(2)
    for idx, (titulo, tabla, suf) in enumerate(metricas_r):
        fig_r = go.Figure()
        for banco in tabla.columns:
            serie = tabla[banco].dropna()
            texts = [""]*len(serie)
            if len(serie)>0:
                texts[0]  = f"{cl(serie.iloc[0],2)}%"
                texts[-1] = f"{cl(serie.iloc[-1],2)}%"
            fig_r.add_trace(go.Scatter(
                x=serie.index, y=serie.values, name=banco,
                mode="lines+markers+text",
                line=dict(color=COLORES.get(banco,"#aaa"), width=2.5),
                marker=dict(size=5),
                text=texts, textposition="top center",
                textfont=dict(size=10, color=COLORES.get(banco,"#aaa")),
                hovertemplate=f"<b>{banco}</b><br>%{{x|%b %Y}}: %{{y:.2f}}%<extra></extra>",
            ))
        periodos_r = list(tabla.index) if not tabla.empty else []
        fig_r.update_layout(**layout_base(titulo, h=300, yaxis_visible=False, periodos=periodos_r))
        fig_r.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, linecolor="#ddd",
                       tickfont=dict(size=11, color="#333")),
            yaxis=dict(visible=False, showgrid=False, zeroline=False),
        )
        (cr1 if idx%2==0 else cr2).plotly_chart(fig_r, use_container_width=True, config=PLOT_CFG)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — RANKING
# ═════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown(f'<div class="section-title">Ranking bancos — {mes_lbl}</div>', unsafe_allow_html=True)
    df_rank = K.ranking_kpis(df_b1_f, df_r1_f, periodo_sel)

    if not df_rank.empty:
        st.dataframe(df_rank.style.format({
            c: (lambda x: pct(x) if "%" in c else mmm(x)) for c in df_rank.columns
        }), use_container_width=True)

        # Radar
        st.markdown('<div class="section-title">Comparativa multidimensional</div>', unsafe_allow_html=True)
        kpis_radar = [c for c in ["NIM (%)","ROA (%)","ROE (%)"] if c in df_rank.columns]
        if len(kpis_radar) >= 2:
            fig_radar = go.Figure()
            for banco in df_rank.index:
                vals = [df_rank.loc[banco,k] for k in kpis_radar]
                vals = [v if not np.isnan(v) else 0 for v in vals]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals+[vals[0]], theta=kpis_radar+[kpis_radar[0]],
                    name=banco, fill="toself",
                    fillcolor=FILL.get(banco,"rgba(170,170,170,0.1)"),
                    line=dict(color=COLORES.get(banco,"#aaa"), width=2.5),
                ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, gridcolor="#e0e0e0",
                                   tickfont=dict(size=11)),
                    angularaxis=dict(gridcolor="#e0e0e0",
                                    tickfont=dict(size=13, color="#1a1a2e")),
                    bgcolor="white",
                ),
                paper_bgcolor="white",
                font=dict(family="Open Sans", size=12),
                margin=dict(l=60, r=60, t=60, b=60),
                legend=dict(orientation="h", y=-0.15,
                            font=dict(size=12), bgcolor="rgba(0,0,0,0)"),
                height=500,
            )
            st.plotly_chart(fig_radar, use_container_width=True, config=PLOT_CFG)

        # Barras horizontales
        st.markdown('<div class="section-title">Detalle por KPI</div>', unsafe_allow_html=True)
        bk1, bk2 = st.columns(2)
        for idx, kpi_col in enumerate(df_rank.columns):
            serie = df_rank[kpi_col].dropna().sort_values(ascending=False)
            es_pct_col = "%" in kpi_col
            # Sin prefijo MMM$ — el título ya indica la unidad
            labels = [pct(v) if es_pct_col else cl(v, 0) for v in serie.values]
            fig_h = go.Figure(go.Bar(
                x=serie.values, y=serie.index, orientation="h",
                marker_color=[COLORES.get(b,"#aaa") for b in serie.index],
                text=labels, textposition="outside",
                textfont=dict(size=12, color="#1a1a2e"),
                hovertemplate="%{y}: %{x:.2f}<extra></extra>",
            ))
            fig_h.update_layout(
                title=dict(text=f"<b>{kpi_col}</b>", font=dict(size=13,color="#1a1a2e"), x=0),
                paper_bgcolor="white", plot_bgcolor="white",
                font=dict(family="Open Sans", size=12),
                margin=dict(l=10, r=220, t=40, b=10),
                height=220,
                xaxis=dict(visible=False, range=[0, serie.max() * 1.35]),
                yaxis=dict(tickfont=dict(size=12, color="#1a1a2e")),
            )
            fig_h.update_traces(cliponaxis=False)
            (bk1 if idx%2==0 else bk2).plotly_chart(fig_h, use_container_width=True, config=PLOT_CFG)

# FOOTER
st.markdown("---")
st.markdown("<small style='color:#94a9be'>Fuente: CMF Chile · Datos públicos · "
            "Uso académico — Diplomado Ciencia de Datos en Finanzas · FEN UCHILE</small>",
            unsafe_allow_html=True)
