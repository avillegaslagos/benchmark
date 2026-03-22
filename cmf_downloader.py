"""
cmf_downloader.py
=================
Descarga y consolida los archivos de Balance y Estado de Situación de Bancos
publicados mensualmente por la CMF Chile.

Sitio fuente:
    https://www.cmfchile.cl/portal/estadisticas/617/w3-propertyvalue-28917.html

Estructura real del ZIP:
    articles-XXXXXX_recurso_1.zip
    └── (carpeta interna con mismo nombre)
        ├── metadata/
        ├── b1202601001     ← tipo B1, año 2026, mes 01, banco 001
        ├── b1202601009     ← tipo B1, año 2026, mes 01, banco 009
        ├── r1202601001     ← tipo R1 (Estado de Resultados), banco 001
        ├── c1202601001     ← tipo C1 (Cartera), banco 001
        └── ...             (un archivo TXT por banco y por tipo)

Nomenclatura:  [tipo2][año4][mes2][banco3]
    Ej: b1202601001  →  B1 | 2026-01 | banco 001

Estructura de salida:
    data/
    ├── raw/
    │   └── 2026_01/
    │       ├── b1_2026_01.csv    ← todos los bancos tipo B1 del período
    │       ├── r1_2026_01.csv
    │       └── c1_2026_01.csv
    ├── consolidado/
    │   ├── B1_historico.csv      ← panel completo (todos los períodos)
    │   ├── R1_historico.csv
    │   └── C1_historico.csv
    └── log/
        └── descarga_log.csv

Uso:
    python cmf_downloader.py                    # solo períodos nuevos (>= 2022)
    python cmf_downloader.py --full             # todo el histórico disponible (>= 2022)
    python cmf_downloader.py --periodo 2026/01  # período específico
    python cmf_downloader.py --no-consolidar    # descarga sin regenerar consolidados
"""

import re
import io
import zipfile
import logging
import argparse
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
BASE_URL  = "https://www.cmfchile.cl/portal/estadisticas/617/"
INDEX_URL = BASE_URL + "w3-propertyvalue-28917.html"

# Tipos de informe a procesar (prefijo de 2 chars del nombre de archivo)
TARGET_TIPOS = ["b1", "r1", "c1"]

# Solo descargar desde este año en adelante
ANIO_DESDE = 2022

# Rutas del proyecto
DATA_DIR   = Path("data")
RAW_DIR    = DATA_DIR / "raw"
CONSOL_DIR = DATA_DIR / "consolidado"
LOG_DIR    = DATA_DIR / "log"
LOG_FILE   = LOG_DIR  / "descarga_log.csv"

PERIODO_COL = "periodo"   # columna identificadora, formato YYYY-MM

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS GENERALES
# ─────────────────────────────────────────────────────────────────────────────
def crear_directorios():
    for d in [RAW_DIR, CONSOL_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def leer_log() -> pd.DataFrame:
    if LOG_FILE.exists():
        return pd.read_csv(LOG_FILE, dtype=str)
    return pd.DataFrame(columns=["periodo", "zip_url", "fecha_descarga", "estado"])


def guardar_log(df: pd.DataFrame):
    df.to_csv(LOG_FILE, index=False)


def periodo_a_label(periodo: str) -> str:
    """'2026/01'  →  '2026_01'  (nombre de carpeta)"""
    return periodo.replace("/", "_")


def periodo_a_col(periodo: str) -> str:
    """'2026/01'  →  '2026-01'  (valor en columna periodo)"""
    return periodo.replace("/", "-")


# ─────────────────────────────────────────────────────────────────────────────
# PARSER DEL NOMBRE DE ARCHIVO CMF
# ─────────────────────────────────────────────────────────────────────────────
def parsear_nombre(nombre: str) -> dict | None:
    """
    'b1202601001'  →  {"tipo": "b1", "anio": "2026", "mes": "01", "banco": "001"}
    Devuelve None si no coincide con el patrón esperado.
    """
    stem = Path(nombre).stem.lower()
    m = re.match(r"^([a-z]\d)(\d{4})(\d{2})(\d{3})$", stem)
    if not m:
        return None
    return {
        "tipo":  m.group(1),   # "b1" / "r1" / "c1"
        "anio":  m.group(2),   # "2026"
        "mes":   m.group(3),   # "01"
        "banco": m.group(4),   # "001"
    }


# ─────────────────────────────────────────────────────────────────────────────
# PARSER DEL CONTENIDO TXT (ANCHO FIJO)
# ─────────────────────────────────────────────────────────────────────────────
def parsear_txt_banco(contenido_bytes: bytes, banco: str, periodo: str) -> pd.DataFrame | None:
    """
    Parsea el TXT de ancho fijo de un banco.

    Línea 1:  "001      BANCO DE CHILE"     → código y nombre
    Resto:    cuenta   valor1  valor2  ...  → una fila por cuenta contable

    Devuelve DataFrame con columnas:
        periodo | banco_codigo | banco_nombre | cuenta | col_1 | col_2 | ...
    """
    # Intentar distintas codificaciones
    texto = None
    for enc in ["latin-1", "utf-8", "cp1252"]:
        try:
            texto = contenido_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue

    if texto is None:
        log.error(f"  No se pudo decodificar archivo del banco {banco}")
        return None

    lineas = texto.splitlines()
    if not lineas:
        return None

    # ── Encabezado: código y nombre del banco ──────────────────────────────
    primera = lineas[0].strip()
    partes_h = primera.split(None, 1)
    # Normalizar a 3 dígitos con ceros a la izquierda ("1" → "001", "16" → "016")
    # El TXT puede traer el código sin padding según el período histórico
    raw_codigo   = partes_h[0] if partes_h else banco
    banco_codigo = str(int(raw_codigo)).zfill(3) if raw_codigo.isdigit() else raw_codigo
    banco_nombre = partes_h[1].strip() if len(partes_h) > 1 else ""

    # ── Cuerpo: líneas de cuentas ──────────────────────────────────────────
    filas = []
    for linea in lineas[1:]:
        linea = linea.rstrip()
        if not linea:
            continue

        partes = linea.split()
        if len(partes) < 2:
            continue

        cuenta  = partes[0]
        valores = partes[1:]

        fila = {
            PERIODO_COL:    periodo_a_col(periodo),
            "banco_codigo": banco_codigo,
            "banco_nombre": banco_nombre,
            "cuenta":       cuenta,
        }
        for i, v in enumerate(valores, start=1):
            fila[f"col_{i}"] = v

        filas.append(fila)

    if not filas:
        return None

    return pd.DataFrame(filas)


# ─────────────────────────────────────────────────────────────────────────────
# SCRAPING DEL ÍNDICE CMF
# ─────────────────────────────────────────────────────────────────────────────
MESES_ES = {
    "enero": "01", "febrero": "02", "marzo": "03", "abril": "04",
    "mayo": "05", "junio": "06", "julio": "07", "agosto": "08",
    "septiembre": "09", "octubre": "10", "noviembre": "11", "diciembre": "12",
}


def _texto_a_periodo(texto: str) -> str | None:
    """'Noviembre 2025'  →  '2025/11'   |   None si no machea"""
    if not texto:
        return None
    t = texto.lower().strip()
    for mes_es, num in MESES_ES.items():
        if mes_es in t:
            m = re.search(r"(20\d{2})", t)
            if m:
                return f"{m.group(1)}/{num}"
    return None


def obtener_periodos_disponibles() -> list[dict]:
    """
    Parsea la página índice de la CMF y devuelve lista de períodos
    disponibles >= ANIO_DESDE, con su URL de ZIP.
    """
    log.info("Consultando índice CMF…")
    resp = requests.get(INDEX_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    periodos = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.endswith(".zip"):
            continue

        # Intentar parsear el período desde el texto del link o del link anterior
        periodo = _texto_a_periodo(a.get_text(strip=True))
        if periodo is None:
            prev = a.find_previous_sibling("a")
            if prev:
                periodo = _texto_a_periodo(prev.get_text(strip=True))
        if periodo is None:
            continue

        # Filtro de año mínimo
        if int(periodo.split("/")[0]) < ANIO_DESDE:
            continue

        zip_url = href if href.startswith("http") else BASE_URL + href
        periodos.append({"periodo": periodo, "zip_url": zip_url})

    log.info(f"Períodos disponibles (>= {ANIO_DESDE}): {len(periodos)}")
    return periodos


# ─────────────────────────────────────────────────────────────────────────────
# DESCARGA Y PARSEO DE UN PERÍODO
# ─────────────────────────────────────────────────────────────────────────────
def descargar_periodo(periodo: str, zip_url: str) -> bool:
    """
    Descarga el ZIP, parsea todos los TXT por tipo y banco,
    y guarda un CSV consolidado por tipo en raw/<periodo>/.
    """
    label   = periodo_a_label(periodo)
    destino = RAW_DIR / label
    destino.mkdir(parents=True, exist_ok=True)

    log.info(f"Descargando {periodo}  ←  {zip_url}")
    try:
        resp = requests.get(zip_url, headers=HEADERS, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.error(f"  Error de red: {e}")
        return False

    # Acumular DataFrames por tipo
    dfs_por_tipo: dict[str, list] = {t: [] for t in TARGET_TIPOS}

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            for nombre_zip in zf.namelist():
                # Ignorar carpeta metadata y directorios
                if nombre_zip.endswith("/") or "metadata" in nombre_zip.lower():
                    continue

                info = parsear_nombre(nombre_zip)
                if info is None or info["tipo"] not in TARGET_TIPOS:
                    continue

                contenido = zf.read(nombre_zip)
                df = parsear_txt_banco(contenido, info["banco"], periodo)
                if df is not None:
                    dfs_por_tipo[info["tipo"]].append(df)

    except zipfile.BadZipFile as e:
        log.error(f"  ZIP inválido: {e}")
        return False

    # Guardar CSV por tipo
    guardados = 0
    for tipo, lista in dfs_por_tipo.items():
        if not lista:
            continue
        df_tipo    = pd.concat(lista, ignore_index=True)
        nombre_csv = f"{tipo}_{label}.csv"
        df_tipo.to_csv(destino / nombre_csv, index=False, encoding="utf-8-sig")
        n_bancos = df_tipo["banco_codigo"].nunique()
        log.info(f"  ✓ {nombre_csv}  ({len(df_tipo):,} filas | {n_bancos} bancos)")
        guardados += 1

    return guardados > 0


# ─────────────────────────────────────────────────────────────────────────────
# CONSOLIDACIÓN
# ─────────────────────────────────────────────────────────────────────────────
def consolidar():
    """
    Lee todos los CSV en raw/ y genera (o actualiza) los archivos
    consolidado/B1_historico.csv, R1_historico.csv, C1_historico.csv.
    Reemplaza los períodos que ya existen para evitar duplicados.
    """
    log.info("Consolidando archivos…")

    for tipo in TARGET_TIPOS:
        partes = []

        for carpeta in sorted(RAW_DIR.iterdir()):
            if not carpeta.is_dir():
                continue
            for csv in carpeta.glob(f"{tipo}_*.csv"):
                df = pd.read_csv(csv, dtype=str, low_memory=False)
                partes.append(df)

        if not partes:
            log.info(f"  Sin datos para {tipo.upper()}, se omite.")
            continue

        nuevo = pd.concat(partes, ignore_index=True)
        ruta_hist = CONSOL_DIR / f"{tipo.upper()}_historico.csv"

        if ruta_hist.exists():
            historico = pd.read_csv(ruta_hist, dtype=str, low_memory=False)
            periodos_nuevos = set(nuevo[PERIODO_COL].unique())
            historico = historico[~historico[PERIODO_COL].isin(periodos_nuevos)]
            final = pd.concat([historico, nuevo], ignore_index=True)
        else:
            final = nuevo

        final = final.sort_values([PERIODO_COL, "banco_codigo"], ignore_index=True)
        final.to_csv(ruta_hist, index=False, encoding="utf-8-sig")

        log.info(
            f"  ✓ {ruta_hist.name}  "
            f"({len(final):,} filas | "
            f"{final[PERIODO_COL].nunique()} períodos | "
            f"{final['banco_codigo'].nunique()} bancos)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=f"Descargador CMF Bancos (períodos >= {ANIO_DESDE})"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Descarga todos los períodos disponibles (ignora log previo)"
    )
    parser.add_argument(
        "--periodo", default=None,
        help="Descarga un período específico (ej: 2026/01)"
    )
    parser.add_argument(
        "--no-consolidar", action="store_true",
        help="Solo descarga; no regenera los archivos consolidados"
    )
    args = parser.parse_args()

    crear_directorios()
    log_df        = leer_log()
    ya_descargados = set(log_df[log_df["estado"] == "ok"]["periodo"].tolist())

    catalogo = obtener_periodos_disponibles()
    if not catalogo:
        log.error("No se encontraron períodos. Revisa conexión o estructura del sitio.")
        return

    # Filtrar según flags
    if args.periodo:
        catalogo = [p for p in catalogo if p["periodo"] == args.periodo]
        if not catalogo:
            log.error(f"Período '{args.periodo}' no encontrado (o es anterior a {ANIO_DESDE}).")
            return
    elif not args.full:
        catalogo = [p for p in catalogo if p["periodo"] not in ya_descargados]

    if not catalogo:
        log.info("✓ Todo al día. No hay períodos nuevos para descargar.")
    else:
        log.info(f"Períodos a descargar: {[p['periodo'] for p in catalogo]}")

    nuevas = []
    for item in catalogo:
        exito = descargar_periodo(item["periodo"], item["zip_url"])
        nuevas.append({
            "periodo":        item["periodo"],
            "zip_url":        item["zip_url"],
            "fecha_descarga": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "estado":         "ok" if exito else "error",
        })

    if nuevas:
        nuevas_df = pd.DataFrame(nuevas)
        log_df = log_df[~log_df["periodo"].isin(set(nuevas_df["periodo"]))]
        log_df = pd.concat([log_df, nuevas_df], ignore_index=True)
        log_df = log_df.sort_values("periodo", ignore_index=True)
        guardar_log(log_df)

    if not args.no_consolidar:
        consolidar()

    ok      = (log_df["estado"] == "ok").sum()
    errores = (log_df["estado"] == "error").sum()
    log.info("─" * 60)
    log.info(f"Finalizado.  Histórico: {ok} períodos OK  |  {errores} con error")
    if errores:
        for _, r in log_df[log_df["estado"] == "error"].iterrows():
            log.warning(f"  ERROR → {r['periodo']}")


if __name__ == "__main__":
    main()
