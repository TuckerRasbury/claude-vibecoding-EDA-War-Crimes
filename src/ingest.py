"""
src/ingest.py
=============
Downloads and stores raw data from:
  1. UCDP GED     — georeferenced conflict events 1989–2024 (public Zenodo download)
  2. HRDAG        — statistical datasets on documented killings (public downloads)
  3. UNHCR        — annual displacement/refugee data (public API)

No API keys or credentials required — all sources are freely accessible.

Usage:
    python src/ingest.py               # pull everything
    python src/ingest.py --source ucdp
    python src/ingest.py --source hrdag
    python src/ingest.py --source unhcr

Outputs:
    data/raw/ucdp_ged.csv             — UCDP GED conflict events (normalised schema)
    data/raw/hrdag_colombia.csv       — HRDAG Colombia dataset
    data/raw/hrdag_guatemala.csv      — HRDAG Guatemala dataset
    data/raw/unhcr_displacement.csv   — UNHCR displacement data
"""

import argparse
import io
import logging
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT    = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# UCDP GED 25.1 — events 1989-01-01 through 2024-12-31
# Hosted on Zenodo (open access, no auth required)
UCDP_GED_VERSION = "25.1"
UCDP_GED_ZIP_URL = (
    "https://zenodo.org/records/15420680/files/GEDEvent_v25_1.csv.zip"
)
UCDP_GED_CSV_NAME = "GEDEvent_v25_1.csv"    # filename inside the ZIP

# Violence type labels (UCDP type_of_violence codes 1–3)
UCDP_VIOLENCE_LABELS = {
    1: "State-based conflict",
    2: "Non-state conflict",
    3: "One-sided violence",
}

# HRDAG public dataset URLs
HRDAG_SOURCES = {
    "colombia": (
        "https://raw.githubusercontent.com/HRDAG/CO-decesos/main/export/co-decesos.csv",
        RAW_DIR / "hrdag_colombia.csv",
    ),
    "guatemala": (
        "https://hrdag.org/wp-content/uploads/2012/01/HRDAG-Guatemala-Data.zip",
        RAW_DIR / "hrdag_guatemala.zip",
    ),
}


# ---------------------------------------------------------------------------
# Generic download helper
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: Path, label: str = "") -> bool:
    """Stream-download a file with a progress bar. Returns True on success."""
    log.info("Downloading %s → %s", label or url, dest.name)
    try:
        resp = requests.get(url, stream=True, timeout=300)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
        ) as bar:
            for chunk in resp.iter_content(chunk_size=65_536):
                fh.write(chunk)
                bar.update(len(chunk))
        return True
    except requests.RequestException as exc:
        log.error("Download failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# UCDP GED ingestion
# ---------------------------------------------------------------------------

def ingest_ucdp() -> pd.DataFrame:
    """
    Download UCDP Georeferenced Event Dataset (GED) v25.1 from Zenodo and
    normalise it to the project's standard column schema.

    No authentication required — Zenodo is open access.

    Column mapping (UCDP → project schema):
        date_start          → event_date
        type_of_violence    → event_type  (human-readable label)
        side_a              → actor1
        side_b              → actor2
        deaths_a + deaths_b + deaths_civilians + deaths_unknown → fatalities
        country, latitude, longitude, admin1, conflict_name kept as-is

    Returns a DataFrame and saves to data/raw/ucdp_ged.csv.
    """
    dest_zip = RAW_DIR / "ucdp_ged.zip"
    dest_csv = RAW_DIR / "ucdp_ged.csv"

    if dest_csv.exists():
        log.info("UCDP GED already downloaded; loading from %s", dest_csv)
        return pd.read_csv(dest_csv, low_memory=False)

    # 1. Download ZIP from Zenodo
    ok = _download_file(UCDP_GED_ZIP_URL, dest_zip, f"UCDP GED {UCDP_GED_VERSION}")
    if not ok:
        log.error(
            "\n  UCDP GED download failed.\n"
            "  Manual steps:\n"
            "    1. Visit: https://ucdp.uu.se/downloads/\n"
            "    2. Download the GED %s CSV ZIP\n"
            "    3. Extract the CSV and save it to: data/raw/ucdp_ged.csv\n",
            UCDP_GED_VERSION,
        )
        return pd.DataFrame()

    # 2. Extract CSV from ZIP
    log.info("Extracting %s from ZIP…", UCDP_GED_CSV_NAME)
    try:
        with zipfile.ZipFile(dest_zip) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                log.error("No CSV found inside UCDP GED ZIP.")
                return pd.DataFrame()
            target = UCDP_GED_CSV_NAME if UCDP_GED_CSV_NAME in csv_names else csv_names[0]
            log.info("  Using: %s", target)
            with zf.open(target) as f:
                raw = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8", errors="replace"),
                                  low_memory=False)
    except Exception as exc:
        log.error("Failed to read UCDP GED ZIP: %s", exc)
        return pd.DataFrame()

    # 3. Normalise to project schema
    df = _normalise_ucdp(raw)

    # 4. Save
    df.to_csv(dest_csv, index=False)
    log.info("Saved UCDP GED: %d events (%d countries) → %s",
             len(df), df["country"].nunique(), dest_csv)

    # Clean up ZIP to save disk space (~200 MB)
    dest_zip.unlink(missing_ok=True)
    return df


def _normalise_ucdp(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Map UCDP GED column names to the project's standard schema.

    Standard schema columns (used by visualize.py chart functions):
        event_date   — datetime
        event_type   — str  (human-readable violence category)
        actor1       — str  (side A, typically government or primary actor)
        actor2       — str  (side B, typically opposition / non-state actor)
        fatalities   — int  (sum of all reported deaths)
        country      — str
        latitude     — float
        longitude    — float
        admin1       — str  (province / state)
        conflict_name— str
        year         — int
        source_article — str  (source references)
        number_of_sources — int
    """
    df = raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # event_date
    date_col = next((c for c in ("date_start", "date_end") if c in df.columns), None)
    if date_col:
        df["event_date"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        df["event_date"] = pd.NaT

    # event_type from type_of_violence
    if "type_of_violence" in df.columns:
        df["event_type"] = (
            df["type_of_violence"]
            .map(UCDP_VIOLENCE_LABELS)
            .fillna("Unknown")
        )
    else:
        df["event_type"] = "Unknown"

    # actor names
    df["actor1"] = df.get("side_a", pd.Series("", index=df.index)).fillna("Unknown")
    df["actor2"] = df.get("side_b", pd.Series("", index=df.index)).fillna("Unknown")

    # fatalities — sum all death columns
    death_cols = [c for c in ("deaths_a", "deaths_b", "deaths_civilians", "deaths_unknown")
                  if c in df.columns]
    if death_cols:
        df["fatalities"] = (
            df[death_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .sum(axis=1)
            .astype(int)
        )
    else:
        df["fatalities"] = 0

    # pass-through columns (rename where needed)
    passthrough = {
        "country":            "country",
        "latitude":           "latitude",
        "longitude":          "longitude",
        "admin1":             "admin1",
        "conflict_name":      "conflict_name",
        "year":               "year",
        "source_article":     "source_article",
        "number_of_sources":  "number_of_sources",
        "deaths_civilians":   "deaths_civilians",
        "type_of_violence":   "type_of_violence",
        "dyad_name":          "dyad_name",
        "id":                 "event_id",
    }
    for src, tgt in passthrough.items():
        if src in df.columns and tgt not in df.columns:
            df[tgt] = df[src]

    # numeric coercions
    for col in ("latitude", "longitude", "year", "fatalities", "number_of_sources",
                "deaths_civilians"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # drop rows with no location
    df = df.dropna(subset=["latitude", "longitude"])

    return df


# ---------------------------------------------------------------------------
# HRDAG ingestion
# ---------------------------------------------------------------------------

def ingest_hrdag_colombia() -> pd.DataFrame:
    """
    Download HRDAG Colombia decesos (deaths) dataset.

    The canonical source is the HRDAG CO-decesos repository on GitHub.
    Returns a DataFrame and saves to data/raw/hrdag_colombia.csv.
    """
    url, dest = HRDAG_SOURCES["colombia"]

    if dest.exists():
        log.info("HRDAG Colombia already downloaded; loading from %s", dest)
        return pd.read_csv(dest, low_memory=False)

    ok = _download_file(url, dest, "HRDAG Colombia")
    if not ok:
        _hrdag_fallback("colombia")
        return pd.DataFrame()

    try:
        df = pd.read_csv(dest, low_memory=False)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        df.to_csv(dest, index=False)
        log.info("Saved HRDAG Colombia: %d rows, %d columns", *df.shape)
        return df
    except Exception as exc:
        log.error("Failed to parse HRDAG Colombia CSV: %s", exc)
        _hrdag_fallback("colombia")
        return pd.DataFrame()


def ingest_hrdag_guatemala() -> pd.DataFrame:
    """
    Download HRDAG Guatemala dataset (ZIP containing CSV exports).

    Covers CEH-era (1960–1996) documented killings and disappearances.
    Returns a DataFrame and saves to data/raw/hrdag_guatemala.csv.
    """
    url, dest_zip = HRDAG_SOURCES["guatemala"]
    dest_csv = RAW_DIR / "hrdag_guatemala.csv"

    if dest_csv.exists():
        log.info("HRDAG Guatemala already downloaded; loading from %s", dest_csv)
        return pd.read_csv(dest_csv, low_memory=False)

    ok = _download_file(url, dest_zip, "HRDAG Guatemala")
    if not ok:
        _hrdag_fallback("guatemala")
        return pd.DataFrame()

    try:
        with zipfile.ZipFile(dest_zip) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                log.warning("No CSV found in HRDAG Guatemala ZIP.")
                _hrdag_fallback("guatemala")
                return pd.DataFrame()
            log.info("Extracting %s from ZIP", csv_names[0])
            with zf.open(csv_names[0]) as f:
                df = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8", errors="replace"))

        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        df.to_csv(dest_csv, index=False)
        log.info("Saved HRDAG Guatemala: %d rows → %s", len(df), dest_csv)
        return df

    except Exception as exc:
        log.error("Failed to process HRDAG Guatemala ZIP: %s", exc)
        _hrdag_fallback("guatemala")
        return pd.DataFrame()


def _hrdag_fallback(dataset: str) -> None:
    """Print human-readable instructions for manual HRDAG download."""
    msgs = {
        "colombia": (
            "\n  HRDAG Colombia download failed.\n"
            "  Manual steps:\n"
            "    1. Visit: https://hrdag.org/colombia-data/\n"
            "    2. Download the CSV dataset.\n"
            "    3. Save it to: data/raw/hrdag_colombia.csv\n"
        ),
        "guatemala": (
            "\n  HRDAG Guatemala download failed.\n"
            "  Manual steps:\n"
            "    1. Visit: https://hrdag.org/guatemala-data/\n"
            "    2. Download the dataset ZIP or CSV.\n"
            "    3. Save CSV as: data/raw/hrdag_guatemala.csv\n"
        ),
    }
    log.warning(msgs.get(dataset, "Manual download required."))


# ---------------------------------------------------------------------------
# UNHCR ingestion
# ---------------------------------------------------------------------------

def ingest_unhcr() -> pd.DataFrame:
    """
    Download UNHCR annual displacement data (refugees + IDPs).

    UNHCR's population API is public — no key required.
    Returns a DataFrame and saves to data/raw/unhcr_displacement.csv.
    """
    dest = RAW_DIR / "unhcr_displacement.csv"

    if dest.exists():
        log.info("UNHCR data already downloaded; loading from %s", dest)
        return pd.read_csv(dest, low_memory=False)

    UNHCR_URL = (
        "https://api.unhcr.org/population/v1/population/"
        "?limit=10000&dataset=population&displayType=totals"
        "&columns[]=refugees&columns[]=idps&columns[]=year"
        "&columns[]=iso3&columns[]=coa_iso3"
        "&yearFrom=2019&yearTo=2024"
    )

    log.info("Downloading UNHCR displacement data…")
    try:
        resp = requests.get(UNHCR_URL, timeout=60, headers={"Accept": "application/json"})
        resp.raise_for_status()
        payload = resp.json()
        items = payload.get("items", payload.get("data", []))
        if not items:
            raise ValueError("Empty response from UNHCR API")

        df = pd.DataFrame(items)
        df.columns = [c.strip().lower() for c in df.columns]

        rename_map = {
            "refugees_under_unhcrs_mandate":        "refugees",
            "asylum-seekers":                       "asylum_seekers",
            "internally_displaced_persons__idps_":  "idps",
            "coo":                                  "iso3_origin",
            "coa":                                  "iso3_asylum",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        for col in ("refugees", "idps", "year"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        df.to_csv(dest, index=False)
        log.info("Saved UNHCR displacement: %d rows → %s", len(df), dest)
        return df

    except Exception as exc:
        log.warning("UNHCR API download failed: %s", exc)
        log.warning(
            "\n  UNHCR download failed.\n"
            "  Manual steps:\n"
            "    1. Visit: https://www.unhcr.org/refugee-statistics/download/\n"
            "    2. Select: Population statistics → Refugees + IDPs, 2019–2024\n"
            "    3. Export as CSV and save to: data/raw/unhcr_displacement.csv\n"
        )
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ingest UCDP, HRDAG, and UNHCR data into data/raw/."
    )
    parser.add_argument(
        "--source",
        choices=["ucdp", "hrdag", "unhcr", "all"],
        default="all",
        help="Which data source to pull (default: all)",
    )
    args = parser.parse_args()

    if args.source in ("ucdp", "all"):
        log.info("=== UCDP GED ingestion ===")
        df_ucdp = ingest_ucdp()
        if not df_ucdp.empty:
            log.info("UCDP GED: %d events, %d countries, years %d–%d",
                     len(df_ucdp),
                     df_ucdp["country"].nunique() if "country" in df_ucdp.columns else 0,
                     int(df_ucdp["year"].min()) if "year" in df_ucdp.columns else 0,
                     int(df_ucdp["year"].max()) if "year" in df_ucdp.columns else 0)

    if args.source in ("hrdag", "all"):
        log.info("=== HRDAG ingestion ===")
        df_col  = ingest_hrdag_colombia()
        df_guat = ingest_hrdag_guatemala()
        log.info("HRDAG Colombia:  %d rows", len(df_col))
        log.info("HRDAG Guatemala: %d rows", len(df_guat))

    if args.source in ("unhcr", "all"):
        log.info("=== UNHCR ingestion ===")
        df_unhcr = ingest_unhcr()
        log.info("UNHCR: %d rows", len(df_unhcr))

    log.info("Done. Raw files are in data/raw/")


if __name__ == "__main__":
    main()
