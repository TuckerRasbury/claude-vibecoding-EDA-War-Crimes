"""
src/ingest.py
=============
Downloads and stores raw data from:
  1. ACLED API  — armed conflict events (requires free API key)
  2. HRDAG      — statistical datasets on documented killings (public downloads)

Usage:
    python src/ingest.py               # pull everything
    python src/ingest.py --source acled
    python src/ingest.py --source hrdag

Outputs:
    data/raw/acled_raw.csv             — ACLED event records
    data/raw/hrdag_colombia.csv        — HRDAG Colombia dataset
    data/raw/hrdag_guatemala.csv       — HRDAG Guatemala dataset
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
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
ACLED_BASE_URL = "https://api.acleddata.com/acled/read"

# Event types most associated with war crimes / IHL violations
ACLED_EVENT_TYPES = [
    "Battles",
    "Explosions/Remote violence",
    "Violence against civilians",
    "Sexual violence",
]

# How many years of ACLED data to pull
ACLED_YEARS_BACK = 5

# HRDAG public dataset URLs
# These are direct links to CSV files hosted on HRDAG's GitHub/public repo.
# If a URL changes, set HRDAG_LOCAL_PATH in .env to point to a local file instead.
HRDAG_SOURCES = {
    "colombia": (
        "https://raw.githubusercontent.com/HRDAG/CO-decesos/main/export/co-decesos.csv",
        RAW_DIR / "hrdag_colombia.csv",
    ),
    "guatemala": (
        # HRDAG Guatemala truth commission data (REMHI / CEH era estimates)
        # Fallback: HRDAG's public data page at https://hrdag.org/guatemala-data/
        "https://hrdag.org/wp-content/uploads/2012/01/HRDAG-Guatemala-Data.zip",
        RAW_DIR / "hrdag_guatemala.zip",
    ),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_acled_credentials() -> tuple[str, str]:
    """Load and validate ACLED credentials from .env. Exit with message if missing."""
    load_dotenv()
    email = os.getenv("ACLED_EMAIL", "").strip()
    key = os.getenv("ACLED_API_KEY", "").strip()
    if not email or not key:
        log.error(
            "\n"
            "  ACLED credentials not found.\n"
            "  1. Register for a free API key at: https://acleddata.com/register/\n"
            "  2. Copy .env.example -> .env\n"
            "  3. Fill in ACLED_EMAIL and ACLED_API_KEY\n"
        )
        sys.exit(1)
    return email, key


def _acled_request(params: dict, session: requests.Session) -> list[dict]:
    """Single paginated ACLED API request with basic retry logic."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = session.get(ACLED_BASE_URL, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if data.get("success") is False:
                log.error("ACLED API error: %s", data.get("error", "unknown"))
                return []
            return data.get("data", [])
        except requests.RequestException as exc:
            wait = 2 ** attempt
            log.warning("Request failed (attempt %d/%d): %s — retrying in %ds",
                        attempt + 1, max_retries, exc, wait)
            time.sleep(wait)
    log.error("All retries exhausted for ACLED request.")
    return []


# ---------------------------------------------------------------------------
# ACLED ingestion
# ---------------------------------------------------------------------------

def ingest_acled() -> pd.DataFrame:
    """
    Pull the last ACLED_YEARS_BACK years of events filtered to war-crimes-relevant
    event types. Paginates automatically (ACLED max page size = 5000).

    Returns a DataFrame and saves to data/raw/acled_raw.csv.
    """
    email, key = _check_acled_credentials()

    start_date = (datetime.utcnow() - timedelta(days=365 * ACLED_YEARS_BACK)).strftime("%Y-%m-%d")
    end_date = datetime.utcnow().strftime("%Y-%m-%d")

    log.info("Pulling ACLED data from %s to %s", start_date, end_date)
    log.info("Event types: %s", ACLED_EVENT_TYPES)

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    all_records: list[dict] = []
    page_size = 5000

    # Pull each event type separately to keep per-type logging clear
    for event_type in ACLED_EVENT_TYPES:
        log.info("  Fetching: %s", event_type)
        page = 1
        type_count = 0

        while True:
            params = {
                "email": email,
                "key": key,
                "event_type": event_type,
                "event_date": f"{start_date}|{end_date}",
                "event_date_where": "BETWEEN",
                "fields": (
                    "event_id_cnty|event_date|year|time_precision|event_type|"
                    "sub_event_type|actor1|assoc_actor_1|inter1|actor2|assoc_actor_2|"
                    "inter2|interaction|country|region|admin1|admin2|admin3|location|"
                    "latitude|longitude|geo_precision|source|notes|fatalities"
                ),
                "limit": page_size,
                "page": page,
            }

            records = _acled_request(params, session)
            if not records:
                break

            all_records.extend(records)
            type_count += len(records)

            if len(records) < page_size:
                # Last page
                break
            page += 1

        log.info("    -> %d records", type_count)

    if not all_records:
        log.warning("No ACLED records retrieved. Check credentials and network.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    # Type coercion
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["fatalities"] = pd.to_numeric(df["fatalities"], errors="coerce").fillna(0).astype(int)

    out_path = RAW_DIR / "acled_raw.csv"
    df.to_csv(out_path, index=False)
    log.info("Saved %d ACLED records to %s", len(df), out_path)
    return df


# ---------------------------------------------------------------------------
# HRDAG ingestion
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: Path, label: str = "") -> bool:
    """Stream-download a file with a progress bar. Returns True on success."""
    log.info("Downloading %s -> %s", label or url, dest.name)
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
        ) as bar:
            for chunk in resp.iter_content(chunk_size=8192):
                fh.write(chunk)
                bar.update(len(chunk))
        return True
    except requests.RequestException as exc:
        log.error("Download failed: %s", exc)
        return False


def ingest_hrdag_colombia() -> pd.DataFrame:
    """
    Download HRDAG Colombia decesos (deaths) dataset.

    The canonical source is the HRDAG CO-decesos repository on GitHub.
    If the URL is unavailable, instructions for manual download are printed.

    Returns a DataFrame and saves to data/raw/hrdag_colombia.csv.
    """
    url, dest = HRDAG_SOURCES["colombia"]

    # Allow override via .env
    local_path = os.getenv("HRDAG_LOCAL_PATH", "").strip()
    if local_path and Path(local_path).exists():
        log.info("Using local HRDAG file: %s", local_path)
        df = pd.read_csv(local_path)
        df.to_csv(dest, index=False)
        return df

    success = _download_file(url, dest, "HRDAG Colombia")
    if not success:
        _hrdag_fallback_instructions("colombia")
        return pd.DataFrame()

    try:
        df = pd.read_csv(dest)
        log.info("Loaded HRDAG Colombia: %d rows, %d columns", *df.shape)

        # Normalize common column names if present
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Save back with normalized headers
        df.to_csv(dest, index=False)
        return df
    except Exception as exc:
        log.error("Failed to parse HRDAG Colombia CSV: %s", exc)
        _hrdag_fallback_instructions("colombia")
        return pd.DataFrame()


def ingest_hrdag_guatemala() -> pd.DataFrame:
    """
    Download HRDAG Guatemala dataset (ZIP containing CSV exports).

    The Guatemala dataset covers CEH-era (1960–1996) documented killings and
    disappearances. If the URL is unavailable, manual download instructions
    are printed.

    Returns a DataFrame and saves to data/raw/hrdag_guatemala.csv.
    """
    import io
    import zipfile

    url, dest_zip = HRDAG_SOURCES["guatemala"]
    dest_csv = RAW_DIR / "hrdag_guatemala.csv"

    success = _download_file(url, dest_zip, "HRDAG Guatemala")
    if not success:
        _hrdag_fallback_instructions("guatemala")
        return pd.DataFrame()

    # Extract first CSV from the zip
    try:
        with zipfile.ZipFile(dest_zip) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                log.warning("No CSV found in HRDAG Guatemala ZIP.")
                _hrdag_fallback_instructions("guatemala")
                return pd.DataFrame()

            log.info("Extracting %s from ZIP", csv_names[0])
            with zf.open(csv_names[0]) as f:
                df = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8", errors="replace"))

        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        df.to_csv(dest_csv, index=False)
        log.info("Saved HRDAG Guatemala: %d rows to %s", len(df), dest_csv)
        return df

    except Exception as exc:
        log.error("Failed to process HRDAG Guatemala ZIP: %s", exc)
        _hrdag_fallback_instructions("guatemala")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# UNHCR ingestion
# ---------------------------------------------------------------------------

def ingest_unhcr() -> pd.DataFrame:
    """
    Download UNHCR annual displacement data (refugees + IDPs) — no API key required.

    UNHCR publishes population statistics as a public CSV. We pull the full
    dataset and filter to refugees and internally displaced persons (IDPs),
    which are the displacement categories most directly linked to armed conflict
    and war-crimes-adjacent violence.

    Returns a DataFrame and saves to data/raw/unhcr_displacement.csv.

    If the download fails, prints manual download instructions.
    """
    # UNHCR API v1 — public, no key required
    # Returns annual population figures by origin country, asylum country, and
    # population type (REF=refugees, IDP=internally displaced, etc.)
    UNHCR_URL = (
        "https://api.unhcr.org/population/v1/population/"
        "?limit=10000&dataset=population&displayType=totals"
        "&columns[]=refugees&columns[]=idps&columns[]=year"
        "&columns[]=iso3&columns[]=coa_iso3"
        "&yearFrom=2019&yearTo=2024"
    )

    dest = RAW_DIR / "unhcr_displacement.csv"
    log.info("Downloading UNHCR displacement data...")

    try:
        resp = requests.get(UNHCR_URL, timeout=60, headers={"Accept": "application/json"})
        resp.raise_for_status()
        payload = resp.json()

        # UNHCR API wraps results in {"items": [...]}
        items = payload.get("items", payload.get("data", []))
        if not items:
            raise ValueError("Empty response from UNHCR API")

        df = pd.DataFrame(items)
        df.columns = [c.strip().lower() for c in df.columns]

        # Normalize column names — API returns vary slightly by version
        rename_map = {
            "refugees_under_unhcrs_mandate": "refugees",
            "asylum-seekers": "asylum_seekers",
            "internally_displaced_persons__idps_": "idps",
            "coo": "iso3_origin",
            "coa": "iso3_asylum",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        for col in ("refugees", "idps", "year"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        df.to_csv(dest, index=False)
        log.info("Saved UNHCR displacement: %d rows to %s", len(df), dest)
        return df

    except Exception as exc:
        log.warning("UNHCR API download failed: %s", exc)
        log.warning(
            "\n"
            "  UNHCR download failed.\n"
            "  Manual steps:\n"
            "    1. Visit: https://www.unhcr.org/refugee-statistics/download/\n"
            "    2. Select: Population statistics → Refugees + IDPs, 2019–2024\n"
            "    3. Export as CSV and save to: data/raw/unhcr_displacement.csv\n"
        )
        return pd.DataFrame()


def _hrdag_fallback_instructions(dataset: str) -> None:
    """Print human-readable instructions for manual HRDAG download."""
    instructions = {
        "colombia": (
            "\n"
            "  HRDAG Colombia download failed.\n"
            "  Manual steps:\n"
            "    1. Visit: https://hrdag.org/colombia-data/\n"
            "    2. Download the CSV dataset.\n"
            "    3. Save it to: data/raw/hrdag_colombia.csv\n"
            "    4. Or set HRDAG_LOCAL_PATH=/path/to/file in your .env\n"
        ),
        "guatemala": (
            "\n"
            "  HRDAG Guatemala download failed.\n"
            "  Manual steps:\n"
            "    1. Visit: https://hrdag.org/guatemala-data/\n"
            "    2. Download the dataset ZIP or CSV.\n"
            "    3. Save CSV as: data/raw/hrdag_guatemala.csv\n"
            "    4. Or set HRDAG_LOCAL_PATH=/path/to/file in your .env\n"
        ),
    }
    log.warning(instructions.get(dataset, "Manual download required."))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ingest ACLED and HRDAG data into data/raw/."
    )
    parser.add_argument(
        "--source",
        choices=["acled", "hrdag", "unhcr", "all"],
        default="all",
        help="Which data source to pull (default: all)",
    )
    args = parser.parse_args()

    if args.source in ("acled", "all"):
        log.info("=== ACLED ingestion ===")
        df_acled = ingest_acled()
        if not df_acled.empty:
            log.info("ACLED: %d total records, %d unique countries",
                     len(df_acled), df_acled["country"].nunique() if "country" in df_acled.columns else 0)

    if args.source in ("hrdag", "all"):
        log.info("=== HRDAG ingestion ===")
        df_col = ingest_hrdag_colombia()
        df_guat = ingest_hrdag_guatemala()
        log.info("HRDAG Colombia: %d rows", len(df_col))
        log.info("HRDAG Guatemala: %d rows", len(df_guat))

    if args.source in ("unhcr", "all"):
        log.info("=== UNHCR ingestion ===")
        df_unhcr = ingest_unhcr()
        log.info("UNHCR: %d rows", len(df_unhcr))

    log.info("Done. Raw files are in data/raw/")


if __name__ == "__main__":
    main()
