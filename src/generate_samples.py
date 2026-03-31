"""
src/generate_samples.py
=======================
Generates synthetic ACLED-format data and exports PNG chart previews
to assets/images/ for embedding in README.md.

No API key required. Run standalone:
    python src/generate_samples.py

Produces 16 PNG files in assets/images/:
  - Calls existing matplotlib chart functions (with OUT_DIR patched to assets/images/)
  - Builds matplotlib equivalents of Plotly/Folium charts for static export

The synthetic data uses seeded RNG for reproducibility — regenerating
always produces the same images.
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths — patch OUT_DIR before importing visualize
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT / "assets" / "images"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# Patch OUT_DIR *before* any chart functions are called so matplotlib
# functions (which use OUT_DIR at call time) save to assets/images/
sys.path.insert(0, str(ROOT))
import src.visualize as viz
viz.OUT_DIR = ASSETS_DIR

from src.visualize import (  # noqa: E402 — must come after patch
    chart_yoy_violence_civilians,
    chart_top20_actors,
    chart_data_completeness,
    chart_fatality_analysis,
    chart_source_analysis,
    chart_escalation_phases,
    _classify_actor_type,
    PALETTE,
    ICC_SITUATION_COUNTRIES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", font_scale=1.0)

# ---------------------------------------------------------------------------
# Synthetic data configuration
# ---------------------------------------------------------------------------

# country → (region, lat_center, lon_center, relative_weight)
COUNTRY_CONFIG = {
    "Syria":                        ("Middle East",      34.8,  38.5, 10),
    "Yemen":                        ("Middle East",      15.5,  48.5,  9),
    "Iraq":                         ("Middle East",      33.0,  43.7,  6),
    "Democratic Republic of Congo": ("Middle Africa",    -4.0,  23.0, 10),
    "Central African Republic":     ("Middle Africa",     6.5,  20.5,  5),
    "Cameroon":                     ("Middle Africa",     5.0,  12.5,  4),
    "Chad":                         ("Middle Africa",    15.0,  19.0,  4),
    "Ethiopia":                     ("Eastern Africa",    9.0,  40.5,  9),
    "Somalia":                      ("Eastern Africa",    5.0,  46.0,  8),
    "South Sudan":                  ("Eastern Africa",    7.0,  31.5,  6),
    "Sudan":                        ("Northern Africa",  15.5,  32.5,  7),
    "Libya":                        ("Northern Africa",  27.0,  17.0,  5),
    "Nigeria":                      ("Western Africa",    9.0,   8.5,  9),
    "Mali":                         ("Western Africa",   17.0,  -2.0,  7),
    "Burkina Faso":                 ("Western Africa",   12.0,  -1.5,  6),
    "Niger":                        ("Western Africa",   17.0,   8.5,  4),
    "Myanmar":                      ("Southeast Asia",   19.0,  96.5,  8),
    "Philippines":                  ("Southeast Asia",   13.0, 122.0,  4),
    "Afghanistan":                  ("South Asia",       33.0,  67.5,  7),
    "Pakistan":                     ("South Asia",       30.5,  69.0,  5),
    "Ukraine":                      ("Eastern Europe",   49.0,  31.0,  9),
    "Colombia":                     ("Americas",          4.5, -74.0,  5),
    "Mexico":                       ("Americas",         23.0,-102.5,  4),
    "Mozambique":                   ("Southern Africa", -18.5,  35.5,  4),
    "India":                        ("South Asia",       20.0,  77.0,  3),
}

# inter1 code → actor name pool
ACTOR_POOLS = {
    "1": ["Syrian Arab Army", "Ukrainian Armed Forces", "Ethiopian National Defence Force",
          "Myanmar Tatmadaw", "Nigerian Military", "Afghan National Army",
          "Iraqi Security Forces", "Sudanese Armed Forces", "Yemeni Government Forces"],
    "2": ["ISIS", "Al-Shabaab", "FARC dissidents", "Wagner Group", "Hamas",
          "Taliban", "Houthi movement", "JNIM", "Boko Haram", "Al-Qaeda in Mali"],
    "3": ["Popular Mobilization Forces", "Tigray Peoples Liberation Front",
          "Arakan Army", "National Resistance Army Uganda", "Seleka coalition"],
    "4": ["Fulani militia", "Dozo hunters", "Oromo militia", "Karen militia",
          "Interahamwe remnants"],
    "7": ["Civilians (unspecified)", "Unknown armed group"],
    "8": ["UN Peacekeepers MINUSCA", "AMISOM", "French Barkhane Forces",
          "US Special Operations Forces"],
}

SUB_EVENTS = {
    "Battles":                    ["Armed clash", "Government regains territory",
                                   "Non-state actor overtakes territory"],
    "Explosions/Remote violence": ["Air/drone strike", "Shelling/artillery/missile attack",
                                   "Remote explosive/landmine/IED", "Suicide bomb"],
    "Violence against civilians": ["Attack", "Abduction/forced disappearance",
                                   "Looting/property destruction"],
    "Sexual violence":            ["Rape", "Sexual assault", "Forced marriage",
                                   "Sexual slavery"],
}

SOURCES = [
    "Reuters", "AP", "BBC", "Al Jazeera", "UN OCHA", "Human Rights Watch",
    "Amnesty International", "ACLED", "France 24", "AFP", "Local newspaper",
    "Radio Free Europe", "The Guardian", "Middle East Eye",
    "NGO situation report", "Twitter/social media", "Government statement",
]
SOURCE_WEIGHTS = np.array([10, 9, 8, 8, 7, 6, 5, 5, 4, 4, 4, 3, 3, 3, 3, 2, 2], dtype=float)
SOURCE_WEIGHTS /= SOURCE_WEIGHTS.sum()


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def generate_synthetic_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic ACLED-format DataFrame for chart previews.
    Seeded for reproducibility — same seed always produces the same charts.
    """
    rng = np.random.default_rng(seed)

    countries = list(COUNTRY_CONFIG.keys())
    country_weights = np.array([COUNTRY_CONFIG[c][3] for c in countries], dtype=float)
    country_weights /= country_weights.sum()
    chosen_countries = rng.choice(countries, size=n, p=country_weights)

    event_types = list(SUB_EVENTS.keys())
    event_type_weights = np.array([0.40, 0.25, 0.30, 0.05])
    chosen_events = rng.choice(event_types, size=n, p=event_type_weights)

    # Dates spanning 2020–2024
    start_ns = pd.Timestamp("2020-01-01").value
    end_ns   = pd.Timestamp("2024-12-31").value
    dates = pd.to_datetime(rng.integers(start_ns, end_ns, size=n))

    inter1_codes = rng.choice(
        ["1", "2", "3", "4", "7", "8"], size=n,
        p=[0.28, 0.30, 0.18, 0.12, 0.08, 0.04],
    )
    inter2_codes = rng.choice(
        ["1", "2", "3", "4", "7", "8"], size=n,
        p=[0.20, 0.30, 0.20, 0.15, 0.10, 0.05],
    )
    actor1 = np.array([rng.choice(ACTOR_POOLS[c]) for c in inter1_codes])
    actor2 = np.array([rng.choice(ACTOR_POOLS[c]) for c in inter2_codes])

    sub_events = np.array([rng.choice(SUB_EVENTS[e]) for e in chosen_events])

    # Coordinates: country center + small jitter
    lats = np.array([COUNTRY_CONFIG[c][1] for c in chosen_countries])
    lons = np.array([COUNTRY_CONFIG[c][2] for c in chosen_countries])
    lats += rng.normal(0, 1.5, size=n)
    lons += rng.normal(0, 1.5, size=n)

    # Fatalities: log-normal, mostly 0–5, occasional high values
    fatalities = np.clip(
        rng.lognormal(mean=0.4, sigma=1.6, size=n), 0, 600
    ).astype(int)

    sources = rng.choice(SOURCES, size=n, p=SOURCE_WEIGHTS)
    regions = [COUNTRY_CONFIG[c][0] for c in chosen_countries]
    admin1  = [f"{c} — Province {rng.integers(1, 9)}" for c in chosen_countries]

    df = pd.DataFrame({
        "event_id_cnty":  [f"SMP{i:06d}" for i in range(n)],
        "event_date":     dates,
        "year":           dates.year,
        "time_precision": rng.integers(1, 4, size=n),
        "event_type":     chosen_events,
        "sub_event_type": sub_events,
        "actor1":         actor1,
        "assoc_actor_1":  np.where(rng.random(n) > 0.7, actor2, ""),
        "inter1":         inter1_codes,
        "actor2":         actor2,
        "assoc_actor_2":  "",
        "inter2":         inter2_codes,
        "interaction":    (inter1_codes.astype(int) * 10
                           + inter2_codes.astype(int)).astype(str),
        "country":        chosen_countries,
        "region":         regions,
        "admin1":         admin1,
        "admin2":         "",
        "admin3":         "",
        "location":       [f"Location {rng.integers(1, 200)}" for _ in range(n)],
        "latitude":       lats,
        "longitude":      lons,
        "geo_precision":  rng.integers(1, 4, size=n),
        "source":         sources,
        "notes":          "Synthetic sample record — not real data.",
        "fatalities":     fatalities,
    })

    # Derived columns expected by some chart functions
    df["year_month"]  = df["event_date"].dt.to_period("M")
    df["actor_type"]  = df["inter1"].apply(_classify_actor_type)
    df["state_vs_nonstate"] = df["actor_type"].apply(
        lambda x: "State" if x in ("State Military", "State Police") else "Non-State"
    )
    return df


# ---------------------------------------------------------------------------
# Static PNG equivalents for Plotly / Folium charts
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, name: str) -> None:
    path = ASSETS_DIR / name
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s (%.1f KB)", name, path.stat().st_size / 1024)


def make_choropleth_png(df: pd.DataFrame) -> None:
    """Top-30 countries by event count as horizontal bar (static map preview)."""
    counts = df["country"].value_counts().head(30).sort_values()
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(counts.index, counts.values, color="#c0392b", alpha=0.85)
    ax.set_title(
        "Total Events by Country (ACLED)\n"
        "Interactive choropleth world map generated on pipeline run",
        fontsize=12,
    )
    ax.set_xlabel("Events")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    _save(fig, "choropleth_world.png")


def make_cluster_map_png(df: pd.DataFrame) -> None:
    """Lat/lon scatter coloured by event type — static preview of Folium cluster map."""
    d = df.dropna(subset=["latitude", "longitude"])
    fig, ax = plt.subplots(figsize=(14, 7), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    color_map = {
        "Battles":                    "#e74c3c",
        "Explosions/Remote violence": "#e67e22",
        "Violence against civilians": "#9b59b6",
        "Sexual violence":            "#2e86de",
    }
    for etype, color in color_map.items():
        sub = d[d["event_type"] == etype]
        ax.scatter(sub["longitude"], sub["latitude"],
                   c=color, s=3, alpha=0.45, label=etype, rasterized=True)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 75)
    ax.set_title(
        "Event Locations by Type — Interactive Folium cluster map generated on pipeline run",
        color="white", fontsize=11,
    )
    ax.set_xlabel("Longitude", color="#aaa", fontsize=9)
    ax.set_ylabel("Latitude",  color="#aaa", fontsize=9)
    ax.tick_params(colors="#888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.4,
              labelcolor="white", facecolor="#222", markerscale=3)
    plt.tight_layout()
    fig.savefig(str(ASSETS_DIR / "event_cluster_map.png"),
                dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    log.info("Saved event_cluster_map.png")


def make_heatmap_png(df: pd.DataFrame) -> None:
    """Hexbin density of lat/lon — static preview of Folium heatmap."""
    d = df.dropna(subset=["latitude", "longitude"])
    fig, ax = plt.subplots(figsize=(14, 7), facecolor="#0d0d1a")
    ax.set_facecolor("#0d0d1a")
    hb = ax.hexbin(
        d["longitude"], d["latitude"],
        gridsize=55, cmap="YlOrRd", mincnt=1, alpha=0.9,
        extent=(-180, 180, -60, 75),
    )
    cb = plt.colorbar(hb, ax=ax, shrink=0.7)
    cb.set_label("Event Density", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 75)
    ax.set_title(
        "Event Density Heatmap — Interactive Folium heatmap generated on pipeline run",
        color="white", fontsize=11,
    )
    ax.set_xlabel("Longitude", color="#aaa", fontsize=9)
    ax.set_ylabel("Latitude",  color="#aaa", fontsize=9)
    ax.tick_params(colors="#888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    plt.tight_layout()
    fig.savefig(str(ASSETS_DIR / "heatmap_density.png"),
                dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
    plt.close(fig)
    log.info("Saved heatmap_density.png")


def make_monthly_line_png(df: pd.DataFrame) -> None:
    """Monthly event counts by type — static version of the Plotly line chart."""
    df2 = df.copy()
    df2["month"] = df2["event_date"].dt.to_period("M").dt.to_timestamp()
    monthly = df2.groupby(["month", "event_type"]).size().reset_index(name="count")

    fig, ax = plt.subplots(figsize=(13, 5))
    for etype, color in PALETTE.items():
        sub = monthly[monthly["event_type"] == etype]
        ax.plot(sub["month"], sub["count"], label=etype, color=color, linewidth=2)
    ax.set_title(
        "Monthly Events by Type (ACLED)\n"
        "Interactive Plotly version generated on pipeline run",
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Events")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(title="Event Type", fontsize=9)
    plt.tight_layout()
    _save(fig, "monthly_events_by_type.png")


def make_animated_static_png(df: pd.DataFrame) -> None:
    """Most-recent-year country bar — static frame of the animated Plotly choropleth."""
    most_recent = int(df["year"].max())
    year_data = (
        df[df["year"] == most_recent]["country"]
        .value_counts()
        .head(20)
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(year_data.index, year_data.values, color="#e67e22", alpha=0.85)
    ax.set_title(
        f"Events by Country — {most_recent} (static frame)\n"
        "Animated year-over-year choropleth generated on pipeline run",
        fontsize=12,
    )
    ax.set_xlabel("Events")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    _save(fig, "animated_timeseries.png")


def make_actor_region_png(df: pd.DataFrame) -> None:
    """Actor type stacked bar by region — static version of the Plotly chart."""
    pivot = (
        df.groupby(["region", "actor_type"])
        .size()
        .unstack(fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white", width=0.8)
    ax.set_title(
        "Actor Type by Region (ACLED)\n"
        "Interactive Plotly version generated on pipeline run"
    )
    ax.set_xlabel("")
    ax.set_ylabel("Events")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Actor Type", fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    _save(fig, "actor_type_by_region.png")


def make_actor_network_png(df: pd.DataFrame) -> None:
    """Force-directed actor network — matplotlib render of the NetworkX/Plotly chart."""
    import networkx as nx

    df2 = df.dropna(subset=["actor1", "actor2"])
    df2 = df2[df2["actor2"].str.strip().ne("")]

    top_actors = set(df2["actor1"].value_counts().head(30).index)
    event_counts = df2["actor1"].value_counts().to_dict()

    edges = (
        df2[df2["actor1"].isin(top_actors) & df2["actor2"].isin(top_actors)]
        .groupby(["actor1", "actor2"])
        .size()
        .reset_index(name="weight")
    )

    G = nx.Graph()
    for _, row in edges.iterrows():
        G.add_edge(row["actor1"], row["actor2"], weight=row["weight"])
    for a in top_actors:
        if a not in G:
            G.add_node(a)

    type_colors = {
        "State Military": "#e74c3c", "State Police":      "#e74c3c",
        "Rebel Group":    "#2ecc71", "Political Militia":  "#f39c12",
        "Communal Militia":"#e67e22","Rioters/Protesters": "#3498db",
        "Civilians":      "#95a5a6", "External/Other":     "#9b59b6",
        "Unknown":        "#bdc3c7",
    }
    lookup = (
        df2.groupby("actor1")["inter1"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "")
        .apply(_classify_actor_type)
        .to_dict()
    )
    node_colors = [type_colors.get(lookup.get(n, "Unknown"), "#bdc3c7") for n in G.nodes()]
    node_sizes  = [max(120, min(1800, event_counts.get(n, 10) * 8)) for n in G.nodes()]

    pos = nx.spring_layout(
        G, weight="weight",
        k=1.8 / np.sqrt(max(G.number_of_nodes(), 1)),
        iterations=100, seed=42,
    )

    max_weight = edges["weight"].max() if not edges.empty else 1
    fig, ax = plt.subplots(figsize=(13, 10))

    for _, row in edges.iterrows():
        x = [pos[row["actor1"]][0], pos[row["actor2"]][0]]
        y = [pos[row["actor1"]][1], pos[row["actor2"]][1]]
        lw = 0.3 + 2.5 * (row["weight"] / max_weight)
        ax.plot(x, y, color="#aaaaaa", linewidth=lw, alpha=0.35, zorder=1)

    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors, node_size=node_sizes,
                           alpha=0.92, linewidths=0.5, edgecolors="white")

    # Label the top 15 most active nodes
    top_nodes = sorted(G.nodes(), key=lambda n: event_counts.get(n, 0), reverse=True)[:15]
    nx.draw_networkx_labels(
        G, pos, {n: n for n in top_nodes}, ax=ax, font_size=6.5,
    )

    ax.set_title(
        "Actor Interaction Network — Force-directed layout (Fruchterman-Reingold)\n"
        "Node size = event count  |  Color = actor type  |  Edge weight = interaction frequency",
        fontsize=11,
    )
    ax.axis("off")
    for atype, color in list(type_colors.items())[:6]:
        ax.scatter([], [], c=color, label=atype, s=80)
    ax.legend(title="Actor Type", fontsize=8, loc="lower left")
    plt.tight_layout()
    _save(fig, "actor_network.png")


def make_accountability_gap_png(df: pd.DataFrame) -> None:
    """ACLED events vs. ICC situations — static bar version of the Plotly chart."""
    top = df["country"].value_counts().head(25).reset_index()
    top.columns = ["country", "events"]
    top["icc"] = top["country"].apply(
        lambda c: "ICC Situation" if c in ICC_SITUATION_COUNTRIES else "No ICC Situation"
    )
    colors = top["icc"].map({"ICC Situation": "#2471a3", "No ICC Situation": "#e74c3c"})

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(top["country"], top["events"], color=colors, alpha=0.85, edgecolor="white")
    ax.set_title(
        "Accountability Gap: ACLED Events vs. ICC Situations (Top 25 Countries)\n"
        "Red = no ICC situation open  |  Blue = ICC situation exists",
        fontsize=12,
    )
    ax.set_ylabel("Events")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xticklabels(top["country"], rotation=40, ha="right", fontsize=8)
    for label, color in [("ICC Situation", "#2471a3"), ("No ICC Situation", "#e74c3c")]:
        ax.bar([], [], color=color, label=label)
    ax.legend()
    plt.tight_layout()
    _save(fig, "accountability_gap.png")


def make_source_diversity_png(df: pd.DataFrame) -> None:
    """Source diversity by region — static bar version of the Plotly chart."""
    if "source" not in df.columns or "region" not in df.columns:
        return
    diversity = (
        df.groupby("region")
        .agg(events=("source", "count"), unique_sources=("source", "nunique"))
        .reset_index()
        .assign(ratio=lambda x: (x["unique_sources"] / x["events"]).round(4))
        .sort_values("ratio")
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(diversity["region"], diversity["ratio"], color="#2471a3", alpha=0.85)
    ax.set_title(
        "Source Diversity by Region\n"
        "Unique sources / event count — lower = higher undercount risk (Weidmann, 2016)",
        fontsize=11,
    )
    ax.set_xlabel("Diversity Ratio")
    plt.tight_layout()
    _save(fig, "source_diversity.png")


def make_escalation_phases_png(df: pd.DataFrame) -> None:
    """30-day rolling event count — static version of the Plotly escalation chart."""
    df2 = df.dropna(subset=["event_date", "country"]).copy().sort_values("event_date")
    df2 = df2.set_index("event_date")

    global_daily = df2.resample("D").size().rename("events").fillna(0)
    global_roll  = global_daily.rolling(30, min_periods=1).mean()

    top5   = df["country"].value_counts().head(5).index.tolist()
    colors = ["#c0392b", "#e67e22", "#2471a3", "#27ae60", "#8e44ad"]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(global_roll.index, global_roll.values,
            color="#333", linewidth=2, linestyle="--", label="Global (30d avg)", alpha=0.7)
    for country, color in zip(top5, colors):
        csub  = df2[df2["country"] == country].resample("D").size().fillna(0)
        croll = csub.rolling(30, min_periods=1).mean()
        ax.plot(croll.index, croll.values, color=color, linewidth=1.5, label=country)

    ax.set_title(
        "Conflict Escalation Phases — 30-Day Rolling Event Count\n"
        "Interactive Plotly version with spike annotations generated on pipeline run",
        fontsize=11,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Events (30d rolling avg)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, "escalation_phases.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("Generating synthetic data (n=5000, seed=42)...")
    df = generate_synthetic_data(n=5000, seed=42)
    log.info("Generated %d rows across %d countries", len(df), df["country"].nunique())

    # ── Existing matplotlib chart functions (OUT_DIR already patched) ──────
    log.info("=== Matplotlib chart functions ===")
    chart_yoy_violence_civilians(df)
    chart_top20_actors(df)
    chart_data_completeness(df)   # uses hardcoded scores, df not needed
    chart_fatality_analysis(df)
    chart_source_analysis(df)     # saves source_analysis.png + source_diversity.html
    chart_escalation_phases(df)   # saves sub_event_breakdown.png + escalation_phases.html

    # ── Static PNG equivalents for Plotly / Folium charts ─────────────────
    log.info("=== Static previews for interactive charts ===")
    make_choropleth_png(df)
    make_cluster_map_png(df)
    make_heatmap_png(df)
    make_monthly_line_png(df)
    make_animated_static_png(df)
    make_actor_region_png(df)
    make_actor_network_png(df)
    make_accountability_gap_png(df)
    make_source_diversity_png(df)
    make_escalation_phases_png(df)

    pngs = sorted(ASSETS_DIR.glob("*.png"))
    log.info("Done. %d PNGs in %s:", len(pngs), ASSETS_DIR)
    for p in pngs:
        log.info("  %-40s  %.1f KB", p.name, p.stat().st_size / 1024)


if __name__ == "__main__":
    main()
