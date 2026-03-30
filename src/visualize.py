"""
src/visualize.py
================
Generates all charts and maps from processed ACLED data and HRDAG datasets.
All outputs are saved to /outputs.

Usage:
    python src/visualize.py              # generate everything
    python src/visualize.py --chart choropleth
    python src/visualize.py --chart all

Requires data/raw/acled_raw.csv to exist (run src/ingest.py first).

Output files:
    outputs/choropleth_world.html         geographic: events by country
    outputs/event_cluster_map.html        geographic: dot map with tooltips
    outputs/heatmap_density.html          geographic: density heatmap
    outputs/monthly_events_by_type.html   temporal: line chart
    outputs/animated_timeseries.html      temporal: animated year map
    outputs/yoy_violence_civilians.png    temporal: YoY bar chart
    outputs/top20_actors.png              actor: horizontal bar
    outputs/actor_type_by_region.html     actor: stacked bar by region
    outputs/actor_network.html            actor: network diagram
    outputs/accountability_gap.html       accountability: ACLED vs ICC
    outputs/data_completeness.png         accountability: completeness heatmap
    outputs/summary_aggregated.csv        aggregated findings export
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving files

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Branding / style constants
# ---------------------------------------------------------------------------
PALETTE = {
    "Battles": "#c0392b",
    "Explosions/Remote violence": "#e67e22",
    "Violence against civilians": "#8e44ad",
    "Sexual violence": "#2471a3",
}
REGION_COLORS = sns.color_palette("tab10", 12)
sns.set_theme(style="whitegrid", font_scale=1.1)

# ICC countries with open investigations (as of 2024)
# Source: https://www.icc-cpi.int/situations
ICC_SITUATION_COUNTRIES = {
    "Uganda", "Democratic Republic of the Congo", "Sudan", "Central African Republic",
    "Kenya", "Libya", "Ivory Coast", "Mali", "Georgia", "Burundi",
    "Bangladesh/Myanmar", "Myanmar", "Palestine", "Venezuela", "Afghanistan",
    "Philippines", "Nigeria", "Guinea", "Colombia", "Bolivia",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_acled(path: Path | None = None) -> pd.DataFrame:
    """Load ACLED raw CSV with type casting. Returns empty DataFrame if missing."""
    path = path or RAW_DIR / "acled_raw.csv"
    if not path.exists():
        log.warning(
            "ACLED data not found at %s. Run `python src/ingest.py` first.", path
        )
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df["year"] = pd.to_numeric(df.get("year", pd.Series()), errors="coerce")
    df["latitude"] = pd.to_numeric(df.get("latitude", pd.Series()), errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude", pd.Series()), errors="coerce")
    df["fatalities"] = pd.to_numeric(df.get("fatalities", pd.Series()), errors="coerce").fillna(0).astype(int)
    df["year_month"] = df["event_date"].dt.to_period("M")
    log.info("Loaded ACLED: %d rows", len(df))
    return df


def _require(df: pd.DataFrame, name: str = "ACLED") -> bool:
    if df.empty:
        log.warning("Skipping chart — %s DataFrame is empty.", name)
        return False
    return True


# ---------------------------------------------------------------------------
# Geographic charts
# ---------------------------------------------------------------------------

def chart_choropleth_world(df: pd.DataFrame) -> None:
    """Interactive choropleth: total war-crime-event count by country."""
    if not _require(df):
        return
    import plotly.express as px

    counts = df.groupby("country").size().reset_index(name="event_count")

    fig = px.choropleth(
        counts,
        locations="country",
        locationmode="country names",
        color="event_count",
        color_continuous_scale="OrRd",
        title="Total War-Crimes-Related Events by Country (ACLED)",
        labels={"event_count": "Events"},
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        coloraxis_colorbar=dict(title="Events"),
    )
    out = OUT_DIR / "choropleth_world.html"
    fig.write_html(str(out))
    log.info("Saved: %s", out)


def chart_event_cluster_map(df: pd.DataFrame, max_points: int = 20_000) -> None:
    """
    Interactive folium dot/cluster map with tooltips showing date, actor, event type.
    Capped at max_points for performance.
    """
    if not _require(df):
        return
    import folium
    from folium.plugins import MarkerCluster

    sample = df.dropna(subset=["latitude", "longitude"])
    if len(sample) > max_points:
        sample = sample.sample(max_points, random_state=42)
        log.info("Cluster map: sampled %d of %d events for performance.", max_points, len(df))

    m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB positron")
    cluster = MarkerCluster(name="Events").add_to(m)

    event_color_map = {
        "Battles": "red",
        "Explosions/Remote violence": "orange",
        "Violence against civilians": "purple",
        "Sexual violence": "blue",
    }

    for _, row in sample.iterrows():
        color = event_color_map.get(str(row.get("event_type", "")), "gray")
        tooltip = (
            f"<b>{row.get('event_type', 'N/A')}</b><br>"
            f"Date: {str(row.get('event_date', 'N/A'))[:10]}<br>"
            f"Actor: {row.get('actor1', 'N/A')}<br>"
            f"Location: {row.get('location', '')}, {row.get('country', '')}<br>"
            f"Fatalities: {row.get('fatalities', 0)}"
        )
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=4,
            color=color,
            fill=True,
            fill_opacity=0.6,
            tooltip=tooltip,
        ).add_to(cluster)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
                padding:10px;border-radius:5px;border:1px solid #ccc;font-size:12px;">
        <b>Event Type</b><br>
        <span style="color:red;">&#9679;</span> Battles<br>
        <span style="color:orange;">&#9679;</span> Explosions/Remote<br>
        <span style="color:purple;">&#9679;</span> Violence vs. Civilians<br>
        <span style="color:blue;">&#9679;</span> Sexual Violence
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    out = OUT_DIR / "event_cluster_map.html"
    m.save(str(out))
    log.info("Saved: %s", out)


def chart_heatmap_density(df: pd.DataFrame, max_points: int = 50_000) -> None:
    """Folium heatmap layer showing geographic event density."""
    if not _require(df):
        return
    import folium
    from folium.plugins import HeatMap

    coords = df.dropna(subset=["latitude", "longitude"])[["latitude", "longitude"]]
    if len(coords) > max_points:
        coords = coords.sample(max_points, random_state=42)

    m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB dark_matter")
    HeatMap(
        coords.values.tolist(),
        radius=8,
        blur=10,
        max_zoom=6,
        gradient={0.2: "blue", 0.4: "lime", 0.6: "yellow", 1.0: "red"},
    ).add_to(m)

    out = OUT_DIR / "heatmap_density.html"
    m.save(str(out))
    log.info("Saved: %s", out)


# ---------------------------------------------------------------------------
# Temporal charts
# ---------------------------------------------------------------------------

def chart_monthly_events_by_type(df: pd.DataFrame) -> None:
    """Interactive Plotly line chart: monthly event counts broken out by event type."""
    if not _require(df):
        return
    import plotly.graph_objects as go

    df2 = df.dropna(subset=["event_date", "event_type"]).copy()
    df2["month"] = df2["event_date"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        df2.groupby(["month", "event_type"])
        .size()
        .reset_index(name="count")
    )

    fig = go.Figure()
    for etype, color in PALETTE.items():
        sub = monthly[monthly["event_type"] == etype]
        fig.add_trace(go.Scatter(
            x=sub["month"],
            y=sub["count"],
            mode="lines",
            name=etype,
            line=dict(color=color, width=2),
        ))

    fig.update_layout(
        title="Monthly War-Crimes-Related Events by Type (ACLED)",
        xaxis_title="Month",
        yaxis_title="Events",
        legend_title="Event Type",
        hovermode="x unified",
        template="plotly_white",
    )
    out = OUT_DIR / "monthly_events_by_type.html"
    fig.write_html(str(out))
    log.info("Saved: %s", out)


def chart_animated_timeseries(df: pd.DataFrame) -> None:
    """Animated scatter-geo map: events per country per year (Plotly Express)."""
    if not _require(df):
        return
    import plotly.express as px

    df2 = df.dropna(subset=["country", "year"]).copy()
    df2["year"] = df2["year"].astype(int)

    annual = (
        df2.groupby(["year", "country"])
        .size()
        .reset_index(name="event_count")
    )

    fig = px.choropleth(
        annual,
        locations="country",
        locationmode="country names",
        color="event_count",
        animation_frame="year",
        color_continuous_scale="YlOrRd",
        title="Annual War-Crimes-Related Events by Country (Animated)",
        labels={"event_count": "Events"},
        range_color=[0, annual["event_count"].quantile(0.95)],
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        coloraxis_colorbar=dict(title="Events"),
    )
    out = OUT_DIR / "animated_timeseries.html"
    fig.write_html(str(out))
    log.info("Saved: %s", out)


def chart_yoy_violence_civilians(df: pd.DataFrame) -> None:
    """Static Matplotlib bar chart: year-over-year change in 'Violence against civilians'."""
    if not _require(df):
        return

    vac = df[df["event_type"] == "Violence against civilians"].copy()
    if vac.empty:
        log.warning("No 'Violence against civilians' events found.")
        return

    annual = vac.groupby("year").size().rename("events")
    annual = annual.sort_index()
    yoy = annual.diff().dropna()

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#c0392b" if v >= 0 else "#27ae60" for v in yoy.values]
    ax.bar(yoy.index.astype(int), yoy.values, color=colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Year-over-Year Change: Violence Against Civilians (ACLED)", fontsize=14)
    ax.set_xlabel("Year")
    ax.set_ylabel("Change in Event Count")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):+,}"))

    # Annotate bars
    for year, val in zip(yoy.index.astype(int), yoy.values):
        ax.text(year, val + (50 if val >= 0 else -100),
                f"{int(val):+,}", ha="center", va="bottom" if val >= 0 else "top",
                fontsize=8)

    plt.tight_layout()
    out = OUT_DIR / "yoy_violence_civilians.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    log.info("Saved: %s", out)


# ---------------------------------------------------------------------------
# Actor analysis charts
# ---------------------------------------------------------------------------

def chart_top20_actors(df: pd.DataFrame) -> None:
    """Static Matplotlib horizontal bar: top 20 actors by event count."""
    if not _require(df):
        return

    top20 = (
        df["actor1"]
        .value_counts()
        .head(20)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(top20.index, top20.values, color="#c0392b", alpha=0.85)
    ax.set_title("Top 20 Named Actors by Event Count (ACLED)", fontsize=14)
    ax.set_xlabel("Events")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    for bar, val in zip(bars, top20.values):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9)

    plt.tight_layout()
    out = OUT_DIR / "top20_actors.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    log.info("Saved: %s", out)


def _classify_actor_type(inter_code) -> str:
    """Map ACLED inter1/inter2 numeric codes to human-readable actor type."""
    mapping = {
        "1": "State Military",
        "2": "Rebel Group",
        "3": "Political Militia",
        "4": "Communal Militia",
        "5": "Rioters/Protesters",
        "6": "Civilians",
        "7": "External/Other",
        "8": "State Police",
    }
    return mapping.get(str(inter_code).strip(), "Unknown")


def chart_actor_type_by_region(df: pd.DataFrame) -> None:
    """Interactive Plotly stacked bar: actor type distribution by region."""
    if not _require(df):
        return
    import plotly.graph_objects as go

    if "inter1" not in df.columns or "region" not in df.columns:
        log.warning("Required columns 'inter1' or 'region' not found. Skipping.")
        return

    df2 = df.dropna(subset=["inter1", "region"]).copy()
    df2["actor_type"] = df2["inter1"].apply(_classify_actor_type)

    pivot = (
        df2.groupby(["region", "actor_type"])
        .size()
        .unstack(fill_value=0)
    )

    fig = go.Figure()
    color_seq = [
        "#c0392b", "#e67e22", "#f1c40f", "#2ecc71",
        "#3498db", "#9b59b6", "#1abc9c", "#e74c3c",
    ]
    for i, atype in enumerate(pivot.columns):
        fig.add_trace(go.Bar(
            name=atype,
            x=pivot.index,
            y=pivot[atype],
            marker_color=color_seq[i % len(color_seq)],
        ))

    fig.update_layout(
        barmode="stack",
        title="Actor Type by Region (ACLED)",
        xaxis_title="Region",
        yaxis_title="Events",
        legend_title="Actor Type",
        xaxis_tickangle=-30,
        template="plotly_white",
    )
    out = OUT_DIR / "actor_type_by_region.html"
    fig.write_html(str(out))
    log.info("Saved: %s", out)


def chart_actor_network(df: pd.DataFrame, top_n: int = 30) -> None:
    """
    Interactive Plotly network diagram: actor vs. actor interactions.
    Nodes = top_n actors; edges = event count between actor1 and actor2.
    """
    if not _require(df):
        return
    import plotly.graph_objects as go

    if "actor1" not in df.columns or "actor2" not in df.columns:
        log.warning("actor1/actor2 columns not found. Skipping network chart.")
        return

    df2 = df.dropna(subset=["actor1", "actor2"]).copy()
    df2 = df2[df2["actor2"].str.strip() != ""]

    top_actors = set(df2["actor1"].value_counts().head(top_n).index)

    edges = (
        df2[df2["actor1"].isin(top_actors) & df2["actor2"].isin(top_actors)]
        .groupby(["actor1", "actor2"])
        .size()
        .reset_index(name="weight")
    )

    if edges.empty:
        log.warning("No actor-vs-actor interactions found for top actors. Skipping network.")
        return

    # Assign positions using a simple circle layout
    all_nodes = list(set(edges["actor1"]) | set(edges["actor2"]))
    n = len(all_nodes)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    node_pos = {node: (np.cos(a), np.sin(a)) for node, a in zip(all_nodes, angles)}

    edge_x, edge_y = [], []
    for _, row in edges.iterrows():
        x0, y0 = node_pos[row["actor1"]]
        x1, y1 = node_pos[row["actor2"]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x = [node_pos[n][0] for n in all_nodes]
    node_y = [node_pos[n][1] for n in all_nodes]
    node_size = [
        max(8, min(40, df2[df2["actor1"] == n].shape[0] / 50))
        for n in all_nodes
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.5, color="#aaaaaa"),
        hoverinfo="none",
        name="Interactions",
    ))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(size=node_size, color="#c0392b", line=dict(width=1, color="white")),
        text=all_nodes,
        textposition="top center",
        textfont=dict(size=8),
        hoverinfo="text",
        name="Actors",
    ))
    fig.update_layout(
        title=f"Actor Interaction Network (Top {top_n} Actors by Event Count)",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    out = OUT_DIR / "actor_network.html"
    fig.write_html(str(out))
    log.info("Saved: %s", out)


# ---------------------------------------------------------------------------
# Accountability gap charts
# ---------------------------------------------------------------------------

def chart_accountability_gap(df: pd.DataFrame) -> None:
    """
    Interactive grouped bar: top countries by ACLED event count, colored by
    whether the ICC has an open situation there.
    """
    if not _require(df):
        return
    import plotly.graph_objects as go

    top_countries = df["country"].value_counts().head(30).reset_index()
    top_countries.columns = ["country", "event_count"]
    top_countries["icc_situation"] = top_countries["country"].apply(
        lambda c: "ICC Open Situation" if c in ICC_SITUATION_COUNTRIES else "No ICC Situation"
    )

    icc_yes = top_countries[top_countries["icc_situation"] == "ICC Open Situation"]
    icc_no = top_countries[top_countries["icc_situation"] == "No ICC Situation"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=icc_yes["country"],
        y=icc_yes["event_count"],
        name="ICC Open Situation",
        marker_color="#2471a3",
    ))
    fig.add_trace(go.Bar(
        x=icc_no["country"],
        y=icc_no["event_count"],
        name="No ICC Situation",
        marker_color="#e74c3c",
        opacity=0.8,
    ))

    fig.update_layout(
        title="Accountability Gap: ACLED Event Count vs. ICC Situation (Top 30 Countries)",
        xaxis_title="Country",
        yaxis_title="ACLED Events",
        barmode="group",
        xaxis_tickangle=-40,
        template="plotly_white",
        legend_title="ICC Status",
        annotations=[dict(
            x=0.5, y=1.08, xref="paper", yref="paper",
            text="Red = documented events with no ICC situation open | Blue = ICC situation exists",
            showarrow=False, font=dict(size=11),
        )],
    )
    out = OUT_DIR / "accountability_gap.html"
    fig.write_html(str(out))
    log.info("Saved: %s", out)


def chart_data_completeness(df: pd.DataFrame) -> None:
    """
    Static Matplotlib heatmap: estimated data completeness/confidence by region
    and event type. Confidence scores are research-based proxies; see inline comments.
    """
    if not _require(df):
        return

    # Confidence proxy scores (0–1) based on ACLED documentation, press freedom
    # indices, and published literature on conflict reporting bias.
    # These are illustrative ordinal estimates, NOT precise measurements.
    regions = [
        "Western Africa",
        "Middle Africa",
        "Eastern Africa",
        "Southern Africa",
        "Northern Africa",
        "Middle East",
        "South Asia",
        "Southeast Asia",
        "Central Asia",
        "Eastern Europe",
        "Western Europe",
        "Americas",
    ]
    event_types = ["Battles", "Explosions/Remote", "Violence vs. Civilians", "Sexual Violence"]

    # Rows = regions, Cols = event types
    # Lower = worse coverage; higher = better coverage
    scores = np.array([
        [0.65, 0.60, 0.55, 0.30],  # Western Africa
        [0.55, 0.50, 0.45, 0.25],  # Middle Africa
        [0.60, 0.58, 0.52, 0.28],  # Eastern Africa
        [0.70, 0.65, 0.60, 0.35],  # Southern Africa
        [0.60, 0.55, 0.50, 0.25],  # Northern Africa
        [0.65, 0.70, 0.55, 0.30],  # Middle East
        [0.70, 0.68, 0.60, 0.35],  # South Asia
        [0.72, 0.70, 0.65, 0.40],  # Southeast Asia
        [0.50, 0.48, 0.42, 0.20],  # Central Asia
        [0.85, 0.82, 0.80, 0.55],  # Eastern Europe
        [0.90, 0.88, 0.85, 0.65],  # Western Europe
        [0.80, 0.75, 0.72, 0.50],  # Americas
    ])

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(scores, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Estimated Completeness (0=Low, 1=High)")

    ax.set_xticks(range(len(event_types)))
    ax.set_xticklabels(event_types, rotation=20, ha="right")
    ax.set_yticks(range(len(regions)))
    ax.set_yticklabels(regions)
    ax.set_title("Estimated Data Completeness by Region & Event Type (ACLED)\n"
                 "(Proxy scores — not official metrics)", fontsize=12)

    for i in range(len(regions)):
        for j in range(len(event_types)):
            ax.text(j, i, f"{scores[i, j]:.2f}",
                    ha="center", va="center", fontsize=9,
                    color="black" if 0.3 < scores[i, j] < 0.75 else "white")

    plt.tight_layout()
    out = OUT_DIR / "data_completeness.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    log.info("Saved: %s", out)


# ---------------------------------------------------------------------------
# Summary export
# ---------------------------------------------------------------------------

def export_summary(df: pd.DataFrame) -> None:
    """Export aggregated summary CSV to outputs/."""
    if not _require(df):
        return

    summary_parts = []

    # By country
    by_country = (
        df.groupby("country")
        .agg(
            event_count=("event_type", "count"),
            total_fatalities=("fatalities", "sum"),
            earliest_event=("event_date", "min"),
            latest_event=("event_date", "max"),
        )
        .reset_index()
        .assign(
            icc_situation=lambda x: x["country"].apply(
                lambda c: True if c in ICC_SITUATION_COUNTRIES else False
            )
        )
        .sort_values("event_count", ascending=False)
    )
    by_country.to_csv(PROC_DIR / "summary_by_country.csv", index=False)

    # By event type and year
    by_type_year = (
        df.groupby(["year", "event_type"])
        .agg(event_count=("event_type", "count"), fatalities=("fatalities", "sum"))
        .reset_index()
    )
    by_type_year.to_csv(PROC_DIR / "summary_by_type_year.csv", index=False)

    # Top actors
    top_actors = (
        df["actor1"]
        .value_counts()
        .head(50)
        .rename_axis("actor")
        .reset_index(name="event_count")
    )
    top_actors.to_csv(PROC_DIR / "summary_top_actors.csv", index=False)

    # Combined summary
    combined = by_country.merge(
        df.groupby("country")["event_type"].value_counts().unstack(fill_value=0),
        on="country",
        how="left",
    )
    out = OUT_DIR / "summary_aggregated.csv"
    combined.to_csv(out, index=False)
    log.info("Saved summary CSVs to %s and %s", PROC_DIR, OUT_DIR)


# ---------------------------------------------------------------------------
# Chart registry
# ---------------------------------------------------------------------------

CHART_REGISTRY = {
    "choropleth": chart_choropleth_world,
    "cluster_map": chart_event_cluster_map,
    "heatmap": chart_heatmap_density,
    "monthly_line": chart_monthly_events_by_type,
    "animated": chart_animated_timeseries,
    "yoy_civilians": chart_yoy_violence_civilians,
    "top20_actors": chart_top20_actors,
    "actor_region": chart_actor_type_by_region,
    "actor_network": chart_actor_network,
    "accountability_gap": chart_accountability_gap,
    "data_completeness": chart_data_completeness,
    "summary": export_summary,
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate all war-crimes EDA visualizations.")
    parser.add_argument(
        "--chart",
        choices=list(CHART_REGISTRY.keys()) + ["all"],
        default="all",
        help="Which chart to generate (default: all)",
    )
    args = parser.parse_args()

    df = load_acled()

    targets = list(CHART_REGISTRY.keys()) if args.chart == "all" else [args.chart]

    for chart_name in targets:
        log.info("--- %s ---", chart_name)
        try:
            CHART_REGISTRY[chart_name](df)
        except Exception as exc:
            log.error("Failed to generate '%s': %s", chart_name, exc, exc_info=True)

    log.info("Done. Check /outputs for results.")


if __name__ == "__main__":
    main()
