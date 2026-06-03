"""
PflegeOptix — Sprint 4: Interactive Streamlit Dashboard
Elderly Care Planning Dashboard for Germany (2025-2040)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
from pulp import (
    LpProblem, LpMinimize, LpVariable, LpStatus, lpSum, value
)

# ────────────────────────────────────────────────────────────────
# 0. PAGE CONFIG & THEME
# ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PflegeOptix — Elderly Care Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject light government-style CSS (Destatis / Eurostat inspired)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ── Global ── */
    html, body, .stApp {
        background-color: #f5f6f8 !important;
        color: #212121;
        font-family: 'Inter', sans-serif !important;
    }
    /* Remove top header bar background */
    header[data-testid="stHeader"] {
        background: #ffffff !important;
        border-bottom: 1px solid #e0e4ea;
    }
    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: #f0f2f5 !important;
        border-right: 1px solid #dde2ea;
    }
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] p {
        color: #5a6473 !important;
    }
    /* ── Metric cards ── */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border-left: 3px solid #1d4e89;
        border-radius: 6px;
        padding: 16px 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    div[data-testid="stMetric"] label {
        color: #5a6473 !important;
        font-size: 0.82rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #1a2a42 !important;
        font-weight: 700;
        font-size: 1.5rem;
    }
    /* ── Tabs (prominent pill style) ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: #e8ecf2;
        border-radius: 8px;
        padding: 5px;
        border-bottom: none;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: #3d4a5c;
        font-weight: 600;
        font-size: 0.92rem;
        padding: 10px 18px;
        border-bottom: none;
        margin-bottom: 0;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(29,78,137,0.08);
        color: #1d4e89;
    }
    .stTabs [aria-selected="true"] {
        background: #1d4e89 !important;
        color: #ffffff !important;
        border-bottom: none !important;
        font-weight: 700;
        box-shadow: 0 2px 6px rgba(29,78,137,0.25);
    }
    /* ── Content panels ── */
    .block-container {
        background-color: #f5f6f8;
    }
    /* ── DataFrames ── */
    .stDataFrame {
        border-radius: 6px;
        border: 1px solid #e0e4ea;
    }
    /* Force dark text in all table cells */
    .stDataFrame [data-testid="stDataFrameResizable"],
    .stDataFrame table,
    .stDataFrame td, .stDataFrame th,
    .stDataFrame [class*="cell"],
    [data-testid="stDataFrame"] * {
        color: #1a2a42 !important;
    }
    /* ── Headers ── */
    h1 { color: #1a2a42 !important; font-weight: 700; }
    h2, h3 { color: #1d4e89 !important; font-weight: 600; }
    h4, h5, h6 { color: #2c3e6b !important; }
    /* ── Plotly chart container ── */
    .js-plotly-plot .plotly .modebar { opacity: 0.5; }
    /* ── SHAP images ── */
    .shap-container img {
        border-radius: 6px;
        border: 1px solid #e0e4ea;
        background: #fff;
    }
    /* ── Insight cards ── */
    .insight-card {
        background: #ffffff;
        border: 1px solid #e0e4ea;
        border-top: 3px solid #1d4e89;
        border-radius: 6px;
        padding: 20px;
        height: 100%;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .insight-card h4 { color: #1d4e89 !important; margin-top: 0; }
    .insight-card p { color: #3d4a5c; line-height: 1.6; }
    /* ── Divider ── */
    hr { border-color: #e0e4ea !important; }
    /* ── Buttons ── */
    .stDownloadButton button {
        background-color: #1d4e89 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
    }
    .stDownloadButton button:hover {
        background-color: #163b6b !important;
    }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────
# 1. DATA LOADING (cached)
# ────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent / "dataset" / "dashboard"
FIG_DIR  = Path(__file__).resolve().parent / "static"


@st.cache_data(show_spinner="Loading master dataset…")
def load_master() -> pd.DataFrame:
    """Load and clean the master dataset, converting German decimals to float."""
    df = pd.read_csv(DATA_DIR / "master_clean.csv", dtype={"AGS": str})
    df["AGS"] = df["AGS"].str.zfill(5)
    df["Region"] = df["Region"].str.strip()
    
    # Pre-clean numeric columns that use German commas as decimal separators
    german_num_cols = ["Population_Density", "Single_Person_Households_Pct"]
    for col in german_num_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False).astype(float)
    return df


@st.cache_data(show_spinner="Loading population forecast…")
def load_forecast() -> pd.DataFrame:
    """Load and clean the population forecast dataset."""
    df = pd.read_csv(DATA_DIR / "forecast_clean.csv", dtype={"AGS": str})
    df["AGS"] = df["AGS"].str.zfill(5)
    df["Region"] = df["Region"].str.strip()
    return df


@st.cache_data(show_spinner="Loading PuLP allocation…")
def load_pulp() -> pd.DataFrame:
    """Load and clean the PuLP resource allocation dataset."""
    df = pd.read_csv(DATA_DIR / "pulp_clean.csv", dtype={"AGS": str})
    df["AGS"] = df["AGS"].str.zfill(5)
    df["Region"] = df["Region"].str.strip()
    return df


@st.cache_data(show_spinner="Loading GeoJSON…")
def load_geojson() -> dict:
    """Load the GeoJSON boundary data for German districts."""
    with open(DATA_DIR / "landkreise_with_ags.geo.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading mapping…")
def load_mapping() -> pd.DataFrame:
    """Load the AGS-to-GeoJSON district mapping file."""
    df = pd.read_csv(DATA_DIR / "ags_geojson_mapping.csv", dtype={"AGS": str})
    df["AGS"] = df["AGS"].str.zfill(5)
    return df


# Load everything once
df_master   = load_master()
df_forecast = load_forecast()
df_pulp     = load_pulp()
geo         = load_geojson()
df_mapping  = load_mapping()

@st.cache_data(show_spinner=False)
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Convert a pandas DataFrame to encoded CSV bytes for download."""
    return df.to_csv(index=False).encode('utf-8')

# Derive state list from master (first 2 digits of AGS → state code)
STATE_CODES = {
    "01": "Schleswig-Holstein",  "02": "Hamburg",
    "03": "Niedersachsen",       "04": "Bremen",
    "05": "Nordrhein-Westfalen", "06": "Hessen",
    "07": "Rheinland-Pfalz",     "08": "Baden-Württemberg",
    "09": "Bayern",              "10": "Saarland",
    "11": "Berlin",              "12": "Brandenburg",
    "13": "Mecklenburg-Vorpommern", "14": "Sachsen",
    "15": "Sachsen-Anhalt",      "16": "Thüringen",
}

df_master["State"] = df_master["AGS"].str[:2].map(STATE_CODES)
df_pulp["State"]   = df_pulp["AGS"].str[:2].map(STATE_CODES)

# Plotly color schemes
PLOTLY_TEMPLATE = "plotly_white"
# Design tokens
COLOR_PRIMARY   = "#1d4e89"   # Navy — bars, lines, accents
COLOR_ACCENT    = "#c0392b"   # Red — urgency / deficit
COLOR_POSITIVE  = "#2e7d32"   # Green — allocation / coverage
COLOR_FORECAST  = "#e67e22"   # Amber — forecast lines

# Global Plotly font override — ensures all chart text is readable
PLOTLY_FONT = dict(
    font=dict(family="Inter, sans-serif", size=13, color="#1a2a42"),
)

# ────────────────────────────────────────────────────────────────
# 2. SIDEBAR
# ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 PflegeOptix")
    st.caption("Elderly Care Planning Dashboard")
    st.divider()

    # State filter (only show states that exist in our clean dataset)
    valid_states = sorted(df_pulp["State"].dropna().unique().tolist())
    states = ["All States"] + valid_states
    selected_state = st.selectbox("🗺️ Select State", states)
    
    missing_states = sorted(list(set(STATE_CODES.values()) - set(valid_states)))
    if missing_states:
        st.caption(f"*(Note: {', '.join(missing_states)} excluded due to missing data)*")


    st.divider()

    # Calculate dynamic slider bounds based on state deficit
    if selected_state != "All States":
        state_def = int(df_pulp[df_pulp["State"] == selected_state]["Deficit"].clip(lower=0).sum())
    else:
        state_def = int(df_pulp["Deficit"].clip(lower=0).sum())
        
    if state_def == 0: 
        state_def = 1000
    
    max_slider = int(state_def * 1.1)
    max_slider = max(500, (max_slider // 500 + 1) * 500)
    
    step_val = max(100, max_slider // 20)
    if step_val > 500: step_val = (step_val // 500) * 500
    elif step_val > 100: step_val = (step_val // 100) * 100
        
    default_val = min(max_slider // 2, max_slider)

    # Budget slider
    budget = st.slider(
        "💰 New Beds Budget",
        min_value=0,
        max_value=max_slider,
        value=default_val,
        step=step_val,
        help="Total new beds the government can allocate (dynamically scaled to state deficit).",
    )

    st.divider()
    
    csv_data = convert_df_to_csv(df_master)
    st.download_button(
        label="📥 Download Master Data",
        data=csv_data,
        file_name="pflegeoptix_master_dataset.csv",
        mime="text/csv",
        width="stretch"
    )
    st.write("") # small spacing
    
    st.markdown(
        "<small style='color:#3d4a5c;'>"
        "📊 Data: INKAR + Pflegestatistik (2015-2023)<br>"
        "📈 Forecast: Prophet (2025-2040)<br>"
        "🧠 Model: XGBoost + SHAP<br>"
        "⚖️ Optimizer: PuLP (Min-Max Fairness)"
        "</small>",
        unsafe_allow_html=True,
    )

# ────────────────────────────────────────────────────────────────
# 3. HELPER: build choropleth for Germany
# ────────────────────────────────────────────────────────────────
def make_germany_choropleth(df_data: pd.DataFrame, color_col: str, color_scale: str, title_bar: str,
                            hover_cols: dict = None) -> go.Figure:
    """Create a well-centered choropleth of Germany using the modern map engine."""
    df_geo = df_data.merge(
        df_mapping[df_mapping["Match_Type"] != "unmatched"][["AGS", "GeoJSON_NAME_3"]],
        on="AGS", how="left",
    )
    df_geo = df_geo.dropna(subset=["GeoJSON_NAME_3"])

    if hover_cols is None:
        hover_cols = {}

    hover_cols["GeoJSON_NAME_3"] = False

    fig = px.choropleth_map(
        df_geo,
        geojson=geo,
        locations="GeoJSON_NAME_3",
        featureidkey="properties.NAME_3",
        color=color_col,
        hover_name="Region",
        hover_data=hover_cols,
        color_continuous_scale=color_scale,
        map_style="carto-positron",
        center={"lat": 51.2, "lon": 10.4},
        zoom=4.8,
        opacity=0.85,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        height=550,
        coloraxis_colorbar=dict(
            title=dict(text=title_bar, font=dict(color="#1a2a42", size=12)),
            bgcolor="rgba(255,255,255,0.9)",
            tickfont=dict(color="#3d4a5c"),
            bordercolor="#e0e4ea",
            borderwidth=1,
        ),
    )
    return fig


# ────────────────────────────────────────────────────────────────
# 4. HELPER: PuLP Real-time solver
# ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Optimizing bed allocation…")
def solve_allocation(budget_beds: int, state: str = "All States") -> pd.DataFrame:
    """Run PuLP Min-Max Fairness optimization scoped to the selected state.
    
    Args:
        budget_beds: Total number of care beds available to allocate.
        state: The selected state to filter by, or 'All States'.
        
    Returns:
        DataFrame with columns 'Beds_Allocated', 'Remaining_Deficit', and 'Unmet_Ratio'.
    """
    if state != "All States":
        df = df_pulp[df_pulp["State"] == state].copy()
    else:
        df = df_pulp.copy()

    districts = df[df["Deficit"] > 0].copy().reset_index(drop=True)

    if districts.empty:
        df["Beds_Allocated"] = 0.0
        df["Remaining_Deficit"] = df["Deficit"].clip(lower=0)
        df["Unmet_Ratio"] = 0.0
        return df

    prob = LpProblem("BedAllocation", LpMinimize)
    n = len(districts)
    x = [LpVariable(f"x_{i}", lowBound=0) for i in range(n)]
    z = LpVariable("z_max", lowBound=0)

    prob += z  # minimize worst-case unmet ratio

    for i in range(n):
        deficit_i = float(districts.loc[i, "Deficit"])
        prob += x[i] <= deficit_i
        prob += z >= (deficit_i - x[i]) / deficit_i

    prob += lpSum(x) <= budget_beds
    prob.solve()

    # Initialize columns
    df["Beds_Allocated"]    = 0.0
    df["Remaining_Deficit"] = df["Deficit"].clip(lower=0)
    df["Unmet_Ratio"]       = 0.0

    if LpStatus[prob.status] == "Optimal":
        for i in range(n):
            ags_i = districts.loc[i, "AGS"]
            alloc = value(x[i])
            mask  = df["AGS"] == ags_i
            df.loc[mask, "Beds_Allocated"]    = alloc
            df.loc[mask, "Remaining_Deficit"] = df.loc[mask, "Deficit"] - alloc
            deficit_val = float(districts.loc[i, "Deficit"])
            if deficit_val > 0:
                df.loc[mask, "Unmet_Ratio"] = (deficit_val - alloc) / deficit_val

    return df


# ────────────────────────────────────────────────────────────────
# 4b. HELPER: Budget Efficiency Curve
# ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Calculating efficiency curve…")
def get_efficiency_curve(state: str) -> pd.DataFrame:
    """Pre-compute coverage % dynamically based on state deficit budget levels.
    
    Args:
        state: Selected state code/name, or 'All States'.
        
    Returns:
        DataFrame containing precalculated Budget vs Coverage % vs Unmet %.
    """
    if state != "All States":
        df = df_pulp[df_pulp["State"] == state]
    else:
        df = df_pulp
        
    total_def = int(df["Deficit"].clip(lower=0).sum())
    if total_def == 0:
        return pd.DataFrame([{"Budget": 0, "Coverage_Pct": 100.0, "Avg_Unmet_Pct": 0.0}])
        
    step = max(500, total_def // 20)
    max_budget = int(total_def * 1.1)
    if max_budget < step: 
        max_budget = step
    
    budgets = list(range(step, max_budget + step, step))
    results = []
    for b in budgets:
        df_sim = solve_allocation(b, state)
        tot_alloc = df_sim["Beds_Allocated"].sum()
        cov = (tot_alloc / total_def * 100) if total_def > 0 else 0
        avg_unmet = df_sim.loc[df_sim["Deficit"] > 0, "Unmet_Ratio"].mean() * 100
        if pd.isna(avg_unmet): 
            avg_unmet = 0.0
        results.append({
            "Budget": b, 
            "Coverage_Pct": round(cov, 1), 
            "Avg_Unmet_Pct": round(avg_unmet, 1)
        })
    return pd.DataFrame(results)


# ────────────────────────────────────────────────────────────────
# 5. HEADER (project name is already in sidebar, so only show subtitle here)
# ────────────────────────────────────────────────────────────────
st.caption("Elderly care bed planning — Germany (2025-2040)")

# ────────────────────────────────────────────────────────────────
# 6. TABS
# ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "📈 Population Forecast",
    "🛏️ Bed Allocation",
    "⚖️ District Comparison",
    "🧠 Model Insights",
    "ℹ️ About Project",
])


# ================================================================
# TAB 1 — OVERVIEW
# ================================================================
with tab1:
    st.subheader("National Overview")

    latest = df_master["Year"].max()
    df_latest = df_master[df_master["Year"] == latest].copy()

    if selected_state != "All States":
        df_latest = df_latest[df_latest["State"] == selected_state]
        df_pulp_filt = df_pulp[df_pulp["State"] == selected_state]
    else:
        df_pulp_filt = df_pulp

    total_districts = df_pulp_filt["AGS"].nunique()
    total_deficit   = int(df_pulp_filt["Deficit"].sum())
    districts_in_deficit = int((df_pulp_filt["Deficit"] > 0).sum())
    pct_in_deficit  = (districts_in_deficit / total_districts * 100) if total_districts > 0 else 0
    total_beds      = int(df_latest["FullStay_Beds"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Districts",       f"{total_districts}")
    c2.metric("Total Deficit",   f"{total_deficit:,} beds")
    c3.metric("Districts in Deficit", f"{districts_in_deficit} ({pct_in_deficit:.0f}%)")
    c4.metric("Current Beds",    f"{total_beds:,}")

    st.markdown("---")

    col_map, col_bar = st.columns([3, 2])

    with col_map:
        st.markdown("##### Care Gap by District")
        fig_map = make_germany_choropleth(
            df_pulp_filt, "Care_Gap", "RdYlGn_r", "Care Gap",
            hover_cols={"Care_Gap": ":.0f", "Deficit": ":.0f"},
        )
        st.plotly_chart(fig_map, key="map_overview")
        
        # Alert for unmapped districts
        unmatched_ags = df_mapping[df_mapping["Match_Type"] == "unmatched"]["AGS"].tolist()
        unmatched_in_filt = df_pulp_filt[df_pulp_filt["AGS"].isin(unmatched_ags)]
        if not unmatched_in_filt.empty:
            with st.expander("⚠️ View districts not shown on map (due to historical boundary changes in GeoJSON):"):
                st.caption("These districts are still fully calculated in all tables and data downloads:")
                st.dataframe(
                    unmatched_in_filt[["Region", "Deficit", "Care_Gap"]]
                    .sort_values("Deficit", ascending=False)
                    .style.format({"Deficit": "{:,.0f}", "Care_Gap": "{:.1f}"}),
                    hide_index=True,
                    width="stretch"
                )

    with col_bar:
        st.markdown("##### Top 10 Districts by Care Gap")
        top10 = df_pulp_filt[df_pulp_filt["Care_Gap"] > 0].nlargest(10, "Care_Gap")
        fig_bar = px.bar(
            top10, x="Care_Gap", y="Region", orientation="h",
            color="Care_Gap", color_continuous_scale="YlOrRd",
            template=PLOTLY_TEMPLATE,
            hover_data={"Care_Gap": ":.1f"}
        )
        fig_bar.update_layout(
            yaxis=dict(autorange="reversed", tickfont=dict(size=11, color="#1a2a42")),
            margin=dict(l=10, r=20, t=10, b=10),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            height=550,
            showlegend=False,
            coloraxis_showscale=False,
            **PLOTLY_FONT,
        )
        st.plotly_chart(fig_bar, key="bar_overview")


# ================================================================
# TAB 2 — POPULATION FORECAST
# ================================================================
with tab2:
    st.subheader("Population Forecast (Prophet 2025-2040)")

    # District picker
    if selected_state != "All States":
        districts_avail = df_forecast[
            df_forecast["AGS"].str[:2].map(STATE_CODES) == selected_state
        ]["Region"].unique()
    else:
        districts_avail = df_forecast["Region"].unique()

    districts_avail = sorted(set(r.strip() for r in districts_avail))
    selected_district = st.selectbox("🔍 Select District", districts_avail)

    ags_sel = df_forecast[df_forecast["Region"].str.strip() == selected_district]["AGS"].iloc[0]

    # Historical
    df_hist = df_master[df_master["AGS"] == ags_sel][["Year", "Total"]].copy()
    df_hist = df_hist.rename(columns={"Total": "Population"})

    # Forecast
    df_fcast = df_forecast[df_forecast["AGS"] == ags_sel][
        ["Year", "Pop_Total_Forecast", "Pop_Total_Lower", "Pop_Total_Upper"]
    ].copy()
    df_fcast = df_fcast.rename(columns={"Pop_Total_Forecast": "Population"})

    # Growth summary cards (placed ABOVE chart)
    if not df_hist.empty and not df_fcast.empty:
        pop_now  = df_hist["Population"].iloc[-1]
        pop_2040 = df_fcast["Population"].iloc[-1]
        growth   = (pop_2040 - pop_now) / pop_now * 100

        m1, m2, m3 = st.columns(3)
        m1.metric("Current Population", f"{pop_now:,.0f}")
        m2.metric("Forecast 2040",      f"{pop_2040:,.0f}")
        m3.metric("Growth",             f"{growth:+.1f}%")

    # Combined line chart
    fig_fc = go.Figure()

    fig_fc.add_trace(go.Scatter(
        x=df_hist["Year"], y=df_hist["Population"],
        mode="lines+markers", name="Historical",
        line=dict(color=COLOR_PRIMARY, width=3),
        marker=dict(size=8),
    ))

    fig_fc.add_trace(go.Scatter(
        x=df_fcast["Year"], y=df_fcast["Population"],
        mode="lines+markers", name="Forecast",
        line=dict(color=COLOR_FORECAST, width=3, dash="dash"),
        marker=dict(size=6),
    ))

    # CI band
    fig_fc.add_trace(go.Scatter(
        x=pd.concat([df_fcast["Year"], df_fcast["Year"][::-1]]),
        y=pd.concat([df_fcast["Pop_Total_Upper"], df_fcast["Pop_Total_Lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(230,126,34,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% CI",
    ))

    fig_fc.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        height=400,
        margin=dict(l=60, r=20, t=30, b=40),
        xaxis_title="Year",
        yaxis_title="Total Population",
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        **PLOTLY_FONT,
    )
    st.plotly_chart(fig_fc, key="chart_forecast")
    
    st.markdown("""
    > **⚠️ Methodological Note on Denominator Consistency:**
    > The population forecast shows the **Total Population** (all ages) rather than only the 65+ demographic. 
    > This is because the historical **Versorgungsquote (VQ)** (the benchmark bed ratio) is defined as *Beds per Total Population*. 
    > To maintain mathematical consistency across Sprints, all forecasts and XGBoost predictors utilize the same total population denominator.
    """)


# ================================================================
# TAB 3 — BED ALLOCATION (PuLP Real-time) — UPGRADED
# ================================================================
with tab3:
    st.subheader(f"Bed Allocation Optimizer — Budget: {budget:,} beds")

    # Run PuLP scoped to selected state
    df_alloc_filt = solve_allocation(budget, selected_state)

    # Summary cards
    total_def_before = int(df_alloc_filt["Deficit"].sum())
    total_allocated  = int(df_alloc_filt["Beds_Allocated"].sum())
    total_remaining  = int(df_alloc_filt["Remaining_Deficit"].sum())
    pct_covered = (
        (total_allocated / total_def_before * 100) if total_def_before > 0 else 0
    )

    # Calculate fairness indicators
    districts_with_deficit = df_alloc_filt[df_alloc_filt["Deficit"] > 0]
    if not districts_with_deficit.empty:
        # Standard deviation of unmet ratio (0 means perfectly equal unmet ratio, which is maximum fairness)
        fairness_sd = districts_with_deficit["Unmet_Ratio"].std() * 100
        # If perfect coverage, max unmet is 0
        if total_allocated >= total_def_before:
            fairness_sd = 0.0
    else:
        fairness_sd = 0.0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Deficit",      f"{total_def_before:,}")
    k2.metric("Beds Allocated",     f"{total_allocated:,}")
    k3.metric("Remaining Deficit",  f"{total_remaining:,}")
    k4.metric("Coverage",           f"{pct_covered:.1f}%")
    
    if pd.isna(fairness_sd):
        fairness_str = "N/A"
    elif fairness_sd < 1.0:
        fairness_str = "Perfect"
    else:
        fairness_str = f"±{fairness_sd:.1f}%"
        
    k5.metric(
        "Inequality (SD of Unmet)", 
        fairness_str,
        help="Standard deviation of the unmet deficit ratio across districts. Lower values indicate more equal allocation."
    )

    st.markdown("---")
    
    st.info(
        "⚖️ **Optimization Strategy: Min-Max Fairness**  \n"
        "Unlike standard greedy allocation which would give all beds to the largest cities (leaving rural areas with 100% shortage), "
        "the PuLP optimizer minimizes the *maximum* unmet deficit ratio. This mathematically guarantees that the burden of shortage "
        "is shared as equally as possible (minimizing the inequality index shown above)."
    )

    # ── Budget Efficiency Curve ──
    with st.expander(f"📈 Budget Efficiency Curve — {selected_state}", expanded=False):
        st.caption(f"Dynamic coverage estimation based on deficit in {selected_state}.")
        df_curve = get_efficiency_curve(selected_state)

        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=df_curve["Budget"], y=df_curve["Coverage_Pct"],
            mode="lines+markers", name="Coverage %",
            line=dict(color=COLOR_PRIMARY, width=3),
            marker=dict(size=8),
            hovertemplate="Budget: %{x:,.0f} beds<br>Coverage: %{y:.1f}%<extra></extra>",
        ))
        # Add current budget marker
        cur_cov = df_curve.loc[df_curve["Budget"] == budget, "Coverage_Pct"]
        if not cur_cov.empty:
            fig_curve.add_trace(go.Scatter(
                x=[budget], y=[cur_cov.iloc[0]],
                mode="markers", name="Current Budget",
                marker=dict(color=COLOR_FORECAST, size=14, symbol="star"),
            ))

        fig_curve.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            height=350,
            margin=dict(l=60, r=20, t=30, b=40),
            xaxis_title="Budget (New Beds)",
            yaxis_title="Coverage (%)",
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
            **PLOTLY_FONT,
        )
        st.plotly_chart(fig_curve, width="stretch", key="chart_curve")

    st.markdown("---")

    col_map2, col_bar2 = st.columns([3, 2])

    with col_map2:
        st.markdown("##### Allocation Map")
        fig_alloc_map = make_germany_choropleth(
            df_alloc_filt, "Beds_Allocated", "Tealgrn", "Beds",
            hover_cols={
                "Deficit": ":.0f",
                "Beds_Allocated": ":.0f",
                "Remaining_Deficit": ":.0f",
            },
        )
        st.plotly_chart(fig_alloc_map, key="map_alloc")
        
        # Alert for unmapped districts
        unmatched_ags = df_mapping[df_mapping["Match_Type"] == "unmatched"]["AGS"].tolist()
        unmatched_in_filt = df_alloc_filt[df_alloc_filt["AGS"].isin(unmatched_ags)]
        if not unmatched_in_filt.empty:
            with st.expander("⚠️ View districts not shown on map (due to historical boundary changes in GeoJSON):"):
                st.caption("These districts are still fully included in the optimizer and all data tables:")
                st.dataframe(
                    unmatched_in_filt[["Region", "Deficit", "Beds_Allocated", "Remaining_Deficit"]]
                    .sort_values("Beds_Allocated", ascending=False)
                    .style.format({"Deficit": "{:,.0f}", "Beds_Allocated": "{:,.0f}", "Remaining_Deficit": "{:,.0f}"}),
                    hide_index=True,
                    width="stretch"
                )

    with col_bar2:
        st.markdown("##### Top 10 Districts by Beds Allocated")
        top10_alloc = df_alloc_filt.nlargest(10, "Beds_Allocated")
        fig_bar2 = px.bar(
            top10_alloc, x="Beds_Allocated", y="Region", orientation="h",
            template=PLOTLY_TEMPLATE,
        )
        fig_bar2.update_traces(marker_color=COLOR_POSITIVE)
        fig_bar2.update_layout(
            yaxis=dict(autorange="reversed", tickfont=dict(size=11, color="#1a2a42")),
            margin=dict(l=10, r=20, t=10, b=10),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            height=550,
            showlegend=False,
            **PLOTLY_FONT,
        )
        st.plotly_chart(fig_bar2, key="bar_alloc")

    # Detail table
    st.markdown("---")
    st.markdown("##### 📋 Full Allocation Table")
    display_cols = [
        "AGS", "Region", "Deficit", "Beds_Allocated",
        "Remaining_Deficit", "Unmet_Ratio",
    ]
    st.dataframe(
        df_alloc_filt[display_cols]
        .sort_values("Beds_Allocated", ascending=False)
        .style.format({
            "Deficit": "{:,.0f}",
            "Beds_Allocated": "{:,.0f}",
            "Remaining_Deficit": "{:,.0f}",
            "Unmet_Ratio": "{:.2%}",
        }),
        hide_index=True,
        height=400,
    )

    # ── Download Allocation Plan ──
    st.markdown("---")
    csv_alloc = convert_df_to_csv(
        df_alloc_filt[display_cols].sort_values("Beds_Allocated", ascending=False)
    )
    st.download_button(
        label="📥 Download Allocation Plan (CSV)",
        data=csv_alloc,
        file_name=f"pflegeoptix_allocation_plan_{budget}_beds.csv",
        mime="text/csv",
        width="stretch",
    )


# ================================================================
# TAB 4 — DISTRICT COMPARISON
# ================================================================
with tab4:
    st.subheader("⚖️ District Comparison")
    st.caption("Compare key metrics and demand between two districts side-by-side.")
    
    if selected_state != "All States":
        comp_districts = df_pulp[df_pulp["State"] == selected_state]["Region"].unique()
    else:
        comp_districts = df_pulp["Region"].unique()
        
    comp_districts = sorted(set(r.strip() for r in comp_districts))
    
    if len(comp_districts) > 1:
        c_sel1, c_sel2 = st.columns(2)
        with c_sel1:
            d1 = st.selectbox("Select District A", comp_districts, index=0)
        with c_sel2:
            d2 = st.selectbox("Select District B", comp_districts, index=min(1, len(comp_districts)-1))
            
        if d1 and d2:
            st.markdown("---")
            
            # Get data for D1
            d1_pulp = df_pulp[df_pulp["Region"].str.strip() == d1].iloc[0]
            ags1 = d1_pulp["AGS"]
            d1_hist_df = df_master[(df_master["AGS"] == ags1) & (df_master["Year"] == df_master["Year"].max())]
            d1_hist = d1_hist_df.iloc[0] if not d1_hist_df.empty else None
            d1_fc_df = df_forecast[(df_forecast["AGS"] == ags1) & (df_forecast["Year"] == 2040)]
            d1_fc = d1_fc_df.iloc[0] if not d1_fc_df.empty else None
            
            # Get data for D2
            d2_pulp = df_pulp[df_pulp["Region"].str.strip() == d2].iloc[0]
            ags2 = d2_pulp["AGS"]
            d2_hist_df = df_master[(df_master["AGS"] == ags2) & (df_master["Year"] == df_master["Year"].max())]
            d2_hist = d2_hist_df.iloc[0] if not d2_hist_df.empty else None
            d2_fc_df = df_forecast[(df_forecast["AGS"] == ags2) & (df_forecast["Year"] == 2040)]
            d2_fc = d2_fc_df.iloc[0] if not d2_fc_df.empty else None
            
            # ── Metric Cards Side-by-Side ──
            c_val1, c_val2 = st.columns(2)
            
            with c_val1:
                st.markdown(f"#### 🏥 {d1}")
                st.metric("Care Gap (Beds Needed)", f"{d1_pulp['Care_Gap']:,.1f}")
                st.metric("Current Deficit", f"{d1_pulp['Deficit']:,.0f}")
                st.metric("Current Beds", f"{d1_pulp['Current_Beds']:,.0f}")
                st.metric("Demand 2040", f"{d1_pulp['Demand_2040']:,.0f}")
                if d1_fc is not None and d1_hist is not None:
                    pop_now1 = d1_hist["Total"]
                    pop_fc1 = d1_fc["Pop_Total_Forecast"]
                    growth1 = (pop_fc1 - pop_now1) / pop_now1 * 100
                    st.metric("Pop. Growth (2040)", f"{growth1:+.1f}%")
                if d1_hist is not None:
                    st.metric("Population Density", f"{d1_hist['Population_Density']:,.0f}")
                    st.metric("Single-Person HH %", f"{d1_hist['Single_Person_Households_Pct']:.1f}%")
                
            with c_val2:
                st.markdown(f"#### 🏥 {d2}")
                st.metric("Care Gap (Beds Needed)", f"{d2_pulp['Care_Gap']:,.1f}")
                st.metric("Current Deficit", f"{d2_pulp['Deficit']:,.0f}")
                st.metric("Current Beds", f"{d2_pulp['Current_Beds']:,.0f}")
                st.metric("Demand 2040", f"{d2_pulp['Demand_2040']:,.0f}")
                if d2_fc is not None and d2_hist is not None:
                    pop_now2 = d2_hist["Total"]
                    pop_fc2 = d2_fc["Pop_Total_Forecast"]
                    growth2 = (pop_fc2 - pop_now2) / pop_now2 * 100
                    st.metric("Pop. Growth (2040)", f"{growth2:+.1f}%")
                if d2_hist is not None:
                    st.metric("Population Density", f"{d2_hist['Population_Density']:,.0f}")
                    st.metric("Single-Person HH %", f"{d2_hist['Single_Person_Households_Pct']:.1f}%")

            st.markdown("---")

            # ── Bar Chart Comparison ──
            st.markdown("##### 📊 Side-by-Side Comparison")
            compare_metrics = ["Care_Gap", "Deficit", "Current_Beds", "Demand_2040"]
            compare_labels  = ["Care Gap", "Deficit", "Current Beds", "Demand 2040"]
            d1_vals = [float(d1_pulp[m]) for m in compare_metrics]
            d2_vals = [float(d2_pulp[m]) for m in compare_metrics]

            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                name=d1, x=compare_labels, y=d1_vals,
                marker_color=COLOR_PRIMARY,
                text=[f"{v:,.0f}" for v in d1_vals], textposition="outside",
            ))
            fig_comp.add_trace(go.Bar(
                name=d2, x=compare_labels, y=d2_vals,
                marker_color=COLOR_FORECAST,
                text=[f"{v:,.0f}" for v in d2_vals], textposition="outside",
            ))
            fig_comp.update_layout(
                barmode="group",
                template=PLOTLY_TEMPLATE,
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
                **PLOTLY_FONT,
            )
            st.plotly_chart(fig_comp, width="stretch", key="chart_comp_bar")

            # ── Policy Recommendation ──
            st.markdown("---")
            st.markdown("##### 💡 Multi-Criteria AI Policy Analysis")
            
            if d1 == d2:
                st.info("⚠️ Same district selected. Please choose two different districts for comparison.")
            else:
                gap1 = float(d1_pulp["Care_Gap"])
                gap2 = float(d2_pulp["Care_Gap"])
                higher_gap = d1 if gap1 > gap2 else d2
                lower_gap  = d2 if gap1 > gap2 else d1
                diff_gap   = abs(gap1 - gap2)
                
                # Densities
                dens1 = float(d1_hist['Population_Density']) if d1_hist is not None else 100.0
                dens2 = float(d2_hist['Population_Density']) if d2_hist is not None else 100.0
                
                # Single HH percentages
                hh1 = float(d1_hist['Single_Person_Households_Pct']) if d1_hist is not None else 30.0
                hh2 = float(d2_hist['Single_Person_Households_Pct']) if d2_hist is not None else 30.0
                
                # Growth
                grow1 = growth1 if (d1_fc is not None and d1_hist is not None) else 0.0
                grow2 = growth2 if (d2_fc is not None and d2_hist is not None) else 0.0
                
                # Classification: Urban (> 500 people/km²) or Rural (<= 500)
                type1 = "Urban 🏙️" if dens1 > 500 else "Rural 🌲"
                type2 = "Urban 🏙️" if dens2 > 500 else "Rural 🌲"
                
                # Draft the dynamic insights
                insights = []
                
                # 1. Care Gap Urgency
                if diff_gap < 10:
                    insights.append(f"🟢 **Care Gap Urgency:** Both districts face a similar Care Gap intensity (difference of only `{diff_gap:,.1f}` beds). A cooperative regional investment approach is recommended.")
                else:
                    insights.append(f"⚠️ **Care Gap Urgency:** **{higher_gap}** has a significantly larger Care Gap (+`{diff_gap:,.1f}` beds) compared to **{lower_gap}**, making it the primary target for immediate capital funding.")
                
                # 2. Growth and Future Pressure
                if abs(grow1 - grow2) > 2:
                    faster_grow = d1 if grow1 > grow2 else d2
                    slower_grow = d2 if grow1 > grow2 else d1
                    insights.append(f"📈 **Future Pressure:** **{faster_grow}**'s population is projected to grow faster (`{max(grow1, grow2):+.1f}%`) than **{slower_grow}**'s (`{min(grow1, grow2):+.1f}%`). This implies that care bed deficits in **{faster_grow}** will accelerate, requiring proactive planning.")
                
                # 3. Social Support (Single households)
                if abs(hh1 - hh2) > 3:
                    higher_single = d1 if hh1 > hh2 else d2
                    insights.append(f"👤 **Vulnerability Profile:** **{higher_single}** has a higher rate of single-person households (`{max(hh1, hh2):.1f}%`). In elderly care planning, this indicates a lower availability of home-based family care, creating higher dependency on institutional care home places.")
                
                # 4. Urban vs Rural Structural Policy
                if type1 != type2:
                    urban_district = d1 if type1 == "Urban 🏙️" else d2
                    rural_district = d2 if type1 == "Urban 🏙️" else d1
                    insights.append(f"🗺️ **Structural Strategy:**  \n"
                                    f"  * For **{urban_district}** (Urban): Focus should be on high-density care centers, optimizing urban land use, and tackling construction costs.  \n"
                                    f"  * For **{rural_district}** (Rural): Focus should be on developing decentralized outpatient networks, mobile care services, and community nursing, since travel times are higher and population is dispersed.")
                else:
                    if type1 == "Urban 🏙️":
                        insights.append("🏙️ **Structural Strategy:** Both are high-density urban areas. Policy should prioritize vertical care facilities, community integration, and easing land-acquisition regulations.")
                    else:
                        insights.append("🌲 **Structural Strategy:** Both are rural districts. Focus on network integration, home-care assistance support, and financial incentives to attract qualified care staff to outlying areas.")
                
                # Display insights
                st.info("\n\n".join(insights))
    else:
        st.info("Not enough districts to compare. Please select a larger state or 'All States'.")


# ================================================================
# TAB 5 — MODEL INSIGHTS (SHAP + Ablation)
# ================================================================
with tab5:
    st.subheader("🧠 Model Reference & Insights")
    st.caption("Technical documentation for AI transparency (Explainable AI).")

    st.markdown("---")

    # ── SHAP Analysis ──
    st.markdown("##### 🧠 SHAP Feature Importance (Clean Model)")
    st.caption("XGBoost Clean Model — 14 features, proxy leakage removed")

    col_bar_shap, col_bee = st.columns(2)

    shap_bar_path = FIG_DIR / "shap_bar_clean.png"
    shap_bee_path = FIG_DIR / "shap_beeswarm_clean.png"

    with col_bar_shap:
        st.markdown("###### Mean |SHAP value| (Feature Importance)")
        if shap_bar_path.exists():
            st.markdown('<div class="shap-container">', unsafe_allow_html=True)
            st.image(str(shap_bar_path), width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("shap_bar_clean.png not found.")

    with col_bee:
        st.markdown("###### SHAP Beeswarm (Value Distribution)")
        if shap_bee_path.exists():
            st.markdown('<div class="shap-container">', unsafe_allow_html=True)
            st.image(str(shap_bee_path), width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("shap_beeswarm_clean.png not found.")

    st.markdown("---")

    # ── Key Insights ──
    st.markdown("##### 🔑 Key Policy Insights from SHAP")
    i1, i2, i3 = st.columns(3)

    with i1:
        st.markdown("""
        <div class="insight-card">
        <h4>🏥 Premature Mortality</h4>
        <p>Districts with higher premature death rates
        show significantly higher care bed demand,
        reflecting poorer overall health outcomes.</p>
        </div>
        """, unsafe_allow_html=True)

    with i2:
        st.markdown("""
        <div class="insight-card">
        <h4>🏙️ Population Density</h4>
        <p>Urban districts have higher institutional care
        demand because limited housing reduces the
        viability of home-based care.</p>
        </div>
        """, unsafe_allow_html=True)

    with i3:
        st.markdown("""
        <div class="insight-card">
        <h4>👤 Single-Person HH</h4>
        <p>Elderly living alone rely more heavily on
        care homes, as there is no family caregiver
        available at home.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Ablation Study ──
    st.markdown("##### 📊 Ablation Study: Full Model vs. Clean Model")
    comparison = pd.DataFrame({
        "Metric": ["R² (Train)", "R² (Test)", "MAE (Test)", "RMSE (Test)"],
        "Full Model (15 feat.)": [0.9164, 0.9036, 0.000451, 0.000940],
        "Clean Model (14 feat.)": [0.5636, 0.4341, 0.001739, 0.002278],
    })
    st.dataframe(
        comparison.style.format({
            "Full Model (15 feat.)":  "{:.4f}",
            "Clean Model (14 feat.)": "{:.4f}",
        }),
        hide_index=True,
    )
    st.caption(
        "⚠️ The Full Model's high R² is inflated by proxy leakage "
        "(Care_Home_Places_per_Pop ≈ 84.8% SHAP dominance). "
        "The Clean Model reflects the true socioeconomic predictive power."
    )

    st.markdown("""
    <div style="background-color:rgba(249, 115, 22, 0.1); border-left: 5px solid #f97316; padding: 15px; border-radius: 4px; margin-top: 15px; margin-bottom: 20px;">
        <h4 style="color:#f97316; margin-top:0;">💡 Understanding the "Ablation Study" (Proxy Leakage)</h4>
        <p style="font-size:0.95rem; line-height:1.5; color:#e0e6ed;">
            <b>Why is a lower R² (0.43) better than a higher R² (0.90)?</b><br>
            In predictive modeling, using existing supply to predict demand creates a feedback loop known as <b>Proxy Leakage</b>. 
            For example, using the number of existing care homes to predict where new care homes are needed will show perfect accuracy (R² = 0.90), 
            but it is completely useless for planning — it just tells you where care homes already are.
        </p>
        <p style="font-size:0.95rem; line-height:1.5; margin-bottom:0; color:#e0e6ed;">
            <i><b>Analogy:</b> Predicting tomorrow's weather by looking at how many people are holding umbrellas today is highly accurate, 
            but it doesn't give you a real weather forecast. It just describes the present.</i>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

# ================================================================
# TAB 6 — ABOUT PROJECT (Methodology & Limitations)
# ================================================================
with tab6:
    st.subheader("ℹ️ About This Project")
    st.caption("Methodology, Data Sources, and Known Limitations.")

    # ── Methodology Pipeline ──
    st.markdown("##### ⚙️ Methodology Pipeline")
    st.markdown("""
    | Sprint | Component | Method | Output |
    |:---:|---|---|---|
    | 1 | Data Engineering | Cleaning + Imputation | `master_clean.csv` (399 districts) |
    | 2 | Population Forecast | Prophet (2025-2040) | `forecast_clean.csv` |
    | 3 | Demand Prediction | XGBoost + SHAP | Feature importance + Care Gap |
    | 3 | Resource Allocation | PuLP (Min-Max Fairness) | Optimal bed distribution |
    | 4 | Dashboard | Streamlit + Plotly + Mapbox | Interactive decision tool |
    """)

    st.markdown("---")

    # ── Data Sources ──
    st.markdown("##### 📚 Data Sources")
    st.markdown("""
    - **INKAR** — Indicators and Maps on Spatial Development (BBSR, 2015-2023)
    - **Pflegestatistik** — German Federal Statistical Office (care facility data)
    - **GeoJSON** — District boundaries via isellsoap/deutschlandGeoJSON (GitHub)
    """)

    st.markdown("---")

    # ── Limitations ──
    st.markdown("##### ⚠️ Known Limitations")
    st.markdown("""
    - **Berlin excluded**: 12 Bezirke in statistical data vs. 1 entity in GeoJSON — data mismatch
    - **16 unmapped districts**: Historical boundary changes prevent GeoJSON matching (96% coverage)
    - **Clean Model R² = 0.43**: Intentionally lower after removing proxy leakage — reflects true socioeconomic signal
    - **Static SHAP**: Pre-computed on training data; does not update with slider changes
    """)


# ────────────────────────────────────────────────────────────────
# FOOTER
# ────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "PflegeOptix © 2026 — Capstone Project | "
    "Data: INKAR, Pflegestatistik | "
    "ML: XGBoost + SHAP | Optimization: PuLP (Min-Max Fairness)"
)
