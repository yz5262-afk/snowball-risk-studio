from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from snowball_engine import (
    SnowballParams,
    barrier_schedule,
    generate_normals,
    greeks_at_spot,
    greeks_curve,
    make_analysis_zip,
    observation_delta_table,
    outcome_analysis,
    seed_stability,
    target_coupon_for_par,
    target_coupon_sensitivity,
)


# ============================================================
# App configuration & theme
# ============================================================

st.set_page_config(
    page_title="Snowball Risk Studio",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #122033;
        --muted: #607086;
        --border: #dce4ec;
        --panel: #f7f9fc;
        --accent: #0f766e;
        --accent-soft: #e7f4f1;
    }
    .block-container { max-width: 1500px; padding-top: 1.1rem; padding-bottom: 2.5rem; }
    .hero {
        padding: 1.35rem 1.5rem 1.2rem 1.5rem;
        border: 1px solid var(--border);
        border-radius: 18px;
        background: linear-gradient(135deg, #f9fbfd 0%, #eef7f5 100%);
        margin-bottom: 1rem;
    }
    .hero-kicker {
        display: inline-block;
        padding: .28rem .55rem;
        border-radius: 999px;
        background: var(--accent-soft);
        color: var(--accent);
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .04em;
        text-transform: uppercase;
        margin-bottom: .55rem;
    }
    .hero-title { font-size: 2.15rem; font-weight: 800; color: var(--ink); margin: 0; }
    .hero-subtitle { color: var(--muted); margin-top: .35rem; font-size: 1rem; }
    .section-card {
        border: 1px solid var(--border);
        border-radius: 16px;
        background: white;
        padding: 1rem 1rem .75rem 1rem;
    }
    .small-note { color: var(--muted); font-size: .86rem; }
    div[data-testid="stMetric"] {
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: .85rem .95rem;
        background: rgba(255,255,255,.9);
    }
    div[data-testid="stMetricLabel"] { color: var(--muted); }
    button[kind="primary"] {
        border-radius: 10px;
        font-weight: 700;
    }
    .sidebar-note {
        padding: .7rem .8rem;
        border-radius: 10px;
        background: #f6f8fa;
        border: 1px solid var(--border);
        font-size: .82rem;
        color: var(--muted);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Helpers
# ============================================================

BASE_PARAMS = SnowballParams()


def pct(x: float | int | None, digits: int = 2) -> str:
    if x is None or not np.isfinite(float(x)):
        return "—"
    return f"{float(x):.{digits}%}"


def num(x: float | int | None, digits: int = 4) -> str:
    if x is None or not np.isfinite(float(x)):
        return "—"
    return f"{float(x):,.{digits}f}"


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def apply_chart_style(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#edf1f5")
    fig.update_yaxes(showgrid=True, gridcolor="#edf1f5")
    return fig


def add_barriers(fig: go.Figure, params: SnowballParams, current_month: float | None = None) -> None:
    fig.add_vline(
        x=params.KI_barrier,
        line_dash="dot",
        line_color="#b45309",
        annotation_text=f"KI {params.KI_barrier:g}",
        annotation_position="top left",
    )
    if current_month is not None:
        month = int(np.floor(current_month))
        if params.lock_months < month < params.tenure_months:
            current_ko = params.ko_barrier(month)
            fig.add_vline(
                x=current_ko,
                line_dash="dash",
                line_color="#0f766e",
                annotation_text=f"KO @ m{month}: {current_ko:g}",
                annotation_position="top right",
            )
    else:
        fig.add_vline(
            x=params.KO_start,
            line_dash="dash",
            line_color="#0f766e",
            annotation_text=f"Initial KO {params.KO_start:g}",
            annotation_position="top right",
        )


def get_active_params() -> SnowballParams:
    return st.session_state.get("active_params", BASE_PARAMS)


@st.cache_data(show_spinner=False)
def cached_normals(n_sims: int, tenure_months: int, seed: int) -> np.ndarray:
    params = SnowballParams(n_sims=n_sims, tenure_months=tenure_months)
    return generate_normals(params, seed)


@st.cache_data(show_spinner=False)
def cached_day0(params: SnowballParams, seed: int, dS: float, target_price: float):
    Z = cached_normals(params.n_sims, params.tenure_months, seed)
    greeks = greeks_at_spot(params.S0, 0.0, Z, params, dS=dS)
    target_coupon = target_coupon_for_par(params, Z, target_price=target_price)
    target_params = replace(params, coupon=target_coupon)
    target_greeks = greeks_at_spot(params.S0, 0.0, Z, target_params, dS=dS)
    return greeks, target_coupon, target_greeks


@st.cache_data(show_spinner=False)
def cached_outcomes(params: SnowballParams, seed: int):
    Z = cached_normals(params.n_sims, params.tenure_months, seed)
    return outcome_analysis(params, Z)


@st.cache_data(show_spinner=False)
def cached_curve(
    params: SnowballParams,
    seed: int,
    spots_tuple: tuple[float, ...],
    time_months: float,
    dS: float,
) -> pd.DataFrame:
    Z = cached_normals(params.n_sims, params.tenure_months, seed)
    return greeks_curve(np.asarray(spots_tuple), time_months / 12.0, Z, params, dS=dS)


@st.cache_data(show_spinner=False)
def cached_observation_table(
    params: SnowballParams,
    seed: int,
    spot: float,
    dS: float,
) -> pd.DataFrame:
    Z = cached_normals(params.n_sims, params.tenure_months, seed)
    return observation_delta_table(spot, params, Z, dS=dS)


@st.cache_data(show_spinner=False)
def cached_target_coupon_sensitivity(
    params: SnowballParams,
    seed: int,
    target_price: float,
    q_grid_tuple: tuple[float, ...],
    vol_grid_tuple: tuple[float, ...],
):
    return target_coupon_sensitivity(
        params,
        seed,
        q_values=q_grid_tuple,
        vol_values=vol_grid_tuple,
        target_price=target_price,
    )


@st.cache_data(show_spinner=False)
def cached_seed_stability(
    params: SnowballParams,
    seeds_tuple: tuple[int, ...],
    target_price: float,
):
    return seed_stability(params, seeds_tuple, target_price)


# ============================================================
# Sidebar: deal ticket
# ============================================================

active_params = get_active_params()

with st.sidebar:
    st.markdown("### ❄️ Snowball Risk Studio")
    st.caption("Interactive European Snowball pricing & risk dashboard")

    with st.form("deal_ticket", clear_on_submit=False):
        st.markdown("#### Structure")
        S0 = st.number_input("Initial underlying S₀", min_value=0.01, value=float(active_params.S0), step=1.0)
        KI_barrier = st.number_input("KI barrier", min_value=0.01, value=float(active_params.KI_barrier), step=1.0)
        KO_start = st.number_input("Initial KO barrier", min_value=0.01, value=float(active_params.KO_start), step=1.0)
        SD_rate = st.number_input("Monthly KO step-down", min_value=0.0, value=float(active_params.SD_rate), step=0.1)
        lock_months = st.number_input("Lock-in period (months)", min_value=0, value=int(active_params.lock_months), step=1)
        tenure_months = st.number_input("Total term (months)", min_value=1, value=int(active_params.tenure_months), step=1)
        coupon_pct = st.number_input("Input coupon (% p.a.)", value=float(active_params.coupon * 100), step=0.25)

        st.markdown("#### Market")
        r_pct = st.number_input("Risk-free rate r (% p.a.)", value=float(active_params.r * 100), step=0.10)
        q_pct = st.number_input("Dividend yield q (% p.a.)", value=float(active_params.q * 100), step=0.10)
        vol_pct = st.number_input("Volatility σ (% p.a.)", min_value=0.0, value=float(active_params.vol * 100), step=0.50)

        st.markdown("#### Numerical controls")
        n_sims = st.number_input(
            "Monte Carlo paths",
            min_value=1_000,
            max_value=200_000,
            value=int(active_params.n_sims),
            step=1_000,
            help="Use 5k–10k for exploration, then 30k+ for the final case.",
        )
        seed = st.number_input("Random seed", min_value=0, value=42, step=1)
        dS = st.number_input("Finite-difference dS", min_value=0.01, value=0.5, step=0.1)
        target_price = st.number_input("Target issue price", min_value=0.01, value=100.0, step=1.0)

        st.markdown("#### Curve range")
        grid_min = st.number_input("Spot range — min", min_value=0.01, value=40.0, step=5.0)
        grid_max = st.number_input("Spot range — max", min_value=0.02, value=120.0, step=5.0)
        grid_points = st.slider("Curve points", min_value=21, max_value=121, value=61, step=2)

        submitted = st.form_submit_button("Run / update analysis", type="primary", use_container_width=True)

    if submitted:
        candidate = SnowballParams(
            S0=float(S0),
            KI_barrier=float(KI_barrier),
            KO_start=float(KO_start),
            SD_rate=float(SD_rate),
            lock_months=int(lock_months),
            tenure_months=int(tenure_months),
            coupon=float(coupon_pct) / 100.0,
            r=float(r_pct) / 100.0,
            q=float(q_pct) / 100.0,
            vol=float(vol_pct) / 100.0,
            n_sims=int(n_sims),
        )
        try:
            candidate.validate()
            if grid_min >= grid_max:
                raise ValueError("Spot range min must be smaller than max.")
            if grid_min <= dS:
                raise ValueError("Spot range min must be greater than dS.")
            st.session_state["active_params"] = candidate
            st.session_state["curve_min"] = float(grid_min)
            st.session_state["curve_max"] = float(grid_max)
            st.session_state["curve_points"] = int(grid_points)
            st.session_state["seed"] = int(seed)
            st.session_state["dS"] = float(dS)
            st.session_state["target_price"] = float(target_price)
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    # Restore active numerical settings after reruns.
    grid_min = float(st.session_state.get("curve_min", grid_min))
    grid_max = float(st.session_state.get("curve_max", grid_max))
    grid_points = int(st.session_state.get("curve_points", grid_points))
    seed = int(st.session_state.get("seed", seed))
    dS = float(st.session_state.get("dS", dS))
    target_price = float(st.session_state.get("target_price", target_price))

    st.markdown(
        '<div class="sidebar-note"><b>Model scope:</b> this is the European Snowball model from the supplied notebook. It is intended for research, education and model demonstration, not formal trade valuation.</div>',
        unsafe_allow_html=True,
    )


params = get_active_params()

try:
    params.validate()
except ValueError as exc:
    st.error(str(exc))
    st.stop()


# ============================================================
# Header
# ============================================================

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-kicker">Structured Products · Quant Risk</div>
        <div class="hero-title">Snowball Risk Studio</div>
        <div class="hero-subtitle">
            Explore valuation, Delta/Gamma, observation-date risk and target coupon behavior for a European Snowball.
            Current case: <b>S₀={params.S0:g}</b> · <b>{params.tenure_months}M</b> · <b>σ={pct(params.vol)}</b> · <b>q={pct(params.q)}</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Core analytics
# ============================================================

with st.spinner("Running Monte Carlo analytics…"):
    day0, fair_coupon, target_greeks = cached_day0(
        params, seed, dS, target_price
    )

npv = day0["price"] - target_price
coupon_gap = fair_coupon - params.coupon

# ============================================================
# KPI strip
# ============================================================

kpi = st.columns(6)
kpi[0].metric("Day-0 PV", num(day0["price"], 3))
kpi[1].metric("NPV vs target", f"{npv:+.3f}")
kpi[2].metric("Target coupon", pct(fair_coupon, 2))
kpi[3].metric("Coupon gap", f"{coupon_gap:+.2%}")
kpi[4].metric("Day-0 Delta", num(day0["delta"], 4))
kpi[5].metric("Day-0 Gamma", num(day0["gamma"], 6))

st.caption(
    "Target coupon = annualized coupon that makes Day-0 Monte Carlo PV equal the selected target issue price. "
    "Greeks use common-random-number central finite differences."
)


# ============================================================
# Navigation tabs
# ============================================================

tab_overview, tab_greeks, tab_event, tab_coupon, tab_method = st.tabs(
    ["Overview", "Greeks & Curves", "Observation Risk", "Target Coupon", "Model & Deployment"]
)


# ============================================================
# Overview
# ============================================================

with tab_overview:
    st.subheader("Executive overview")
    st.markdown(
        "Use this page as the quick read: structure schedule, outcome probabilities, and the current deal ticket."
    )

    with st.spinner("Estimating KO and maturity outcome probabilities…"):
        ko_df, outcome_df, outcome_summary = cached_outcomes(params, seed)

    a, b, c, d = st.columns(4)
    a.metric("KO probability", pct(outcome_summary["ko_probability"]))
    b.metric("Maturity safe", pct(outcome_summary["maturity_safe_probability"]))
    c.metric("Maturity loss", pct(outcome_summary["maturity_loss_probability"]))
    d.metric(
        "Expected KO month | KO",
        "—" if np.isnan(outcome_summary["expected_ko_month_conditional"]) else f"M{outcome_summary['expected_ko_month_conditional']:.1f}",
    )

    left, right = st.columns([1.1, 1])
    with left:
        barrier_df = barrier_schedule(params)
        fig_barrier = go.Figure()
        fig_barrier.add_trace(
            go.Scatter(
                x=barrier_df["Observation_Month"],
                y=barrier_df["KO_Barrier"],
                mode="lines+markers",
                name="KO barrier",
                line=dict(color="#0f766e", width=3),
                marker=dict(size=7),
            )
        )
        fig_barrier.add_hline(
            y=params.KI_barrier,
            line_dash="dot",
            line_color="#b45309",
            annotation_text=f"KI = {params.KI_barrier:g}",
        )
        fig_barrier.update_layout(
            title="Knock-out barrier schedule",
            xaxis_title="Observation month",
            yaxis_title="Barrier level",
        )
        st.plotly_chart(apply_chart_style(fig_barrier, 430), use_container_width=True)

    with right:
        fig_outcome = go.Figure(
            go.Bar(
                x=outcome_df["Outcome"],
                y=outcome_df["Probability"],
                text=[f"{x:.1%}" for x in outcome_df["Probability"]],
                textposition="outside",
                marker_color=["#0f766e", "#64748b", "#b45309"],
            )
        )
        fig_outcome.update_layout(
            title="Monte Carlo outcome mix",
            xaxis_title="Outcome",
            yaxis_title="Probability",
        )
        fig_outcome.update_yaxes(tickformat=".0%", range=[0, max(1.0, float(outcome_df["Probability"].max()) * 1.2)])
        st.plotly_chart(apply_chart_style(fig_outcome, 430), use_container_width=True)

    fig_ko = px.bar(
        ko_df,
        x="Observation_Month",
        y="First_KO_Probability",
        hover_data=["KO_Barrier"],
        title="First knock-out probability by observation month",
    )
    fig_ko.update_yaxes(tickformat=".1%")
    st.plotly_chart(apply_chart_style(fig_ko, 400), use_container_width=True)

    overview_tables = {
        "barrier_schedule": barrier_df,
        "first_KO_probability": ko_df,
        "outcome_probabilities": outcome_df,
    }
    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "target_price": target_price,
        "analysis": "Snowball Risk Studio overview",
    }
    zip_data = make_analysis_zip(params, overview_tables, metadata)
    st.download_button(
        "Download overview analysis package (.zip)",
        data=zip_data,
        file_name="snowball_overview_analysis.zip",
        mime="application/zip",
        use_container_width=False,
    )


# ============================================================
# Greeks & Curves
# ============================================================

with tab_greeks:
    st.subheader("Greeks & scenario curves")
    st.markdown(
        "Visualize how the Snowball's Price, Delta and Gamma change with the underlying level and the valuation time."
    )

    controls = st.columns([1.2, 1, 1])
    with controls[0]:
        default_times = sorted(
            {
                0,
                min(params.lock_months, params.tenure_months - 1),
                min(6, params.tenure_months - 1),
                min(12, params.tenure_months - 1),
                min(18, params.tenure_months - 1),
            }
        )
        default_times = [x for x in default_times if 0 <= x <= params.tenure_months]
        selected_times = st.multiselect(
            "Valuation time (months)",
            list(range(params.tenure_months + 1)),
            default=default_times,
        )
    with controls[1]:
        reference_spot = st.number_input(
            "Reference spot",
            min_value=float(grid_min),
            max_value=float(grid_max),
            value=float(np.clip(params.S0, grid_min, grid_max)),
            step=1.0,
        )
    with controls[2]:
        metric_choice = st.selectbox("Primary curve", ["Delta", "Price", "Gamma"], index=0)

    spots = np.linspace(grid_min, grid_max, grid_points)
    if selected_times:
        frames = []
        with st.spinner("Computing scenario curves…"):
            for month in selected_times:
                frame = cached_curve(params, seed, tuple(float(x) for x in spots), float(month), dS).copy()
                frame["Time_Label"] = f"t = {month}M"
                frames.append(frame)
        curve_df = pd.concat(frames, ignore_index=True)

        fig_main = px.line(
            curve_df,
            x="Spot",
            y=metric_choice,
            color="Time_Label",
            title=f"{metric_choice} vs underlying spot",
        )
        add_barriers(fig_main, params)
        fig_main.add_vline(x=params.S0, line_dash="dot", line_color="#475569", annotation_text="S₀")
        if metric_choice != "Price":
            fig_main.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
        st.plotly_chart(apply_chart_style(fig_main, 540), use_container_width=True)

        nearest = (
            curve_df.assign(Distance=lambda x: np.abs(x["Spot"] - reference_spot))
            .sort_values(["Time_Months", "Distance"])
            .groupby("Time_Label", as_index=False)
            .first()[["Time_Label", "Spot", "Price", "Delta", "Gamma"]]
        )
        st.markdown("#### Reference-point snapshot")
        st.dataframe(nearest, use_container_width=True, hide_index=True)

        d1, d2 = st.columns(2)
        with d1:
            fig_price = px.line(curve_df, x="Spot", y="Price", color="Time_Label", title="Price vs spot")
            add_barriers(fig_price, params)
            st.plotly_chart(apply_chart_style(fig_price, 420), use_container_width=True)
        with d2:
            fig_gamma = px.line(curve_df, x="Spot", y="Gamma", color="Time_Label", title="Gamma vs spot")
            add_barriers(fig_gamma, params)
            fig_gamma.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
            st.plotly_chart(apply_chart_style(fig_gamma, 420), use_container_width=True)

        st.download_button(
            "Download Greeks curves CSV",
            data=csv_bytes(curve_df),
            file_name="snowball_greeks_curves.csv",
            mime="text/csv",
        )

        st.markdown("#### Delta heatmap")
        run_heatmap = st.checkbox("Generate Delta heatmap", value=False)
        if run_heatmap:
            heat_cols = st.columns(2)
            with heat_cols[0]:
                heat_step = st.slider("Time step (months)", 1, max(1, min(6, params.tenure_months)), 3)
            with heat_cols[1]:
                heat_points = st.slider("Spot points", 21, 61, 31, step=2)

            heat_times = list(range(0, params.tenure_months, heat_step))
            if params.tenure_months - 1 not in heat_times:
                heat_times.append(max(0, params.tenure_months - 1))
            heat_spots = np.linspace(grid_min, grid_max, heat_points)

            rows = []
            with st.spinner("Computing heatmap…"):
                for month in sorted(set(heat_times)):
                    hdf = cached_curve(params, seed, tuple(float(x) for x in heat_spots), float(month), dS)
                    rows.append(hdf[["Spot", "Delta"]].assign(Time_Months=month))
            heat_df = pd.concat(rows, ignore_index=True)
            pivot = heat_df.pivot(index="Time_Months", columns="Spot", values="Delta")
            fig_heat = px.imshow(
                pivot,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                labels={"x": "Spot", "y": "Valuation time (months)", "color": "Delta"},
                title="Delta surface across spot and time",
            )
            st.plotly_chart(apply_chart_style(fig_heat, 560), use_container_width=True)
    else:
        st.info("Select at least one valuation time to render the curves.")


# ============================================================
# Observation risk
# ============================================================

with tab_event:
    st.subheader("Observation-date Delta risk")
    st.markdown(
        "Compare Delta one day before and one day after each monthly observation. "
        "The jump is a model-time discontinuity around the observation event under the notebook's discrete observation convention."
    )

    obs_spot = st.number_input(
        "Spot used for observation-date analysis",
        min_value=0.01,
        value=float(params.S0),
        step=1.0,
        key="obs_spot",
    )

    with st.spinner("Computing observation-date Delta jumps…"):
        obs_df = cached_observation_table(params, seed, float(obs_spot), dS)

    line_df = obs_df.melt(
        id_vars=["Observation_Month", "KO_Barrier"],
        value_vars=["Delta_Before", "Delta_After"],
        var_name="Position",
        value_name="Delta",
    )
    fig_obs = px.line(
        line_df,
        x="Observation_Month",
        y="Delta",
        color="Position",
        markers=True,
        title="Delta immediately around observation dates",
    )
    fig_obs.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
    st.plotly_chart(apply_chart_style(fig_obs, 440), use_container_width=True)

    fig_jump = px.bar(
        obs_df.dropna(subset=["Delta_Jump"]),
        x="Observation_Month",
        y="Delta_Jump",
        hover_data=["KO_Barrier", "Delta_Before", "Delta_After"],
        title="Delta jump = Delta(after) − Delta(before)",
    )
    fig_jump.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
    st.plotly_chart(apply_chart_style(fig_jump, 420), use_container_width=True)

    event_cols = st.columns(4)
    ranked = obs_df.dropna(subset=["Delta_Jump"]).assign(Abs_Delta_Jump=lambda x: x["Delta_Jump"].abs())
    if not ranked.empty:
        max_abs = ranked.loc[ranked["Abs_Delta_Jump"].idxmax()]
        max_pos = ranked.loc[ranked["Delta_Jump"].idxmax()]
        max_neg = ranked.loc[ranked["Delta_Jump"].idxmin()]
        event_cols[0].metric("Largest |Delta jump|", num(max_abs["Abs_Delta_Jump"], 5))
        event_cols[1].metric("Largest jump month", f"M{int(max_abs['Observation_Month'])}")
        event_cols[2].metric("Largest positive jump", f"M{int(max_pos['Observation_Month'])} · {max_pos['Delta_Jump']:+.5f}")
        event_cols[3].metric("Largest negative jump", f"M{int(max_neg['Observation_Month'])} · {max_neg['Delta_Jump']:+.5f}")

    display_obs = obs_df.copy()
    display_obs["Abs_Delta_Jump"] = display_obs["Delta_Jump"].abs()
    display_obs = display_obs.sort_values("Abs_Delta_Jump", ascending=False)
    st.dataframe(display_obs, use_container_width=True, hide_index=True)
    st.download_button(
        "Download observation-date Delta table CSV",
        data=csv_bytes(obs_df),
        file_name="snowball_observation_delta_jump.csv",
        mime="text/csv",
    )


# ============================================================
# Target coupon & scenarios
# ============================================================

with tab_coupon:
    st.subheader("Target coupon & scenario analysis")
    st.markdown(
        "The target coupon is the annual coupon that makes the Day-0 Monte Carlo PV equal the selected target issue price."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Input coupon", pct(params.coupon))
    c2.metric("Target coupon", pct(fair_coupon))
    c3.metric("Coupon gap", f"{coupon_gap:+.2%}")
    c4.metric("PV at target coupon", num(target_greeks["price"], 4))

    if fair_coupon < 0:
        st.warning("The target coupon is negative under this scenario. Check the structure, market inputs and target issue price.")
    elif fair_coupon > 0.50:
        st.warning("The target coupon exceeds 50% p.a. Under this model this indicates a very demanding structure/market combination.")

    st.markdown("#### Sensitivity of target coupon")
    sens_cols = st.columns(2)
    with sens_cols[0]:
        q_low, q_high = st.slider(
            "Dividend yield range (%)",
            min_value=0.0,
            max_value=15.0,
            value=(0.0, max(7.0, min(15.0, params.q * 100 + 4.0))),
            step=0.5,
        )
    with sens_cols[1]:
        vol_low, vol_high = st.slider(
            "Volatility range (%)",
            min_value=1.0,
            max_value=80.0,
            value=(max(1.0, params.vol * 100 - 10.0), min(80.0, params.vol * 100 + 10.0)),
            step=1.0,
        )

    q_grid = np.linspace(q_low / 100.0, q_high / 100.0, 9)
    vol_grid = np.linspace(vol_low / 100.0, vol_high / 100.0, 9)
    with st.spinner("Computing target-coupon sensitivity…"):
        q_sens, vol_sens = cached_target_coupon_sensitivity(
            params,
            seed,
            target_price,
            tuple(float(x) for x in q_grid),
            tuple(float(x) for x in vol_grid),
        )

    s1, s2 = st.columns(2)
    with s1:
        fig_q = px.line(q_sens, x="Dividend_Yield", y="Target_Coupon", markers=True, title="Target coupon vs dividend yield")
        fig_q.update_xaxes(tickformat=".1%")
        fig_q.update_yaxes(tickformat=".1%")
        st.plotly_chart(apply_chart_style(fig_q, 400), use_container_width=True)
    with s2:
        fig_v = px.line(vol_sens, x="Volatility", y="Target_Coupon", markers=True, title="Target coupon vs volatility")
        fig_v.update_xaxes(tickformat=".1%")
        fig_v.update_yaxes(tickformat=".1%")
        st.plotly_chart(apply_chart_style(fig_v, 400), use_container_width=True)

    st.markdown("#### Seed stability")
    seeds = (42, 123, 456, 789, 1000)
    with st.spinner("Running seed-stability check…"):
        seed_df = cached_seed_stability(params, seeds, target_price)
    s_mean = float(seed_df["Target_Coupon"].mean())
    s_std = float(seed_df["Target_Coupon"].std(ddof=1))
    ss1, ss2, ss3 = st.columns(3)
    ss1.metric("Mean target coupon", pct(s_mean))
    ss2.metric("Std. dev. across seeds", pct(s_std, 3))
    ss3.metric("Range", f"{pct(seed_df['Target_Coupon'].min())} – {pct(seed_df['Target_Coupon'].max())}")
    st.dataframe(seed_df, use_container_width=True, hide_index=True)

    combined_sens = pd.concat(
        [
            q_sens.rename(columns={"Dividend_Yield": "Input", "Target_Coupon": "Target_Coupon"}).assign(Scenario="Dividend yield"),
            vol_sens.rename(columns={"Volatility": "Input", "Target_Coupon": "Target_Coupon"}).assign(Scenario="Volatility"),
        ],
        ignore_index=True,
    )[["Scenario", "Input", "Target_Coupon"]]

    st.download_button(
        "Download target-coupon sensitivity CSV",
        data=csv_bytes(combined_sens),
        file_name="snowball_target_coupon_sensitivity.csv",
        mime="text/csv",
    )


# ============================================================
# Model & deployment
# ============================================================

with tab_method:
    st.subheader("Model assumptions & scope")
    st.markdown(
        "This dashboard is a presentation layer around the pricing logic in the supplied `Snowball_Pricing(2).ipynb`."
    )

    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            **Core model**

            - Risk-neutral GBM with drift `r - q`.
            - Monthly KO observation after the lock-in period.
            - Linear KO step-down by `SD_rate` price points per month.
            - European KI: the terminal underlying level is checked only at maturity.
            - If knocked out, payoff is principal plus accrued coupon through the KO month.
            - If not knocked out and terminal spot is above KI, principal plus full-term coupon is paid.
            - If not knocked out and terminal spot is below KI, redemption follows `S_T / S0`.
            - Delta and Gamma use common-random-number central finite differences.
            """
        )
    with right:
        st.markdown(
            """
            **Important limitations**

            - Intermediate-time valuation assumes the note is still alive and does not take a real historical path state as input.
            - Time is measured in model months using `1/12` years; real calendars, holidays and business-day conventions are not modeled.
            - No volatility surface, skew, stochastic rates, funding costs, transaction costs or hedging slippage.
            - Monte Carlo results retain sampling error even with common random numbers.
            - This dashboard is for research, education and model demonstration; it is not formal trading or valuation advice.
            """
        )

    st.markdown("#### Current deal ticket")
    ticket = pd.DataFrame(
        {
            "Parameter": list(params.as_dict().keys()),
            "Value": list(params.as_dict().values()),
        }
    )
    st.dataframe(ticket, use_container_width=True, hide_index=True)

    st.markdown("#### Deployment")
    st.info(
        "The project is structured for Streamlit Community Cloud or another Streamlit host. "
        "The main entry point is `app.py`; dependencies are in `requirements.txt`."
    )
    st.code(
        "git add .\n"
        "git commit -m \"Build Snowball Risk Studio\"\n"
        "git push\n\n"
        "# Local only\n"
        "streamlit run app.py",
        language="bash",
    )
    st.caption(
        "Once deployed from GitHub, you can share the app URL with others; the dashboard runs in the browser and users do not need your local Python environment."
    )
