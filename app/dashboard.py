from __future__ import annotations

import json
import inspect
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    METRICS_DIR,
    MODELS_DIR,
    PRIMARY_DATASET_CSV,
    PROJECT_ROOT as CONFIG_PROJECT_ROOT,
    TARGET_COL,
    TIMESTAMP_COL,
)
from src.data_loader import (  # noqa: E402
    extract_features_targets,
    load_dataset,
    locate_dataset,
    validate_dataset,
)

PASS_COLOR = "#168C87"
FAIL_COLOR = "#D45B53"
NAVY = "#183247"
BLUE = "#3D7EA6"
AMBER = "#D99A36"
NEUTRAL = "#9AAAB5"
PLOT_TEMPLATE = "plotly_white"
PLOT_WIDTH_ARGS = (
    {"width": "stretch"}
    if "width" in inspect.signature(st.plotly_chart).parameters
    else {"use_container_width": True}
)
TABLE_WIDTH_ARGS = (
    {"width": "stretch"}
    if "width" in inspect.signature(st.dataframe).parameters
    else {"use_container_width": True}
)
METRIC_LABELS = {
    "balanced_accuracy_mean": "Balanced accuracy",
    "precision_mean": "Precision",
    "recall_mean": "Recall",
    "f1_mean": "F1",
    "roc_auc_mean": "ROC-AUC",
    "average_precision_mean": "Average precision",
}

st.set_page_config(
    page_title="Semiconductor Yield Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --navy: #183247; --teal: #168C87; --line: #DCE5E9; --muted: #617583; }
    .stApp { background: #F4F7F8; color: var(--navy); }
    [data-testid="stHeader"] { background: rgba(244,247,248,0.94); }
    [data-testid="stSidebar"] { background: #EDF2F4; border-right: 1px solid var(--line); }
    .block-container { max-width: 1440px; padding-top: 1.65rem; padding-bottom: 2rem; }
    h1, h2, h3 { color: var(--navy); letter-spacing: -0.02em; }
    h1 { font-size: 2.15rem; margin-bottom: .12rem; }
    h2 { font-size: 1.42rem; margin-top: .55rem; }
    .hero-subtitle { color: #527082; font-size: 1.05rem; margin-bottom: .5rem; }
    .hero-copy { color: var(--muted); max-width: 980px; line-height: 1.55; }
    .section-note { color: var(--muted); font-size: .92rem; margin: -.25rem 0 .85rem 0; }
    div[data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid var(--line); border-radius: 10px;
        padding: 13px 15px; box-shadow: 0 2px 8px rgba(24,50,71,.035);
    }
    div[data-testid="stMetricLabel"] { color: #617583; }
    div[data-testid="stMetricValue"] { color: var(--navy); }
    div[data-testid="stTabs"] button[data-baseweb="tab"] { color: #526A79 !important; font-weight: 600; }
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] { color: var(--teal) !important; }
    div[data-baseweb="tab-highlight"] { background-color: var(--teal) !important; }
    .status-line { color: #527082; font-size: .86rem; padding: .3rem 0 .75rem; }
    .stPlotlyChart { background: #FFFFFF; border: 1px solid var(--line); border-radius: 10px; padding: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_dataset() -> tuple[pd.DataFrame, str]:
    path = locate_dataset(PRIMARY_DATASET_CSV)
    if path is None:
        raise FileNotFoundError("No combined SECOM dataset was found in data/raw or the project root.")
    return load_dataset(path), str(path.resolve())


@st.cache_data(show_spinner=False)
def read_csv_artifact(path: str) -> pd.DataFrame | None:
    file_path = Path(path)
    if not file_path.is_file():
        return None
    frame = pd.read_csv(file_path)
    if frame.empty:
        return None
    return frame


@st.cache_data(show_spinner=False)
def read_json_artifact(path: str) -> dict | None:
    file_path = Path(path)
    if not file_path.is_file():
        return None
    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


@st.cache_data(show_spinner=False)
def get_test_scores(dataset_path: str, model_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Run inference only on the fixed held-out split; never fit or tune here."""
    data = load_dataset(dataset_path)
    features, target = extract_features_targets(data)
    model = joblib.load(model_path)
    feature_names = list(getattr(model, "feature_names_in_", []))
    if not feature_names:
        raise ValueError("The saved model does not expose its fitted sensor feature names.")
    missing_features = sorted(set(feature_names).difference(features.columns))
    if missing_features:
        raise ValueError(f"Saved model features are absent from the dataset: {missing_features[:5]}")
    _, X_test, _, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )
    probabilities = model.predict_proba(X_test.loc[:, feature_names])[:, 1]
    return y_test.to_numpy(dtype=int), probabilities


def chart_layout(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        margin=dict(l=20, r=20, t=54, b=24),
        font=dict(family="Arial, sans-serif", color=NAVY, size=12),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E9EEF0", zerolinecolor="#DCE5E9")
    fig.update_yaxes(showgrid=True, gridcolor="#E9EEF0", zerolinecolor="#DCE5E9")
    return fig


def show_empty_state(message: str) -> None:
    st.info(message)


def render_chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, **PLOT_WIDTH_ARGS)


def render_table(frame: pd.DataFrame, hide_index: bool = False) -> None:
    st.dataframe(frame, hide_index=hide_index, **TABLE_WIDTH_ARGS)


def draw_overview(df: pd.DataFrame, sensors: list[str], missing: pd.DataFrame) -> None:
    st.subheader("Manufacturing dataset at a glance")
    st.markdown(
        '<p class="section-note">A verified snapshot of the SECOM manufacturing records. '
        'The target is a quality-test outcome, not a live production signal.</p>',
        unsafe_allow_html=True,
    )
    labels = df[TARGET_COL].astype(str).str.strip()
    counts = labels.value_counts()
    pass_count = int(counts.get("Pass", 0))
    fail_count = int(counts.get("Fail", 0))
    missing_cells = int(missing["missing_count"].sum())
    denominator = int(df.shape[0] * len(sensors))
    missing_pct = (100 * missing_cells / denominator) if denominator else 0

    metrics = st.columns(5)
    metrics[0].metric("Observations", f"{len(df):,}")
    metrics[1].metric("Sensor features", f"{len(sensors):,}")
    metrics[2].metric("Pass", f"{pass_count:,}")
    metrics[3].metric("Fail", f"{fail_count:,}")
    metrics[4].metric("Sensor cells missing", f"{missing_pct:.2f}%")
    st.caption(
        f"Missing-value percentage = {missing_cells:,} missing sensor cells / "
        f"{denominator:,} total sensor cells; Timestamp and target are excluded."
    )

    left, right = st.columns([1.05, 1.35], gap="large")
    with left:
        st.markdown("#### Pass / Fail composition")
        pie = go.Figure(go.Pie(
            labels=["Pass", "Fail"],
            values=[pass_count, fail_count],
            hole=0.62,
            sort=False,
            marker=dict(colors=[PASS_COLOR, FAIL_COLOR], line=dict(color="white", width=2)),
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
        ))
        chart_layout(pie, 330)
        pie.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.08, x=0.25))
        render_chart(pie)
    with right:
        st.markdown("#### Actual class counts")
        class_frame = pd.DataFrame({"Outcome": ["Pass", "Fail"], "Observations": [pass_count, fail_count]})
        bars = px.bar(
            class_frame,
            x="Outcome",
            y="Observations",
            color="Outcome",
            color_discrete_map={"Pass": PASS_COLOR, "Fail": FAIL_COLOR},
            text="Observations",
        )
        bars.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
        bars.update_yaxes(title="Manufacturing observations", rangemode="tozero")
        chart_layout(bars, 330)
        render_chart(bars)
        st.caption("Fail is the minority class. Accuracy alone can therefore overstate model usefulness.")

    st.markdown("#### Dataset composition")
    composition = pd.DataFrame({
        "Column type": ["Sensor measurements", "Timestamp metadata", "Pass/Fail target"],
        "Columns": [len(sensors), int(TIMESTAMP_COL in df.columns), int(TARGET_COL in df.columns)],
    })
    composition_chart = px.bar(
        composition,
        x="Column type",
        y="Columns",
        color="Column type",
        color_discrete_sequence=[BLUE, NEUTRAL, AMBER],
        text="Columns",
    )
    composition_chart.update_traces(textposition="outside")
    composition_chart.update_layout(showlegend=False)
    composition_chart.update_yaxes(title="Number of columns", rangemode="tozero")
    chart_layout(composition_chart, 290)
    render_chart(composition_chart)

    with st.expander("Inspect a small preview of actual records"):
        preview_cols = st.multiselect(
            "Columns to preview",
            options=[TIMESTAMP_COL, *sensors, TARGET_COL],
            default=[TIMESTAMP_COL, *sensors[:3], TARGET_COL],
            max_selections=7,
        )
        if preview_cols:
            st.caption("These are source-dataset rows, not live or recent production records.")
            render_table(df.loc[:, preview_cols].head(8), hide_index=True)


def draw_quality(df: pd.DataFrame, sensors: list[str], missing: pd.DataFrame, audit: dict | None) -> None:
    st.subheader("Data quality and sensor availability")
    total_cells = len(df) * len(sensors)
    missing_cells = int(missing["missing_count"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Missing sensor measurements", f"{missing_cells:,}")
    c2.metric("Sensor cells unavailable", f"{(100 * missing_cells / total_cells):.2f}%" if total_cells else "N/A")
    c3.metric("Exact duplicate rows", f"{int(df.duplicated().sum()):,}")
    st.caption(
        "Availability bands are defined by per-sensor missing fraction: low <5%, moderate 5–20%, "
        "high 20–50%, and very high ≥50%."
    )

    missing["missing_pct"] = missing["missing_count"] / len(df) * 100
    top_n = st.slider("Sensors to show by missingness", min_value=5, max_value=min(40, len(sensors)), value=min(15, len(sensors)), step=1)
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        top = missing.nlargest(top_n, "missing_pct").sort_values("missing_pct")
        top_chart = px.bar(
            top,
            x="missing_pct",
            y="feature",
            orientation="h",
            color="missing_pct",
            color_continuous_scale=["#B8D9D5", "#168C87", "#D99A36"],
            hover_data={"missing_count": True, "missing_pct": ":.2f"},
        )
        top_chart.update_layout(coloraxis_showscale=False)
        top_chart.update_xaxes(title="Missing measurements (%)")
        top_chart.update_yaxes(title="Sensor")
        chart_layout(top_chart, max(360, top_n * 24))
        render_chart(top_chart)
    with right:
        distribution = px.histogram(
            missing,
            x="missing_pct",
            nbins=24,
            color_discrete_sequence=[BLUE],
            labels={"missing_pct": "Missing values per sensor (%)", "count": "Number of sensors"},
        )
        distribution.update_layout(bargap=0.08)
        chart_layout(distribution, 360)
        render_chart(distribution)

    missing["availability_band"] = pd.cut(
        missing["missing_pct"],
        bins=[-0.001, 5, 20, 50, 100],
        labels=["Low (<5%)", "Moderate (5–20%)", "High (20–50%)", "Very high (≥50%)"],
        include_lowest=True,
    )
    bands = missing["availability_band"].value_counts(sort=False).rename_axis("Band").reset_index(name="Sensors")
    band_chart = px.bar(
        bands,
        x="Band",
        y="Sensors",
        color="Band",
        color_discrete_sequence=[PASS_COLOR, BLUE, AMBER, FAIL_COLOR],
        text="Sensors",
    )
    band_chart.update_traces(textposition="outside")
    band_chart.update_layout(showlegend=False)
    band_chart.update_yaxes(title="Number of sensors", rangemode="tozero")
    chart_layout(band_chart, 300)
    st.markdown("#### Feature availability bands")
    render_chart(band_chart)

    st.markdown("#### Integrity checks")
    target_missing = int(df[TARGET_COL].isna().sum())
    timestamp_missing = int(df[TIMESTAMP_COL].isna().sum()) if TIMESTAMP_COL in df.columns else None
    converted = df.loc[:, sensors].apply(pd.to_numeric, errors="coerce")
    invalid_numeric = int((df.loc[:, sensors].notna() & converted.isna()).to_numpy().sum())
    checks = [
        {"Check": "Dataset schema and numeric sensors", "Result": "Pass" if invalid_numeric == 0 else "Review", "Finding": f"{invalid_numeric:,} nonnumeric non-missing sensor values"},
        {"Check": "Target labels", "Result": "Pass" if target_missing == 0 else "Review", "Finding": f"{target_missing:,} missing target labels"},
        {"Check": "Timestamp", "Result": "Pass" if timestamp_missing == 0 else "Review", "Finding": f"{timestamp_missing:,} missing timestamps" if timestamp_missing is not None else "Timestamp column unavailable"},
        {"Check": "Exact duplicate rows", "Result": "Pass" if df.duplicated().sum() == 0 else "Review", "Finding": f"{int(df.duplicated().sum()):,} duplicates"},
        {"Check": "Sensor missingness", "Result": "Review" if missing_cells else "Pass", "Finding": f"{missing_cells:,} missing cells remain"},
    ]
    render_table(pd.DataFrame(checks), hide_index=True)
    if audit:
        audited_rows = audit.get("dataset", {}).get("rows")
        audited_missing = audit.get("missingness", {}).get("count")
        if audited_rows != len(df) or audited_missing != int(df.isna().sum().sum()):
            st.warning("The saved audit report does not match the currently loaded dataset; displayed quality measures were recalculated from the loaded data.")
        else:
            st.caption("The saved audit report agrees with the current dataset dimensions and missing-value total.")
    st.info("Zero duplicates does not mean the dataset is complete: sensor missingness and strong Pass/Fail imbalance remain important limitations.")


def draw_sensor_explorer(df: pd.DataFrame, sensors: list[str]) -> None:
    st.subheader("Exploratory sensor analysis")
    st.markdown(
        '<p class="section-note">Explore observed sensor distributions and compare recorded Pass and Fail groups. '
        'Distribution differences are descriptive and are not, by themselves, evidence of statistical significance or causation.</p>',
        unsafe_allow_html=True,
    )
    sensor = st.selectbox("Select a sensor", sensors, index=0, key="explorer_sensor")
    values = pd.to_numeric(df[sensor], errors="coerce")
    missing_count = int(values.isna().sum())
    observed = values.dropna()
    stats_cols = st.columns(4)
    stats_cols[0].metric("Observed values", f"{len(observed):,}")
    stats_cols[1].metric("Mean", f"{observed.mean():.4g}" if len(observed) else "N/A")
    stats_cols[2].metric("Median", f"{observed.median():.4g}" if len(observed) else "N/A")
    stats_cols[3].metric("Std. deviation", f"{observed.std():.4g}" if len(observed) > 1 else "N/A")
    st.caption(f"{missing_count:,} missing values ({100 * missing_count / len(df):.2f}% of records); units are not specified in the dataset.")

    left, right = st.columns(2, gap="large")
    with left:
        hist = px.histogram(
            pd.DataFrame({"Value": observed}),
            x="Value",
            nbins=40,
            color_discrete_sequence=[BLUE],
            title=f"Observed distribution: {sensor}",
        )
        hist.update_layout(yaxis_title="Observations")
        chart_layout(hist, 360)
        render_chart(hist)
    with right:
        comparison = pd.DataFrame({"Value": values, "Outcome": df[TARGET_COL].astype(str).str.strip()}).dropna()
        group_counts = comparison.groupby("Outcome")["Value"].size().to_dict()
        box = px.box(
            comparison,
            x="Outcome",
            y="Value",
            color="Outcome",
            points="outliers",
            color_discrete_map={"Pass": PASS_COLOR, "Fail": FAIL_COLOR},
            category_orders={"Outcome": ["Pass", "Fail"]},
            title=f"Observed values by outcome: {sensor}",
        )
        box.update_yaxes(title="Observed sensor value")
        chart_layout(box, 360)
        render_chart(box)
        st.caption(f"Valid observations used: Pass {group_counts.get('Pass', 0):,}; Fail {group_counts.get('Fail', 0):,}.")

    st.markdown("#### Correlation among leading sensors")
    analysis_path = METRICS_DIR / "sensor_feature_analysis.csv"
    analysis = read_csv_artifact(str(analysis_path))
    if analysis is not None and {"feature", "importance_mean"}.issubset(analysis.columns):
        chosen = analysis.dropna(subset=["importance_mean"]).nlargest(12, "importance_mean")["feature"].tolist()
        if len(chosen) >= 2:
            corr = df.loc[:, chosen].apply(pd.to_numeric, errors="coerce").corr()
            heat = px.imshow(
                corr,
                zmin=-1,
                zmax=1,
                color_continuous_scale="RdBu",
                origin="lower",
                aspect="auto",
                labels={"color": "Correlation"},
            )
            heat.update_layout(title="Pairwise correlation among top permutation-ranked sensors")
            chart_layout(heat, 520)
            render_chart(heat)
            st.caption("The subset is selected by held-out validation permutation importance. Correlation is descriptive and does not establish causation.")
        else:
            show_empty_state("At least two ranked sensors are needed for a correlation view.")
    else:
        show_empty_state("Run the explain sensor features task to create a ranked sensor subset for correlation analysis.")


def draw_explainability(top_n: int) -> None:
    st.subheader("Explainable sensor analysis")
    st.markdown(
        '<p class="section-note">Complementary statistical, predictive, and stability evidence. '
        'Anonymous sensor identifiers are not physical interpretations.</p>',
        unsafe_allow_html=True,
    )
    combined = read_csv_artifact(str(METRICS_DIR / "sensor_feature_analysis.csv"))
    if combined is None:
        show_empty_state("Sensor analysis is not available yet. Run `python run_project.py --task explain` or choose “Explain sensor features” in the launcher.")
        return

    available_methods = []
    if {"importance_mean", "importance_std"}.issubset(combined.columns) and combined["importance_mean"].notna().any():
        available_methods.append("Permutation importance")
    if "selection_frequency" in combined and combined["selection_frequency"].notna().any():
        available_methods.append("Selection stability")
    if {"adjusted_p_value", "effect_size"}.issubset(combined.columns) and combined["adjusted_p_value"].notna().any():
        available_methods.append("Statistical ranking")
    method = st.selectbox("Rank sensors by", available_methods or ["No ranking available"], key="sensor_method")
    n_show = st.slider("Leading sensors to display", min_value=5, max_value=25, value=min(top_n, 15), key="feature_top_n")
    left, right = st.columns([1.3, 1], gap="large")
    selected_features: list[str] = []

    with left:
        if method == "Permutation importance":
            ranking = combined.dropna(subset=["importance_mean"]).nlargest(n_show, "importance_mean").sort_values("importance_mean")
            selected_features = ranking["feature"].tolist()
            chart = go.Figure(go.Bar(
                x=ranking["importance_mean"],
                y=ranking["feature"],
                orientation="h",
                error_x=dict(type="data", array=ranking["importance_std"].fillna(0), visible=True),
                marker_color=BLUE,
                hovertemplate="%{y}<br>Mean decrease: %{x:.4f}<extra></extra>",
            ))
            chart.update_layout(title="Validation permutation importance")
            chart.update_xaxes(title="Decrease in balanced accuracy after shuffling")
        elif method == "Selection stability":
            ranking = combined.dropna(subset=["selection_frequency"]).nlargest(n_show, "selection_frequency").sort_values("selection_frequency")
            selected_features = ranking["feature"].tolist()
            chart = px.bar(ranking, x="selection_frequency", y="feature", orientation="h", color_discrete_sequence=[PASS_COLOR])
            chart.update_layout(title="Selection frequency across stratified training folds")
            chart.update_xaxes(title="Fold selection frequency", range=[0, 1])
        elif method == "Statistical ranking":
            ranking = combined.dropna(subset=["adjusted_p_value", "effect_size"]).nsmallest(n_show, "adjusted_p_value").sort_values("effect_size")
            selected_features = ranking["feature"].tolist()
            ranking["minus_log10_q"] = -np.log10(ranking["adjusted_p_value"].clip(lower=1e-300))
            chart = px.scatter(
                ranking,
                x="effect_size",
                y="minus_log10_q",
                color="significant_after_fdr",
                hover_name="feature",
                color_discrete_map={True: FAIL_COLOR, False: BLUE},
                labels={"effect_size": "Rank-biserial effect size", "minus_log10_q": "−log10(FDR-adjusted p-value)", "significant_after_fdr": "FDR < 0.05"},
            )
            chart.update_layout(title="Effect size and adjusted statistical evidence")
        else:
            ranking = pd.DataFrame()
            chart = go.Figure()
            show_empty_state("No supported feature ranking is present in the saved analysis.")
        if not ranking.empty:
            chart_layout(chart, max(390, n_show * 25))
            render_chart(chart)

    with right:
        st.markdown("#### Missingness of selected sensors")
        missing_selected = combined[combined["feature"].isin(selected_features)].copy()
        if not missing_selected.empty:
            missing_selected = missing_selected.sort_values("missing_fraction")
            miss_chart = px.bar(
                missing_selected,
                x="missing_fraction",
                y="feature",
                orientation="h",
                color_discrete_sequence=[AMBER],
                labels={"missing_fraction": "Missing fraction", "feature": "Sensor"},
            )
            miss_chart.update_xaxes(tickformat=".0%")
            chart_layout(miss_chart, max(300, len(missing_selected) * 23))
            render_chart(miss_chart)
        else:
            show_empty_state("Missingness values are not available for this ranking.")
        st.caption(
            "Permutation importance estimates predictive utility, not causation. Correlated sensors may share or mask importance. "
            "Statistical significance and out-of-sample predictive usefulness are distinct."
        )

    with st.expander("Detailed sensor-analysis results"):
        render_table(combined.head(100), hide_index=True)
        st.download_button(
            "Download sensor analysis CSV",
            data=(METRICS_DIR / "sensor_feature_analysis.csv").read_bytes(),
            file_name="sensor_feature_analysis.csv",
            mime="text/csv",
        )


def draw_models(dataset_path: str, df: pd.DataFrame) -> None:
    st.subheader("Model performance")
    cv = read_csv_artifact(str(METRICS_DIR / "cv_summary.csv"))
    final = read_csv_artifact(str(METRICS_DIR / "model_metrics.csv"))
    summary = read_json_artifact(str(MODELS_DIR / "model_summary.json"))

    if cv is None:
        show_empty_state("Cross-validation metrics are missing. Run the training task to generate model comparisons.")
    else:
        metric_cols = [col for col in METRIC_LABELS if col in cv.columns and cv[col].notna().any()]
        if not metric_cols:
            show_empty_state("The saved cross-validation table has no usable model metric columns.")
        else:
            model_choice = st.selectbox("Models to compare", ["All models", *cv["model"].astype(str).tolist()], key="model_compare")
            plot_data = cv.copy() if model_choice == "All models" else cv[cv["model"].astype(str) == model_choice].copy()
            long = plot_data.melt(id_vars="model", value_vars=metric_cols, var_name="metric", value_name="score")
            long["Metric"] = long["metric"].map(METRIC_LABELS)
            long["std_column"] = long["metric"].map({
                "balanced_accuracy_mean": "balanced_accuracy_std",
                "precision_mean": "precision_std",
                "recall_mean": "recall_std",
                "f1_mean": "f1_std",
                "roc_auc_mean": "roc_auc_std",
                "average_precision_mean": "average_precision_std",
            })
            long["std"] = [
                float(row[std_col]) if std_col in row and pd.notna(row[std_col]) else 0.0
                for _, row in long.iterrows()
                for std_col in [row["std_column"]]
            ]
            comparison = px.bar(
                long,
                x="Metric",
                y="score",
                color="model",
                barmode="group",
                error_y="std",
                color_discrete_sequence=[NAVY, BLUE, PASS_COLOR, AMBER, "#765D9A"],
                labels={"score": "Mean cross-validation score"},
            )
            comparison.update_yaxes(range=[0, 1], title="Mean score (error bars: ±1 SD)")
            comparison.update_layout(title="Stratified cross-validation model comparison")
            chart_layout(comparison, 450)
            render_chart(comparison)
            missing_metric_rows = cv.loc[:, metric_cols].isna().any(axis=1)
            if missing_metric_rows.any():
                missing_models = cv.loc[missing_metric_rows, "model"].astype(str).tolist()
                st.warning(
                    "Some saved model metrics are unavailable and therefore have no bar for affected metrics: "
                    + ", ".join(missing_models)
                )
            st.caption("Scores are fold means; error bars show fold standard deviation. Average precision should be read relative to the minority-class prevalence.")

    if final is None:
        show_empty_state("Final held-out test metrics are unavailable. Run the evaluation task to generate them.")
        return
    result = final.iloc[0]
    model_name = str(result["model"])
    st.markdown(f"#### Final held-out test: `{model_name}`")
    metric_names = [
        ("balanced_accuracy", "Balanced accuracy"),
        ("precision", "Precision"),
        ("recall", "Fail recall"),
        ("f1", "F1"),
        ("roc_auc", "ROC-AUC"),
        ("average_precision", "Average precision"),
    ]
    metric_cols = st.columns(len(metric_names))
    for column, (key, label) in zip(metric_cols, metric_names):
        if key in result and pd.notna(result[key]):
            column.metric(label, f"{float(result[key]):.3f}")

    confusion_values = {
        "True negative": int(result["true_negative_count"]),
        "False positive": int(result["false_positive_count"]),
        "False negative": int(result["false_negative_count"]),
        "True positive": int(result["true_positive_count"]),
    }
    matrix = np.array([
        [confusion_values["True negative"], confusion_values["False positive"]],
        [confusion_values["False negative"], confusion_values["True positive"]],
    ])
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.markdown("#### Confusion matrix")
        heat = go.Figure(go.Heatmap(
            z=matrix,
            x=["Predicted Pass", "Predicted Fail"],
            y=["Actual Pass", "Actual Fail"],
            text=matrix,
            texttemplate="%{text:,}",
            colorscale=[[0, "#EAF1F3"], [0.55, "#70B5AF"], [1, "#168C87"]],
            showscale=False,
            hovertemplate="%{y}<br>%{x}: %{z:,}<extra></extra>",
        ))
        chart_layout(heat, 360)
        render_chart(heat)
        st.caption(
            f"TN {confusion_values['True negative']:,} · FP {confusion_values['False positive']:,} · "
            f"FN {confusion_values['False negative']:,} · TP {confusion_values['True positive']:,}"
        )

    with right:
        artifact = MODELS_DIR / f"{model_name}.joblib"
        scores: tuple[np.ndarray, np.ndarray] | None = None
        if artifact.is_file():
            try:
                scores = get_test_scores(dataset_path, str(artifact))
            except (FileNotFoundError, OSError, EOFError, ValueError, KeyError, AttributeError) as exc:
                st.warning(f"Could not derive test-set curves from the saved model: {exc}")
        if scores is None:
            show_empty_state("ROC and precision-recall curves require saved, reproducible held-out prediction scores. No curve is drawn without them.")
        else:
            y_test, probabilities = scores
            from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

            predicted = (probabilities >= float(result.get("selected_threshold", 0.5))).astype(int)
            computed = sklearn_confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()
            saved = np.array([
                confusion_values["True negative"],
                confusion_values["False positive"],
                confusion_values["False negative"],
                confusion_values["True positive"],
            ])
            if not np.array_equal(computed, saved):
                st.warning("Saved model predictions do not reproduce the saved confusion counts. Curves are withheld to avoid presenting mismatched results.")
            else:
                fpr, tpr, _ = roc_curve(y_test, probabilities)
                precision, recall, _ = precision_recall_curve(y_test, probabilities)
                roc_auc = auc(fpr, tpr)
                pr_auc = auc(recall[::-1], precision[::-1])
                curve = go.Figure()
                curve.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC {roc_auc:.3f})", line=dict(color=BLUE, width=3)))
                curve.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance baseline", line=dict(color=NEUTRAL, dash="dash")))
                curve.update_layout(title="Held-out test ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
                chart_layout(curve, 340)
                render_chart(curve)
                pr = go.Figure()
                pr.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"PR (area {pr_auc:.3f})", line=dict(color=PASS_COLOR, width=3)))
                prevalence = float(np.mean(y_test))
                pr.add_trace(go.Scatter(x=[0, 1], y=[prevalence, prevalence], mode="lines", name=f"Fail prevalence ({prevalence:.3f})", line=dict(color=NEUTRAL, dash="dash")))
                pr.update_layout(title="Held-out test precision–recall curve", xaxis_title="Fail recall", yaxis_title="Fail precision", yaxis=dict(range=[0, 1]))
                chart_layout(pr, 340)
                render_chart(pr)
                st.caption("The precision–recall baseline equals Fail prevalence in the test split. Curves use the fixed held-out labels and are not used to tune the model.")

    st.markdown("#### Validation threshold trade-off")
    threshold_models = (summary or {}).get("cv_summary", {}).get("models", [])
    threshold_model = next((item for item in threshold_models if item.get("model") == model_name), None)
    threshold_rows = (threshold_model or {}).get("threshold_analysis", [])
    if threshold_rows:
        threshold_data = pd.DataFrame(threshold_rows)
        if {"threshold", "precision", "recall", "f1"}.issubset(threshold_data.columns):
            st.caption("This control uses out-of-fold training/validation threshold results only. It does not change or re-evaluate the held-out test result.")
            default_threshold = float(threshold_model.get("selected_threshold", 0.5))
            selected_threshold = st.slider(
                "Explore validation threshold",
                min_value=float(threshold_data["threshold"].min()),
                max_value=float(threshold_data["threshold"].max()),
                value=default_threshold,
                step=0.01,
                key="validation_threshold",
            )
            index = (threshold_data["threshold"] - selected_threshold).abs().idxmin()
            threshold_row = threshold_data.loc[index]
            values = st.columns(5)
            for col, key, label in [
                (values[0], "precision", "Precision"),
                (values[1], "recall", "Fail recall"),
                (values[2], "f1", "F1"),
                (values[3], "false_positive_count", "False positives"),
                (values[4], "false_negative_count", "False negatives"),
            ]:
                if key in threshold_row:
                    col.metric(label, f"{float(threshold_row[key]):.3f}" if key in {"precision", "recall", "f1"} else f"{int(threshold_row[key]):,}")
            line = px.line(
                threshold_data,
                x="threshold",
                y=[key for key in ["precision", "recall", "f1", "balanced_accuracy"] if key in threshold_data],
                labels={"value": "Validation score", "variable": "Metric"},
            )
            line.add_vline(x=selected_threshold, line_dash="dash", line_color=AMBER)
            line.update_yaxes(range=[0, 1])
            line.update_layout(title="Out-of-fold validation metrics across thresholds")
            chart_layout(line, 340)
            render_chart(line)
    else:
        show_empty_state("Saved out-of-fold threshold results are unavailable. Run training to generate them; no test-set threshold tuning is performed.")

    st.info(
        "False negatives are Fail observations predicted as Pass; false positives are Pass observations flagged as Fail. "
        "Their operational costs differ. These results are exploratory and do not establish factory readiness."
    )


def draw_summary(df: pd.DataFrame, audit: dict | None, cv: pd.DataFrame | None, final: pd.DataFrame | None, sensors: pd.DataFrame | None) -> None:
    st.subheader("Project results and methodology")
    audit_status = "Available and consistent" if audit else "Audit report not generated"
    if audit:
        dataset_info = audit.get("dataset", {})
        missing_info = audit.get("missingness", {})
        if dataset_info.get("rows") != len(df) or missing_info.get("count") != int(df.isna().sum().sum()):
            audit_status = "Saved report differs from loaded data"
    model_count = int(cv["model"].nunique()) if cv is not None and "model" in cv else 0
    selected = str(final.iloc[0]["model"]) if final is not None and not final.empty else "Not evaluated"
    selected_rule = "Highest mean balanced accuracy, with mean F1 as tie-breaker" if selected != "Not evaluated" else "Not available"
    leading = []
    if sensors is not None and "importance_mean" in sensors:
        leading = sensors.dropna(subset=["importance_mean"]).nlargest(5, "importance_mean")["feature"].tolist()
    m1, m2, m3 = st.columns(3)
    m1.metric("Dataset audit", audit_status)
    m2.metric("Models compared", f"{model_count}")
    m3.metric("Selected model", selected)
    st.caption(f"Selection rule recorded by the current workflow: {selected_rule}.")
    if final is not None and not final.empty:
        row = final.iloc[0]
        st.markdown(
            f"Final test performance: balanced accuracy **{float(row['balanced_accuracy']):.3f}**, "
            f"Fail precision **{float(row['precision']):.3f}**, Fail recall **{float(row['recall']):.3f}**, "
            f"F1 **{float(row['f1']):.3f}**, average precision **{float(row['average_precision']):.3f}**."
        )
    st.markdown(f"Leading validation-importance sensors: {', '.join(leading) if leading else 'Not available'}.")
    st.markdown(
        "**Limitations:** observational data; 104 Fail cases in the full dataset; substantial sensor missingness; "
        "anonymous sensor names; correlated measurements; no live-factory validation. Associations do not establish causation."
    )
    report_path = CONFIG_PROJECT_ROOT / "reports" / "project_report.md"
    if report_path.is_file():
        st.download_button(
            "Download academic project report",
            data=report_path.read_bytes(),
            file_name="project_report.md",
            mime="text/markdown",
        )
    if final is not None:
        metrics_path = METRICS_DIR / "model_metrics.csv"
        st.download_button(
            "Download final model metrics",
            data=metrics_path.read_bytes(),
            file_name="model_metrics.csv",
            mime="text/csv",
        )


try:
    raw_df, dataset_path = get_dataset()
    sensor_columns = validate_dataset(raw_df)
    extract_features_targets(raw_df)
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Dataset could not be loaded or validated: {exc}")
    st.info("Check the combined dataset in data/raw and run the dataset audit before opening the dashboard.")
    st.stop()

if not raw_df.empty:
    sensor_frame = raw_df.loc[:, sensor_columns].apply(pd.to_numeric, errors="coerce")
    missing_by_sensor = pd.DataFrame({
        "feature": sensor_columns,
        "missing_count": sensor_frame.isna().sum().to_numpy(),
    })
else:
    missing_by_sensor = pd.DataFrame(columns=["feature", "missing_count"])

audit_data = read_json_artifact(str(CONFIG_PROJECT_ROOT / "reports" / "data_quality_report.json"))
cv_data = read_csv_artifact(str(METRICS_DIR / "cv_summary.csv"))
final_data = read_csv_artifact(str(METRICS_DIR / "model_metrics.csv"))
sensor_data = read_csv_artifact(str(METRICS_DIR / "sensor_feature_analysis.csv"))
dataset_source_label = Path(dataset_path).relative_to(PROJECT_ROOT).as_posix()

st.title("Semiconductor Yield Intelligence")
st.markdown('<p class="hero-subtitle"><em>Sensor Analytics, Defect Detection and Manufacturing Quality Insights</em></p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-copy">This application analyzes the SECOM semiconductor manufacturing dataset to investigate '
    'sensor patterns associated with failed manufacturing tests. It combines data-quality review, exploratory '
    'analysis, explainable sensor rankings, and leakage-aware model evaluation.</p>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="status-line">Dataset source: <code>{dataset_source_label}</code> &nbsp;·&nbsp; '
    f'Loaded {len(raw_df):,} observations and validated {len(sensor_columns):,} sensor columns</div>',
    unsafe_allow_html=True,
)

tabs = st.tabs(["Overview", "Data quality", "Sensor explorer", "Explainable sensors", "Model performance", "Results summary"])
with tabs[0]:
    draw_overview(raw_df, sensor_columns, missing_by_sensor)
with tabs[1]:
    draw_quality(raw_df, sensor_columns, missing_by_sensor.copy(), audit_data)
with tabs[2]:
    draw_sensor_explorer(raw_df, sensor_columns)
with tabs[3]:
    draw_explainability(15)
with tabs[4]:
    draw_models(dataset_path, raw_df)
with tabs[5]:
    draw_summary(raw_df, audit_data, cv_data, final_data, sensor_data)
