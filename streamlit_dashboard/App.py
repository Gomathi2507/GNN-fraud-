"""Elliptic fraud detection dashboard.

Reads the files saved by the training script from ./results. Nothing is retrained.
Run:  streamlit run app.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

RESULTS = Path(__file__).parent / "results"
RUNS = {"All 165 features": "all_features", "Local 93 features only": "local_features_only"}

# ----------------------------------------------------------------------------- palette
# Espresso & brass dark palette: warm coffee-brown page, leather panels, parchment-cream text.
# Model colours are chosen to stand apart from the brown background (brass, verdigris, ivory).
PAGE, PANEL, LINE = "#15100D", "#201914", "#3D3027"
INK, MUTED = "#F1E7D6", "#A89885"
GRID = "#2C231C"
ALERT = "#E5484D"
# label: (probability column, config key, colour, row name in the metrics table)
MODELS = {
    "XGBoost": ("xgboost_prob", "xgboost", "#E0A83A", "XGBoost"),
    "GraphSAGE": ("graphsage_prob", "graphsage", "#4FB39A", "GraphSAGE (mean over seeds)"),
    "Ensemble": ("ensemble_prob", "ensemble", "#EDE3D0", "Ensemble"),
}
RUN_BAR_COLORS = ["#C8693F", "#8E7F70"]  # all features / local only in the grouped bar chart
COLLAPSE_F1 = 0.2  # a step counts as "collapsed" when every model's F1 stays below this

st.set_page_config(page_title="Elliptic fraud detection", page_icon="🔎", layout="wide")
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
html, body, .stApp, [class*="css"] {{font-family:'IBM Plex Sans', system-ui, sans-serif; color:{INK};}}
.stApp {{background:{PAGE};}}
#MainMenu, footer {{visibility:hidden;}}
header[data-testid="stHeader"] {{background:transparent;}}
.block-container {{padding-top:2.2rem; max-width:1240px;}}
section[data-testid="stSidebar"] {{background:{PANEL}; border-right:1px solid {LINE};}}
h1 {{font-weight:700 !important; letter-spacing:-0.025em; font-size:2.1rem !important; margin-bottom:.2rem;}}
h2, h3 {{font-weight:600 !important; letter-spacing:-0.01em;}}
.verdict {{font-size:1.08rem; color:#D3C7B6; max-width:44rem; line-height:1.6; margin:.3rem 0 1.5rem;}}
.verdict b {{color:{INK};}}
.kpis {{display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:1.6rem;}}
.kpi {{background:{PANEL}; border:1px solid {LINE}; border-left:5px solid var(--c); border-radius:8px; padding:14px 18px;}}
.kpi .l {{font-size:.85rem; color:{MUTED}; line-height:1.3; min-height:2.2em;}}
.kpi .v {{font-size:2rem; font-weight:700; letter-spacing:-0.03em; font-variant-numeric:tabular-nums; line-height:1.15;}}
.mini {{display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin:.5rem 0 .4rem;}}
.mini > div {{background:{PANEL}; border:1px solid {LINE}; border-radius:6px; padding:7px 9px;}}
.mini .l {{font-size:.72rem; color:{MUTED}; white-space:nowrap;}}
.mini .v {{font-size:1.2rem; font-weight:600; font-variant-numeric:tabular-nums;}}
.mhead {{font-weight:600; font-size:1.05rem; padding-bottom:.3rem; border-bottom:3px solid var(--c); margin-bottom:.2rem;}}
.mhead span {{font-weight:400; color:{MUTED}; font-size:.9rem; margin-left:.5rem;}}
.stTabs [data-baseweb="tab-list"] {{gap:1.8rem; border-bottom:1px solid {LINE};}}
.stTabs [data-baseweb="tab"] {{padding:.55rem 0; font-weight:500; color:{MUTED};}}
.stTabs [aria-selected="true"] {{color:{INK}; font-weight:600;}}
[data-testid="stCaptionContainer"] {{color:{MUTED};}}
@media (max-width:900px) {{.kpis {{grid-template-columns:repeat(2,1fr);}}}}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------- helpers
def required_files():
    names = []
    for key in RUNS.values():
        names += [f"model_comparison_{key}.csv", f"f1_per_test_step_{key}.csv",
                  f"test_predictions_{key}.csv", f"chosen_config_{key}.json"]
    return names


missing = [n for n in required_files() if not (RESULTS / n).exists()]
if missing:
    st.error("Result files not found in the results/ folder next to app.py:")
    st.code("\n".join(missing))
    st.stop()


@st.cache_data
def load(key):
    metrics = pd.read_csv(RESULTS / f"model_comparison_{key}.csv", index_col=0)
    steps = pd.read_csv(RESULTS / f"f1_per_test_step_{key}.csv", index_col=0)
    preds = pd.read_csv(RESULTS / f"test_predictions_{key}.csv")
    cfg = json.loads((RESULTS / f"chosen_config_{key}.json").read_text())
    return metrics, steps, preds, cfg


def confusion(y, p, t):
    pred = p >= t
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1)


def detect_shift(steps):
    """First test step after which every model's F1 stays below COLLAPSE_F1."""
    for s in steps.index:
        if steps.loc[s:].to_numpy().max() < COLLAPSE_F1:
            return int(s)
    return int(steps.index.max()) + 1


def tint(hex_color, amount=0.45):
    """Mix a colour into the dark panel colour so light text stays readable on top of it."""
    h = hex_color.lstrip("#")
    rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    base = [int(PANEL.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{int(b + (c - b) * amount):02X}" for c, b in zip(rgb, base))


def style(fig, height=380):
    fig.update_layout(
        template="plotly_dark",
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PANEL,
        font=dict(family="IBM Plex Sans, sans-serif", color=INK, size=14),
        legend=dict(orientation="h", y=1.14, x=0),
        hoverlabel=dict(bgcolor=PANEL, font_color=INK, bordercolor=LINE),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=LINE, title_font_color=MUTED)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=LINE, title_font_color=MUTED)
    return fig


def show(fig, key, height=380):
    style(fig, height)
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except Exception:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True, key=key)


def kpi(label, value, color):
    return f"<div class='kpi' style='--c:{color}'><div class='l'>{label}</div><div class='v'>{value}</div></div>"


runs = {label: load(key) for label, key in RUNS.items()}

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Elliptic fraud detection")
    st.caption("XGBoost vs GraphSAGE on Bitcoin transactions. Reads saved results; nothing is retrained.")
    run_label = st.radio("Feature set", list(RUNS))
    st.caption("Local only drops the pre-aggregated neighbour features, so only GraphSAGE can see the graph.")
    st.markdown("---")
    st.markdown("**Model colours**")
    for n, (_, _, col, _) in MODELS.items():
        st.markdown(f"<span style='color:{col};font-size:1.1rem'>■</span> {n}", unsafe_allow_html=True)

metrics, steps, preds, cfg = runs[run_label]
shift = detect_shift(runs["All 165 features"][1])

# ----------------------------------------------------------------------------- header
f1_by_run = {lab: {m: r[0].loc[v[3], "F1"] for m, v in MODELS.items()} for lab, r in runs.items()}
best_all = max(f1_by_run["All 165 features"], key=f1_by_run["All 165 features"].get)
best_loc = max(f1_by_run["Local 93 features only"], key=f1_by_run["Local 93 features only"].get)
p_all, c_all = runs["All 165 features"][2], runs["All 165 features"][3]
post = p_all[p_all.time_step >= shift]
post_f1 = confusion(post.true_label.to_numpy(), post[MODELS[best_all][0]].to_numpy(),
                    c_all["thresholds"][MODELS[best_all][1]])["f1"]

st.title("Fraud detection on the Elliptic Bitcoin graph")
st.markdown(
    f"<div class='verdict'><b>{best_all}</b> scores highest on all features and <b>{best_loc}</b> on local "
    f"features only. From time step <b>{shift}</b> onward every model fails: the best one falls to "
    f"F1 <b style='color:{ALERT}'>{post_f1:.2f}</b>.</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='kpis'>"
    + kpi(f"{best_all} F1, all features", f"{f1_by_run['All 165 features'][best_all]:.2f}", MODELS[best_all][2])
    + kpi("GraphSAGE F1, all features", f"{f1_by_run['All 165 features']['GraphSAGE']:.2f}", MODELS["GraphSAGE"][2])
    + kpi(f"{best_loc} F1, local only", f"{f1_by_run['Local 93 features only'][best_loc]:.2f}", MODELS[best_loc][2])
    + kpi(f"{best_all} F1 from step {shift}", f"{post_f1:.2f}", ALERT)
    + "</div>",
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Threshold explorer", "Transactions", "All results"])

# ----------------------------------------------------------------------------- overview
with tab1:
    st.subheader("F1 at every test time step")
    st.caption(f"Feature set: {run_label}. The shaded area marks the collapse found on the all-features run.")
    fig = go.Figure()
    for name, (_, _, color, _) in MODELS.items():
        fig.add_trace(go.Scatter(
            x=steps.index, y=steps[name], name=name, mode="lines+markers",
            line=dict(color=color, width=2 if name == "Ensemble" else 3, dash="dot" if name == "Ensemble" else "solid"),
            marker=dict(size=7), hovertemplate="%{y:.3f}"))
    fig.add_vrect(x0=shift - 0.5, x1=steps.index.max() + 0.5, fillcolor=ALERT, opacity=0.15, line_width=0,
                  annotation_text=f"Collapse from step {shift}", annotation_position="top left",
                  annotation_font_color=ALERT)
    fig.update_layout(hovermode="x unified", xaxis_title="Test time step", yaxis_title="F1", yaxis_range=[0, 1.02])
    show(fig, "f1_steps", 400)

    st.subheader("With and without the pre-aggregated features")
    metric = st.radio("Metric", ["F1", "Precision", "Recall", "PR_AUC", "ROC_AUC"], horizontal=True)
    fig = go.Figure()
    for (label, r), color in zip(runs.items(), RUN_BAR_COLORS):
        vals = [r[0].loc[v[3], metric] for v in MODELS.values()]
        fig.add_trace(go.Bar(x=list(MODELS), y=vals, name=label, marker_color=color,
                             text=[f"{v:.2f}" for v in vals], textposition="outside"))
    fig.update_layout(barmode="group", yaxis_range=[0, 1.1], yaxis_title=metric, bargap=0.35)
    show(fig, "feature_bars", 360)

# ----------------------------------------------------------------------------- threshold explorer
with tab2:
    c1, c2 = st.columns(2)
    period = c1.radio("Test period", ["All test steps", f"Before step {shift}", f"From step {shift}"], horizontal=True)
    mode = c2.radio("Threshold", ["Chosen on validation", "Custom per model"], horizontal=True)
    if period.startswith("Before"):
        sub = preds[preds.time_step < shift]
    elif period.startswith("From"):
        sub = preds[preds.time_step >= shift]
    else:
        sub = preds
    y = sub.true_label.to_numpy()
    st.caption(f"{len(sub):,} transactions, {int(y.sum()):,} illicit. "
               "Matrix colour shows the share of each actual class; numbers are counts.")

    used = {}
    for name, (pcol, ckey, color, _) in MODELS.items():
        base_t = float(cfg["thresholds"][ckey])
        left, right = st.columns([5, 4], gap="large")
        with left:
            st.markdown(f"<div class='mhead' style='--c:{color}'>{name}<span>validation threshold {base_t:.2f}</span></div>",
                        unsafe_allow_html=True)
            t = st.slider("Threshold", 0.0, 1.0, base_t, 0.01, key=f"thr_{ckey}_{run_label}",
                          label_visibility="collapsed") if mode.startswith("Custom") else base_t
            used[name] = t
            c = confusion(y, sub[pcol].to_numpy(), t)
            flagged_per_1k = (c["tp"] + c["fp"]) / max(len(sub), 1) * 1000
            st.markdown(
                "<div class='mini'>"
                + "".join(f"<div><div class='l'>{l}</div><div class='v'>{v}</div></div>" for l, v in [
                    ("F1", f"{c['f1']:.2f}"), ("Precision", f"{c['precision']:.2f}"),
                    ("Recall", f"{c['recall']:.2f}"), ("Flagged /1k", f"{flagged_per_1k:.0f}")])
                + "</div>", unsafe_allow_html=True)
            st.caption(f"Caught {c['tp']:,} of {c['tp'] + c['fn']:,} illicit transactions "
                       f"with {c['fp']:,} false alarms at threshold {t:.2f}.")
        with right:
            z = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
            rate = z / np.maximum(z.sum(axis=1, keepdims=True), 1)
            text = [[f"<b>{z[i, j]:,}</b><br>{rate[i, j]:.0%}" for j in range(2)] for i in range(2)]
            fig = go.Figure(go.Heatmap(
                z=rate, x=["Predicted licit", "Predicted illicit"], y=["Actual licit", "Actual illicit"],
                text=text, texttemplate="%{text}", showscale=False, zmin=0, zmax=1, xgap=3, ygap=3,
                textfont=dict(color="#FFFFFF", size=14),
                colorscale=[[0, "#120D0A"], [1, tint(color, 0.5)]], hoverinfo="skip"))
            fig.update_yaxes(autorange="reversed")
            show(fig, f"cm_{ckey}", 250)

    st.subheader("F1 across thresholds")
    st.caption("Dots mark the threshold currently in use. A narrow peak means the model is sensitive to threshold choice.")
    grid = np.linspace(0.02, 0.98, 49)
    fig = go.Figure()
    for name, (pcol, _, color, _) in MODELS.items():
        p = sub[pcol].to_numpy()
        fig.add_trace(go.Scatter(x=grid, y=[confusion(y, p, g)["f1"] for g in grid], name=name,
                                 line=dict(color=color, width=3), hovertemplate="%{y:.3f}"))
        fig.add_trace(go.Scatter(x=[used[name]], y=[confusion(y, p, used[name])["f1"]], mode="markers",
                                 showlegend=False, marker=dict(size=13, color=color, line=dict(width=3, color=PANEL)),
                                 hovertemplate=f"{name} at %{{x:.2f}}: F1 %{{y:.3f}}<extra></extra>"))
    fig.update_layout(hovermode="x unified", xaxis_title="Threshold", yaxis_title="F1", yaxis_range=[0, 1.02])
    show(fig, "f1_sweep", 340)

# ----------------------------------------------------------------------------- transactions
with tab3:
    xt = cfg["thresholds"]["xgboost"]
    flagged = preds.xgboost_prob >= xt
    kinds = {
        "Illicit, caught by XGBoost": (preds.true_label == 1) & flagged,
        "Illicit, missed by XGBoost": (preds.true_label == 1) & ~flagged,
        "False alarms from XGBoost": (preds.true_label == 0) & flagged,
        "All transactions": preds.true_label >= 0,
    }
    kind = st.selectbox("Show", list(kinds))
    pool = preds[kinds[kind]].sort_values("xgboost_prob", ascending=False).head(300).set_index("txId")
    if pool.empty:
        st.info("No transactions match this filter.")
    else:
        st.caption(f"Top {len(pool)} by XGBoost probability.")
        tx = st.selectbox("Transaction", pool.index.tolist(),
                          format_func=lambda i: f"{i}  (step {int(pool.loc[i, 'time_step'])})")
        row = pool.loc[tx]
        truth = "illicit" if row.true_label == 1 else "licit"
        st.markdown(f"Time step **{int(row.time_step)}**. Actually **{truth}**.")
        fig = go.Figure()
        for name, (pcol, ckey, color, _) in MODELS.items():
            fig.add_trace(go.Bar(x=[name], y=[row[pcol]], marker_color=color, showlegend=False,
                                 text=f"{row[pcol]:.2f}", textposition="outside"))
            fig.add_trace(go.Scatter(x=[name], y=[cfg["thresholds"][ckey]], mode="markers", showlegend=False,
                                     marker=dict(symbol="line-ew", size=70, line=dict(width=3, color=ALERT)),
                                     hovertemplate="Validation threshold %{y:.2f}<extra></extra>"))
        fig.update_layout(yaxis_range=[0, 1.1], yaxis_title="Predicted probability of illicit")
        show(fig, "tx_bars", 360)
        st.caption("Bars are predicted probabilities. The red tick is each model's validation-chosen threshold: "
                   "a bar above its tick means the model flags the transaction.")

# ----------------------------------------------------------------------------- all results
with tab4:
    st.subheader("Test metrics")
    for label, r in runs.items():
        st.markdown(f"**{label}**")
        st.dataframe(r[0].round(4))
        st.download_button("Download CSV", r[0].to_csv(), f"model_comparison_{RUNS[label]}.csv", key=f"dl_{label}")

    st.subheader("Settings chosen on validation")
    rows = {}
    for label, r in runs.items():
        c = r[3]
        rows[label] = {
            "XGBoost class weight": c["xgboost"]["class_weight_name"], "XGBoost trees": c["xgboost"]["trees"],
            "GraphSAGE class weight": c["graphsage"]["class_weight_name"], "GraphSAGE epochs": c["graphsage"]["epochs"],
            "Ensemble GNN share": round(c["ensemble_gnn_share"], 2),
            **{f"Threshold {k}": round(v, 3) for k, v in c["thresholds"].items()}}
    st.dataframe(pd.DataFrame(rows).astype(str))

    st.subheader("F1 per test time step")
    both = pd.concat({lab: r[1] for lab, r in runs.items()}, axis=1)
    st.dataframe(both)

    st.subheader(f"Precision-recall and ROC curves ({run_label})")
    y = preds.true_label.to_numpy()
    pr_fig, roc_fig = go.Figure(), go.Figure()
    for name, (pcol, _, color, _) in MODELS.items():
        pr, rc, _t = precision_recall_curve(y, preds[pcol])
        fpr, tpr, _t = roc_curve(y, preds[pcol])
        pr_fig.add_trace(go.Scatter(x=rc, y=pr, name=name, line=dict(color=color, width=2.5)))
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, line=dict(color=color, width=2.5)))
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], showlegend=False, line=dict(color=MUTED, width=1, dash="dash")))
    pr_fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
    roc_fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
    left, right = st.columns(2)
    with left:
        show(pr_fig, "pr_curve", 340)
    with right:
        show(roc_fig, "roc_curve", 340)
