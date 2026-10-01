"""Business Profit Prediction - Streamlit dashboard (Linear Regression).
Run:  streamlit run app.py
"""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from model import FEATURES, TARGET, load_data, predict_profit, train_model

st.set_page_config(page_title="Business Profit Prediction", page_icon="■", layout="wide", initial_sidebar_state="auto")

# ----------------------------------------------------------------- styling (black & white)
st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-family: Arial, Helvetica, sans-serif; }
    #MainMenu, footer {visibility: hidden;}
    [data-testid="stToolbar"], [data-testid="stDecoration"] {display:none;}
    header[data-testid="stHeader"] {background: transparent; height: 2.5rem;}
    [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"],
    [data-testid="stExpandSidebarButton"] {visibility: visible !important; display: flex !important;}
    .block-container { padding-top: 0.6rem; max-width: 1250px; }
    .titlebar { background:#000; color:#fff; padding:12px 20px; margin-bottom:10px;
                display:flex; justify-content:space-between; align-items:center; }
    .titlebar b { font-size:18px; letter-spacing:.5px; }
    .titlebar span { font-size:12px; opacity:.85; }
    .credit { border:1px solid #000; padding:6px 14px; font-size:12px; margin-bottom:10px;
              display:flex; justify-content:space-between; flex-wrap:wrap; gap:6px; }
    .footer { background:#000; color:#fff; padding:12px 20px; margin-top:28px; font-size:12px;
              display:flex; justify-content:space-between; flex-wrap:wrap; gap:6px; }
    .card { border:1px solid #000; padding:14px 16px; background:#fff; }
    .card .lbl { font-size:11px; text-transform:uppercase; letter-spacing:1px; color:#444; }
    .card .val { font-size:28px; font-weight:700; margin-top:2px; }
    .card .sub { font-size:11px; color:#555; }
    .section { font-size:13px; font-weight:700; text-transform:uppercase; letter-spacing:1px;
               border-bottom:1px solid #000; padding-bottom:4px; margin:18px 0 10px 0; }
    .result { background:#000; color:#fff; padding:22px; text-align:center; }
    .result .val { font-size:40px; font-weight:700; }
    .result .lbl { font-size:12px; letter-spacing:1px; text-transform:uppercase; opacity:.8; }
    div.stButton > button { border:1px solid #000; border-radius:0; font-weight:600; width:100%; }
    div.stButton > button[kind="primary"] { background:#000; color:#fff; }
    div.stButton > button[kind="secondary"] { background:#fff; color:#000; }
    div.stButton > button:hover { background:#000; color:#fff; border:1px solid #000; }
    [data-testid="stSidebar"] { border-right:1px solid #000; }
    div.stButton > button, div.stDownloadButton > button { min-height: 44px; }
    .card .val, .result .val { overflow-wrap:anywhere; }
    .card, .result { margin-bottom: 8px; }
    img, canvas, svg { max-width: 100%; }

    @media (max-width: 768px) {
      .block-container { padding: 0.4rem 0.7rem 2rem 0.7rem; }
      .titlebar { padding: 10px 12px; flex-direction: column; align-items: flex-start; gap: 4px; }
      .titlebar b { font-size: 15px; }
      .credit, .footer { padding: 8px 12px; font-size: 11px; flex-direction: column; }
      .card { padding: 10px 12px; }
      .card .val { font-size: 22px; }
      .result { padding: 16px; }
      .result .val { font-size: 30px; }
      .section { font-size: 12px; margin: 14px 0 8px 0; }
      [data-testid="stSidebar"] { max-width: 85vw; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "black",
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.facecolor": "white", "figure.facecolor": "white", "axes.grid": True,
    "grid.color": "#D9D9D9", "grid.linewidth": 0.5,
})


def money(x):
    return f"${x:,.0f}"


def card(label, value, sub=""):
    return f'<div class="card"><div class="lbl">{label}</div><div class="val">{value}</div><div class="sub">{sub}</div></div>'


def show(fig):
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ----------------------------------------------------------------- sidebar: model settings only
with st.sidebar:
    st.markdown("### Settings")
    test_size = st.slider("Test data size", 0.1, 0.5, 0.2, 0.05)
    seed = st.number_input("Random seed", 0, 9999, 42)
    st.caption("Dataset upload is on the Dashboard page.")

# ----------------------------------------------------------------- session state
for key, default in {"page": "Dashboard", "data_bytes": None, "data_name": None, "uk": 0}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ----------------------------------------------------------------- title bar (filled in after data loads) + credits + ribbon
titlebar = st.empty()
CREDITS = "Priyaranjan &nbsp;|&nbsp; Fardin &nbsp;|&nbsp; Arif &nbsp;|&nbsp; Deep"
st.markdown(
    f'<div class="credit"><span>Design and developed by</span><b>{CREDITS}</b></div>',
    unsafe_allow_html=True,
)
n1, n2, _ = st.columns([1, 1, 4])
if n1.button("Dashboard", type="primary" if st.session_state.page == "Dashboard" else "secondary"):
    st.session_state.page = "Dashboard"
    st.rerun()
if n2.button("Predictor", type="primary" if st.session_state.page == "Predictor" else "secondary"):
    st.session_state.page = "Predictor"
    st.rerun()

# ----------------------------------------------------------------- dataset upload (Dashboard page)
if st.session_state.page == "Dashboard":
    st.markdown('<div class="section">Dataset source</div>', unsafe_allow_html=True)
    u1, u2 = st.columns([3, 1])
    up = u1.file_uploader(
        "Upload your dataset (CSV)", type="csv", key=f"up_{st.session_state.uk}",
        help="Required columns: R&D Spend, Administration, Marketing Spend, Profit",
    )
    if up is not None and up.name != st.session_state.data_name:
        st.session_state.data_bytes = up.getvalue()
        st.session_state.data_name = up.name
    u2.write("")
    u2.write("")
    if u2.button("Use sample data"):
        st.session_state.data_bytes = None
        st.session_state.data_name = None
        st.session_state.uk += 1
        st.rerun()
    st.caption("Required columns: R&D Spend, Administration, Marketing Spend, Profit. "
               "Without an upload, the sample dataset is used.")

# ----------------------------------------------------------------- load data + train
try:
    if st.session_state.data_bytes is not None:
        df = load_data(io.BytesIO(st.session_state.data_bytes))
        source_label = st.session_state.data_name
    else:
        df = load_data("data/startups.csv")
        source_label = "sample dataset"
except Exception as e:  # noqa: BLE001
    st.error(f"Could not use that file ({e}). Showing the sample dataset instead.")
    st.session_state.data_bytes = None
    st.session_state.data_name = None
    df = load_data("data/startups.csv")
    source_label = "sample dataset"

res = train_model(df, test_size, int(seed))
model = res["model"]

titlebar.markdown(
    '<div class="titlebar"><b>BUSINESS PROFIT PREDICTION</b>'
    f"<span>Linear Regression &nbsp;|&nbsp; {len(df)} records &nbsp;|&nbsp; R² {res['r2']:.3f}"
    f" &nbsp;|&nbsp; {source_label}</span></div>",
    unsafe_allow_html=True,
)
if st.session_state.page == "Dashboard":
    st.success(f"Using: {source_label} ({len(df)} records)")

# ================================================================= PAGE 1: DASHBOARD
if st.session_state.page == "Dashboard":
    k = st.columns(4)
    k[0].markdown(card("R² Score", f"{res['r2']:.3f}", "variation explained"), unsafe_allow_html=True)
    k[1].markdown(card("MAE", money(res["mae"]), "mean absolute error"), unsafe_allow_html=True)
    k[2].markdown(card("MSE", f"{res['mse']:,.0f}", "mean squared error"), unsafe_allow_html=True)
    k[3].markdown(card("Train / Test", f"{res['n_train']} / {res['n_test']}", "records"), unsafe_allow_html=True)

    st.markdown('<div class="section">Model performance</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 3.6))
        y, p = res["y_test"], res["pred"]
        lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, linestyle="--", label="Perfect fit")
        ax.scatter(y, p, color="black", s=28, label="Test records")
        ax.set_xlabel("Actual profit ($)")
        ax.set_ylabel("Predicted profit ($)")
        ax.set_title("Actual vs Predicted", loc="left", fontweight="bold")
        ax.legend(frameon=False)
        show(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(5, 3.6))
        per_k = [res["coefs"][f] * 1000 for f in FEATURES]
        bars = ax.barh(FEATURES, per_k, color="black")
        ax.axvline(0, color="black", linewidth=0.8)
        for b, v in zip(bars, per_k):
            ax.text(v + (20 if v >= 0 else -20), b.get_y() + b.get_height() / 2, f"{v:,.0f}",
                    va="center", ha="left" if v >= 0 else "right")
        ax.set_xlabel("Change in profit ($) for +$1,000 spend")
        ax.set_title("Effect of each expense", loc="left", fontweight="bold")
        ax.margins(x=0.15)
        show(fig)

    st.markdown('<div class="section">Data exploration</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        feat = st.selectbox("Spend vs profit", FEATURES)
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.scatter(df[feat], df[TARGET], color="black", s=22)
        m, b = np.polyfit(df[feat], df[TARGET], 1)
        xs = np.linspace(df[feat].min(), df[feat].max(), 50)
        ax.plot(xs, m * xs + b, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel(f"{feat} ($)")
        ax.set_ylabel("Profit ($)")
        show(fig)
    with c4:
        st.write("Correlation matrix")
        corr = df[FEATURES + [TARGET]].corr()
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.grid(False)
        im = ax.imshow(corr, cmap="Greys", vmin=-1, vmax=1)
        labels = ["R&D", "Admin", "Marketing", "Profit"]
        ax.set_xticks(range(4), labels)
        ax.set_yticks(range(4), labels)
        for i in range(4):
            for j in range(4):
                v = corr.iloc[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if abs(v) > 0.5 else "black")
        ax.spines[:].set_visible(False)
        show(fig)

    st.markdown('<div class="section">Dataset & predictions</div>', unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["Dataset", "Summary statistics", "Actual vs Predicted (test set)"])
    with t1:
        st.dataframe(df, use_container_width=True, height=280)
        st.download_button("Download dataset (CSV)", df.to_csv(index=False), "dataset.csv", "text/csv")
    with t2:
        st.dataframe(df[FEATURES + [TARGET]].describe().T.round(2), use_container_width=True)
    with t3:
        cmp_df = res["X_test"].copy()
        cmp_df["Actual Profit"] = res["y_test"]
        cmp_df["Predicted Profit"] = res["pred"]
        cmp_df["Error"] = cmp_df["Actual Profit"] - cmp_df["Predicted Profit"]
        st.dataframe(cmp_df.round(2), use_container_width=True)

# ================================================================= PAGE 2: PREDICTOR
else:
    st.markdown('<div class="section">Enter planned expenditure</div>', unsafe_allow_html=True)
    means = df[FEATURES].mean()
    i1, i2, i3 = st.columns(3)
    rd = i1.number_input("R&D Spend ($)", 0.0, float(df[FEATURES[0]].max() * 2), float(round(means.iloc[0], -3)), 1000.0)
    ad = i2.number_input("Administration ($)", 0.0, float(df[FEATURES[1]].max() * 2), float(round(means.iloc[1], -3)), 1000.0)
    mk = i3.number_input("Marketing Spend ($)", 0.0, float(df[FEATURES[2]].max() * 2), float(round(means.iloc[2], -3)), 1000.0)

    b1, _ = st.columns([1, 4])
    go = b1.button("Predict profit", type="primary")
    if go or "last_pred" in st.session_state:
        if go:
            st.session_state.last_pred = predict_profit(model, rd, ad, mk)
        r1, r2 = st.columns([1, 2])
        r1.markdown(
            f'<div class="result"><div class="lbl">Estimated profit</div>'
            f'<div class="val">{money(st.session_state.last_pred)}</div></div>',
            unsafe_allow_html=True,
        )
        c = res["coefs"]
        r2.markdown(
            f'<div class="card"><div class="lbl">Fitted model</div>'
            f'<div style="font-size:15px;margin-top:6px">Profit = {res["intercept"]:,.0f} '
            f'+ {c[FEATURES[0]]:.4f} × R&D {c[FEATURES[1]]:+.4f} × Admin '
            f'{c[FEATURES[2]]:+.4f} × Marketing</div>'
            f'<div class="sub" style="margin-top:6px">Predictions are estimates; real results also depend on other business factors.</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section">Compare spending scenarios</div>', unsafe_allow_html=True)
    st.caption("Edit the table, then press Compare.")
    base = pd.DataFrame({
        "Scenario": ["A - Current plan", "B - More R&D", "C - More Marketing"],
        "R&D Spend": [rd, rd * 1.2, rd],
        "Administration": [ad, ad, ad],
        "Marketing Spend": [mk, mk, mk * 1.2],
    })
    edited = st.data_editor(base, use_container_width=True, hide_index=True, num_rows="dynamic")
    b2, _ = st.columns([1, 4])
    if b2.button("Compare scenarios", type="primary"):
        out = edited.dropna().copy()
        out["Predicted Profit"] = model.predict(out[FEATURES]).round(0)
        out["Change vs first"] = (out["Predicted Profit"] - out["Predicted Profit"].iloc[0]).round(0)
        s1, s2 = st.columns([3, 2])
        s1.dataframe(out, use_container_width=True, hide_index=True)
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        ax.bar(out["Scenario"].str.split(" ").str[0], out["Predicted Profit"], color="black", width=0.5)
        ax.set_ylabel("Predicted profit ($)")
        ax.set_title("Scenario comparison", loc="left", fontweight="bold")
        with s2:
            show(fig)
        st.download_button("Download scenarios (CSV)", out.to_csv(index=False), "scenarios.csv", "text/csv")

# ----------------------------------------------------------------- footer credits (both pages)
st.markdown(
    f'<div class="footer"><span>Design and developed by</span><b>{CREDITS}</b></div>',
    unsafe_allow_html=True,
)
