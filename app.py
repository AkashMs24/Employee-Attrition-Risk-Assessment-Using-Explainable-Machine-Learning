import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# ==============================
# LOAD MODEL
# ==============================

MODEL_PATH = "rf_model_4_features.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model file `{MODEL_PATH}` not found. "
            "Please place the trained model file in the project root directory."
        )
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="AttritionIQ — HR Decision Support",
    page_icon="📊",
    layout="centered",
)

# ==============================
# PREMIUM DARK THEME CSS
# ==============================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

    :root {
        --bg:      #080808;
        --surface: #111111;
        --surf2:   #181818;
        --border:  #242424;
        --accent:  #c8ff00;
        --danger:  #ff4444;
        --warn:    #ffb800;
        --ok:      #00e676;
        --text:    #f0f0f0;
        --muted:   #555555;
        --radius:  14px;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif !important;
        background-color: var(--bg) !important;
        color: var(--text) !important;
    }

    .main .block-container {
        background: var(--bg);
        max-width: 860px;
        padding: clamp(1rem, 4vw, 2.5rem);
    }

    /* ── Hero ── */
    .hero {
        text-align: center;
        padding: clamp(2.5rem, 8vw, 4.5rem) 1rem clamp(1.5rem, 4vw, 2.5rem);
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        inset: 0;
        background: radial-gradient(ellipse 70% 50% at 50% 0%,
            rgba(200,255,0,0.06) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(200,255,0,0.08);
        border: 1px solid rgba(200,255,0,0.22);
        color: var(--accent);
        font-size: clamp(0.58rem, 1.6vw, 0.7rem);
        font-weight: 500;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        padding: 0.28rem 0.9rem;
        border-radius: 100px;
        margin-bottom: 1.1rem;
    }
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: clamp(1.9rem, 6.5vw, 3.6rem);
        font-weight: 800;
        line-height: 1.05;
        color: var(--text);
        margin: 0 0 0.45rem;
        letter-spacing: -0.02em;
    }
    .hero-title span { color: var(--accent); }
    .hero-sub {
        font-size: clamp(0.82rem, 2.2vw, 0.97rem);
        color: var(--muted);
        font-weight: 300;
        margin: 0;
    }

    /* ── Section labels ── */
    .sec-label {
        font-family: 'Syne', sans-serif;
        font-size: clamp(0.6rem, 1.6vw, 0.7rem);
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--muted);
        margin: 2rem 0 0.75rem;
        padding-bottom: 0.45rem;
        border-bottom: 1px solid var(--border);
    }

    /* ── Cards ── */
    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: clamp(1rem, 3vw, 1.5rem);
        margin-bottom: 1rem;
    }
    .card-danger { border-left: 3px solid var(--danger); background: rgba(255,68,68,0.04); }
    .card-warn   { border-left: 3px solid var(--warn);   background: rgba(255,184,0,0.04); }
    .card-ok     { border-left: 3px solid var(--ok);     background: rgba(0,230,118,0.04); }
    .card-accent { border-left: 3px solid var(--accent); }

    /* ── Inputs ── */
    [data-testid="stNumberInput"] input {
        background: var(--surf2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.9rem !important;
    }
    [data-testid="stNumberInput"] input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px rgba(200,255,0,0.1) !important;
    }
    [data-testid="stNumberInput"] label,
    [data-testid="stSelectbox"] label,
    [data-testid="stSlider"] label {
        color: var(--muted) !important;
        font-size: 0.75rem !important;
        font-family: 'DM Mono', monospace !important;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    /* Selectbox */
    [data-testid="stSelectbox"] > div > div {
        background: var(--surf2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
        font-family: 'DM Mono', monospace !important;
    }

    /* Slider */
    [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
        background: var(--accent) !important;
    }

    /* ── Button ── */
    [data-testid="stButton"] button,
    [data-testid="stDownloadButton"] button {
        background: var(--accent) !important;
        color: #000 !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: clamp(0.88rem, 2.2vw, 0.98rem) !important;
        letter-spacing: 0.04em;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.7rem 1.8rem !important;
        transition: opacity 0.2s, transform 0.15s !important;
    }
    [data-testid="stButton"] button:hover,
    [data-testid="stDownloadButton"] button:hover {
        opacity: 0.85 !important;
        transform: translateY(-1px) !important;
    }
    [data-testid="stDownloadButton"] button {
        background: var(--surface) !important;
        color: var(--accent) !important;
        border: 1px solid var(--accent) !important;
    }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 1rem 1.2rem !important;
        transition: border-color 0.2s, transform 0.2s;
    }
    [data-testid="stMetric"]:hover {
        border-color: #3a3a3a !important;
        transform: translateY(-1px);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.68rem !important;
        color: var(--muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-family: 'DM Mono', monospace !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Syne', sans-serif !important;
        font-size: clamp(1.3rem, 3.5vw, 1.8rem) !important;
        font-weight: 700 !important;
        color: var(--accent) !important;
    }

    /* ── Progress ── */
    [data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, var(--ok), var(--warn), var(--danger)) !important;
        border-radius: 4px !important;
    }
    [data-testid="stProgressBar"] > div {
        background: var(--surf2) !important;
        border-radius: 4px !important;
        height: 6px !important;
    }

    /* ── Alerts ── */
    [data-testid="stAlert"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        color: var(--text) !important;
        font-size: 0.88rem;
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
    }
    [data-testid="stExpander"] summary {
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.88rem !important;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: var(--surface) !important;
        border: 1px dashed var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 1rem;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border-radius: var(--radius) !important;
        overflow: hidden;
    }

    /* ── Divider ── */
    hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

    /* ── Caption ── */
    [data-testid="stCaptionContainer"] {
        color: var(--muted) !important;
        font-size: 0.72rem !important;
        font-family: 'DM Mono', monospace !important;
    }

    /* ── Headers ── */
    h1,h2,h3 {
        font-family: 'Syne', sans-serif !important;
        color: var(--text) !important;
        letter-spacing: -0.01em;
    }
    h2 { font-size: clamp(1.05rem, 2.8vw, 1.35rem) !important; font-weight: 700 !important; }
    h3 { font-size: clamp(0.92rem, 2.2vw, 1.08rem) !important; font-weight: 600 !important; }

    /* ── Risk score big display ── */
    .risk-score-wrap { text-align: center; padding: 1.5rem 0 1rem; }
    .risk-score-num {
        font-family: 'Syne', sans-serif;
        font-size: clamp(3rem, 10vw, 5.5rem);
        font-weight: 800;
        line-height: 1;
    }
    .risk-score-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: var(--muted);
        margin-top: 0.3rem;
    }
    .confidence-pill {
        display: inline-block;
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        padding: 0.25rem 0.8rem;
        border-radius: 100px;
        margin-top: 0.8rem;
        background: rgba(200,255,0,0.08);
        border: 1px solid rgba(200,255,0,0.2);
        color: var(--accent);
    }

    /* ── Adjusted income badge ── */
    .income-badge {
        display: inline-block;
        background: var(--surf2);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.3rem 0.8rem;
        font-family: 'DM Mono', monospace;
        font-size: 0.82rem;
        color: var(--accent);
        margin-top: 0.3rem;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: var(--surface); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

    /* ── Mobile ── */
    @media (max-width: 600px) {
        .main .block-container { padding: 0.75rem !important; }
        .hero { padding: 2rem 0.5rem 1.5rem; }
    }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ──────────────────────────────────────────────────
mpl.rcParams.update({
    "figure.facecolor":  "#111111",
    "axes.facecolor":    "#111111",
    "axes.edgecolor":    "#242424",
    "axes.labelcolor":   "#888888",
    "xtick.color":       "#555555",
    "ytick.color":       "#888888",
    "text.color":        "#f0f0f0",
    "grid.color":        "#1e1e1e",
    "grid.linestyle":    "--",
    "font.family":       "monospace",
})

# ==============================
# HERO
# ==============================

st.markdown("""
<div class="hero">
    <div class="hero-badge">Explainable ML · Random Forest · HR Analytics</div>
    <div class="hero-title">Attrition<span>IQ</span></div>
    <p class="hero-sub">Employee attrition risk assessment powered by explainable machine learning</p>
</div>
""", unsafe_allow_html=True)

# ==============================
# INPUT SECTION
# ==============================

st.markdown("<div class='sec-label'>Employee Details</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age              = st.number_input("Age", 18, 60, 30)
    years_at_company = st.number_input("Years at Company", 0, 40, 5)

with col2:
    monthly_income = st.number_input("Monthly Income", 1000, 200000, 30000)
    overtime       = st.selectbox("OverTime", ["No", "Yes"])

overtime_enc = 1 if overtime == "Yes" else 0

# ==============================
# WHAT-IF ANALYSIS
# ==============================

st.markdown("<div class='sec-label'>What-If Salary Simulation</div>", unsafe_allow_html=True)

salary_change = st.slider(
    "Simulate salary change (%)",
    min_value=-20,
    max_value=50,
    value=0,
    step=5,
)

adjusted_income = int(monthly_income * (1 + salary_change / 100))

st.markdown(
    f"<div class='income-badge'>Adjusted Monthly Income: ₹{adjusted_income:,}</div>",
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# ==============================
# PREDICT BUTTON
# ==============================

if st.button("🔍 Predict Attrition Risk", use_container_width=True):

    input_data = np.array([[age, adjusted_income, years_at_company, overtime_enc]])
    proba      = model.predict_proba(input_data)[0][1]
    risk_pct   = int(proba * 100)

    # Confidence
    if proba < 0.25 or proba > 0.75:
        confidence = "High"
    elif proba < 0.4 or proba > 0.6:
        confidence = "Medium"
    else:
        confidence = "Low"

    # Color based on risk
    if proba >= 0.6:
        score_color = "#ff4444"
        card_class  = "card-danger"
        verdict     = "High Risk"
        advice      = "Immediate HR attention recommended. Consider compensation review, workload audit, or retention conversation."
    elif proba >= 0.35:
        score_color = "#ffb800"
        card_class  = "card-warn"
        verdict     = "Medium Risk"
        advice      = "Monitor and engage this employee. Check-in on job satisfaction and career development opportunities."
    else:
        score_color = "#00e676"
        card_class  = "card-ok"
        verdict     = "Low Risk"
        advice      = "Employee is likely to stay. Continue standard engagement practices."

    # ── Big score display ──────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="risk-score-wrap">
        <div class="risk-score-num" style="color:{score_color};">{risk_pct}%</div>
        <div class="risk-score-label">Attrition Risk Score</div>
        <div class="confidence-pill">Model Confidence: {confidence}</div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(risk_pct)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metrics ───────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Risk Score",      f"{risk_pct}%")
    c2.metric("Confidence",      confidence)
    c3.metric("Adjusted Income", f"₹{adjusted_income:,}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Verdict card ──────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="card {card_class}">
        <div style="font-family:Syne,sans-serif; font-weight:700; font-size:1rem;
                    color:{score_color}; margin-bottom:0.4rem;">
            {verdict}
        </div>
        <div style="font-size:0.87rem; color:#aaa; line-height:1.65;">
            {advice}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Download report ────────────────────────────────────────────────────
    report_df = pd.DataFrame({
        "Age":                        [age],
        "Monthly Income":             [adjusted_income],
        "Years at Company":           [years_at_company],
        "OverTime":                   ["Yes" if overtime_enc == 1 else "No"],
        "Attrition Risk Probability": [round(proba, 2)],
    })
    csv = report_df.to_csv(index=False)

    st.download_button(
        "📥 Download Prediction Report",
        csv,
        "attrition_prediction_report.csv",
        "text/csv",
        use_container_width=True,
    )

    st.caption("⚠️ Predictions are probabilistic and should be used as decision support, not final judgment.")

# ==============================
# FEATURE IMPORTANCE
# ==============================

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div class='sec-label'>Feature Importance</div>", unsafe_allow_html=True)

feature_names = ["Age", "Monthly Income", "Years at Company", "OverTime"]
importances   = model.feature_importances_

fi_df = pd.DataFrame({
    "Feature":    feature_names,
    "Importance": importances,
}).sort_values(by="Importance", ascending=True)

fig, ax = plt.subplots(figsize=(7, 2.8))
ax.barh(
    fi_df["Feature"],
    fi_df["Importance"],
    color=["#c8ff00" if v == fi_df["Importance"].max() else "#2a2a2a" for v in fi_df["Importance"]],
    edgecolor="#111",
    height=0.5,
)
ax.set_xlabel("Importance Score", fontsize=9, color="#555")
ax.set_title("Feature Importance — Random Forest", fontsize=10, color="#888", pad=10)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.spines["bottom"].set_color("#242424")
ax.tick_params(colors="#666", labelsize=9)
ax.xaxis.set_tick_params(color="#242424")
fig.tight_layout()

st.pyplot(fig)

# ==============================
# EXPANDERS
# ==============================

with st.expander("🧠 How is this prediction made?"):
    st.markdown("""
    - The model is trained on historical HR data.
    - It uses a **Random Forest** classifier with 4 key features.
    - The output is a probability score between 0–100%.
    - Higher probability indicates higher attrition risk.
    """)

# ==============================
# BATCH PREDICTION — CSV Upload
# ==============================

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div class='sec-label'>Batch Prediction — CSV Upload</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload employee CSV", type=["csv"])

if uploaded_file:
    batch_df         = pd.read_csv(uploaded_file)
    required_columns = ["Age", "MonthlyIncome", "YearsAtCompany", "OverTime"]

    if not all(col in batch_df.columns for col in required_columns):
        st.error("CSV must contain: Age, MonthlyIncome, YearsAtCompany, OverTime")
    else:
        batch_df["OverTime"]       = batch_df["OverTime"].map({"Yes": 1, "No": 0})
        X_batch                    = batch_df[required_columns]
        batch_df["Attrition Risk"] = model.predict_proba(X_batch)[:, 1].round(2)
        st.dataframe(batch_df, use_container_width=True)

with st.expander("ℹ️ Feature Explanation"):
    st.markdown("""
    - **Age** — Career stage indicator
    - **Monthly Income** — Compensation satisfaction proxy
    - **Years at Company** — Employee loyalty & stability
    - **OverTime** — Workload & burnout signal
    """)

# ==============================
# FOOTER
# ==============================

st.markdown("""
<hr>
<div style='text-align:center; padding:1rem 0 0.5rem;'>
    <div style='font-family:DM Mono,monospace; font-size:0.68rem; color:#2a2a2a; letter-spacing:0.1em;'>
        ATTRITIONIQ &nbsp;·&nbsp; v1.0 &nbsp;·&nbsp; Portfolio Demonstration &nbsp;·&nbsp; Built by Akash M S
    </div>
</div>
""", unsafe_allow_html=True)
