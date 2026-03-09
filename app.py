import streamlit as st
from joblib import load
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediScan AI · Disease Predictor",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model        = load("disease_model.joblib")
    symptoms_list = load("symptoms_list.joblib")
    return model, symptoms_list

try:
    model, symptoms_list = load_assets()
except Exception as e:
    st.error(f"⚠️ Could not load model files: {e}")
    st.stop()

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #05080f !important;
    font-family: 'DM Sans', sans-serif;
    color: #e8edf5;
}

[data-testid="stHeader"] { display: none !important; }

/* ── Main container ── */
[data-testid="stAppViewContainer"] > .main > .block-container {
    max-width: 760px;
    padding: 3rem 2rem 5rem;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 3rem 0 2rem;
    position: relative;
}
.hero-badge {
    display: inline-block;
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: #4fffb0;
    border: 1px solid #4fffb026;
    background: #4fffb010;
    padding: 5px 14px;
    border-radius: 100px;
    margin-bottom: 1.4rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.4rem, 6vw, 3.8rem);
    font-weight: 800;
    line-height: 1.1;
    background: linear-gradient(135deg, #ffffff 30%, #4fffb0 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: .7rem;
}
.hero-sub {
    font-size: 1.05rem;
    color: #7a8a9e;
    font-weight: 300;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.65;
}

/* ── Section label ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .16em;
    text-transform: uppercase;
    color: #4fffb0;
    margin-bottom: .5rem;
}

/* ── Streamlit multiselect ── */
.stMultiSelect [data-baseweb="select"] > div {
    background: #0d1320 !important;
    border: 1.5px solid #1e2d42 !important;
    border-radius: 14px !important;
    padding: 6px 10px !important;
    transition: border-color .2s;
}
.stMultiSelect [data-baseweb="select"] > div:focus-within {
    border-color: #4fffb0 !important;
    box-shadow: 0 0 0 3px #4fffb018 !important;
}
.stMultiSelect [data-baseweb="tag"] {
    background: #4fffb018 !important;
    border: 1px solid #4fffb040 !important;
    color: #4fffb0 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
}
.stMultiSelect label { display: none !important; }

/* ── Predict button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #4fffb0, #00c97a) !important;
    color: #05080f !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    letter-spacing: .04em;
    border: none !important;
    border-radius: 14px !important;
    padding: 1rem 0 !important;
    cursor: pointer;
    transition: transform .15s, box-shadow .15s !important;
    box-shadow: 0 8px 28px #4fffb030 !important;
    margin-top: .5rem;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 14px 36px #4fffb050 !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Result card ── */
.result-card {
    background: linear-gradient(135deg, #0d1a2a, #0a1520);
    border: 1.5px solid #4fffb030;
    border-radius: 20px;
    padding: 2rem 2.2rem;
    margin: 2rem 0 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 50% -20%, #4fffb012 0%, transparent 70%);
}
.result-label {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: #4fffb0;
    margin-bottom: .6rem;
    font-family: 'DM Sans', sans-serif;
}
.result-disease {
    font-family: 'Syne', sans-serif;
    font-size: clamp(1.8rem, 5vw, 2.8rem);
    font-weight: 800;
    color: #fff;
    line-height: 1.15;
}
.result-conf {
    display: inline-block;
    margin-top: .8rem;
    background: #4fffb018;
    color: #4fffb0;
    border: 1px solid #4fffb030;
    border-radius: 100px;
    font-size: .88rem;
    font-weight: 500;
    padding: 4px 16px;
}

/* ── Prob section header ── */
.prob-header {
    font-family: 'Syne', sans-serif;
    font-size: .85rem;
    font-weight: 600;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: #4a5a6e;
    margin-bottom: 1rem;
    margin-top: 2rem;
}

/* ── Probability row ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}
.prob-name {
    min-width: 160px;
    font-size: .88rem;
    color: #c0ccd8;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-weight: 400;
}
.prob-track {
    flex: 1;
    height: 7px;
    background: #0d1320;
    border-radius: 100px;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 100px;
    transition: width .6s cubic-bezier(.22,.68,0,1.2);
}
.prob-pct {
    min-width: 46px;
    text-align: right;
    font-size: .84rem;
    font-weight: 500;
    color: #7a8a9e;
}

/* ── Warning / info ── */
.stAlert > div {
    border-radius: 12px !important;
    border: 1.5px solid #2a3a50 !important;
    background: #0d1320 !important;
    color: #7a8a9e !important;
    font-size: .88rem !important;
}

/* ── Divider ── */
.divider {
    border: none;
    border-top: 1px solid #1a2638;
    margin: 2rem 0;
}

/* ── Footer ── */
.footer {
    text-align: center;
    font-size: .78rem;
    color: #2e4060;
    margin-top: 3rem;
    line-height: 2;
}
.footer a { color: #4fffb050; text-decoration: none; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #05080f; }
::-webkit-scrollbar-thumb { background: #1e2d42; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# ── Hero section ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">🧬 Machine Learning · Medical AI</div>
  <div class="hero-title">AI Disease<br>Predictor</div>
  <p class="hero-sub">
    Select your symptoms below and the trained classifier will predict the most likely condition — instantly.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Symptom input ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Step 1 — Choose Symptoms</p>', unsafe_allow_html=True)

# Sort for UX
sorted_symptoms = sorted(symptoms_list)
selected_symptoms = st.multiselect(
    label="symptoms",
    options=sorted_symptoms,
    placeholder="🔍  Search and select symptoms…",
)

# Live counter
n = len(selected_symptoms)
if n > 0:
    st.markdown(
        f'<p style="font-size:.82rem;color:#4fffb0;margin-top:.3rem;">'
        f'✓ {n} symptom{"s" if n>1 else ""} selected</p>',
        unsafe_allow_html=True,
    )

st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-label">Step 2 — Run Prediction</p>', unsafe_allow_html=True)

predict_clicked = st.button("Predict Disease →", key="predict_btn")

# ── Prediction logic ──────────────────────────────────────────────────────────
if predict_clicked:
    if not selected_symptoms:
        st.warning("Please select at least one symptom before predicting.")
    else:
        # Build input vector
        input_vector = np.zeros(len(symptoms_list), dtype=int)
        for symptom in selected_symptoms:
            idx = list(symptoms_list).index(symptom)
            input_vector[idx] = 1

        with st.spinner("Analysing symptoms…"):
            prediction = model.predict([input_vector])[0]
            probs      = model.predict_proba([input_vector])[0]
            classes    = model.classes_

        top_prob = max(probs) * 100

        # ── Main result card ──
        st.markdown(f"""
        <div class="result-card">
          <div class="result-label">Predicted Condition</div>
          <div class="result-disease">{prediction}</div>
          <span class="result-conf">Confidence {top_prob:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Probability breakdown ──
        pairs = sorted(zip(probs, classes), reverse=True)

        st.markdown('<p class="prob-header">Full probability breakdown</p>', unsafe_allow_html=True)

        for prob, cls in pairs:
            pct    = prob * 100
            is_top = cls == prediction
            fill_color = "#4fffb0" if is_top else "#1e3a52"
            name_color = "#ffffff" if is_top else "#c0ccd8"
            pct_color  = "#4fffb0" if is_top else "#4a5a6e"

            st.markdown(f"""
            <div class="prob-row">
              <span class="prob-name" style="color:{name_color};font-weight:{'600' if is_top else '400'}">
                {"▶ " if is_top else ""}{cls}
              </span>
              <div class="prob-track">
                <div class="prob-fill" style="width:{pct:.1f}%;background:{fill_color};"></div>
              </div>
              <span class="prob-pct" style="color:{pct_color}">{pct:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<div class="footer">
  ⚠️ For demonstration purposes only — not a substitute for professional medical advice.<br>
  Built with Streamlit · Scikit-learn · Python &nbsp;|&nbsp; <span style="color:#4fffb060">MediScan AI</span>
</div>
""", unsafe_allow_html=True)