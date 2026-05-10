import os
import streamlit as st
import numpy as np
import joblib
from pathlib import Path

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Mental Health AI",
    page_icon="🧠",
    layout="centered",
)

# ---------------------------
# GLASSMORPHISM + ANIMATION UI
# ---------------------------
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #0f172a, #1e293b, #0f172a);
        background-size: 400% 400%;
        animation: gradientBG 12s ease infinite;
    }

    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    .title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #ffffff;
        text-align: center;
        margin-bottom: 8px;
        text-shadow: 0px 4px 20px rgba(0,0,0,0.6);
    }

    .subtitle {
        font-size: 1.1rem;
        color: #cbd5e1;
        text-align: center;
        margin-bottom: 20px;
    }

    .fade-box {
        text-align: center;
        margin-bottom: 25px;
        padding: 15px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        backdrop-filter: blur(10px);
        color: #cbd5e1;
        font-size: 0.95rem;

        opacity: 0;
        transform: translateY(10px);
        animation: fadeIn 1.2s ease forwards;
    }

    @keyframes fadeIn {
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .card {
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border-radius: 18px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.37);
        margin-bottom: 18px;
    }

    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #ff4d6d, #7c3aed);
        color: white;
        font-size: 16px;
        font-weight: 600;
        padding: 12px;
        border-radius: 12px;
        border: none;
        transition: 0.3s ease;
    }

    .stButton>button:hover {
        transform: scale(1.03);
        box-shadow: 0px 0px 20px rgba(124, 58, 237, 0.6);
    }

    label, p {
        color: #e2e8f0 !important;
        font-weight: 500;
    }

    hr {
        border: 1px solid rgba(255,255,255,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# MODEL LOADING (Random Forest)
# ---------------------------
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "models" / "mental_health(rf).pkl"
    return joblib.load(model_path)

model = load_model()

# ---------------------------
# OPTIONS
# ---------------------------
GENDER_OPTIONS = ["Female", "Male"]

COUNTRY_OPTIONS = [
    "Australia","Belgium","Bosnia and Herzegovina","Brazil","Canada",
    "Colombia","Costa Rica","Croatia","Czech Republic","Denmark",
    "Finland","France","Georgia","Germany","Greece","India",
    "Ireland","Israel","Italy","Mexico","Moldova","Netherlands",
    "New Zealand","Nigeria","Philippines","Poland","Portugal",
    "Russia","Singapore","South Africa","Sweden","Switzerland",
    "Thailand","United Kingdom","United States"
]

OCCUPATION_OPTIONS = ["Business", "Corporate", "Housewife", "Others", "Student"]
YES_NO = ["No", "Yes"]
MAYBE_NO_YES = ["Maybe", "No", "Yes"]
MOOD_OPTIONS = ["High", "Low", "Medium"]

DAYS_INDOORS_OPTIONS = [
    "Go out Every day",
    "1-14 days",
    "15-30 days",
    "31-60 days",
    "More than 2 months",
]

# ---------------------------
# ENCODING MAP
# ---------------------------
ENCODING_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "country": {v: i for i, v in enumerate(COUNTRY_OPTIONS)},
    "occupation": {v: i for i, v in enumerate(OCCUPATION_OPTIONS)},
    "self_employed": {"No": 0, "Yes": 1},
    "family_history": {"No": 0, "Yes": 1},
    "days_indoors": {
        "Go out Every day": 0,
        "1-14 days": 1,
        "15-30 days": 2,
        "31-60 days": 3,
        "More than 2 months": 4,
    },
    "growing_stress": {"Maybe": 0, "No": 1, "Yes": 2},
    "changes_habits": {"Maybe": 0, "No": 1, "Yes": 2},
    "mental_health_history": {"Maybe": 0, "No": 1, "Yes": 2},
    "mood_swings": {"High": 0, "Low": 1, "Medium": 2},
    "coping_struggles": {"No": 0, "Yes": 1},
    "work_interest": {"Maybe": 0, "No": 1, "Yes": 2},
    "social_weakness": {"Maybe": 0, "No": 1, "Yes": 2},
    "mental_health_interview": {"Maybe": 0, "No": 1, "Yes": 2},
    "care_options": {"No": 0, "Not sure": 1, "Yes": 2},
}

FEATURE_ORDER = list(ENCODING_MAP.keys())

# ---------------------------
# HEADER
# ---------------------------
st.markdown("<div class='title'>🧠 Mental Health AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered Random Forest prediction system</div>", unsafe_allow_html=True)

# ---------------------------
# INTRO BOX (ANIMATED)
# ---------------------------
st.markdown(
    """
    <div class="fade-box">
        This system analyzes behavioral, lifestyle, and mental health indicators to estimate the likelihood of requiring mental health support.  
        It is designed for awareness and educational purposes only.
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# FORM
# ---------------------------
with st.container():

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("👤 Profile Information")

        gender = st.selectbox("Gender", GENDER_OPTIONS)
        country = st.selectbox("Country", COUNTRY_OPTIONS)
        occupation = st.selectbox("Occupation", OCCUPATION_OPTIONS)
        self_employed = st.selectbox("Self Employed", YES_NO)
        family_history = st.selectbox("Family History", YES_NO)

        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🧠 Mental Health Factors")

        days_indoors = st.selectbox("Days Indoors", DAYS_INDOORS_OPTIONS)
        growing_stress = st.selectbox("Growing Stress", MAYBE_NO_YES)
        changes_habits = st.selectbox("Changes in Habits", MAYBE_NO_YES)
        mental_health_history = st.selectbox("Mental Health History", MAYBE_NO_YES)
        mood_swings = st.selectbox("Mood Swings", MOOD_OPTIONS)
        coping_struggles = st.selectbox("Coping Struggles", YES_NO)

        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("💼 Support & Awareness")

        work_interest = st.selectbox("Work Interest", MAYBE_NO_YES)
        social_weakness = st.selectbox("Social Weakness", MAYBE_NO_YES)
        mental_health_interview = st.selectbox("Interview Interest", MAYBE_NO_YES)
        care_options = st.selectbox("Care Options", ["No", "Not sure", "Yes"])

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# INPUT VECTOR
# ---------------------------
def build_input():
    user_data = {
        "gender": gender,
        "country": country,
        "occupation": occupation,
        "self_employed": self_employed,
        "family_history": family_history,
        "days_indoors": days_indoors,
        "growing_stress": growing_stress,
        "changes_habits": changes_habits,
        "mental_health_history": mental_health_history,
        "mood_swings": mood_swings,
        "coping_struggles": coping_struggles,
        "work_interest": work_interest,
        "social_weakness": social_weakness,
        "mental_health_interview": mental_health_interview,
        "care_options": care_options,
    }

    return np.array([ENCODING_MAP[f][user_data[f]] for f in FEATURE_ORDER]).reshape(1, -1)

# ---------------------------
# PREDICTION
# ---------------------------
st.write("---")

if st.button("🔍 Analyze Mental Health Risk"):

    X = build_input()

    prediction = model.predict(X)[0]

    try:
        prob = model.predict_proba(X)[0]
        confidence = float(np.max(prob) * 100)
    except:
        confidence = 0.0

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Result")

    if prediction == 1:
        st.error("⚠ Higher likelihood of needing mental health support")
    else:
        st.success("✅ Lower likelihood of needing treatment")

    st.metric("Confidence", f"{confidence:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("Fill the form and click analyze to get prediction.")