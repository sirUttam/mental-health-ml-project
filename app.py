import os
import streamlit as st
import numpy as np
import joblib

st.set_page_config(
    page_title="Mental Health Treatment Predictor",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {background: linear-gradient(180deg, #f7fbff 0%, #e8f1ff 100%);}
    .stButton>button {background-color: #0f4c81; color: white;}
    .st-badge {background: #0f4c81; color: white;}
    .title {font-size: 2.8rem; font-weight: 800; color: #0f4c81;}
    .subtitle {font-size: 1.1rem; color: #334e68;}
    .card {background: white; border-radius: 18px; padding: 20px; box-shadow: 0 12px 35px rgba(15, 76, 129, 0.12);}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model():
    model_path = os.path.join("models", "Mental_Health_Project.pkl")
    return joblib.load(model_path)

model = load_model()

GENDER_OPTIONS = ["Female", "Male"]
COUNTRY_OPTIONS = [
    "Australia",
    "Belgium",
    "Bosnia and Herzegovina",
    "Brazil",
    "Canada",
    "Colombia",
    "Costa Rica",
    "Croatia",
    "Czech Republic",
    "Denmark",
    "Finland",
    "France",
    "Georgia",
    "Germany",
    "Greece",
    "India",
    "Ireland",
    "Israel",
    "Italy",
    "Mexico",
    "Moldova",
    "Netherlands",
    "New Zealand",
    "Nigeria",
    "Philippines",
    "Poland",
    "Portugal",
    "Russia",
    "Singapore",
    "South Africa",
    "Sweden",
    "Switzerland",
    "Thailand",
    "United Kingdom",
    "United States",
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

ENCODING_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "country": {value: idx for idx, value in enumerate(COUNTRY_OPTIONS)},
    "occupation": {value: idx for idx, value in enumerate(OCCUPATION_OPTIONS)},
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
FEATURE_ORDER = [
    "gender",
    "country",
    "occupation",
    "self_employed",
    "family_history",
    "days_indoors",
    "growing_stress",
    "changes_habits",
    "mental_health_history",
    "mood_swings",
    "coping_struggles",
    "work_interest",
    "social_weakness",
    "mental_health_interview",
    "care_options",
]

st.markdown("<div class='title'>Mental Health Treatment Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Enter the profile details below and get a professional prediction from your trained model.</div>", unsafe_allow_html=True)
st.write("---")

with st.container():
    left, right = st.columns([2, 1])

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Profile Summary")
        gender = st.selectbox("Gender", GENDER_OPTIONS)
        country = st.selectbox("Country", COUNTRY_OPTIONS, index=COUNTRY_OPTIONS.index("United States"))
        occupation = st.selectbox("Occupation", OCCUPATION_OPTIONS)
        self_employed = st.selectbox("Self Employed", YES_NO)
        family_history = st.selectbox("Family History of Mental Health", YES_NO)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card' style='margin-top: 18px;'>", unsafe_allow_html=True)
        st.subheader("Lifestyle & Wellbeing")
        days_indoors = st.selectbox("Days Indoors", DAYS_INDOORS_OPTIONS)
        growing_stress = st.selectbox("Growing Stress", MAYBE_NO_YES, index=0)
        changes_habits = st.selectbox("Changes in Habits", MAYBE_NO_YES, index=0)
        mental_health_history = st.selectbox("History of Mental Health Issues", MAYBE_NO_YES, index=0)
        mood_swings = st.selectbox("Mood Swings", MOOD_OPTIONS, index=2)
        coping_struggles = st.selectbox("Coping Struggles", YES_NO)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card' style='margin-top: 18px;'>", unsafe_allow_html=True)
        st.subheader("Support & Interest")
        work_interest = st.selectbox("Interest in Work", MAYBE_NO_YES, index=0)
        social_weakness = st.selectbox("Social Weakness", MAYBE_NO_YES, index=0)
        mental_health_interview = st.selectbox("Mental Health Interview Interest", MAYBE_NO_YES, index=0)
        care_options = st.selectbox("Care Options Awareness", ["No", "Not sure", "Yes"], index=2)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Model Overview")
        st.write(
            "This interface uses your trained Decision Tree model from `models/Mental_Health_Project.pkl`."
        )
        st.write("The model evaluates a complete mental health profile and estimates whether treatment is likely needed.")
        st.markdown("<hr>")
        st.write("**Input encoding rules:**")
        st.write("- Gender and occupation are label encoded.")
        st.write("- Days indoors uses an ordinal mapping from lowest to highest risk.")
        st.write("- Categorical features use the same values from your training dataset.")
        st.markdown("</div>", unsafe_allow_html=True)

prediction_button = st.button("Predict Treatment Need")

def build_feature_vector():
    user_input = {
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
    feature_vector = [ENCODING_MAP[name][user_input[name]] for name in FEATURE_ORDER]
    return np.array(feature_vector).reshape(1, -1)

if prediction_button:
    input_vector = build_feature_vector()
    prediction = model.predict(input_vector)[0]
    probability = model.predict_proba(input_vector)[0]
    score = float(np.max(probability) * 100)
    label = "Treatment Needed" if prediction == 1 else "Treatment Not Needed"
    color = "success" if prediction == 0 else "warning"

    st.write("---")
    st.markdown(f"<div class='card' style='padding: 24px;'>", unsafe_allow_html=True)
    st.markdown(f"### Prediction Result: {label}")
    st.metric("Confidence", f"{score:.1f}%")
    st.write(
        "This prediction is based on your trained `Mental_Health_Project.pkl` model and the same preprocessing used during training."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if prediction == 1:
        st.warning(
            "The model indicates a higher likelihood that the person may need mental health treatment. Use this as a supportive guide, not a diagnosis."
        )
    else:
        st.success(
            "The model indicates a lower likelihood that mental health treatment is needed based on the provided profile."
        )

    if score > 80:
        st.balloons()
else:
    st.info("Fill in the details and click ‘Predict Treatment Need’ to see the result.")
