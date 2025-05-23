import pandas as pd
import numpy as np
import streamlit as st
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from googletrans import Translator

# Page setup
st.set_page_config(page_title='🩺 AI Healthcare Chatbot', layout='centered')
st.markdown("""
    <h1 style='text-align: center; color: #0d6efd;'>🩺 AI Healthcare Chatbot</h1>
    <p style='text-align: center; color: gray;'>Describe your symptoms. Get a diagnosis. Find the right doctor.</p>
""", unsafe_allow_html=True)

# Medical-themed background
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1588776814546-ec7d7fae36e1?fit=crop&w=1350&q=80");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load data
train_df = pd.read_csv("Training.csv")
doctors_df = pd.read_csv("doctors_dataset.csv", header=None, names=["Doctor", "Link"])

X = train_df.drop(columns=["prognosis"])
y = train_df["prognosis"]
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y_encoded)

# Disease info mock
disease_info = {
    "Heart attack": {
        "about": "A serious condition where the blood supply to the heart is suddenly blocked.",
        "precautions": ["Seek immediate medical help", "Take aspirin", "Avoid stress"],
        "clinics": ["City Heart Hospital", "Apollo Cardiac Center"]
    },
    "Diabetes": {
        "about": "A chronic condition that affects how your body processes blood sugar.",
        "precautions": ["Monitor blood sugar", "Healthy diet", "Regular exercise"],
        "clinics": ["Diabetes Wellness Clinic", "Sugar Control Center"]
    },
    # Add more diseases if needed
}

# Translator setup
translator = Translator()

# Language selection
lang = st.selectbox("Select language", ["English", "Hindi", "Telugu"])
translate = lambda text: translator.translate(text, dest="hi" if lang == "Hindi" else "te" if lang == "Telugu" else "en").text

# Chat-style interaction state
if "step" not in st.session_state:
    st.session_state.step = 1
    st.session_state.symptoms = []
    st.session_state.duration = None
    st.session_state.severity = {}

# Step 1: Symptoms input
if st.session_state.step == 1:
    user_symptoms = st.text_input(translate("Step 1: What symptoms are you experiencing?"))
    if st.button(translate("Next")) and user_symptoms:
        st.session_state.symptoms = re.findall(r'\w+', user_symptoms.lower())
        st.session_state.step = 2

# Step 2: Duration
if st.session_state.step == 2:
    st.session_state.duration = st.selectbox(translate("Step 2: How long have you had these symptoms?"),
                                             ["1-2 days", "3-7 days", "1+ week"])
    if st.button(translate("Next"), key="next2"):
        st.session_state.step = 3

# Step 3: Severity input
if st.session_state.step == 3:
    st.markdown(translate("Step 3: Rate the severity of each symptom:"))
    for sym in st.session_state.symptoms:
        st.session_state.severity[sym] = st.select_slider(sym, options=["None", "Mild", "Moderate", "Severe"])
    if st.button(translate("Diagnose")):
        st.session_state.step = 4

# Step 4: Prediction and Output
if st.session_state.step == 4:
    # Convert to binary vector
    symptoms_present = [s for s, sev in st.session_state.severity.items() if sev != "None"]
    input_vector = [1 if s in symptoms_present else 0 for s in X.columns]

    # Predict disease
    prediction = model.predict([input_vector])
    disease = label_encoder.inverse_transform(prediction)[0]

    st.success(translate(f"You may have: **{disease}**"))

    # Disease info
    if disease in disease_info:
        info = disease_info[disease]
        st.markdown(f"**{translate('About the disease')}:** {translate(info['about'])}")
        st.markdown(f"**{translate('Precautions')}:**")
        for tip in info["precautions"]:
            st.markdown(f"- {translate(tip)}")
        st.markdown(f"**{translate('Suggested Clinics')}:**")
        for clinic in info["clinics"]:
            st.markdown(f"- {translate(clinic)}")
    else:
        st.info(translate("No detailed info available for this disease."))

    # Recommend doctor
    specialization = disease  # simplification — normally mapped
    doctor_match = doctors_df[doctors_df["Doctor"].str.contains(specialization, case=False, na=False)]
    if not doctor_match.empty:
        doc = doctor_match.sample(1).values[0]
        st.info(translate(f"Recommended Doctor: [{doc[0]}]({doc[1]})"))
    else:
        st.warning(translate("No doctor found for this specialization."))

    if st.button(translate("Restart")):
        for key in ["step", "symptoms", "duration", "severity"]:
            st.session_state.pop(key, None)
        st.experimental_rerun()
