import pandas as pd
import numpy as np
import streamlit as st
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# UI Branding
st.set_page_config(page_title='🩺 AI Healthcare Chatbot', layout='centered')
st.markdown("""
    <h1 style='text-align: center; color: #0d6efd;'>🩺 AI Healthcare Chatbot</h1>
    <p style='text-align: center; color: gray;'>Describe your symptoms & get insights based on health patterns.</p>
""", unsafe_allow_html=True)

# Background Styling
st.markdown("""
    <style>
        .stApp {
            background-image: url("https://www.transparenttextures.com/patterns/paper-fibers.png");
            background-attachment: fixed;
            background-size: cover;
        }
    </style>
""", unsafe_allow_html=True)

# Load datasets
train_df = pd.read_csv('Training.csv')
doctors_df = pd.read_csv('doctors_dataset.csv', header=None, names=['Doctor', 'Link'])
education_df = pd.read_csv('disease_education.csv')

# Disease Mapping
disease_specialization_map = {
    'Fungal infection': 'Dermatologist',
    'Allergy': 'Allergist',
    'GERD': 'Gastroenterologist',
    'Bronchial Asthma': 'Pulmonologist',
    # Add more mappings as needed
}

# Model setup
X = train_df.drop(columns=['prognosis'])
y = train_df['prognosis']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y_encoded)

# Helper functions
def predict_disease(symptoms):
    input_vector = [1 if symptom in symptoms else 0 for symptom in X.columns]
    prediction = model.predict([input_vector])
    return label_encoder.inverse_transform(prediction)[0]

def fetch_doctor(disease):
    specialization = disease_specialization_map.get(disease, 'General Physician')
    matched = doctors_df[doctors_df['Doctor'].str.contains(specialization, case=False, na=False)]
    return matched.sample(1).values[0] if not matched.empty else doctors_df.sample(1).values[0]

def extract_symptoms(text):
    words = re.findall(r'\w+', text.lower())
    return [symptom for symptom in X.columns if symptom.replace('_', ' ') in ' '.join(words)]

def get_education_info(disease):
    match = education_df[education_df["Disease"].str.lower() == disease.lower()]
    if not match.empty:
        precautions = match["Precautions"].values[0].split(";")
        did_you_know = match["DidYouKnow"].values[0]
        return precautions, did_you_know
    return [], ""

def detect_emotional_health(text):
    stress_indicators = ["tired", "overwhelmed", "anxious", "depressed", "fatigued"]
    return any(word in text.lower() for word in stress_indicators)

# 1️⃣ User History
if "user_history" not in st.session_state:
    st.session_state.user_history = []

# Chat Input
user_input = st.chat_input("Describe your symptoms:")

# 2️⃣ Additional Inputs
severity = st.selectbox("Symptom Severity:", ["Mild", "Moderate", "Severe"])
duration = st.slider("Duration (Days):", min_value=1, max_value=30, value=5)

# Main logic
if user_input:
    extracted_symptoms = extract_symptoms(user_input)
    st.session_state.user_history.append(user_input)

    if extracted_symptoms:
        diagnosis = predict_disease(extracted_symptoms)
        doc_name, doc_link = fetch_doctor(diagnosis)
        st.success(f"🩺 Possible condition: **{diagnosis}**")
        st.info(f"👨‍⚕️ Recommended Doctor: [{doc_name}]({doc_link})")

        # 3️⃣ Show Education Info
        precautions, fact = get_education_info(diagnosis)
        if precautions:
            st.markdown("### 🛡️ Precautionary Steps:")
            for p in precautions:
                st.markdown(f"- {p.strip()}")
        if fact:
            st.markdown("### 📘 Did You Know?")
            st.info(fact)

        # 4️⃣ Emotional Health
        if detect_emotional_health(user_input):
            st.warning("💙 You seem stressed. Take breaks, practice mindfulness, and consider professional advice.")

    else:
        st.warning("Couldn't detect known symptoms. Try being more specific.")

# 5️⃣ Show Health History
if st.button("View My Health Insights"):
    if st.session_state.user_history:
        st.markdown("### Your Recent Symptom Reports:")
        for entry in st.session_state.user_history[-5:]:
            st.markdown(f"- {entry}")
    else:
        st.info("No previous symptom reports found.")
