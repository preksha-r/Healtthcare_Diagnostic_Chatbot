import pandas as pd
import numpy as np
import streamlit as st
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# UI Branding
st.markdown("""
    <h1 style='text-align: center; color: #0d6efd;'>🩺 AI Healthcare Chatbot</h1>
    <p style='text-align: center; color: gray;'>Describe your symptoms, severity & duration for a better diagnosis.</p>
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

# Disease Mapping
disease_specialization_map = {
    'Fungal infection': 'Dermatologist',
    'Allergy': 'Allergist',
    'GERD': 'Gastroenterologist',
    'Bronchial Asthma': 'Pulmonologist',
}

# Model setup
X = train_df.drop(columns=['prognosis'])
y = train_df['prognosis']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y_encoded)

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

# Chat-like Input
user_input = st.chat_input("Describe your symptoms:")

# Severity and Duration Inputs
severity = st.selectbox("Symptom Severity:", ["Mild", "Moderate", "Severe"])
duration = st.slider("Duration (Days):", min_value=1, max_value=30, value=5)

if user_input:
    extracted_symptoms = extract_symptoms(user_input)
    if extracted_symptoms:
        diagnosis = predict_disease(extracted_symptoms)
        doc_name, doc_link = fetch_doctor(diagnosis)
        st.success(f"🩺 Possible condition: **{diagnosis}**")
        st.info(f"👨‍⚕️ Recommended Doctor: [{doc_name}]({doc_link})")
        st.markdown(f"""
        **Precautionary Steps:**  
        - Maintain hygiene  
        - Avoid allergens  
        - Regular exercise  
        - Follow prescribed medication  
        """, unsafe_allow_html=True)
    else:
        st.warning("Couldn't detect known symptoms. Try being more specific.")

