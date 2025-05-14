
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import re

# UI Branding and background
st.markdown("""
    <h1 style='text-align: center; color: #0d6efd;'>🩺 AI Healthcare Chatbot</h1>
    <p style='text-align: center; color: gray;'>Describe your symptoms. Get a probable diagnosis. Find the right doctor.</p>
""", unsafe_allow_html=True)

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://www.transparenttextures.com/patterns/paper-fibers.png");
             background-attachment: fixed;
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

# Load datasets
train_df = pd.read_csv('Training.csv')
test_df = pd.read_csv('Testing.csv')
doctors_df = pd.read_csv('doctors_dataset.csv', header=None, names=['Doctor', 'Link'])

# Disease to specialist mapping
disease_specialization_map = {
    'Fungal infection': 'Dermatologist',
    'Allergy': 'Allergist',
    'GERD': 'Gastroenterologist',
    'Chronic cholestasis': 'Gastroenterologist',
    'Drug Reaction': 'General Physician',
    'Peptic ulcer diseae': 'Gastroenterologist',
    'AIDS': 'Infectious Disease Specialist',
    'Diabetes ': 'Endocrinologist',
    'Gastroenteritis': 'Gastroenterologist',
    'Bronchial Asthma': 'Pulmonologist',
}

# Model setup
X = train_df.drop(columns=['prognosis'])
y = train_df['prognosis']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y_encoded)

def predict_disease(symptom_input):
    input_vector = [1 if symptom in symptom_input else 0 for symptom in X.columns]
    prediction = model.predict([input_vector])
    return label_encoder.inverse_transform(prediction)[0]

def fetch_doctor(disease):
    specialization = disease_specialization_map.get(disease, 'General Physician')
    matched = doctors_df[doctors_df['Doctor'].str.contains(specialization, case=False, na=False)]
    if not matched.empty:
        doc = matched.sample(1).values[0]
    else:
        doc = doctors_df.sample(1).values[0]
    return doc[0], doc[1]

def extract_symptoms(text):
    words = re.findall(r'\w+', text.lower())
    return [symptom for symptom in X.columns if symptom.replace('_', ' ') in ' '.join(words)]

# Chatbot logic
user_input = st.text_area("What's going on? Describe your symptoms:")

if st.button("Get Advice"):
    if user_input.strip():
        extracted = extract_symptoms(user_input)
        if extracted:
            diagnosis = predict_disease(extracted)
            doc_name, doc_link = fetch_doctor(diagnosis)
            st.success(f"🩺 It sounds like you may have **{diagnosis}**.")
            st.info(f"👨‍⚕️ You should consult: [{doc_name}]({doc_link})")
        else:
            st.warning("I couldn't detect any known symptoms. Try being more specific.")
    else:
        st.warning("Please type your symptoms above.")
