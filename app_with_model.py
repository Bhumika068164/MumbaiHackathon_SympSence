import streamlit as st
import joblib
import re

# Load the trained model
model = joblib.load("models/sympsense_model.joblib")

# Text cleaning function (same as in train_model.py)
def clean_text_simple(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# App UI
st.title("🩺 Sympsense – Smart Symptom Analyzer")

st.write("Enter your symptoms below, and the model will predict the possible disease.")

# Take input from user
symptoms_input = st.text_area("Describe your symptoms (e.g. fever, cough, fatigue):")

# Optional: let user add extra symptom tags
tags = st.text_input("Add additional keywords (optional, comma-separated):")
tags_list = [t.strip() for t in tags.split(",") if t.strip()]

# When user clicks the button
if st.button("Analyze Symptoms"):
    if not symptoms_input.strip():
        st.warning("Please enter your symptoms first.")
    else:
        # Combine inputs
        text = (symptoms_input + " " + " ".join(tags_list)).lower()
        text_clean = clean_text_simple(text)

        # Predict using model
        prediction = model.predict([text_clean])[0]

        st.success(f"🧠 Possible disease prediction: **{prediction}**")
