
import streamlit as st
import re
import pickle
import numpy as np

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# --- Text Cleaning ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r'\s+', ' ', text)
    return text

# --- Stopword removal ---
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

# --- Prediction Function ---
def predict_email(email_text):
    cleaned = clean_text(email_text)
    cleaned = remove_stopwords(cleaned)
    vectorized = vectorizer.transform([cleaned]).toarray()
    prediction = model.predict(vectorized)
    return "🛑 Phishing" if prediction[0] == 1 else "✅ Legitimate"

# --- Streamlit UI ---
st.set_page_config(page_title="Email Phishing Detector", layout="centered")
st.title("📧 Email Phishing Detector")
st.write("Paste an email message below to detect if it's phishing or legitimate.")

email_input = st.text_area("✉️ Email Content")

if st.button("Predict"):
    if email_input.strip() == "":
        st.warning("Please enter an email message.")
    else:
        result = predict_email(email_input)
        st.success(f"Prediction: {result}")
