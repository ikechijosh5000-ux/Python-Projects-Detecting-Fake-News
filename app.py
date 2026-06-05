import streamlit as st
import joblib
import re

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    text = re.sub(r"\s+", " ", text)

    return text


def predict_news(text):

    cleaned = clean_text(text)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector)

    confidence = float(probability.max() * 100)

    return prediction, confidence


st.title("📰 Fake News Detector")

st.write("Paste a news article below and let AI analyze it.")

article = st.text_area("Enter News Article", height=250)

if st.button("Analyze News"):
    if article.strip():
        prediction, confidence = predict_news(article)

        if prediction == "REAL":
            st.success("✅ REAL NEWS")

        else:
            st.error("❌ FAKE NEWS")

        st.info(f"Confidence: {confidence:.2f}%")
        st.progress(int(confidence))
        st.write(f"Words: {len(article.split())}")


st.sidebar.title("About")

st.sidebar.write(
    "Machine Learning Fake News Detector using TF-IDF and Logistic Regression."
)

st.session_state
