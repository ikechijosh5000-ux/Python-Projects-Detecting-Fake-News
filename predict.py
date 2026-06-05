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


while True:
    article = input("\nEnter a news article: ")

    prediction, confidence = predict_news(article)

    print("\n----------------------")
    print(f"Prediction : {prediction}")
    print(f"Confidence : {confidence:.2f}%")
    print("----------------------")
