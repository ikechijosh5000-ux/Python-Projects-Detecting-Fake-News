import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import cross_val_score
import joblib

# Load the dataset

news_data = pd.read_csv("data/news.csv")
print(news_data.head(10))
print(news_data.isnull().sum())
news_data = news_data.dropna()
news_data.info()
print(news_data.shape)
news_data["label"].value_counts()

# We will clean the text data by converting it to lowercase, removing URLs, special characters, and extra whitespace.


def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    text = re.sub(r"\s+", " ", text)

    return text


news_data["text"] = news_data["text"].apply(clean_text)

# Now we will separate the labels from the text data and split the dataset into training and testing sets.

labels = news_data.label
labels.head(10)
# First, we split the dataset into train & test samples:
x_train, x_test, y_train, y_test = train_test_split(
    news_data["text"], labels, test_size=0.2, random_state=42, stratify=labels
)

# Then we’ll initialize TfidfVectorizer with English stop words
vectorizer = TfidfVectorizer(
    stop_words="english", max_df=0.7, min_df=2, ngram_range=(1, 2), max_features=5000
)
tfidf_train = vectorizer.fit_transform(x_train)

tfidf_test = vectorizer.transform(x_test)
# Create a LogisticRegression model
logistic = model = LogisticRegression(max_iter=1000)
logistic.fit(tfidf_train, y_train)

scores = cross_val_score(model, tfidf_train, y_train, cv=5)

print(scores)
print(scores.mean())

y_pred = logistic.predict(tfidf_test)

# Create a confusion matrix
matrix = confusion_matrix(y_test, y_pred, labels=["FAKE", "REAL"])
matrix
# Visualize the confusion matrix
sns.heatmap(matrix, annot=True)
plt.show()
# Calculate the model's accuracy
Accuracy = accuracy_score(y_test, y_pred)
Accuracy * 100
Report = classification_report(y_test, y_pred)
print(Report)
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
model = joblib.load("fake_news_model.pkl")


def predict_news(text):

    cleaned = clean_text(text)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector)

    confidence = probability.max() * 100

    return prediction, confidence


print(news_data["text"].iloc[0])
print(tfidf_train.shape)
print(model)
print(accuracy_score(y_test, y_pred))
print(scores)
print(scores.mean())

while True:
    article = input("\nEnter a news article: ")

    prediction, confidence = predict_news(article)

    print(f"\nPrediction: {prediction}")
    print(f"Confidence: {confidence:.2f}%")
