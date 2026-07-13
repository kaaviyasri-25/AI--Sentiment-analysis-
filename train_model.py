import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load dataset
data = pd.read_csv("dataset/sentiment_dataset.csv")

# Features and labels
X = data["comment"]
y = data["sentiment"]

# Convert text to numbers
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words=None
)
X_vector = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vector, y)

# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model trained successfully!")