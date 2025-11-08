import os
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Setup
nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))
ps = PorterStemmer()

# Paths
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(root_dir, "model", "fake_news_model.pkl")
vectorizer_path = os.path.join(root_dir, "model", "tfidf_vectorizer.pkl")

# Load model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def clean_text(text):
    """Clean and preprocess input text just like training data."""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower().split()
    text = [ps.stem(w) for w in text if w not in STOPWORDS]
    return " ".join(text)

def predict_news(text):
    """Predict if a given news article is Fake or True."""
    cleaned = clean_text(text)
    tfidf = vectorizer.transform([cleaned])
    prediction = model.predict(tfidf)[0]
    label = "📰 True News" if prediction == 1 else "🚨 Fake News"
    print(f"\nPrediction: {label}")

if __name__ == "__main__":
    print("Enter a news article or headline below:")
    user_input = input(">>> ")
    predict_news(user_input)
