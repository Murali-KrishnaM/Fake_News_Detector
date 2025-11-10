import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load preprocessed data
print("Loading preprocessed data...")
df = pd.read_csv("dataSets/preprocessed.csv")
df["text"] = df["text"].fillna("")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=420
)

# TF-IDF Vectorization
print("Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
print("Training PassiveAggressiveClassifier...")
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_tfidf, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_tfidf)
score = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {round(score * 100, 2)}%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# Resolve project root (one level up from src/)
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(root_dir, "model")

# Ensure directory exists
os.makedirs(model_dir, exist_ok=True)

# Save model and vectorizer in project root/model
joblib.dump(model, os.path.join(model_dir, "fake_news_model.pkl"))
joblib.dump(vectorizer, os.path.join(model_dir, "tfidf_vectorizer.pkl"))

print(f"✅ Model and vectorizer saved successfully in {model_dir}")