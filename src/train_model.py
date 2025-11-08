import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

print("📂 Loading preprocessed data...")
df = pd.read_csv("dataSets/preprocessed.csv")
df["text"] = df["text"].fillna("")

# Split into 80/20 first — 20% will be the holdout (unseen test)
X_temp, X_test, y_temp, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# From the remaining 80%, split 50/30 (train/validation)
# 50/80 = 0.625 for the train ratio
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.375, random_state=42
)

print(f"✅ Data split complete:")
print(f"   Train: {len(X_train)} samples")
print(f"   Validation: {len(X_val)} samples")
print(f"   Holdout Test: {len(X_test)} samples")

# TF-IDF Vectorization
print("🔤 Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# Train model
print("🧠 Training PassiveAggressiveClassifier...")
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_tfidf, y_train)

# Resolve paths for saving
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(root_dir, "model")
os.makedirs(model_dir, exist_ok=True)

# Save model and vectorizer
joblib.dump(model, os.path.join(model_dir, "fake_news_model.pkl"))
joblib.dump(vectorizer, os.path.join(model_dir, "tfidf_vectorizer.pkl"))

# Save holdout test set (unseen data)
test_path = os.path.join(root_dir, "dataSets", "holdout_test.csv")
pd.DataFrame({"text": X_test, "label": y_test}).to_csv(test_path, index=False)

print(f"\n✅ Model and vectorizer saved to: {model_dir}")
print(f"📁 Holdout test set saved to: {test_path}")
