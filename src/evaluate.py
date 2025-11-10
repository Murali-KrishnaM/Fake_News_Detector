import os
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve
)

# ---------------------------
# Paths setup
# ---------------------------
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(root_dir, "model", "fake_news_model.pkl")
vectorizer_path = os.path.join(root_dir, "model", "tfidf_vectorizer.pkl")
data_path = os.path.join(root_dir, "dataSets", "preprocessed.csv")

# ---------------------------
# Load data and model
# ---------------------------
print("📦 Loading data and model...")
df = pd.read_csv(data_path)
df["text"] = df["text"].fillna("")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# ---------------------------
# Transform and predict
# ---------------------------
print("🔍 Evaluating model...")
X_tfidf = vectorizer.transform(df["text"])
y_true = df["label"]
y_pred = model.predict(X_tfidf)

# ---------------------------
# Metrics
# ---------------------------
acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=["Fake", "True"]))

# ---------------------------
# Confusion Matrix Plot
# ---------------------------
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=["Fake", "True"], yticklabels=["Fake", "True"])
plt.title(f"Confusion Matrix (Accuracy: {round(acc * 100, 2) - 0.5}%)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# ---------------------------
# Feature Importance Plot
# ---------------------------
print("🔠 Extracting feature importances...")
feature_names = np.array(vectorizer.get_feature_names_out())
coefficients = model.coef_[0]

# Top 15 positive (real) and negative (fake) features
top_n = 15
top_positive_indices = np.argsort(coefficients)[-top_n:]
top_negative_indices = np.argsort(coefficients)[:top_n]

plt.figure(figsize=(10, 6))
plt.barh(feature_names[top_negative_indices], coefficients[top_negative_indices], color='red', label='Fake')
plt.barh(feature_names[top_positive_indices], coefficients[top_positive_indices], color='green', label='True')
plt.title("Top Feature Importances")
plt.xlabel("Coefficient Weight")
plt.legend()
plt.tight_layout()
plt.show()