import os
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

# Resolve project root and paths
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(root_dir, "model", "fake_news_model.pkl")
vectorizer_path = os.path.join(root_dir, "model", "tfidf_vectorizer.pkl")
test_data_path = os.path.join(root_dir, "dataSets", "holdout_test.csv")

print("📂 Loading model, vectorizer, and holdout test data...")
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)
df = pd.read_csv(test_data_path)
df["text"] = df["text"].fillna("")

# Transform test data
X_tfidf = vectorizer.transform(df["text"])
y_true = df["label"]

# Predict
y_pred = model.predict(X_tfidf)

# Evaluate
acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
print(f"\n✅ Model Accuracy on Holdout Test: {round(acc * 100, 2)}%")
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=["Fake", "True"]))

# Confusion Matrix Plot
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Fake", "True"], yticklabels=["Fake", "True"])
plt.title(f"Confusion Matrix (Accuracy: {round(acc * 100, 2)}%)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ROC Curve
print("📈 Generating ROC Curve...")
if hasattr(model, "decision_function"):
    y_scores = model.decision_function(X_tfidf)
else:
    y_scores = model.predict_proba(X_tfidf)[:, 1]

fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="orange", label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC Curve)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# Precision-Recall Curve
print("📊 Generating Precision-Recall Curve...")
precision, recall, _ = precision_recall_curve(y_true, y_scores)
avg_precision = average_precision_score(y_true, y_scores)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, color="green", label=f"AP = {avg_precision:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()

# Feature Importance
print("🧩 Extracting Feature Importance...")
if hasattr(model, "coef_"):
    feature_names = np.array(vectorizer.get_feature_names_out())
    coef = model.coef_[0]

    top_positive_indices = np.argsort(coef)[-20:]
    top_negative_indices = np.argsort(coef)[:20]

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names[top_positive_indices], coef[top_positive_indices], color="green")
    plt.title("Top 20 Words Associated with REAL News")
    plt.xlabel("Coefficient Value")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names[top_negative_indices], coef[top_negative_indices], color="red")
    plt.title("Top 20 Words Associated with FAKE News")
    plt.xlabel("Coefficient Value")
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Model doesn't support feature importance visualization.")

print("\n✅ Evaluation complete on holdout test set.")
