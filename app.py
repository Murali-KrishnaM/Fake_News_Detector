import os
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# --- Load model and vectorizer ---
root_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(root_dir, "model", "fake_news_model.pkl")
vectorizer_path = os.path.join(root_dir, "model", "tfidf_vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    news_text = ""

    if request.method == "POST":
        news_text = request.form.get("news", "").strip()

        if news_text:
            # Transform and predict
            X = vectorizer.transform([news_text])
            pred = model.predict(X)[0]

            print("Model output:", pred)  # debug log

            # Handle both numeric and string predictions
            if isinstance(pred, (int, float)):
                result = "🟢 Real News" if pred == 1 else "🔴 Fake News"
            else:
                pred_str = str(pred).lower()
                result = "🟢 Real News" if pred_str in ["true", "real", "1"] else "🔴 Fake News"

    return render_template("index.html", result=result, news=news_text)

if __name__ == "__main__":
    app.run(debug=True)
