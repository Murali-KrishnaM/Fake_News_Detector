# 📰 Fake News Detection using Machine Learning & NLP

A Machine Learning project that detects **real vs fake news articles** using **Natural Language Processing (NLP)** and **Flask web deployment**. The model achieves **~99.37% accuracy** using the **PassiveAggressiveClassifier** and TF-IDF vectorization.

---

## 🚀 Project Overview

The project aims to combat misinformation by building a text classification system that can identify whether a given news article or headline is **real** or **fake**. It follows a complete ML pipeline — from **data preprocessing** to **model training**, **evaluation**, and **Flask deployment**.

---

## 🧠 Key Features

- 🧹 **Text Preprocessing** with NLTK (stopword removal, stemming, normalization)
- 🧾 **TF-IDF Vectorization** for feature extraction
- ⚡ **PassiveAggressiveClassifier** for efficient text classification
- 📊 **Evaluation** using Accuracy, Confusion Matrix, ROC Curve, and Precision-Recall Curve
- 🌐 **Flask Web App** for real-time predictions
- 💾 **Model Persistence** using `joblib`

---

## 📂 Project Structure

```
Fake-News-Detection/
│
├── dataSets/
│   ├── True.csv
│   ├── Fake.csv
│   └── preprocessed.csv
│
├── model/
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   └── evaluate.py
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Murali-KrishnaM/Fake_News_Detector
cd Fake-News-Detection
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Preprocess the dataset

```bash
python src/preprocess.py
```

### 4️⃣ Train the model

```bash
python src/train_model.py
```

### 5️⃣ Evaluate performance

```bash
python src/evaluate.py
```

### 6️⃣ Run the Flask app

```bash
python app.py
```

Then visit **http://127.0.0.1:5000/** in your browser.

---

## 🧩 Tools & Libraries Used

| Category | Libraries |
|----------|-----------|
| Data Handling | `pandas`, `numpy` |
| NLP | `nltk` |
| Machine Learning | `scikit-learn` |
| Visualization | `matplotlib`, `seaborn` |
| Deployment | `flask` |
| Model Storage | `joblib` |

---

## 📈 Model Performance

- **Accuracy:** ~99.37%
- **Algorithm:** PassiveAggressiveClassifier
- **Feature Extraction:** TF-IDF Vectorization
- **Evaluation Metrics:** Accuracy, F1-Score, ROC, Confusion Matrix

---

## 💻 Flask Web App Preview

- 🟢 **Real News** → Model predicts genuine articles
- 🔴 **Fake News** → Model flags potentially false content

The frontend is built using simple HTML/CSS with a clean and responsive design.

---

## 🧪 Future Improvements

- Integrate Deep Learning models (e.g., LSTM, BERT)
- Deploy to Render, Railway, or AWS for public access
- Expand dataset for better generalization
- Add confidence score visualization on predictions

---

## 👨‍💻 Author

**Murali Krishna**  
B.Tech Artificial Intelligence & Data Science  
Rajalakshmi Engineering College, Chennai

 ML Enthusiast | Flask Developer

🔗 [LinkedIn](https://linkedin.com/in/murali-krishna-m893) • 💻 [GitHub](https://github.com/Murali-KrishnaM)

---

⭐ **If you find this project useful, don't forget to star the repo!**
