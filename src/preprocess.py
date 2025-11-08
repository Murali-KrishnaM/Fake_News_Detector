import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

STOPWORDS = set(stopwords.words("english"))
ps = PorterStemmer()

def load_data():
    true_df = pd.read_csv("dataSets/True.csv")
    fake_df = pd.read_csv("dataSets/Fake.csv")
    true_df["label"] = 1
    fake_df["label"] = 0
    df = pd.concat([true_df, fake_df], axis=0).reset_index(drop=True)
    return df

def clean_texts(texts):
    cleaned = []
    for t in texts:
        t = re.sub(r"http\S+", "", t)
        t = re.sub(r"[^a-zA-Z]", " ", t)
        t = t.lower().split()
        t = [ps.stem(w) for w in t if w not in STOPWORDS]
        cleaned.append(" ".join(t))
    return cleaned

def preprocess_data(df):
    df["text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).astype(str)
    df["text"] = clean_texts(df["text"].tolist())
    return df[["text", "label"]]

if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print("Preprocessing (this might take a bit)...")
    df = preprocess_data(df)
    print("✅ Done! Here's a sample:")
    print(df.head())
    df.to_csv("dataSets/preprocessed.csv", index=False)

