import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords only if missing
nltk.download('stopwords', quiet=True)

STOPWORDS = set(stopwords.words("english"))
ps = PorterStemmer()

def load_data():
    true_df = pd.read_csv("dataSets/True.csv")
    fake_df = pd.read_csv("dataSets/Fake.csv")
    
    true_df["label"] = 1  # 1 = real news
    fake_df["label"] = 0  # 0 = fake news

    df = pd.concat([true_df, fake_df], axis=0, ignore_index=True)
    return df

def clean_text(text):
    text = re.sub(r"http\S+", "", str(text))           # remove URLs
    text = re.sub(r"[^a-zA-Z]", " ", text)             # remove non-letters
    text = text.lower().split()
    text = [ps.stem(w) for w in text if w not in STOPWORDS]
    return " ".join(text)

def preprocess_data(df):
    # Combine title + text, handle missing data safely
    df["text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).astype(str)
    df["text"] = df["text"].apply(clean_text)
    
    # Drop empty or null rows
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).fillna("")
    
    return df[["text", "label"]]

if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print("Preprocessing text (this might take a while)...")
    df = preprocess_data(df)
    df["text"] = df["text"].fillna("")
    print("Done! Sample:")
    print(df.head())
    df.to_csv("dataSets/preprocessed.csv", index=False)
    print("Saved preprocessed data to dataSets/preprocessed.csv")
