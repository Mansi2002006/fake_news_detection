import os
import pickle
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def heavy_clean_text(text):
    if not isinstance(text, str):
        return ""
    # 1. Strip Reuters location stamps at start (e.g., "WASHINGTON (Reuters) - ")
    text = re.sub(r'^.*?\(\s*Reuters\s*\)\s*-\s*', '', text, flags=re.IGNORECASE)
    # 2. Strip bracketed or parenthesised agency markers
    text = re.sub(r'\[.*?Reuters.*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(.*?Reuters.*?\)', '', text, flags=re.IGNORECASE)
    # 3. Clear URLs and excess spacing
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

def train_unbiased_model():
    print("=" * 60)
    print("TRAINING HIGH-ACCURACY ML PIPELINE (LOGISTIC REGRESSION)")
    print("=" * 60)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FAKE_PATH = os.path.join(BASE_DIR, "Fake.csv")
    TRUE_PATH = os.path.join(BASE_DIR, "True.csv")

    if not os.path.exists(FAKE_PATH) or not os.path.exists(TRUE_PATH):
        print("ERROR: Datasets 'Fake.csv' or 'True.csv' are missing.")
        return

    print("Ingesting dataset rows...")
    fake_df = pd.read_csv(FAKE_PATH)
    true_df = pd.read_csv(TRUE_PATH)

    # Use clean explicit string labels to prevent swapped class indices
    fake_df['label'] = "FAKE"
    true_df['label'] = "REAL"

    for frame in (fake_df, true_df):
        frame["title"] = frame.get("title", "").fillna("")
        frame["text"] = frame.get("text", "").fillna("")
        frame["combined_text"] = frame["title"] + " " + frame["text"]

    fake_processed = fake_df[['combined_text', 'label']].rename(columns={"combined_text": "text"}).copy()
    true_processed = true_df[['combined_text', 'label']].rename(columns={"combined_text": "text"}).copy()

    # Balance datasets cleanly to eliminate statistical bias
    sample_size = min(len(fake_processed), len(true_processed))
    balanced_df = pd.concat([
        fake_processed.sample(sample_size, random_state=42),
        true_processed.sample(sample_size, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    print("Cleaning data layers...")
    balanced_df['text'] = balanced_df['text'].apply(heavy_clean_text)
    balanced_df = balanced_df[balanced_df['text'].str.strip() != ""]

    print("Creating validation splits (75/25)...")
    X_train, X_test, y_train, y_test = train_test_split(
        balanced_df['text'], 
        balanced_df['label'], 
        test_size=0.25, 
        stratify=balanced_df['label'], 
        random_state=42
    )

    print("Extracting TF-IDF token matrices...")
    vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 3),
    max_features=20000,
    min_df=2,
    max_df=0.85,
    sublinear_tf=True
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Logistic Regression hyperplane parameters...")
    model = LogisticRegression(
    class_weight='balanced',
    max_iter=3000,
    C=2.0,
    random_state=42
    )

    model.fit(X_train_vec, y_train)

    print("-" * 60)
    y_pred = model.predict(X_test_vec)
    print(f"Verified Performance Accuracy on Test Set: {model.score(X_test_vec, y_test) * 100:.2f}%")
    print("Confusion Matrix [FAKE, REAL]:")
    print(confusion_matrix(y_test, y_pred, labels=["FAKE", "REAL"]))
    print(classification_report(y_test, y_pred, digits=3))
    print("-" * 60)

    # Output aligned binaries via pickle
    with open(os.path.join(BASE_DIR, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(BASE_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    print("Aligned model.pkl and vectorizer.pkl generated successfully!")
    print("=" * 60)

if __name__ == "__main__":
    train_unbiased_model()
