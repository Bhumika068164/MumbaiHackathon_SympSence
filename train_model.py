# train_model.py (updated for your CSV: Code, Name, Symptoms, Treatments)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import re

# Path to your dataset file (update the filename if yours is different)
DATA_PATH = "data/augmented_Symptoms.csv"  # <- put your CSV filename here

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_data(path):
    # load CSV
    df = pd.read_csv(path)

    # Normalize column headers: strip whitespace and lowercase
    df.columns = df.columns.str.strip().str.lower()

    # Now standardize names we expect:
    # 'name' -> label, 'symptoms' -> symptoms, 'treatments' -> treatments (optional)
    # If the dataset used 'name' and 'symptoms' already this will work.
    if 'name' in df.columns:
        df.rename(columns={'name': 'label'}, inplace=True)
    if 'symptoms' in df.columns:
        df.rename(columns={'symptoms': 'symptoms'}, inplace=True)  # no-op but explicit
    if 'treatments' in df.columns:
        df.rename(columns={'treatments': 'treatments'}, inplace=True)

    # Check required columns
    if 'symptoms' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain columns named 'Symptoms' and 'Name' (or 'symptoms' and 'label').")

    # Create cleaned text column
    df['symptoms_clean'] = df['symptoms'].astype(str).apply(clean_text)

    # Optionally drop rows with empty label or symptoms
    df = df[df['label'].notna() & df['symptoms_clean'].str.strip().astype(bool)].reset_index(drop=True)

    return df

def train_and_save(path):
    df = load_data(path)
    X = df['symptoms_clean']
    y = df['label']

    # If you have many unique labels but a small dataset, stratify may fail.
    # If you see an error about stratify, remove the 'stratify=y' argument below.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        # fallback without stratify if there are too few examples per class
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=5000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    print("Training model on {} examples...".format(len(X_train)))
    pipeline.fit(X_train, y_train)

    print("Evaluating...")
    preds = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    out_dir = "models"
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "sympsense_model.joblib")
    joblib.dump(pipeline, model_path)
    print("Saved model to:", model_path)

if __name__ == "__main__":
    train_and_save(DATA_PATH)
