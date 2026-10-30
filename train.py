"""Baseline hate speech classifier: TF-IDF + Logistic Regression."""
import sys, joblib, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from preprocess import clean_text

path = sys.argv[1] if len(sys.argv) > 1 else "data_sample.csv"
df = pd.read_csv(path)
df["clean"] = df["text"].apply(clean_text)

X_tr, X_te, y_tr, y_te = train_test_split(
    df["clean"], df["label"], test_size=0.3, random_state=42, stratify=df["label"])

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
model.fit(X_tr, y_tr)
print(classification_report(y_te, model.predict(X_te), zero_division=0))
joblib.dump(model, "model.joblib")
print("Saved model.joblib")
