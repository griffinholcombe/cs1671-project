
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import warnings
warnings.filterwarnings("ignore")


def load_data():
    train = pd.read_csv("train.csv")
    val = pd.read_csv("val.csv")
    return train, val


def model_tfidf(train, val):
    X_train_text = train["text"].astype(str)
    X_val_text = val["text"].astype(str)
    y_train = train["label"].values
    y_val = val["label"].values
    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 1),
                                 max_features=5000, sublinear_tf=True,
                                 min_df=2, max_df=0.95)

    t = time.time()
    print("Fitting vectorizer...", flush=True)
    X_train_feat = vectorizer.fit_transform(X_train_text)
    print(f"  Done in {time.time() - t:.1f}s", flush=True)

    t = time.time()
    print("Transforming validation set...", flush=True)
    X_val_feat = vectorizer.transform(X_val_text)
    print(f"  Done in {time.time() - t:.1f}s", flush=True)

    print(f"Feature matrix shape: {X_train_feat.shape}", flush=True)

    t = time.time()
    print("Training logistic regression...", flush=True)
    clf = LogisticRegression(max_iter=500, solver="liblinear", C=1.0)
    clf.fit(X_train_feat, y_train)
    print(f"  Done in {time.time() - t:.1f}s", flush=True)

    t = time.time()
    print("Predicting on validation split...", flush=True)
    y_val_pred = clf.predict(X_val_feat)
    val_metrics = evaluate(y_val, y_val_pred, "Model 2: Simple TF-IDF LR (Validation)")
    print(f"  Done in {time.time() - t:.1f}s", flush=True)

    return clf, vectorizer, val_metrics


def evaluate(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1)
    rec = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{model_name}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"              Predicted Human  Predicted MGT")
    print(f"Actual Human       {cm[0][0]:>5d}          {cm[0][1]:>5d}")
    print(f"Actual MGT         {cm[1][0]:>5d}          {cm[1][1]:>5d}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "cm": cm}


def main():
    total_start = time.time()

    train, val = load_data()
    print(f"Train: {len(train)}")
    print(f"Val: {len(val)}")
    print(f"Train label distribution:\n{train['label'].value_counts().rename({0: 'Human', 1: 'MGT'})}")
    print(f"Val label distribution:\n{val['label'].value_counts().rename({0: 'Human', 1: 'MGT'})}")

    clf, vectorizer, val_m = model_tfidf(train, val)

    print(f"\n=== Summary ===")
    print(f"Validation — Accuracy: {val_m['accuracy']:.4f}, F1: {val_m['f1']:.4f}")
    print(f"Total time: {time.time() - total_start:.1f}s")


if __name__ == "__main__":
    main()
