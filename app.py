#!/usr/bin/env python3
"""
app.py - Phishing Email Classifier

Provides:
- Data preparation: prepare-data
- Training: train
- Evaluation: evaluate
- Single prediction (CLI): predict
- Run web UI: runserver (Flask)
"""

import os
import re
import argparse
import csv
from typing import List, Tuple, Dict, Optional

from flask import Flask, request, render_template, flash

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

MODEL_PATH = os.path.join("models", "phishing_model.joblib")
CLEAN_CSV = os.path.join("data", "clean_phishing_emails.csv")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "change-me-for-production")

# Load model at startup if available
GLOBAL_MODEL: Optional[Pipeline] = None
if os.path.exists(MODEL_PATH):
    try:
        GLOBAL_MODEL = joblib.load(MODEL_PATH)
    except Exception as e:
        GLOBAL_MODEL = None


def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)


def prepare_data(raw_csv_path: str, output_csv_path: str = CLEAN_CSV) -> None:
    """
    Read the raw Kaggle CSV and produce a cleaned CSV with columns: label,text
    Normalizes labels to 'phishing' and 'ham'.
    """
    df = pd.read_csv(raw_csv_path, encoding="utf-8", low_memory=False)
    # Attempt to find text and label columns
    text_col = None
    label_col = None
    for c in df.columns:
        lc = c.lower()
        if "email text" in lc or lc == "text" or "content" in lc or "body" in lc:
            text_col = c
        if "email type" in lc or lc == "label" or "type" == lc or "class" in lc:
            label_col = c
    # Fallback: heuristics
    if text_col is None:
        # choose the longest string-like column
        candidates = [c for c in df.columns if df[c].dtype == object]
        if candidates:
            # pick one with most average length
            avg_lengths = {c: df[c].astype(str).map(len).mean() for c in candidates}
            text_col = max(avg_lengths, key=avg_lengths.get)
    if label_col is None:
        # choose small-cardinality object column
        candidates = [c for c in df.columns if df[c].dtype == object and df[c].nunique() < 50]
        if candidates:
            label_col = min(candidates, key=lambda c: df[c].nunique())

    if text_col is None or label_col is None:
        raise ValueError("Could not find text or label columns automatically. Please check the CSV headers.")

    cleaned = pd.DataFrame()
    cleaned["text"] = df[text_col].astype(str).fillna("").map(lambda s: s.strip())
    raw_labels = df[label_col].astype(str).fillna("").map(lambda s: s.strip())

    def normalize_label(v: str) -> str:
        v_low = v.lower()
        if "phish" in v_low or "malicious" in v_low or "fraud" in v_low:
            return "phishing"
        if "safe" in v_low or "legit" in v_low or "ham" in v_low or "legitimate" in v_low:
            return "ham"
        # some datasets use 'spam' - treat as phishing-ish (adjustable)
        if "spam" in v_low:
            return "phishing"
        # if unknown, default to ham but log
        return "ham"

    cleaned["label"] = raw_labels.map(normalize_label)
    cleaned = cleaned[["label", "text"]]
    cleaned.to_csv(output_csv_path, index=False, encoding="utf-8")
    print(f"Saved cleaned dataset to {output_csv_path}")


def load_dataset(csv_path: str) -> Tuple[List[str], List[int]]:
    """
    Load clean_phishing_emails.csv and return (texts, labels).
    Map label 'phishing' -> 1, 'ham' -> 0.
    """
    df = pd.read_csv(csv_path, encoding="utf-8")
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Clean CSV must contain 'label' and 'text' columns.")
    texts = df["text"].astype(str).tolist()
    label_map = {"phishing": 1, "ham": 0}
    labels = [label_map.get(str(l).lower(), 0) for l in df["label"]]
    return texts, labels


def build_pipeline() -> Pipeline:
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        strip_accents="unicode",
        lowercase=True,
        min_df=2,
        max_df=0.95
    )
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
    pipe = Pipeline([
        ("tfidf", tfidf),
        ("clf", clf)
    ])
    return pipe


def train_model(csv_path: str = CLEAN_CSV, model_path: str = MODEL_PATH) -> None:
    texts, labels = load_dataset(csv_path)
    pipeline = build_pipeline()
    pipeline.fit(texts, labels)
    joblib.dump(pipeline, model_path)
    print(f"Model trained and saved to {model_path}")


def evaluate_model(csv_path: str = CLEAN_CSV, model_path: str = MODEL_PATH) -> Dict[str, object]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Run train first.")
    texts, labels = load_dataset(csv_path)
    model: Pipeline = joblib.load(model_path)
    preds = model.predict(texts)
    probs = None
    try:
        probs = model.predict_proba(texts)
    except Exception:
        probs = None
    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, target_names=["ham", "phishing"], output_dict=False)
    cm = confusion_matrix(labels, preds)
    print("Evaluation results")
    print("------------------")
    print(f"Accuracy: {acc:.4f}")
    print()
    print("Classification report:")
    print(report)
    print()
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    return {"accuracy": acc, "report": report, "confusion_matrix": cm, "probs_sample": (probs[0].tolist() if probs is not None else None)}


URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
URGENT_WORDS = {"urgent", "immediately", "asap", "suspend", "suspended", "verify", "action required", "limited time", "failure", "important"}
SENSITIVE_WORDS = {"password", "account", "bank", "payment", "invoice", "ssn", "social security", "credentials", "login"}


def analyze_security_cues(text: str) -> List[str]:
    """
    Analyze the email text for simple phishing-like cues.
    """
    cues: List[str] = []
    if not text or not text.strip():
        return cues
    # URLs
    urls = URL_RE.findall(text)
    if urls:
        if len(urls) >= 3:
            cues.append("Contains multiple links, which is common in phishing campaigns.")
        else:
            cues.append(f"Contains {len(urls)} link(s). Inspect destination URLs carefully.")
    # Urgent language
    lowered = text.lower()
    found_urgent = [w for w in URGENT_WORDS if w in lowered]
    if found_urgent:
        cues.append("Uses urgent language (e.g., " + ", ".join(found_urgent[:5]) + "), a common phishing tactic.")
    # Sensitive keywords
    found_sensitive = [w for w in SENSITIVE_WORDS if w in lowered]
    if found_sensitive:
        cues.append("Mentions sensitive/account-related terms (e.g., " + ", ".join(found_sensitive[:5]) + ").")
    # Exclamation usage
    exclaims = text.count("!")
    if exclaims >= 3:
        cues.append("Multiple exclamation marks used to create urgency.")
    elif exclaims > 0:
        cues.append("Uses exclamation marks; be cautious with overly emphatic language.")
    # Mismatched sender/suspicious patterns - best-effort: look for 'http' shown as text
    if "click here" in lowered or "login to your account" in lowered:
        cues.append("Contains actionable phrases like 'click here' or 'login to your account' prompting immediate action.")
    return cues


def predict_text(text: str, model_path: str = MODEL_PATH, threshold: float = 0.5) -> Dict[str, object]:
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Please train first: python app.py train")
    model: Pipeline = joblib.load(model_path)
    probs = None
    try:
        probs = model.predict_proba([text])[0]
    except Exception:
        probs = None
    pred = model.predict([text])[0]
    label_map_rev = {1: "phishing", 0: "ham"}
    phishing_prob = float(probs[1]) if probs is not None else (1.0 if pred == 1 else 0.0)
    ham_prob = float(probs[0]) if probs is not None else (1.0 - phishing_prob)
    chosen = "phishing" if phishing_prob >= threshold else "ham"
    cues = analyze_security_cues(text)
    return {
        "label": label_map_rev.get(pred, chosen),
        "phishing_prob": phishing_prob,
        "ham_prob": ham_prob,
        "cues": cues
    }


@app.route("/", methods=["GET", "POST"])
def index():
    global GLOBAL_MODEL
    model_missing = GLOBAL_MODEL is None
    result = None
    text_in = ""
    if request.method == "POST":
        text_in = request.form.get("email_text", "").strip()
        if not text_in:
            flash("Please paste the email text to classify.", "warning")
        else:
            try:
                if GLOBAL_MODEL is None and os.path.exists(MODEL_PATH):
                    GLOBAL_MODEL = joblib.load(MODEL_PATH)
                if GLOBAL_MODEL is None:
                    flash("Model not available. Run `python app.py train` to create the model.", "danger")
                else:
                    probs = GLOBAL_MODEL.predict_proba([text_in])[0]
                    pred = GLOBAL_MODEL.predict([text_in])[0]
                    label_map_rev = {1: "phishing", 0: "ham"}
                    phishing_prob = float(probs[1])
                    ham_prob = float(probs[0])
                    cues = analyze_security_cues(text_in)
                    result = {
                        "label": label_map_rev.get(pred, "phishing" if phishing_prob >= 0.5 else "ham"),
                        "phishing_prob": phishing_prob,
                        "ham_prob": ham_prob,
                        "cues": cues
                    }
            except Exception as e:
                flash(f"Error during prediction: {e}", "danger")
    return render_template("index.html", result=result, model_missing=model_missing, email_text=text_in)


def cli_prepare(args):
    prepare_data(args.input_csv, args.output_csv)


def cli_train(args):
    train_model(args.csv if args.csv else CLEAN_CSV, args.model if args.model else MODEL_PATH)


def cli_evaluate(args):
    r = evaluate_model(args.csv if args.csv else CLEAN_CSV, args.model if args.model else MODEL_PATH)
    return r


def cli_predict(args):
    r = predict_text(args.text, args.model if args.model else MODEL_PATH, threshold=args.threshold)
    print("Prediction:")
    print(f"Label: {r['label']}")
    print(f"Phishing probability: {r['phishing_prob']:.4f}")
    print(f"Ham probability: {r['ham_prob']:.4f}")
    if r["cues"]:
        print("Security cues:")
        for c in r["cues"]:
            print(" -", c)
    else:
        print("No obvious phishing-like patterns were detected by simple rules.")


def main():
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Phishing Email Classifier")
    sub = parser.add_subparsers(dest="command")

    p_prep = sub.add_parser("prepare-data", help="Prepare/clean raw Kaggle CSV")
    p_prep.add_argument("input_csv", help="Raw CSV path (e.g., data/phishing_emails_raw.csv)")
    p_prep.add_argument("--output-csv", default=CLEAN_CSV, help="Output cleaned CSV path")

    p_train = sub.add_parser("train", help="Train model")
    p_train.add_argument("--csv", default=CLEAN_CSV, help="Clean CSV path")
    p_train.add_argument("--model", default=MODEL_PATH, help="Output model path")

    p_eval = sub.add_parser("evaluate", help="Evaluate model on clean dataset")
    p_eval.add_argument("--csv", default=CLEAN_CSV, help="Clean CSV path")
    p_eval.add_argument("--model", default=MODEL_PATH, help="Model path")

    p_pred = sub.add_parser("predict", help="Predict a single text from CLI")
    p_pred.add_argument("--text", required=True, help="Email text to classify")
    p_pred.add_argument("--model", default=MODEL_PATH, help="Model path")
    p_pred.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for phishing label")

    p_run = sub.add_parser("runserver", help="Run Flask development server")
    p_run.add_argument("--host", default="127.0.0.1", help="Host")
    p_run.add_argument("--port", type=int, default=5000, help="Port")
    p_run.add_argument("--debug", action="store_true", help="Enable debug")

    args = parser.parse_args()
    if args.command == "prepare-data":
        cli_prepare(args)
    elif args.command == "train":
        cli_train(args)
    elif args.command == "evaluate":
        cli_evaluate(args)
    elif args.command == "predict":
        cli_predict(args)
    elif args.command == "runserver":
        # Start Flask server
        app.run(host=args.host, port=args.port, debug=args.debug)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
