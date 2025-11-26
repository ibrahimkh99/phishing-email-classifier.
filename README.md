# Phishing Email Classifier

A small machine learning + security project that detects phishing emails using text analysis, and provides a web UI to paste emails and get a phishing/ham prediction with simple security explanations.

## Features

- Uses the Kaggle "Phishing Email Detection" dataset (by subhajournal).
- TF-IDF + Logistic Regression text classifier.
- Web UI (Flask) to classify emails interactively.
- Shows phishing probability and simple security cues (links, urgency, sensitive terms).
- CLI functions for preparing data, training, evaluating, and predicting.

## Dataset

This project uses the Kaggle dataset "Phishing Email Detection" by subhajournal.

Steps:
1. Download the dataset CSV from Kaggle.
2. Place the CSV in the project `data/` directory, e.g. `data/phishing_emails_raw.csv`.

## Installation

```powershell
git clone <repo-url>
cd phishing-email-classifier
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

(On macOS/Linux use `source venv/bin/activate`)

## Preparing the data

Assuming you placed the raw CSV at `data/phishing_emails_raw.csv`:

```powershell
python app.py prepare-data data/phishing_emails_raw.csv --output-csv data/clean_phishing_emails.csv
```

This will create `data/clean_phishing_emails.csv` with columns: `label,text` and labels normalized to `phishing` and `ham`.

## Training the model

```powershell
python app.py train --csv data/clean_phishing_emails.csv --model models/phishing_model.joblib
```

This trains a TF-IDF + Logistic Regression pipeline and saves it to `models/phishing_model.joblib`.

## Evaluating the model

```powershell
python app.py evaluate --csv data/clean_phishing_emails.csv --model models/phishing_model.joblib
```

This prints accuracy, classification report, and confusion matrix.

## Running the web app

```powershell
python app.py runserver --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000` in your browser. Paste the email text and click "Classify".

## CLI Predict Example

```powershell
python app.py predict --text "Dear user, your account will be suspended. Click here: http://example.com/verify"
```

Outputs the label, phishing probability and heuristic security cues.

## Disclaimer / Ethics

This is an educational project. The model is trained on one dataset and may not generalize. Do not use it as the only decision factor for real-world security.

## Future Improvements

- Use richer NLP (embeddings, transformers) for better performance.
- Add feature-level explainability (feature importance, LIME/SHAP).
- Integrate with an email client for real-time scanning (prototype).
