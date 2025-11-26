"""
model_utils.py - small helpers for model persistence and inference
"""

import os
from typing import Optional
import joblib
from sklearn.pipeline import Pipeline

def save_model(model: Pipeline, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(model, path)

def load_model(path: str) -> Optional[Pipeline]:
    if not os.path.exists(path):
        return None
    return joblib.load(path)
