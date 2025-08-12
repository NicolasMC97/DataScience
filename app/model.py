import os
import joblib
import numpy as np
from typing import Any
import warnings

warnings.filterwarnings('ignore')

_model: Any | None = None

def load_model() -> Any:
    global _model
    if _model is not None:
        return _model
    model_path = os.getenv("MODEL_PATH", "/Users/nicolasmedina/Documents/Personal GIT/DataScience/models_trained/finalized_model_KNN.sav")
    _model = joblib.load(model_path)  # works for .pkl/.joblib/.sav
    return _model

def predict_label(lat: float, lon: float):
    model = load_model()
    X = np.array([[lat, lon]], dtype=float)

    # scikit-learn KNN expects 2D array; adapt if you engineered features
    y_pred = model.predict(X)[0]

    proba = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X)[0]
            # probability for the predicted class
            class_index = list(model.classes_).index(y_pred)
            proba = float(probs[class_index])
        except Exception:
            proba = None

    return str(y_pred), proba
