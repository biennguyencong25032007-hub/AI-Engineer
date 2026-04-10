import joblib
import numpy as np
from .config import MODEL_PATH

def predict(data):
    model, scaler = joblib.load(MODEL_PATH)
    data_scaled = scaler.transform([data])
    return model.predict(data_scaled)[0]