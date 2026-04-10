import joblib
from sklearn.metrics import classification_report

from src.data_loader import load_data
from src.config import MODEL_PATH


def evaluate():
    print("📊 Evaluating model...")

    df = load_data()

    if "target" not in df.columns:
        raise ValueError("❌ Dataset phải có cột 'target'")

    X = df.drop(columns=["target"])
    y = df["target"]

    model, scaler = joblib.load(MODEL_PATH)

    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)

    print("\n📊 Classification Report:")
    print(classification_report(y, preds))


# 👉 test riêng
if __name__ == "__main__":
    evaluate()