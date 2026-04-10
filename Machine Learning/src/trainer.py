import joblib
from sklearn.metrics import accuracy_score

# 👉 import tuyệt đối để chạy file riêng không lỗi
from src.data_loader import load_data
from src.preprocessing import preprocess, split
from src.model import get_models
from src.config import MODEL_PATH


def train():
    print("🚀 Start training...")

    df = load_data()

    if "target" not in df.columns:
        raise ValueError("❌ Dataset phải có cột 'target'")

    X = df.drop(columns=["target"])
    y = df["target"]

    X_scaled, scaler = preprocess(X)
    X_train, X_test, y_train, y_test = split(X_scaled, y)

    best_model = None
    best_score = 0

    for name, model in get_models().items():
        print(f"\n🔹 Training: {name}")

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        print(f"✅ Accuracy: {acc:.4f}")

        if acc > best_score:
            best_score = acc
            best_model = model

    joblib.dump((best_model, scaler), MODEL_PATH)

    print("\n🔥 DONE TRAINING")
    print(f"🏆 Best Accuracy: {best_score:.4f}")


# 👉 chạy riêng file vẫn OK
if __name__ == "__main__":
    train()