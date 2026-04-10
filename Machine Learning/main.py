import sys
import os

# 👉 đảm bảo nhận src (fix mọi lỗi import)
sys.path.append(os.path.abspath("."))

from src.trainer import train
from src.evaluator import evaluate
from src.predictor import predict


def menu():
    print("\n===== AI MENU =====")
    print("1. Train model")
    print("2. Evaluate model")
    print("3. Predict")
    print("0. Exit")


def main():
    while True:
        menu()
        choice = input("👉 Chọn: ")

        if choice == "1":
            train()

        elif choice == "2":
            evaluate()

        elif choice == "3":
            data = list(map(float, input("Nhập data (cách nhau bằng space): ").split()))
            result = predict(data)
            print("🎯 Prediction:", result)

        elif choice == "0":
            print("👋 Bye!")
            break

        else:
            print("❌ Lựa chọn không hợp lệ")


if __name__ == "__main__":
    main()