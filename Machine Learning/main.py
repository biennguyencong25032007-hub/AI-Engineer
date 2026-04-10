import argparse
import os
import torch

from src.config import Config, DataConfig, ModelConfig, TrainingConfig
from src.data_loader import load_data
from src.model import BinaryClassifier
from src.trainer import Trainer
from src.evaluator import Evaluator
from src.predictor import Predictor
from src.logger import get_logger

logger = get_logger("main")

PREPROCESSOR_PATH = "checkpoints/preprocessor.pkl"


# ──────────────────────────────────────────────────────────────────────
def resolve_device(cfg: Config) -> torch.device:
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


# ──────────────────────────────────────────────────────────────────────
def build_config() -> Config:
    """
    Edit this function (or load from YAML/JSON) to customise your run.
    """
    return Config(
        data=DataConfig(
            data_path="data/dataset.csv",
            target_column="target",
            test_size=0.15,
            val_size=0.15,
            # numerical_columns=[...],   # leave empty for auto-detect
            # categorical_columns=[...],
        ),
        model=ModelConfig(
            hidden_dims=[256, 128, 64],
            dropout_rate=0.3,
            batch_norm=True,
        ),
        training=TrainingConfig(
            epochs=100,
            batch_size=64,
            learning_rate=1e-3,
            early_stopping_patience=10,
            lr_scheduler="cosine",
        ),
        device="auto",
    )


# ──────────────────────────────────────────────────────────────────────
def mode_train(cfg: Config) -> None:
    device = resolve_device(cfg)
    logger.info(f"Device: {device}")

    train_loader, val_loader, test_loader, preprocessor = load_data(cfg)

    # Save preprocessor for later inference
    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
    preprocessor.save(PREPROCESSOR_PATH)

    model   = BinaryClassifier(cfg.model)
    trainer = Trainer(model, cfg, device)
    trainer.train(train_loader, val_loader)

    evaluator = Evaluator(model, device)
    evaluator.find_best_threshold(val_loader)
    evaluator.evaluate(test_loader, split="test")


# ──────────────────────────────────────────────────────────────────────
def mode_evaluate(cfg: Config, ckpt_path: str) -> None:
    device = resolve_device(cfg)
    _, _, test_loader, _ = load_data(cfg)

    model = BinaryClassifier(cfg.model)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    evaluator = Evaluator(model, device)
    evaluator.evaluate(test_loader, split="test")


# ──────────────────────────────────────────────────────────────────────
def mode_predict(cfg: Config, ckpt_path: str, input_csv: str) -> None:
    import pandas as pd

    predictor = Predictor.from_checkpoint(cfg, ckpt_path, PREPROCESSOR_PATH)
    df = pd.read_csv(input_csv)
    df["prediction"] = predictor.predict(df)
    df["probability"] = predictor.predict_proba(df)

    out_path = input_csv.replace(".csv", "_predictions.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"Predictions saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",  choices=["train", "evaluate", "predict"], default="train")
    parser.add_argument("--ckpt",  default="checkpoints/best_model.pt")
    parser.add_argument("--input", default=None, help="CSV for predict mode")
    args = parser.parse_args()

    cfg = build_config()

    if args.mode == "train":
        mode_train(cfg)
    elif args.mode == "evaluate":
        mode_evaluate(cfg, args.ckpt)
    elif args.mode == "predict":
        if not args.input:
            raise ValueError("--input CSV path required for predict mode.")
        mode_predict(cfg, args.ckpt, args.input)


if __name__ == "__main__":
    main()