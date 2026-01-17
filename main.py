# main.py
import argparse
from pathlib import Path

from src.data_preprocessing import preprocess_dataset
from src.train_random_forest import train_random_forest
from src.train_xgboost import train_xgboost
from src.predict import predict_from_csv


def main():
    parser = argparse.ArgumentParser(
        description="MS_Project: Materials Strength Prediction (Preprocess -> Train -> Predict)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ----------------------------
    # PREPROCESS
    # ----------------------------
    pprep = sub.add_parser("preprocess", help="Clean raw CSV -> data/processed/cleaned_dataset.csv")
    pprep.add_argument("--input", default="data/raw/al_data.csv", help="Raw dataset path")
    pprep.add_argument("--output", default="data/processed/cleaned_dataset.csv", help="Cleaned dataset path")
    pprep.add_argument("--drop", nargs="*", default=[], help="Extra columns to drop (optional)")

    # ----------------------------
    # TRAIN
    # ----------------------------
    ptrain = sub.add_parser("train", help="Train RF and XGBoost for one target column")
    ptrain.add_argument("--data", default="data/processed/cleaned_dataset.csv", help="Cleaned dataset path")
    ptrain.add_argument("--target", required=True, help="Target column name (e.g., 'Tensile Strength (MPa)')")
    ptrain.add_argument("--test_size", type=float, default=0.2)
    ptrain.add_argument("--seed", type=int, default=42)

    # ----------------------------
    # PREDICT
    # ----------------------------
    ppred = sub.add_parser("predict", help="Predict using a saved .joblib model on new samples CSV")
    ppred.add_argument("--model", required=True, help="Path to trained .joblib model (in models/)")
    ppred.add_argument("--input", required=True, help="CSV with input features (NO target column)")
    ppred.add_argument("--output", default="results/predictions.csv", help="Where to save predictions CSV")

    args = parser.parse_args()

    if args.cmd == "preprocess":
        preprocess_dataset(
            input_csv=Path(args.input),
            output_csv=Path(args.output),
            extra_drop_cols=args.drop
        )
        print(f"✅ Preprocessing done -> {args.output}")

    elif args.cmd == "train":
        print("=== Random Forest ===")
        train_random_forest(
            cleaned_csv=Path(args.data),
            target=args.target,
            test_size=args.test_size,
            seed=args.seed
        )

        print("\n=== XGBoost ===")
        train_xgboost(
            cleaned_csv=Path(args.data),
            target=args.target,
            test_size=args.test_size,
            seed=args.seed
        )

        print("\n✅ Training complete. Check: models/, results/metrics/, results/plots/")

    elif args.cmd == "predict":
        predict_from_csv(
            model_path=Path(args.model),
            input_csv=Path(args.input),
            output_csv=Path(args.output)
        )
        print(f"✅ Prediction complete -> {args.output}")


if __name__ == "__main__":
    main()
