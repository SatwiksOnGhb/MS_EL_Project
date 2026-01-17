# src/improve_xgboost.py
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor

TARGET_COLUMNS = [
    "Tensile Strength (MPa)",
    "Yield Strength (MPa)",
    "Elongation (%)"
]


def safe_name(s: str) -> str:
    # Your filenames currently look like: xgb_cv_Tensile_Strength_MPa.joblib
    # This keeps it consistent.
    return (
        s.replace("(", "")
         .replace(")", "")
         .replace("%", "")
         .replace(" ", "_")
         .replace("-", "_")
    )


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include="number").columns
    cat_cols = X.select_dtypes(exclude="number").columns

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop"
    )


def improve_xgboost(
    cleaned_csv: Path,
    target: str,
    seed: int = 42,
    n_iter: int = 20,
    cv: int = 5
) -> Path:
    df = pd.read_csv(cleaned_csv)
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]

    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found. Available columns: {list(df.columns)}")

    # --- Clean y (prevents the NaN label crash you saw) ---
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df[target] = df[target].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[target])
    df = df[df[target].abs() < 1e9]  # safety

    y = df[target]
    X = df.drop(columns=[c for c in TARGET_COLUMNS if c in df.columns])

    preprocessor = build_preprocessor(X)

    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=-1
    )

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    param_dist = {
        "model__n_estimators": [300, 500, 800],
        "model__max_depth": [2, 3, 4, 5],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.8, 0.9, 1.0],
        "model__min_child_weight": [1, 3, 5, 7],
        "model__reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="r2",
        verbose=1,
        n_jobs=-1,
        error_score="raise"
    )

    print(f"\n=== Improving XGBoost for: {target} ===")
    print(f"Rows used after cleaning: {len(df)}")
    search.fit(X, y)

    best_model = search.best_estimator_

    Path("models").mkdir(parents=True, exist_ok=True)
    out = Path(f"models/xgb_cv_{safe_name(target)}.joblib")
    joblib.dump(best_model, out)

    print("✅ Best CV R²:", round(search.best_score_, 4))
    print("✅ Best parameters:", search.best_params_)
    print("✅ Saved model:", out)

    return out


def main():
    parser = argparse.ArgumentParser(description="Improve XGBoost via CV + Randomized Search")
    parser.add_argument("--data", default="data/processed/cleaned_dataset.csv", help="Cleaned dataset path")
    parser.add_argument("--target", help='One target column (e.g. "Tensile Strength (MPa)")')
    parser.add_argument("--all", action="store_true", help="Run for all three targets")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n_iter", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    data_path = Path(args.data)

    if args.all:
        for t in TARGET_COLUMNS:
            improve_xgboost(data_path, t, seed=args.seed, n_iter=args.n_iter, cv=args.cv)
    else:
        if not args.target:
            raise SystemExit('Provide --target "..." or use --all')
        improve_xgboost(data_path, args.target, seed=args.seed, n_iter=args.n_iter, cv=args.cv)


if __name__ == "__main__":
    main()
