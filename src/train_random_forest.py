from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

from src.evaluate_model import compute_metrics, save_metrics, plot_pred_vs_actual, plot_feature_importance


# All targets in your dataset (we will never use any of these as input features)
TARGET_COLUMNS = ["Tensile Strength (MPa)", "Yield Strength (MPa)", "Elongation (%)"]


def safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s)


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    cat_pipe = Pipeline(steps=[
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


def train_random_forest(cleaned_csv: Path, target: str, test_size: float = 0.2, seed: int = 42) -> None:
    df = pd.read_csv(cleaned_csv)
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]

    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found. Available columns: {list(df.columns)}")

    # Drop rows where selected target is missing
    df = df.dropna(subset=[target])

    y = df[target]

    # IMPORTANT FIX:
    # Drop ALL target columns from features (including the one we're predicting).
    drop_cols = [c for c in TARGET_COLUMNS if c in df.columns]
    X = df.drop(columns=drop_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    preprocessor = _build_preprocessor(X)

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=seed,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    metrics = compute_metrics(y_test, preds)
    metrics.update({"model": "RandomForest", "target": target, "test_size": test_size, "seed": seed})

    # Save model + results
    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, Path(f"models/rf_{safe_name(target)}.joblib"))

    save_metrics(metrics, Path(f"results/metrics/rf_metrics_{safe_name(target)}.json"))

    plot_pred_vs_actual(
        y_test, preds,
        Path(f"results/plots/pred_vs_actual_rf_{safe_name(target)}.png"),
        title=f"Pred vs Actual (RF) - {target}"
    )

    # Feature importance (index-based because one-hot expands features)
    plot_feature_importance(
        pipe.named_steps["model"].feature_importances_,
        Path(f"results/plots/feat_importance_rf_{safe_name(target)}.png"),
        title=f"Feature Importance (RF) - {target}",
        top_k=20
    )

    print("✅ RF metrics:", metrics)
