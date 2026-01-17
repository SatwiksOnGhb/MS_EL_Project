from pathlib import Path
import pandas as pd
import joblib


def predict_from_csv(model_path: Path, input_csv: Path, output_csv: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    model = joblib.load(model_path)
    df = pd.read_csv(input_csv)

    predictions = model.predict(df)

    out = df.copy()
    out["Prediction"] = predictions

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    print(f"✅ Predictions saved to {output_csv}")
