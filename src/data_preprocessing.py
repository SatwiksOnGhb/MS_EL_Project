from __future__ import annotations
from pathlib import Path
import pandas as pd


def preprocess_dataset(input_csv: Path, output_csv: Path, extra_drop_cols: list[str] | None = None) -> None:
    extra_drop_cols = extra_drop_cols or []

    if not input_csv.exists():
        raise FileNotFoundError(f"Raw dataset not found: {input_csv}")

    df = pd.read_csv(input_csv)

    # Clean column names (strip whitespace + BOM)
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]

    # Drop index-like first column if present (common in your file)
    # It often appears as an empty name or 'Unnamed: 0'
    first_col = df.columns[0]
    if first_col == "" or first_col.lower().startswith("unnamed"):
        df = df.drop(columns=[first_col])

    # Drop any user-specified junk columns
    for c in extra_drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Drop completely empty columns
    df = df.dropna(axis=1, how="all")

    # Drop duplicate rows
    df = df.drop_duplicates()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("=== Preprocess Summary ===")
    print("Saved:", output_csv)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
