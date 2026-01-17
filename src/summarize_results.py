# src/summarize_results.py
from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


def summarize_metrics(metrics_dir: Path = Path("results/metrics")) -> pd.DataFrame:
    if not metrics_dir.exists():
        raise FileNotFoundError(f"Metrics directory not found: {metrics_dir}")

    rows = []
    for fp in sorted(metrics_dir.glob("*.json")):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            # skip any non-json or corrupted files
            continue

        # expected keys: MAE, RMSE, R2, model, target
        row = {
            "file": fp.name,
            "target": data.get("target"),
            "model": data.get("model"),
            "MAE": data.get("MAE"),
            "RMSE": data.get("RMSE"),
            "R2": data.get("R2"),
            "test_size": data.get("test_size"),
            "seed": data.get("seed"),
        }
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No metrics json files found in {metrics_dir}")

    df = pd.DataFrame(rows)

    # Clean ordering + rounding
    for col in ["MAE", "RMSE", "R2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["target", "model"], na_position="last").reset_index(drop=True)
    df["MAE"] = df["MAE"].round(4)
    df["RMSE"] = df["RMSE"].round(4)
    df["R2"] = df["R2"].round(4)

    return df


def save_summary(df: pd.DataFrame, out_dir: Path = Path("results/metrics")) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "summary_metrics.csv"
    md_path = out_dir / "summary_metrics.md"

    df.to_csv(csv_path, index=False)

    # Report-friendly markdown table (sorted)
    md_table = df[["target", "model", "R2", "MAE", "RMSE"]].to_markdown(index=False)
    md_path.write_text(md_table, encoding="utf-8")

    print(f"✅ Saved: {csv_path}")
    print(f"✅ Saved: {md_path}")
    print("\n=== Quick View (best per target by R2) ===")
    best = (
        df.sort_values(["target", "R2"], ascending=[True, False])
          .groupby("target", as_index=False)
          .head(1)[["target", "model", "R2", "MAE", "RMSE"]]
    )
    print(best.to_string(index=False))


if __name__ == "__main__":
    df = summarize_metrics()
    save_summary(df)
