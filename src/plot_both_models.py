# src/plot_both_models.py
# Clean + correct for YOUR MS_Project:
# - Uses your real CSV: data/raw/al_data.csv
# - Uses your real target column names: "Tensile Strength (MPa)", "Yield Strength (MPa)", "Elongation (%)"
# - Uses your EXACT model filenames (FIXED to match actual files)
# - Robust paths (works when script is inside src/)
# - Saves parity plots in results/plots and metrics in results/metrics

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# -----------------------
# PATHS (robust)
# -----------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # MS_Project/
DATA_CSV = os.path.join(BASE_DIR, "data", "raw", "al_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


# -----------------------
# TARGETS (exact CSV names)
# -----------------------
TARGET_COLS = {
    "Tensile_Strength_MPa": "Tensile Strength (MPa)",
    "Yield_Strength_MPa":   "Yield Strength (MPa)",
    "Elongation":           "Elongation (%)",
}

# -----------------------
# MODELS (FIXED to match actual filenames)
# -----------------------
MODEL_FILES = {
    "Tensile_Strength_MPa": {
        "RF":  "rf_Tensile_Strength__MPa_.joblib",
        "XGB": "xgb_cv_Tensile_Strength_MPa.joblib",  # FIXED: single underscore, no trailing _
    },
    "Yield_Strength_MPa": {
        "RF":  "rf_Yield_Strength__MPa_.joblib",
        "XGB": "xgb_cv_Yield_Strength_MPa.joblib",  # FIXED: single underscore, no trailing _
    },
    "Elongation": {
        "RF":  "rf_Elongation____.joblib",
        "XGB": "xgb_cv_Elongation_.joblib",  # This one was already correct
    },
}

TEST_SIZE = 0.2
RANDOM_STATE = 42


# -----------------------
# Helpers
# -----------------------
def safe_filename(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in s)

def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing model file: {path}")
    return joblib.load(path)

def align_features_for_model(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    If model stores feature_names_in_, select/reorder X to match.
    If not present, return X unchanged.
    Adds missing columns with zeros if needed.
    """
    feat_names = getattr(model, "feature_names_in_", None)
    if feat_names is None:
        return X

    missing = [c for c in feat_names if c not in X.columns]
    if missing:
        print(f"   ⚠️  Warning: Model expects {len(missing)} missing column(s): {missing}")
        print(f"   ➜  Adding missing columns with zeros...")
        # Add missing columns with zeros
        for col in missing:
            X[col] = 0

    return X.loc[:, list(feat_names)]

def parity_plot(y_true, preds_dict, title, save_path):
    plt.figure(figsize=(8, 6))

    y_all = [y_true] + list(preds_dict.values())
    mn = float(min(np.min(a) for a in y_all))
    mx = float(max(np.max(a) for a in y_all))

    plt.plot([mn, mx], [mn, mx], 'k--', label='Perfect Prediction', linewidth=2)
    
    colors = ['#1f77b4', '#ff7f0e']  # Blue for RF, Orange for XGB
    for (label, y_pred), color in zip(preds_dict.items(), colors):
        plt.scatter(y_true, y_pred, label=label, alpha=0.6, s=30, color=color, edgecolors='white', linewidth=0.5)

    plt.xlabel("Actual", fontsize=12)
    plt.ylabel("Predicted", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved plot: {os.path.relpath(save_path, BASE_DIR)}")


def create_metrics_summary_plot(metrics_df, save_path):
    """Create a comprehensive bar chart comparing RF vs XGB across all targets and metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    metrics = ['R2', 'MAE', 'RMSE']
    titles = ['R² Score (Higher is Better)', 'Mean Absolute Error (Lower is Better)', 
              'Root Mean Squared Error (Lower is Better)']
    
    # Simplified target names for display
    target_names = {
        'Tensile Strength (MPa)': 'Tensile',
        'Yield Strength (MPa)': 'Yield',
        'Elongation (%)': 'Elongation'
    }
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        
        # Prepare data
        data = metrics_df.pivot(index='Target', columns='Model', values=metric)
        data.index = [target_names.get(t, t) for t in data.index]
        
        # Create bar chart
        x = np.arange(len(data.index))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, data['RF'], width, label='Random Forest', 
                      color='#1f77b4', alpha=0.8, edgecolor='white', linewidth=1.5)
        bars2 = ax.bar(x + width/2, data['XGB'], width, label='XGBoost', 
                      color='#ff7f0e', alpha=0.8, edgecolor='white', linewidth=1.5)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Target Property', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric, fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(data.index, fontsize=10)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set y-axis to start from 0 for MAE and RMSE
        if metric in ['MAE', 'RMSE']:
            ax.set_ylim(bottom=0)
    
    plt.suptitle('Model Performance Comparison: Random Forest vs XGBoost', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Saved summary plot: {os.path.relpath(save_path, BASE_DIR)}")


# -----------------------
# Main
# -----------------------
def main():
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"Dataset not found: {DATA_CSV}")

    df = pd.read_csv(DATA_CSV)

    # Drop junk columns if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Ensure targets exist
    for col in TARGET_COLS.values():
        if col not in df.columns:
            raise ValueError(f"Target column not found in CSV: '{col}'")
    
    # Check for missing values
    print(f"Dataset shape: {df.shape}")
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        print(f"\n⚠️  Warning: Found {missing_counts.sum()} missing values:")
        print(missing_counts[missing_counts > 0])
        print("\n➜  Dropping rows with missing values...")
        df = df.dropna()
        print(f"✓ New dataset shape: {df.shape}")
    
    # IMPORTANT: Reset index after dropping NaN rows
    df = df.reset_index(drop=True)

    # Build X by removing only targets (keep 'class' as a feature if present)
    drop_cols = list(TARGET_COLS.values())
    # Note: 'class' is kept as a feature since models were trained with it

    X = df.drop(columns=drop_cols)

    # One consistent split index for all targets (now works with reset index)
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_test = X.iloc[test_idx].copy()

    metrics_rows = []

    for key, target_col in TARGET_COLS.items():
        print(f"\n{'='*60}")
        print(f"Processing: {target_col}")
        print('='*60)

        rf_path = os.path.join(MODELS_DIR, MODEL_FILES[key]["RF"])
        xgb_path = os.path.join(MODELS_DIR, MODEL_FILES[key]["XGB"])

        print(f"   Loading RF model:  {os.path.basename(rf_path)}")
        rf_model = load_model(rf_path)
        
        print(f"   Loading XGB model: {os.path.basename(xgb_path)}")
        xgb_model = load_model(xgb_path)

        # Align feature columns if models store them
        X_test_rf = align_features_for_model(rf_model, X_test)
        X_test_xgb = align_features_for_model(xgb_model, X_test)

        y_test = df.loc[test_idx, target_col].values

        y_pred_rf = rf_model.predict(X_test_rf)
        y_pred_xgb = xgb_model.predict(X_test_xgb)

        # Metrics
        print(f"\n   Performance Metrics:")
        for name, y_pred in [("RF", y_pred_rf), ("XGB", y_pred_xgb)]:
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            print(f"   {name:4s} - R²: {r2:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")
            
            metrics_rows.append({
                "Target": target_col,
                "Model": name,
                "R2": float(r2),
                "MAE": float(mae),
                "RMSE": float(rmse),
            })

        # Plot
        plot_path = os.path.join(PLOTS_DIR, safe_filename(f"{key}_RF_vs_XGB.png"))
        parity_plot(
            y_true=y_test,
            preds_dict={"Random Forest": y_pred_rf, "XGBoost": y_pred_xgb},
            title=f"Parity Plot: {target_col}",
            save_path=plot_path
        )

    # Save metrics
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = os.path.join(METRICS_DIR, "both_models_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    # Create summary bar chart
    summary_plot_path = os.path.join(PLOTS_DIR, "metrics_summary_RF_vs_XGB.png")
    create_metrics_summary_plot(metrics_df, summary_plot_path)

    print(f"\n{'='*60}")
    print("✅ All done!")
    print(f"{'='*60}")
    print(f"📊 Metrics saved to: {os.path.relpath(metrics_csv, BASE_DIR)}")
    print(f"📁 Plots saved in:   {os.path.relpath(PLOTS_DIR, BASE_DIR)}")
    print(f"\nSummary:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()