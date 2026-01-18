# Aluminum Alloy Properties Prediction

A machine learning project for predicting mechanical properties of aluminum alloys based on their composition and processing methods.

## Features

- **Machine Learning Models**: Random Forest and XGBoost models for predicting:
  - Elongation (%)
  - Tensile Strength (MPa)
  - Yield Strength (MPa)

- **Web Application**: User-friendly interface for making predictions
  - Upload Excel or CSV files with aluminum alloy data
  - Batch prediction support (multiple samples)
  - Download results as CSV
  - Real-time predictions with trained models

## Project Structure

```
MS_Project - Copy/
├── app.py                      # Flask web application
├── main.py                     # CLI tool for training and prediction
├── data/
│   ├── raw/                    # Raw datasets
│   └── processed/              # Cleaned datasets
├── models/                     # Trained ML models (.joblib)
├── results/
│   ├── metrics/                # Model performance metrics
│   └── plots/                  # Visualization plots
├── src/
│   ├── data_preprocessing.py   # Data cleaning utilities
│   ├── train_random_forest.py # Random Forest training
│   ├── train_xgboost.py       # XGBoost training
│   ├── predict.py             # Prediction utilities
│   └── summarize_results.py   # Results summarization
├── templates/                  # HTML templates for web app
├── static/                     # CSS and frontend assets
└── requirements.txt            # Python dependencies
```

## Installation

1. Clone or download this repository

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Web Application (Recommended)

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Upload your Excel or CSV file containing aluminum alloy composition data

4. View and download predictions

**Note**: Input files should contain all composition columns (Processing, Ag, Al, B, Be, Bi, Cd, Co, Cr, Cu, Er, Eu, Fe, Ga, Li, Mg, Mn, Ni, Pb, Sc, Si, Sn, Ti, V, Zn, Zr, class) but **NOT** the target columns (Elongation, Tensile Strength, Yield Strength).

See [WEB_APP_INSTRUCTIONS.md](WEB_APP_INSTRUCTIONS.md) for detailed web app documentation.

### Option 2: Command Line Interface

#### 1. Preprocess Data
```bash
python main.py preprocess --input data/raw/al_data.csv --output data/processed/cleaned_dataset.csv
```

#### 2. Train Models
```bash
python main.py train --data data/processed/cleaned_dataset.csv --target "Tensile Strength (MPa)"
python main.py train --data data/processed/cleaned_dataset.csv --target "Yield Strength (MPa)"
python main.py train --data data/processed/cleaned_dataset.csv --target "Elongation (%)"
```

#### 3. Make Predictions
```bash
python main.py predict --model models/rf_Tensile_Strength__MPa_.joblib --input data/raw/new_sample.csv --output results/predictions.csv
```

#### 4. Summarize Results
```bash
python main.py summarize --metrics_dir results/metrics
```

## Input Data Format

Your input file should contain the following columns:

- **Processing**: Processing method (e.g., "Solutionised + Artificially peak aged")
- **Chemical Composition**: Ag, Al, B, Be, Bi, Cd, Co, Cr, Cu, Er, Eu, Fe, Ga, Li, Mg, Mn, Ni, Pb, Sc, Si, Sn, Ti, V, Zn, Zr
- **class**: Alloy classification

**For predictions**, do NOT include: Elongation (%), Tensile Strength (MPa), Yield Strength (MPa) - these will be predicted.

## Models

The project uses two types of machine learning models:

1. **Random Forest (RF)**: Ensemble learning method using decision trees
2. **XGBoost**: Gradient boosting framework with regularization

Trained models are saved in the `models/` directory as `.joblib` files.

## Web Application Features

- Upload Excel (.xlsx, .xls) or CSV files
- Process multiple samples in a single upload
- View predictions in an organized table
- Download results as CSV for further analysis
- Responsive design for desktop and mobile
- Error handling and validation

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- joblib
- xgboost
- flask
- openpyxl
- werkzeug

## Notes

- The web application runs locally and does not store data permanently
- Uploaded files are processed and immediately deleted after prediction
- All predictions are made using pre-trained models from the `models/` directory
- For production deployment, consider using a production WSGI server instead of Flask's development server

## License

This project is for educational and research purposes.
