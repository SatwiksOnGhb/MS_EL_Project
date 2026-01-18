from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from pathlib import Path
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'xlsx', 'xls', 'csv'}

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained models (both Random Forest and XGBoost)
MODELS = {
    'rf': {
        'elongation': joblib.load('models/rf_Elongation____.joblib'),
        'tensile_strength': joblib.load('models/rf_Tensile_Strength__MPa_.joblib'),
        'yield_strength': joblib.load('models/rf_Yield_Strength__MPa_.joblib')
    },
    'xgb': {
        'elongation': joblib.load('models/xgb_cv_Elongation_.joblib'),
        'tensile_strength': joblib.load('models/xgb_cv_Tensile_Strength_MPa.joblib'),
        'yield_strength': joblib.load('models/xgb_cv_Yield_Strength_MPa.joblib')
    }
}

# Expected feature columns (all columns except the 3 target columns)
EXPECTED_FEATURES = [
    'Processing', 'Ag', 'Al', 'B', 'Be', 'Bi', 'Cd', 'Co', 'Cr', 'Cu', 'Er',
    'Eu', 'Fe', 'Ga', 'Li', 'Mg', 'Mn', 'Ni', 'Pb', 'Sc', 'Si', 'Sn', 'Ti',
    'V', 'Zn', 'Zr', 'class'
]


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def preprocess_input_data(df):
    """Preprocess the uploaded data to match the training data format"""
    # Clean column names
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]

    # Drop unnamed/index columns
    first_col = df.columns[0]
    if first_col == "" or first_col.lower().startswith("unnamed"):
        df = df.drop(columns=[first_col])

    # Verify all required features are present
    missing_cols = set(EXPECTED_FEATURES) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Select only the expected features in the correct order
    df = df[EXPECTED_FEATURES]

    return df


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload Excel (.xlsx, .xls) or CSV file'}), 400

        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Read the file
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)

        # Preprocess the data
        df_processed = preprocess_input_data(df)

        # Make predictions for all three targets using both models
        predictions = []
        for idx, row in df_processed.iterrows():
            # Convert row to DataFrame for prediction (models expect DataFrame)
            X = pd.DataFrame([row], columns=df_processed.columns)

            # Get predictions from both Random Forest and XGBoost
            pred = {
                'row': idx + 1,
                'rf_elongation': round(float(MODELS['rf']['elongation'].predict(X)[0]), 2),
                'rf_tensile_strength': round(float(MODELS['rf']['tensile_strength'].predict(X)[0]), 2),
                'rf_yield_strength': round(float(MODELS['rf']['yield_strength'].predict(X)[0]), 2),
                'xgb_elongation': round(float(MODELS['xgb']['elongation'].predict(X)[0]), 2),
                'xgb_tensile_strength': round(float(MODELS['xgb']['tensile_strength'].predict(X)[0]), 2),
                'xgb_yield_strength': round(float(MODELS['xgb']['yield_strength'].predict(X)[0]), 2)
            }
            predictions.append(pred)

        # Clean up uploaded file
        os.remove(filepath)

        return jsonify({
            'success': True,
            'predictions': predictions,
            'num_samples': len(predictions)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
