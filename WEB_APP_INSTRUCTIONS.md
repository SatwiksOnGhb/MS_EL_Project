# Aluminum Alloy Properties Predictor - Web Application

## Overview
This web application allows users to upload Excel or CSV files containing aluminum alloy composition data and predicts three mechanical properties:
- Elongation (%)
- Tensile Strength (MPa)
- Yield Strength (MPa)

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Web Application

1. Make sure you're in the project directory:
```bash
cd "c:\Users\satwi\OneDrive\Desktop\MS_Project - Copy"
```

2. Run the Flask application:
```bash
python app.py
```

3. Open your web browser and navigate to:
```
http://localhost:5000
```

## Using the Web Application

### Preparing Your Input File

Your Excel or CSV file must contain the following columns (but NOT the three target columns):

**Required Columns:**
- Processing (text: e.g., "Solutionised + Artificially peak aged")
- Ag, Al, B, Be, Bi, Cd, Co, Cr, Cu (numeric values)
- Er, Eu, Fe, Ga, Li, Mg, Mn, Ni (numeric values)
- Pb, Sc, Si, Sn, Ti, V, Zn, Zr (numeric values)
- class (numeric value)

**Do NOT include these columns (they will be predicted):**
- Elongation (%)
- Tensile Strength (MPa)
- Yield Strength (MPa)

### Example Input Format

```csv
Processing,Ag,Al,B,Be,Bi,Cd,Co,Cr,Cu,Er,Eu,Fe,Ga,Li,Mg,Mn,Ni,Pb,Sc,Si,Sn,Ti,V,Zn,Zr,class
Solutionised + Artificially peak aged,0,0.88011,0,0,0,0,0,0,0.0198,0,0,0.00055,0,0,0.0212,0,0,0,0,0.00034,0,0,0,0.0768,0.0012,2
```

### Steps to Use

1. Click on the "Choose a file" button or drag and drop your file
2. Select your Excel (.xlsx, .xls) or CSV (.csv) file
3. Click the "Predict Properties" button
4. Wait for the predictions to be processed
5. View the results in the table showing:
   - Sample number
   - Predicted Elongation (%)
   - Predicted Tensile Strength (MPa)
   - Predicted Yield Strength (MPa)
6. Optionally download the results as a CSV file using the "Download Results as CSV" button

## Features

- **File Upload**: Supports Excel (.xlsx, .xls) and CSV formats
- **Batch Prediction**: Process multiple samples at once
- **Real-time Results**: Instant predictions displayed in a clean table format
- **Download Results**: Export predictions as CSV for further analysis
- **Error Handling**: Clear error messages if file format is incorrect or columns are missing
- **Responsive Design**: Works on desktop and mobile devices

## Models Used

The application uses Random Forest models trained on aluminum alloy data:
- `rf_Elongation____.joblib` - For Elongation prediction
- `rf_Tensile_Strength__MPa_.joblib` - For Tensile Strength prediction
- `rf_Yield_Strength__MPa_.joblib` - For Yield Strength prediction

## Troubleshooting

### Common Issues

1. **"Missing required columns" error**
   - Ensure your file has all required columns listed above
   - Check for typos in column names
   - Make sure you're NOT including the three target columns

2. **"Invalid file type" error**
   - Only .xlsx, .xls, and .csv files are supported
   - Check your file extension

3. **Server won't start**
   - Make sure port 5000 is not already in use
   - Check that all dependencies are installed correctly
   - Ensure the models exist in the `models/` directory

4. **Predictions seem incorrect**
   - Verify your input data is in the correct format
   - Check that composition values are normalized/scaled properly
   - Ensure the "class" column has appropriate values

## File Structure

```
MS_Project - Copy/
├── app.py                  # Flask application
├── templates/
│   └── index.html         # Web interface
├── static/
│   └── style.css          # Styling
├── models/
│   ├── rf_Elongation____.joblib
│   ├── rf_Tensile_Strength__MPa_.joblib
│   └── rf_Yield_Strength__MPa_.joblib
├── uploads/               # Temporary file storage (created automatically)
└── requirements.txt       # Python dependencies
```

## Support

If you encounter any issues, please check:
1. All required packages are installed
2. The trained models exist in the `models/` directory
3. Your input file follows the correct format
4. Python version is compatible (Python 3.7+)
