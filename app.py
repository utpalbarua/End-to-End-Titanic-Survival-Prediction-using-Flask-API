"""
app.py — Titanic Survival Prediction Flask API
Endpoints:
  GET  /          → serve HTML frontend
  POST /predict   → accept JSON, return survival prediction
  GET  /health    → health check for Render
"""

import os
import logging
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

# ── App setup
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Feature constants (must match training) 
NUMERIC_FEATURES     = ['Age', 'Fare', 'FamilySize', 'SibSp', 'Parch']
CATEGORICAL_FEATURES = ['Sex', 'Embarked', 'Title', 'Pclass']
ALL_FEATURES         = NUMERIC_FEATURES + CATEGORICAL_FEATURES

VALID_SEX      = {'male', 'female'}
VALID_EMBARKED = {'S', 'C', 'Q'}
VALID_TITLE    = {'Mrs', 'Miss', 'Rare'}
VALID_PCLASS   = {1, 2, 3}

# ── Load model once at startup
MODEL_PATH = 'C:/Users/lenovo/OneDrive/Desktop/Titanic_Survival_Prediction/model/model.pkl'

try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully from %s", MODEL_PATH)
except FileNotFoundError:
    model = None
    logger.warning("model.pkl not found — run notebook/titanic_model.py first")


# ── Helper: engineer features from raw input 
def prepare_input(data: dict) -> pd.DataFrame:
    """
    Accepts raw passenger dict, derives FamilySize if not provided,
    returns a single-row DataFrame with ALL_FEATURES columns.
    """
    row = {}

    # Numeric
    row['Age']        = float(data.get('Age', 29))
    row['Fare']       = float(data.get('Fare', 14.5))
    row['SibSp']      = int(data.get('SibSp', 0))
    row['Parch']      = int(data.get('Parch', 0))
    row['FamilySize'] = int(data.get('FamilySize',
    row['SibSp'] + row['Parch'] + 1))

    # Categorical
    row['Sex']      = str(data.get('Sex', 'male')).lower()
    row['Embarked'] = str(data.get('Embarked', 'S')).upper()
    row['Title']    = str(data.get('Title', 'Mr'))
    row['Pclass']   = int(data.get('Pclass', 3))

    return pd.DataFrame([row])[ALL_FEATURES]


# ── Input validation
def validate_input(data: dict) -> list[str]:
    errors = []

    age = data.get('Age')
    if age is not None:
        try:
            if not (0 < float(age) < 120):
                errors.append("Age must be between 0 and 120.")
        except (ValueError, TypeError):
            errors.append("Age must be a number.")

    fare = data.get('Fare')
    if fare is not None:
        try:
            if float(fare) < 0:
                errors.append("Fare must be non-negative.")
        except (ValueError, TypeError):
            errors.append("Fare must be a number.")

    pclass = data.get('Pclass')
    if pclass is not None:
        try:
            if int(pclass) not in VALID_PCLASS:
                errors.append("Pclass must be 1, 2, or 3.")
        except (ValueError, TypeError):
            errors.append("Pclass must be an integer.")

    sex = data.get('Sex')
    if sex and str(sex).lower() not in VALID_SEX:
        errors.append(f"Sex must be one of {VALID_SEX}.")

    embarked = data.get('Embarked')
    if embarked and str(embarked).upper() not in VALID_EMBARKED:
        errors.append(f"Embarked must be one of {VALID_EMBARKED}.")

    title = data.get('Title')
    if title and str(title) not in VALID_TITLE:
        errors.append(f"Title must be one of {VALID_TITLE}.")

    return errors


# ── Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify(
        status='ok',
        model_loaded=(model is not None)
    ), 200


@app.route('/predict', methods=['POST'])
def predict():
    # 1. Parse JSON body
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify(error='Request body must be valid JSON.'), 400

    # 2. Validate
    errors = validate_input(data)
    if errors:
        return jsonify(errors=errors), 422

    # 3. Model check
    if model is None:
        return jsonify(error='Model not loaded. Run the training script first.'), 503

    # 4. Prepare features → predict
    try:
        df    = prepare_input(data)
        prob  = float(model.predict_proba(df)[0][1])
        pred  = int(prob >= 0.5)

        logger.info("Prediction: survived=%d  prob=%.3f  input=%s", pred, prob, data)

        return jsonify(
            survived    = pred,
            probability = round(prob, 3),
            confidence  = f"{round(prob * 100, 1)}%",
            message     = "Survived" if pred == 1 else "Did not survive"
        ), 200

    except Exception as exc:
        logger.exception("Prediction error: %s", exc)
        return jsonify(error='Prediction failed. Check your input values.'), 500


# ── Dev server ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)