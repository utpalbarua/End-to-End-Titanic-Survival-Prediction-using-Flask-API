# Titanic Survival Predictor — End-to-End ML Project

A production-grade ML project that goes beyond the Kaggle notebook:
Random Forest classifier wrapped in a Flask API, HTML frontend, GitHub Actions CI, and Render deployment.

---

## Project structure

```
titanic-app/
├── app.py                        # Flask API (GET /, POST /predict, GET /health)
├── model.pkl                     # Serialized sklearn Pipeline (generated)
├── requirements.txt
├── Procfile                      # gunicorn start command for Render
├── render.yaml                   # Render infrastructure-as-code config
├── templates/
│   └── index.html                # Prediction frontend
├── notebook/
│   └── titanic_model.py          # EDA + feature engineering + training script
├── tests/
│   ├── conftest.py
│   └── test_app.py               # pytest suite (health, predict, validation, unit)
└── .github/
    └── workflows/
        └── ci.yml                # GitHub Actions: test on push, deploy on main
```

---

## Quick start (local)

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/titanic-app.git
cd titanic-app
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the model

```bash
cd notebook
python titanic_model.py
# → saves ../model.pkl
# → saves eda_plots/eda_overview.png, confusion_matrix.png, feature_importance.png
cd ..
```

### 3. Run the app

```bash
python app.py
# → http://localhost:5000
```

### 4. Run tests

```bash
pytest tests/ -v --cov=app --cov-report=term-missing
```

---

## API reference

### `GET /`
Returns the HTML prediction form.

### `GET /health`
```json
{ "status": "ok", "model_loaded": true }
```

### `POST /predict`

**Request body (JSON):**

| Field      | Type    | Required | Description                          |
|------------|---------|----------|--------------------------------------|
| Age        | float   | No       | Passenger age (1–120)                |
| Fare       | float   | No       | Ticket fare (≥ 0)                    |
| SibSp      | int     | No       | Siblings/spouses aboard              |
| Parch      | int     | No       | Parents/children aboard              |
| Sex        | string  | No       | `"male"` or `"female"`               |
| Embarked   | string  | No       | `"S"`, `"C"`, or `"Q"`              |
| Title      | string  | No       | `"Mr"`, `"Mrs"`, `"Miss"`, `"Master"`, `"Rare"` |
| Pclass     | int     | No       | `1`, `2`, or `3`                     |
| FamilySize | int     | No       | Auto-derived as SibSp + Parch + 1    |

**Example request:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"Age":28,"Fare":75,"Sex":"female","Pclass":1,"Title":"Miss","Embarked":"C","SibSp":0,"Parch":0}'
```

**Example response:**
```json
{
  "survived": 1,
  "probability": 0.923,
  "confidence": "92.3%",
  "message": "Survived"
}
```

**Validation error (422):**
```json
{ "errors": ["Age must be between 0 and 120.", "Pclass must be 1, 2, or 3."] }
```

---

## Feature engineering

Two key engineered features beyond the raw Titanic columns:

- **FamilySize** = `SibSp + Parch + 1` — captures traveling alone vs. with family
- **Title** — extracted from `Name` (e.g. "Braund, Mr. Owen") using regex `r' ([A-Za-z]+)\.'`
  - Rare titles (Dr, Rev, Major, etc.) grouped into `"Rare"`
  - Mlle → Miss, Mme → Mrs

---

## Model details

| Step             | Choice                              |
|------------------|-------------------------------------|
| Imputation       | Median (numeric), Mode (categorical) |
| Encoding         | OneHotEncoder (unknown → ignore)     |
| Scaling          | StandardScaler on numeric features   |
| Model            | RandomForestClassifier               |
| Tuning           | GridSearchCV, cv=5, scoring=accuracy |
| Serialization    | Full Pipeline via joblib             |

Hyperparameter grid:
- `n_estimators`: [100, 200]
- `max_depth`: [4, 6, 8, None]
- `min_samples_split`: [2, 5]
- `min_samples_leaf`: [1, 2]

---

## Deployment (Render)

### One-time setup

1. Push this repo to GitHub
2. Create a new **Web Service** on [render.com](https://render.com), connect the repo
3. Render reads `render.yaml` automatically — no manual configuration needed
4. The build command trains the model; start command launches gunicorn

### Auto-deploy on merge to main

1. In Render dashboard → your service → **Settings → Deploy hook** → copy the URL
2. In GitHub → **Settings → Secrets → Actions** → add secret `RENDER_DEPLOY_HOOK` with that URL
3. Now every merge to `main` that passes CI will trigger an automatic deploy

### Branch protection (recommended)

In GitHub → **Settings → Branches → Branch protection rules** for `main`:
- [x] Require status checks to pass before merging
- [x] Select the `test` CI job as required
- [x] Require branches to be up to date before merging

---

## CI/CD pipeline

```
git push (any branch)
    ↓
GitHub Actions: CI job
    pip install -r requirements.txt
    pytest tests/ --cov=app --cov-fail-under=60
    ↓
Tests pass?
    NO  → Block PR / notify
    YES → Allow merge to main
              ↓
         Deploy job
         curl $RENDER_DEPLOY_HOOK
              ↓
         Render pulls repo
         buildCommand (train model)
         startCommand (gunicorn)
              ↓
         App live at your-app.onrender.com
```
