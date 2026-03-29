"""
Titanic Survival Prediction — Model Training Script
Run this script to produce model.pkl used by the Flask API.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 55)
print("  TITANIC SURVIVAL PREDICTION — TRAINING PIPELINE")
print("=" * 55)

DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

try:
    df = pd.read_csv(DATA_URL)
    print(f"\n[1] Data loaded from URL  |  shape: {df.shape}")
except Exception:
    # Fallback: generate synthetic data so pipeline still runs
    print("\n[1] Network unavailable — generating synthetic data for demo")
    np.random.seed(42)
    n = 891
    df = pd.DataFrame({
        'Survived':  np.random.randint(0, 2, n),
        'Pclass':    np.random.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55]),
        'Name':      ['Smith, Mr. John'] * (n // 3) +
                     ['Jones, Mrs. Alice'] * (n // 3) +
                     ['Brown, Miss. Kate'] * (n - 2 * (n // 3)),
        'Sex':       np.random.choice(['male', 'female'], n, p=[0.65, 0.35]),
        'Age':       np.where(np.random.rand(n) < 0.2, np.nan,
                              np.random.normal(30, 14, n).clip(1, 80)),
        'SibSp':     np.random.choice([0,1,2,3], n, p=[0.68,0.23,0.07,0.02]),
        'Parch':     np.random.choice([0,1,2], n, p=[0.76,0.13,0.11]),
        'Fare':      np.where(np.random.rand(n) < 0.01, np.nan,
                              np.abs(np.random.exponential(35, n))),
        'Embarked':  np.random.choice(['S','C','Q', None], n,
                                      p=[0.72, 0.19, 0.08, 0.01]),
    })

# ─────────────────────────────────────────────
# 2. EDA
# ─────────────────────────────────────────────
print("\n[2] EDA Summary")
print(f"    Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")
print(f"    Survival rate: {df['Survived'].mean():.1%}")
print("\n    Missing values:")
miss = df.isnull().sum()
for col, cnt in miss[miss > 0].items():
    print(f"      {col:<12} {cnt:>4}  ({cnt/len(df):.1%})")

os.makedirs("eda_plots", exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Titanic EDA", fontsize=14, fontweight='bold')

survival_by_class = df.groupby('Pclass')['Survived'].mean()
axes[0].bar(survival_by_class.index, survival_by_class.values, color=['#7F77DD','#1D9E75','#D85A30'])
axes[0].set_title("Survival rate by class")
axes[0].set_xlabel("Pclass"); axes[0].set_ylabel("Survival rate")

df.groupby('Sex')['Survived'].mean().plot(kind='bar', ax=axes[1], color=['#7F77DD','#D85A30'], rot=0)
axes[1].set_title("Survival rate by sex")
axes[1].set_ylabel("Survival rate")

df['Age'].dropna().hist(bins=30, ax=axes[2], color='#378ADD', edgecolor='white')
axes[2].set_title("Age distribution")
axes[2].set_xlabel("Age")

plt.tight_layout()
plt.savefig("eda_plots/eda_overview.png", dpi=100, bbox_inches='tight')
plt.close()
print("\n    EDA plot saved → eda_plots/eda_overview.png")

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[3] Feature engineering")

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Family size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone']    = (df['FamilySize'] == 1).astype(int)

    # Title extraction from Name
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    rare_titles = ['Lady','Countess','Capt','Col','Don','Dr',
                   'Major','Rev','Sir','Jonkheer','Dona']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

    # Age bands (kept as numeric for pipeline)
    df['AgeBand'] = pd.cut(df['Age'], bins=[0,12,18,35,60,100],
                           labels=[0,1,2,3,4]).astype(float)

    return df

df = engineer_features(df)
print(f"    Title counts:\n{df['Title'].value_counts().to_string()}")
print(f"    FamilySize range: {df['FamilySize'].min()} – {df['FamilySize'].max()}")

# ─────────────────────────────────────────────
# 4. DEFINE FEATURES & SPLIT
# ─────────────────────────────────────────────
NUMERIC_FEATURES    = ['Age', 'Fare', 'FamilySize', 'SibSp', 'Parch']
CATEGORICAL_FEATURES = ['Sex', 'Embarked', 'Title', 'Pclass']
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

X = df[ALL_FEATURES]
y = df['Survived']

print(f"\n[4] Feature matrix  X: {X.shape}  |  Target y: {y.shape}")

# ─────────────────────────────────────────────
# 5. PREPROCESSING PIPELINE
# ─────────────────────────────────────────────
print("\n[5] Building preprocessing pipeline")

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer,    NUMERIC_FEATURES),
    ('cat', categorical_transformer, CATEGORICAL_FEATURES)
])

# ─────────────────────────────────────────────
# 6. TRAIN WITH GRIDSEARCHCV
# ─────────────────────────────────────────────
print("\n[6] Training RandomForest with GridSearchCV (cv=5) ...")

full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier',   RandomForestClassifier(random_state=42, n_jobs=-1))
])

param_grid = {
    'classifier__n_estimators':    [100, 200],
    'classifier__max_depth':       [4, 6, 8, None],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf':  [1, 2],
}

grid_search = GridSearchCV(
    full_pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X, y)
best_model = grid_search.best_estimator_

print(f"    Best params:   {grid_search.best_params_}")
print(f"    CV accuracy:   {grid_search.best_score_:.4f}")

# ─────────────────────────────────────────────
# 7. EVALUATION
# ─────────────────────────────────────────────
print("\n[7] Evaluation on full training set (in-sample for reference)")

y_pred = best_model.predict(X)
y_prob = best_model.predict_proba(X)[:, 1]

print(f"    Accuracy : {accuracy_score(y, y_pred):.4f}")
print(f"    ROC-AUC  : {roc_auc_score(y, y_prob):.4f}")
print("\n    Classification report:")
print(classification_report(y, y_pred, target_names=['Did not survive', 'Survived']))

# Confusion matrix plot
cm = confusion_matrix(y, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Did not survive','Survived'],
            yticklabels=['Did not survive','Survived'], ax=ax)
ax.set_title("Confusion matrix")
ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
plt.tight_layout()
plt.savefig("eda_plots/confusion_matrix.png", dpi=100, bbox_inches='tight')
plt.close()

# Feature importance
ohe_cols = (best_model.named_steps['preprocessor']
            .transformers_[1][1]
            .named_steps['encoder']
            .get_feature_names_out(CATEGORICAL_FEATURES))
feature_names = NUMERIC_FEATURES + list(ohe_cols)
importances   = best_model.named_steps['classifier'].feature_importances_

feat_df = (pd.DataFrame({'feature': feature_names, 'importance': importances})
           .sort_values('importance', ascending=False)
           .head(12))

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(feat_df['feature'][::-1], feat_df['importance'][::-1], color='#378ADD')
ax.set_title("Top 12 feature importances")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("eda_plots/feature_importance.png", dpi=100, bbox_inches='tight')
plt.close()
print("\n    Plots saved → eda_plots/")

# ─────────────────────────────────────────────
# 8. SAVE MODEL
# ─────────────────────────────────────────────
output_path = "../model.pkl"
joblib.dump(best_model, output_path)
print(f"\n[8] Model saved → {output_path}")
print("\n    Done! Run the Flask app with: python app.py")
print("=" * 55)
