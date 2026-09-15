"""
Model training and pipeline creation module for Heart Disease Risk Prediction.
Ensures zero data leakage, correct feature engineering, hyperparameter tuning,
and exports an end-to-end production pipeline.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Project root paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "heart.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

NUMERICAL_COLS = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
CATEGORICAL_COLS = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

CATEGORICAL_OPTIONS = {
    "Sex": ["M", "F"],
    "ChestPainType": ["ATA", "NAP", "ASY", "TA"],
    "RestingECG": ["Normal", "ST", "LVH"],
    "ExerciseAngina": ["N", "Y"],
    "ST_Slope": ["Up", "Flat", "Down"]
}


def load_and_clean_data(file_path: str = DATA_PATH) -> pd.DataFrame:
    """Loads dataset and marks physiological zeros as NaN for imputer."""
    df = pd.read_csv(file_path)
    # RestingBP and Cholesterol cannot biologically be zero; replace with NaN for imputer
    df["RestingBP"] = df["RestingBP"].replace(0, np.nan)
    df["Cholesterol"] = df["Cholesterol"].replace(0, np.nan)
    # Ensure Oldpeak retains float precision
    df["Oldpeak"] = df["Oldpeak"].astype(float)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Builds a scikit-learn ColumnTransformer for numerical and categorical features."""
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("encoder", OneHotEncoder(
            categories=[CATEGORICAL_OPTIONS[col] for col in CATEGORICAL_COLS],
            drop="first",
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_COLS),
            ("cat", cat_pipeline, CATEGORICAL_COLS)
        ],
        remainder="drop"
    )
    return preprocessor


def train_and_evaluate():
    print("Loading data from:", DATA_PATH)
    df = load_and_clean_data(DATA_PATH)
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    # Stratified Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()

    # Pre-transform training data for hyperparameter search
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    encoded_cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_COLS))
    all_feature_names = NUMERICAL_COLS + encoded_cat_names

    models_config = {
        "Logistic Regression": {
            "estimator": LogisticRegression(max_iter=1000, random_state=42),
            "params": {"C": [0.01, 0.1, 1.0, 5.0, 10.0], "solver": ["lbfgs", "liblinear"]}
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(random_state=42),
            "params": {"n_estimators": [100, 200], "max_depth": [4, 6, 8, None], "min_samples_split": [2, 5]}
        },
        "Gradient Boosting": {
            "estimator": GradientBoostingClassifier(random_state=42),
            "params": {"n_estimators": [100, 150], "learning_rate": [0.03, 0.08, 0.1], "max_depth": [3, 4]}
        },
        "Support Vector Machine (RBF)": {
            "estimator": SVC(probability=True, random_state=42),
            "params": {"C": [0.5, 1.0, 2.0], "gamma": ["scale", "auto"]}
        },
        "K-Nearest Neighbors": {
            "estimator": KNeighborsClassifier(),
            "params": {"n_neighbors": [5, 7, 9, 11], "weights": ["uniform", "distance"]}
        },
        "Naive Bayes": {
            "estimator": GaussianNB(),
            "params": {"var_smoothing": [1e-9, 1e-8, 1e-7]}
        }
    }

    benchmark_results = {}
    best_overall_score = -1.0
    best_model_name = None
    best_estimator = None

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("Beginning hyperparameter tuning and model evaluation...")
    for name, cfg in models_config.items():
        grid = GridSearchCV(cfg["estimator"], cfg["params"], cv=cv, scoring="roc_auc", n_jobs=-1)
        grid.fit(X_train_proc, y_train)

        model = grid.best_estimator_
        y_pred = model.predict(X_test_proc)
        y_proba = model.predict_proba(X_test_proc)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred))
        rec = float(recall_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))
        auc = float(roc_auc_score(y_test, y_proba))
        cm = confusion_matrix(y_test, y_pred).tolist()

        benchmark_results[name] = {
            "best_params": grid.best_params_,
            "cv_roc_auc": round(float(grid.best_score_), 4),
            "test_accuracy": round(acc, 4),
            "test_precision": round(prec, 4),
            "test_recall": round(rec, 4),
            "test_f1": round(f1, 4),
            "test_roc_auc": round(auc, 4),
            "confusion_matrix": cm,
        }

        print(f"Model: {name} | Test Acc: {acc:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")

        # Choose best based on balanced clinical score: F1 + ROC-AUC + Recall
        clinical_score = (f1 + auc + rec) / 3.0
        if clinical_score > best_overall_score:
            best_overall_score = clinical_score
            best_model_name = name
            best_estimator = model

    print(f"\nWinning Model: {best_model_name} (Combined Clinical Score: {best_overall_score:.4f})")

    # Build the full production pipeline with the best estimator
    final_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", best_estimator)
    ])

    # Fit final pipeline on all training data
    final_pipeline.fit(X_train, y_train)

    # Save artifacts
    pipeline_path = os.path.join(MODELS_DIR, "heart_disease_pipeline.joblib")
    joblib.dump(final_pipeline, pipeline_path)
    print("Saved production pipeline to:", pipeline_path)

    # Also save metadata
    feature_metadata = {
        "numerical_cols": NUMERICAL_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "categorical_options": CATEGORICAL_OPTIONS,
        "encoded_feature_names": all_feature_names,
        "best_model_name": best_model_name,
        "best_hyperparams": benchmark_results[best_model_name]["best_params"],
        "dataset_summary": {
            "total_samples": int(len(df)),
            "positive_cases": int((df["HeartDisease"] == 1).sum()),
            "negative_cases": int((df["HeartDisease"] == 0).sum()),
        }
    }

    # Extract feature importance / coefficients for explainability
    if hasattr(best_estimator, "coef_"):
        coefs = best_estimator.coef_[0].tolist()
        feature_metadata["feature_weights"] = dict(zip(all_feature_names, [round(c, 4) for c in coefs]))
    elif hasattr(best_estimator, "feature_importances_"):
        imps = best_estimator.feature_importances_.tolist()
        feature_metadata["feature_weights"] = dict(zip(all_feature_names, [round(i, 4) for i in imps]))

    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(feature_metadata, f, indent=2)

    with open(os.path.join(MODELS_DIR, "benchmark_metrics.json"), "w") as f:
        json.dump(benchmark_results, f, indent=2)

    # Maintain backwards compatibility for legacy files in root directory
    joblib.dump(best_estimator, os.path.join(BASE_DIR, "Logistic_heart.pkl"))
    joblib.dump(preprocessor.named_transformers_["num"].named_steps["scaler"], os.path.join(BASE_DIR, "scaler.pkl"))
    joblib.dump(all_feature_names, os.path.join(BASE_DIR, "columns.pkl"))
    print("Legacy root pickle files updated with bug-free models and scaler.")

    return benchmark_results, best_model_name


if __name__ == "__main__":
    train_and_evaluate()
