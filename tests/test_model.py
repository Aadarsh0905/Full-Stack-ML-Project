"""
Unit tests for the Heart Disease ML Pipeline.
"""

import os
import pytest
import pandas as pd
import numpy as np
import joblib
from src.train_pipeline import load_and_clean_data, DATA_PATH
from src.predictor import get_predictor


def test_data_cleaning():
    """Verify that physiological zeros in RestingBP and Cholesterol are properly converted to NaN."""
    df = load_and_clean_data(DATA_PATH)
    assert df is not None
    assert len(df) == 918
    # There should be no raw 0s in RestingBP or Cholesterol after load_and_clean_data
    assert (df["RestingBP"] == 0).sum() == 0
    assert (df["Cholesterol"] == 0).sum() == 0
    # Oldpeak must remain float
    assert df["Oldpeak"].dtype in [np.float64, np.float32, float]


def test_pipeline_artifact_exists():
    """Verify that production pipeline artifact is saved and loadable."""
    predictor = get_predictor()
    assert predictor.pipeline is not None
    assert hasattr(predictor.pipeline, "predict")
    assert hasattr(predictor.pipeline, "predict_proba")


def test_pipeline_healthy_prediction():
    """Verify that low-risk parameters predict class 0 with high confidence."""
    predictor = get_predictor()
    patient = {
        "Age": 30,
        "Sex": "F",
        "ChestPainType": "ATA",
        "RestingBP": 110,
        "Cholesterol": 170,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 180,
        "ExerciseAngina": "N",
        "Oldpeak": 0.0,
        "ST_Slope": "Up"
    }
    res = predictor.predict(patient)
    assert res["prediction"] == 0
    assert res["risk_tier"] == "LOW"
    assert res["probability_percent"] < 35.0
    assert len(res["protective_factors"]) > 0


def test_pipeline_high_risk_prediction():
    """Verify that severe risk parameters predict class 1 with high confidence."""
    predictor = get_predictor()
    patient = {
        "Age": 68,
        "Sex": "M",
        "ChestPainType": "ASY",
        "RestingBP": 165,
        "Cholesterol": 290,
        "FastingBS": 1,
        "RestingECG": "ST",
        "MaxHR": 100,
        "ExerciseAngina": "Y",
        "Oldpeak": 3.0,
        "ST_Slope": "Flat"
    }
    res = predictor.predict(patient)
    assert res["prediction"] == 1
    assert res["risk_tier"] == "HIGH"
    assert res["probability_percent"] > 65.0
    assert len(res["risk_factors"]) > 0


def test_pipeline_edge_case_imputation():
    """Verify that passing 0 for Cholesterol or RestingBP does not crash the pipeline."""
    predictor = get_predictor()
    patient = {
        "Age": 50,
        "Sex": "M",
        "ChestPainType": "NAP",
        "RestingBP": 0,  # missing / zero
        "Cholesterol": 0,  # missing / zero
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 140,
        "ExerciseAngina": "N",
        "Oldpeak": 0.5,
        "ST_Slope": "Up"
    }
    res = predictor.predict(patient)
    assert res["prediction"] in [0, 1]
    assert 0.0 <= res["probability_percent"] <= 100.0
