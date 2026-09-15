"""
Prediction service engine for Heart Disease Detection.
Loads trained scikit-learn pipeline, processes patient records,
calculates risk tiers and patient-specific risk factor breakdowns.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_PATH = os.path.join(BASE_DIR, "models", "heart_disease_pipeline.joblib")
METADATA_PATH = os.path.join(BASE_DIR, "models", "metadata.json")
BENCHMARK_PATH = os.path.join(BASE_DIR, "models", "benchmark_metrics.json")


class HeartDiseasePredictor:
    def __init__(self, pipeline_path: str = PIPELINE_PATH):
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Model pipeline not found at {pipeline_path}. Run train_pipeline.py first.")
        self.pipeline = joblib.load(pipeline_path)
        
        self.metadata = {}
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "r") as f:
                self.metadata = json.load(f)

        self.benchmarks = {}
        if os.path.exists(BENCHMARK_PATH):
            with open(BENCHMARK_PATH, "r") as f:
                self.benchmarks = json.load(f)

    def analyze_factors(self, data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Identifies key clinical warning signs and protective attributes."""
        risk_factors = []
        protective_factors = []

        # ST Slope
        if data.get("ST_Slope") == "Flat":
            risk_factors.append("Flat ST Slope during peak exercise (common sign of myocardial ischemia)")
        elif data.get("ST_Slope") == "Down":
            risk_factors.append("Downsloping ST Segment during peak exercise (high correlation with cardiac abnormalities)")
        elif data.get("ST_Slope") == "Up":
            protective_factors.append("Upsloping ST Slope (favorable physiological response)")

        # Exercise Angina
        if data.get("ExerciseAngina") == "Y":
            risk_factors.append("Exercise-induced angina present (chest pain during physical exertion)")
        else:
            protective_factors.append("No exercise-induced angina reported")

        # Chest Pain Type
        if data.get("ChestPainType") == "ASY":
            risk_factors.append("Asymptomatic chest pain type (historically high correlation with silent CAD)")
        elif data.get("ChestPainType") in ["ATA", "NAP"]:
            protective_factors.append(f"Chest pain type '{data.get('ChestPainType')}' typically carries lower risk than asymptomatic CAD")

        # Oldpeak
        oldpeak = float(data.get("Oldpeak", 0.0))
        if oldpeak >= 1.5:
            risk_factors.append(f"Elevated ST depression Oldpeak ({oldpeak} mm) indicates significant ischemic response")
        elif oldpeak <= 0.5:
            protective_factors.append(f"Low ST depression Oldpeak ({oldpeak} mm) is within typical normal ranges")

        # Fasting Blood Sugar
        if int(data.get("FastingBS", 0)) == 1:
            risk_factors.append("Fasting blood sugar > 120 mg/dl (hyperglycemia increases cardiovascular risk)")
        else:
            protective_factors.append("Normal fasting blood sugar levels (<= 120 mg/dl)")

        # Max Heart Rate
        max_hr = int(data.get("MaxHR", 140))
        age = int(data.get("Age", 50))
        expected_max = 220 - age
        if max_hr < 110:
            risk_factors.append(f"Low peak heart rate achieved ({max_hr} bpm; expected ~{expected_max} bpm)")
        elif max_hr > 150:
            protective_factors.append(f"Good peak heart rate capacity achieved ({max_hr} bpm)")

        # Blood Pressure
        resting_bp = int(data.get("RestingBP", 120))
        if resting_bp >= 140:
            risk_factors.append(f"Elevated resting blood pressure ({resting_bp} mm Hg - stage 2 hypertension threshold)")
        elif resting_bp <= 120:
            protective_factors.append(f"Optimal resting blood pressure ({resting_bp} mm Hg)")

        # Cholesterol
        chol = int(data.get("Cholesterol", 0))
        if chol > 240:
            risk_factors.append(f"High serum cholesterol level ({chol} mg/dl - hypercholesterolemia)")
        elif 120 <= chol <= 200:
            protective_factors.append(f"Normal total cholesterol level ({chol} mg/dl)")

        return {
            "risk_factors": risk_factors,
            "protective_factors": protective_factors
        }

    def predict(self, patient_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes prediction on single patient input dictionary.
        Returns comprehensive clinical report dictionary.
        """
        # Form single-row DataFrame
        df_row = pd.DataFrame([patient_dict])

        # Execute pipeline
        pred_class = int(self.pipeline.predict(df_row)[0])
        probabilities = self.pipeline.predict_proba(df_row)[0]
        prob_heart_disease = float(probabilities[1])
        prob_healthy = float(probabilities[0])

        # Determine clinical risk tier
        if prob_heart_disease < 0.30:
            risk_tier = "LOW"
            badge_color = "success"
            summary = "Low probability of heart disease. Patient exhibits healthy cardiac markers."
            action_plan = "Maintain healthy lifestyle habits, balanced diet, and schedule routine annual physicals."
        elif prob_heart_disease < 0.65:
            risk_tier = "MODERATE"
            badge_color = "warning"
            summary = "Borderline / Moderate probability of heart disease. Secondary indicators present."
            action_plan = "Recommend comprehensive lipid panel, 24-hour ambulatory BP monitoring, and cardiologist follow-up."
        else:
            risk_tier = "HIGH"
            badge_color = "danger"
            summary = "High probability of coronary heart disease detected by diagnostic indicators."
            action_plan = "Immediate cardiologist consultation, diagnostic echocardiogram or coronary angiography strongly advised."

        factor_analysis = self.analyze_factors(patient_dict)

        return {
            "prediction": pred_class,
            "prediction_label": "Heart Disease Detected" if pred_class == 1 else "Normal (No Heart Disease)",
            "probability_percent": round(prob_heart_disease * 100, 1),
            "probability_healthy_percent": round(prob_healthy * 100, 1),
            "risk_tier": risk_tier,
            "badge_color": badge_color,
            "clinical_summary": summary,
            "action_plan": action_plan,
            "risk_factors": factor_analysis["risk_factors"],
            "protective_factors": factor_analysis["protective_factors"],
            "model_name": self.metadata.get("best_model_name", "Random Forest Classifier"),
            "input_snapshot": patient_dict
        }

    def get_benchmarks(self) -> Dict[str, Any]:
        return self.benchmarks

    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata


# Global singleton instance
_predictor_instance = None


def get_predictor() -> HeartDiseasePredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = HeartDiseasePredictor()
    return _predictor_instance
