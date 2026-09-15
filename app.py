"""
CardioSense AI - Full Stack Heart Disease Clinical Risk Prediction Platform
Flask Application providing Web Interface and RESTful API endpoints.
"""

import io
import os
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for
from src.validator import validate_patient_data
from src.predictor import get_predictor

app = Flask(__name__)
app.config["SECRET_KEY"] = "cardiosense-production-secret-key-2026"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max for batch files

# Initialize predictor
predictor = get_predictor()


# -------------------------------------------------------------
# Web Page Routes
# -------------------------------------------------------------

@app.route("/")
def index():
    """Main Diagnostic Assessment Page."""
    metadata = predictor.get_metadata()
    return render_template("index.html", metadata=metadata)


@app.route("/models")
def models_view():
    """Model Benchmarking, Comparative Analysis, and Error Audit."""
    benchmarks = predictor.get_benchmarks()
    metadata = predictor.get_metadata()
    return render_template("models.html", benchmarks=benchmarks, metadata=metadata)


@app.route("/api-docs")
def api_docs():
    """Interactive REST API Documentation & Playground."""
    return render_template("api_docs.html")


# -------------------------------------------------------------
# REST API Endpoints
# -------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "CardioSense Heart Disease Prediction API",
        "version": "1.0.0",
        "model_loaded": predictor.metadata.get("best_model_name", "Random Forest Classifier")
    }), 200


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    JSON API for single patient risk prediction.
    Expects JSON payload with patient diagnostic features.
    """
    if not request.is_json:
        return jsonify({
            "success": False,
            "error": "Request body must be valid JSON with Content-Type: application/json"
        }), 400

    payload = request.get_json()
    is_valid, errors, sanitized_data = validate_patient_data(payload)

    if not is_valid:
        return jsonify({
            "success": False,
            "errors": errors
        }), 422

    try:
        result = predictor.predict(sanitized_data)
        return jsonify({
            "success": True,
            "data": result
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Prediction error: {str(e)}"
        }), 500


@app.route("/api/batch-predict", methods=["POST"])
def api_batch_predict():
    """
    Batch prediction endpoint supporting JSON list or CSV file upload.
    """
    records = []
    
    # Check if a CSV file was uploaded
    if "file" in request.files:
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected"}), 400
        if not file.filename.endswith(".csv"):
            return jsonify({"success": False, "error": "Only CSV files are supported"}), 400
        try:
            df = pd.read_csv(io.StringIO(file.stream.read().decode("UTF8")))
            records = df.to_dict(orient="records")
        except Exception as e:
            return jsonify({"success": False, "error": f"Failed to parse CSV: {str(e)}"}), 400
    elif request.is_json:
        payload = request.get_json()
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict) and "patients" in payload and isinstance(payload["patients"], list):
            records = payload["patients"]
        else:
            return jsonify({"success": False, "error": "Expected a JSON list of patient records or {'patients': [...]}"}), 400
    else:
        return jsonify({"success": False, "error": "Provide either a JSON list or a CSV file upload"}), 400

    if not records:
        return jsonify({"success": False, "error": "No records found to process"}), 400

    results = []
    validation_failures = 0

    for idx, row in enumerate(records[:500]):  # Cap at 500 for responsiveness
        is_valid, errors, sanitized = validate_patient_data(row)
        if not is_valid:
            validation_failures += 1
            results.append({
                "index": idx,
                "success": False,
                "errors": errors,
                "input": row
            })
        else:
            pred_res = predictor.predict(sanitized)
            results.append({
                "index": idx,
                "success": True,
                "prediction": pred_res["prediction"],
                "prediction_label": pred_res["prediction_label"],
                "probability_percent": pred_res["probability_percent"],
                "risk_tier": pred_res["risk_tier"],
                "summary": pred_res["clinical_summary"]
            })

    return jsonify({
        "success": True,
        "total_processed": len(records),
        "successful_predictions": len(records) - validation_failures,
        "validation_failures": validation_failures,
        "results": results
    }), 200


@app.route("/api/models", methods=["GET"])
def api_models():
    """Returns comparative benchmarking metrics across all evaluated ML models."""
    return jsonify({
        "success": True,
        "best_model": predictor.metadata.get("best_model_name"),
        "benchmarks": predictor.get_benchmarks(),
        "metadata": predictor.get_metadata()
    }), 200


if __name__ == "__main__":
    # Run development server
    port = int(os.environ.get("PORT", 5000))
    print(f"CardioSense AI Web Server running on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
