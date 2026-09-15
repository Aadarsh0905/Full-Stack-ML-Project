"""
Entry point script to launch CardioSense AI Web Server.
Usage:
    python run.py
"""

import sys
import os

# Add root directory to python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 65)
    print(" CARDIO-SENSE AI: FULL STACK HEART DISEASE RISK PREDICTOR ")
    print("=" * 65)
    print(f" -> Web Dashboard:  http://127.0.0.1:{port}")
    print(f" -> Benchmark View: http://127.0.0.1:{port}/models")
    print(f" -> REST API Docs:  http://127.0.0.1:{port}/api-docs")
    print(f" -> Health Endpoint:http://127.0.0.1:{port}/api/health")
    print("=" * 65)
    app.run(host="0.0.0.0", port=port, debug=False)
