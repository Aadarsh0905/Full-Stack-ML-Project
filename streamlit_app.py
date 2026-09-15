"""
CardioSense AI - Streamlit Web Application
For 100% free deployment on Hugging Face Spaces (Streamlit SDK) or Streamlit Community Cloud.
No Docker required.
"""

import streamlit as st
import pandas as pd
from src.predictor import get_predictor

st.set_page_config(
    page_title="CardioSense AI - Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: rgba(239, 68, 68, 0.2);
        border: 1px solid #ef4444;
        padding: 1rem;
        border-radius: 10px;
        color: #fca5a5;
    }
    .risk-mod {
        background-color: rgba(245, 158, 11, 0.2);
        border: 1px solid #f59e0b;
        padding: 1rem;
        border-radius: 10px;
        color: #fde68a;
    }
    .risk-low {
        background-color: rgba(16, 185, 129, 0.2);
        border: 1px solid #10b981;
        padding: 1rem;
        border-radius: 10px;
        color: #a7f3d0;
    }
</style>
""", unsafe_allow_html=True)

predictor = get_predictor()
benchmarks = predictor.get_benchmarks()
metadata = predictor.get_metadata()

# Navigation tabs
tab_assess, tab_benchmarks, tab_batch = st.tabs([
    "🩺 Patient Assessment",
    "📊 Model Benchmarks & Audit",
    "📁 Batch Screening (CSV)"
])

# -------------------------------------------------------------
# Tab 1: Patient Assessment
# -------------------------------------------------------------
with tab_assess:
    st.markdown('<div class="main-header">Cardiovascular Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-biomarker hemodynamic evaluation powered by Scikit-Learn</div>', unsafe_allow_html=True)

    # Preset Quick Load
    st.write("**Quick Preset Test Patients:**")
    col_p1, col_p2, col_p3 = st.columns(3)
    preset_choice = None
    if col_p1.button("🟢 Load Low Risk Profile"):
        preset_choice = "healthy"
    if col_p2.button("🟡 Load Moderate Risk Profile"):
        preset_choice = "moderate"
    if col_p3.button("🔴 Load High Risk Profile"):
        preset_choice = "high_risk"

    defaults = {
        "healthy": {"Age": 36, "Sex": "F", "CP": "ATA", "BP": 118, "Chol": 185, "FBS": 0, "ECG": "Normal", "HR": 172, "Ang": "N", "Old": 0.0, "Slope": "Up"},
        "moderate": {"Age": 52, "Sex": "M", "CP": "NAP", "BP": 138, "Chol": 235, "FBS": 0, "ECG": "Normal", "HR": 142, "Ang": "N", "Old": 1.0, "Slope": "Flat"},
        "high_risk": {"Age": 63, "Sex": "M", "CP": "ASY", "BP": 165, "Chol": 295, "FBS": 1, "ECG": "ST", "HR": 108, "Ang": "Y", "Old": 2.8, "Slope": "Flat"}
    }

    current = defaults.get(preset_choice, defaults["moderate"])

    with st.form("assessment_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.slider("Age (Years)", 18, 100, int(current["Age"]))
            sex = st.selectbox("Biological Sex", ["M", "F"], index=0 if current["Sex"] == "M" else 1)
            cpt = st.selectbox(
                "Chest Pain Type",
                ["ASY", "NAP", "ATA", "TA"],
                index=["ASY", "NAP", "ATA", "TA"].index(current["CP"]),
                help="ASY: Asymptomatic, NAP: Non-Anginal, ATA: Atypical Angina, TA: Typical Angina"
            )
            bp = st.number_input("Resting Blood Pressure (mm Hg)", 60, 240, int(current["BP"]))

        with col2:
            chol = st.number_input("Serum Cholesterol (mg/dl)", 0, 650, int(current["Chol"]), help="Enter 0 if unknown")
            fbs = st.selectbox("Fasting Blood Sugar", [0, 1], index=int(current["FBS"]), format_func=lambda x: "Normal (<= 120 mg/dl)" if x == 0 else "High (> 120 mg/dl)")
            ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"], index=["Normal", "ST", "LVH"].index(current["ECG"]))
            max_hr = st.slider("Max Heart Rate Achieved (bpm)", 50, 220, int(current["HR"]))

        with col3:
            angina = st.selectbox("Exercise-Induced Angina", ["N", "Y"], index=0 if current["Ang"] == "N" else 1)
            oldpeak = st.slider("Oldpeak ST Depression (mm)", -2.5, 6.5, float(current["Old"]), step=0.1)
            slope = st.selectbox("Peak Exercise ST Slope", ["Up", "Flat", "Down"], index=["Up", "Flat", "Down"].index(current["Slope"]))

        submitted = st.form_submit_button("🫀 Run Diagnostic Assessment", use_container_width=True)

    if submitted or preset_choice is not None:
        patient_data = {
            "Age": age, "Sex": sex, "ChestPainType": cpt,
            "RestingBP": bp, "Cholesterol": chol, "FastingBS": fbs,
            "RestingECG": ecg, "MaxHR": max_hr, "ExerciseAngina": angina,
            "Oldpeak": oldpeak, "ST_Slope": slope
        }

        res = predictor.predict(patient_data)

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            tier_class = "risk-high" if res["risk_tier"] == "HIGH" else ("risk-mod" if res["risk_tier"] == "MODERATE" else "risk-low")
            st.markdown(f"""
            <div class="{tier_class}">
                <h3 style="margin:0; font-size:1.4rem;">{res['risk_tier']} RISK</h3>
                <h1 style="margin:0.2rem 0; font-size:3rem; font-weight:800;">{res['probability_percent']}%</h1>
                <p style="margin:0; font-weight:600;">{res['prediction_label']}</p>
            </div>
            """, unsafe_allow_html=True)
            st.metric(label="Model in Use", value=res["model_name"])

        with res_col2:
            st.subheader("Clinical Summary")
            st.write(res["clinical_summary"])
            st.info(f"**Action Plan:** {res['action_plan']}")

            c_risk, c_prot = st.columns(2)
            with c_risk:
                st.write("**⚠️ Observed Risk Markers:**")
                if res["risk_factors"]:
                    for rf in res["risk_factors"]:
                        st.markdown(f"- {rf}")
                else:
                    st.success("No critical warning signs observed.")

            with c_prot:
                st.write("**🛡️ Protective Indicators:**")
                if res["protective_factors"]:
                    for pf in res["protective_factors"]:
                        st.markdown(f"- {pf}")
                else:
                    st.warning("Few protective indicators observed.")

# -------------------------------------------------------------
# Tab 2: Model Benchmarks & Audit
# -------------------------------------------------------------
with tab_benchmarks:
    st.subheader("Model Benchmarks Across 6 Algorithms")
    st.write("Results evaluated on the 918-patient cohort using 5-Fold Stratified Cross Validation:")

    df_b = pd.DataFrame([
        {
            "Model": k,
            "Accuracy": f"{v['test_accuracy']*100:.2f}%",
            "Recall (Sensitivity)": f"{v['test_recall']*100:.2f}%",
            "Precision": f"{v['test_precision']*100:.2f}%",
            "F1-Score": f"{v['test_f1']*100:.2f}%",
            "ROC-AUC": v['test_roc_auc']
        }
        for k, v in benchmarks.items()
    ])
    st.dataframe(df_b, use_container_width=True)

    st.subheader("Autopsy of Notebook Errors in Project2.ipynb")
    st.error("**1. Scaler Data Leakage**: `scaler.fit_transform(X_test)` recomputed statistics on test data, corrupting the saved `scaler.pkl`.")
    st.error("**2. Feature Truncation**: `df_encoded.astype(int)` truncated `Oldpeak` floats (e.g. 1.5 -> 1), destroying diagnostic ST depression signals.")
    st.error("**3. Zero Imputation Leakage**: Mean calculation on 0-values performed before train/test split.")
    st.success("**Fixes Applied**: Pipeline ColumnTransformer with SimpleImputer and OneHotEncoder fitted exclusively on training data, preserving float precision.")

# -------------------------------------------------------------
# Tab 3: Batch Screening
# -------------------------------------------------------------
with tab_batch:
    st.subheader("Batch Patient Screening (CSV Upload)")
    uploaded_file = st.file_uploader("Upload patient cohort CSV file", type=["csv"])
    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df_batch)} records.")
        if st.button("Analyze Cohort"):
            results = []
            for _, row in df_batch.iterrows():
                try:
                    res = predictor.predict(row.to_dict())
                    results.append({
                        "Age": row.get("Age"),
                        "Sex": row.get("Sex"),
                        "Prediction": res["prediction_label"],
                        "Risk Tier": res["risk_tier"],
                        "Probability (%)": res["probability_percent"]
                    })
                except Exception as e:
                    results.append({"Prediction": "Error", "Risk Tier": "N/A", "Probability (%)": 0})
            st.dataframe(pd.DataFrame(results), use_container_width=True)
