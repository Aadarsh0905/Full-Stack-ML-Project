---
title: CardioSense AI - Heart Disease Risk Predictor
emoji: 🫀
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# CardioSense AI: Full-Stack Heart Disease Risk Prediction Platform

An end-to-end clinical decision support platform built on the UCI Heart Disease dataset (918 patient cohort). Features a production machine learning pipeline, full model benchmarking suite, interactive medical dashboard, and RESTful inference API.

---

## 🔬 Audit of `Project2.ipynb`: Errors & Model Verification

### 1. Was Your "Best Model" Selection Correct?
**Answer: Partially correct in outcome, but arrived at through flawed methodology.**

In your original notebook, **Logistic Regression** achieved **87.07% accuracy** and was chosen as the champion model. However:
1. The score was computed on an evaluation that suffered from **test data leakage** and **severe feature truncation**.
2. Only a single arbitrary train/test split (`test_size=0.320`) without stratification was tested.
3. Modern tree ensembles (**Random Forest**, **Gradient Boosting**) were either not tested or unpruned (single Decision Tree).

#### Results After Fixing All Errors & Hyperparameter Tuning (Stratified 5-Fold CV):
| Model Algorithm | Test Accuracy | Recall (Sensitivity) | Precision | F1-Score | ROC-AUC | Optimal Hyperparameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Forest (Champion)** | **89.67%** | **93.14%** | 87.88% | **0.9091** | **0.9322** | `n_estimators=200`, `min_samples_split=5` |
| **Logistic Regression (Runner-up)**| 88.59% | 91.18% | 89.42% | 0.8986 | 0.9327 | `C=5.0`, `solver='liblinear'` |
| **Gradient Boosting** | 88.59% | 89.22% | 89.00% | 0.8966 | 0.9240 | `n_estimators=100`, `learning_rate=0.03`, `max_depth=3` |
| **Support Vector Machine (RBF)** | 87.50% | 91.18% | 86.41% | 0.8900 | 0.9393 | `C=2.0`, `gamma='auto'` |
| **K-Nearest Neighbors** | 87.50% | 87.25% | 87.25% | 0.8856 | 0.9331 | `n_neighbors=11`, `weights='distance'` |
| **Naive Bayes** | 83.70% | 84.31% | 90.72% | 0.8515 | 0.8863 | `var_smoothing=1e-09` |

> **Key Clinical Takeaway**: In cardiovascular diagnostics, **Recall / Sensitivity** is the most critical clinical metric because a false negative (failing to diagnose a heart disease patient) carries life-threatening risks. The corrected **Random Forest Classifier** achieved **93.14% Recall**, successfully detecting 95 out of 102 heart disease patients in the held-out test cohort, while **Logistic Regression** remains an excellent, highly interpretable alternative (91.18% Recall).

---

### 2. All Errors Identified in `Project2.ipynb`

#### ❌ Error 1: Test Data Leakage & Saving the Test Scaler
* **Code in Cell 34**:
  ```python
  scaler = StandardScaler()
  x_train_scaled = scaler.fit_transform(X_train)
  x_test_scaled = scaler.fit_transform(X_test)  # <-- BUG
  ...
  joblib.dump(scaler, 'scaler.pkl')             # <-- Corrupted artifact
  ```
* **Why it's an error**: `scaler.fit_transform(X_test)` recomputes mean and variance using the test set, violating the premise of simulating unseen data. Even worse, `scaler.pkl` saved the test set's distribution parameters.
* **The Fix**: Use `scaler.transform(X_test)`. In the full-stack app, this is bundled into a scikit-learn `Pipeline` to prevent data leakage.

#### ❌ Error 2: Truncation of Continuous Features via `astype(int)`
* **Code in Cell 26**:
  ```python
  df_encoded = df_encoded.astype(int)
  ```
* **Why it's an error**: The feature `Oldpeak` (ST depression) is a continuous float (`0.0, 1.5, 2.4, 3.4, -2.6`). Casting to `int` truncated all decimals (`1.5 -> 1`, `2.8 -> 2`), discarding crucial diagnostic resolution.
* **The Fix**: Preserve `Oldpeak` as `float64` and avoid blanket type casting.

#### ❌ Error 3: Zero-Imputation Data Leakage Before Train/Test Split
* **Code in Cells 12-15**:
  ```python
  ch_mean = df.loc[df['Cholesterol'] != 0, 'Cholesterol'].mean()
  df['Cholesterol'] = df['Cholesterol'].replace(0, ch_mean)
  ```
* **Why it's an error**: 0 values in `Cholesterol` (172 patients) and `RestingBP` (1 patient) represent missing values. Calculating the replacement mean across the entire dataset before splitting causes test distribution info to leak into the training set.
* **The Fix**: Convert 0s to `np.nan` and use `SimpleImputer(strategy='median')` fitted exclusively on training data.

#### ❌ Error 4: Global `pd.get_dummies()` and Fragile Inferences
* **Code in Cell 24**:
  ```python
  df_encoded = pd.get_dummies(df, drop_first=True)
  ```
* **Why it's an error**: `pd.get_dummies()` does not retain category definitions. If a single incoming request is missing one category or contains an unseen level, column dimensions mismatch and crash.
* **The Fix**: Production `OneHotEncoder(drop='first', handle_unknown='ignore')` within `ColumnTransformer`.

#### ❌ Error 5: Unstratified Single Split Without K-Fold Cross Validation
* **Code in Cell 33**:
  ```python
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.320, random_state=42)
  ```
* **Why it's an error**: An unstratified 32% test size split can easily yield misleading accuracy figures. No hyperparameter tuning was conducted.
* **The Fix**: 5-Fold Stratified Cross Validation with `GridSearchCV`.

---

## 🏛️ Full Stack Project Architecture

```
Full Stack ML Project/
├── app.py                      # Flask Application (Web UI + RESTful API)
├── run.py                      # Production startup entrypoint
├── heart.csv                   # UCI Heart Disease Dataset (918 rows)
├── requirements.txt            # Python dependencies
├── src/
│   ├── train_pipeline.py       # Production training pipeline & artifact builder
│   ├── predictor.py            # Prediction service with risk tiers & factor analysis
│   └── validator.py            # Strict data validation & bounds checking
├── models/
│   ├── heart_disease_pipeline.joblib  # Production end-to-end pipeline artifact
│   ├── metadata.json           # Model configuration, features & weights
│   └── benchmark_metrics.json  # Comprehensive 6-model benchmark scores
├── templates/
│   ├── index.html              # Diagnostic Assessment Dashboard & Presets
│   ├── models.html             # Model Benchmarking & Error Audit Page
│   └── api_docs.html           # Interactive REST API Documentation
├── static/
│   ├── css/style.css           # Modern clinical glassmorphism design system
│   └── js/main.js              # Client-side validation, AJAX & UI animations
└── tests/
    ├── test_model.py           # Unit tests for ML pipeline & predictions
    └── test_api.py             # Integration tests for Web & API endpoints
```

---

## 🚀 Quickstart: Running the Application

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Web Application
```bash
python run.py
```
Open your browser at **[http://127.0.0.1:5000](http://127.0.0.1:5000)**.

### 3. Run Automated Tests
```bash
pytest tests/
```
All 13 unit and integration tests pass verifying model inference, validation bounds, and API responses.

---

## 🔌 RESTful API Reference

### Health Check
```bash
curl -X GET http://127.0.0.1:5000/api/health
```

### Single Patient Risk Prediction
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 62,
    "Sex": "M",
    "ChestPainType": "ASY",
    "RestingBP": 150,
    "Cholesterol": 280,
    "FastingBS": 1,
    "RestingECG": "ST",
    "MaxHR": 115,
    "ExerciseAngina": "Y",
    "Oldpeak": 2.2,
    "ST_Slope": "Flat"
  }'
```

#### Sample Response:
```json
{
  "success": true,
  "data": {
    "prediction": 1,
    "prediction_label": "Heart Disease Detected",
    "probability_percent": 97.4,
    "risk_tier": "HIGH",
    "badge_color": "danger",
    "clinical_summary": "High probability of coronary heart disease detected by diagnostic indicators.",
    "action_plan": "Immediate cardiologist consultation, diagnostic echocardiogram or coronary angiography strongly advised.",
    "risk_factors": [
      "Flat ST Slope during peak exercise (common sign of myocardial ischemia)",
      "Exercise-induced angina present (chest pain during physical exertion)",
      "Asymptomatic chest pain type (historically high correlation with silent CAD)",
      "Elevated ST depression Oldpeak (2.2 mm) indicates significant ischemic response"
    ],
    "protective_factors": [],
    "model_name": "Random Forest"
  }
}
```

### Batch Patient Screening
```bash
curl -X POST http://127.0.0.1:5000/api/batch-predict \
  -F "file=@heart.csv"
```
