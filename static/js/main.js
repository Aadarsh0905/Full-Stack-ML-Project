/**
 * CardioSense AI - Frontend Interactivity & API Client
 */

// Presets for rapid clinical testing
const PRESETS = {
  healthy: {
    Age: 36,
    Sex: "F",
    ChestPainType: "ATA",
    RestingBP: 118,
    Cholesterol: 185,
    FastingBS: 0,
    RestingECG: "Normal",
    MaxHR: 172,
    ExerciseAngina: "N",
    Oldpeak: 0.0,
    ST_Slope: "Up"
  },
  moderate: {
    Age: 52,
    Sex: "M",
    ChestPainType: "NAP",
    RestingBP: 138,
    Cholesterol: 235,
    FastingBS: 0,
    RestingECG: "Normal",
    MaxHR: 142,
    ExerciseAngina: "N",
    Oldpeak: 1.0,
    ST_Slope: "Flat"
  },
  high_risk: {
    Age: 63,
    Sex: "M",
    ChestPainType: "ASY",
    RestingBP: 165,
    Cholesterol: 295,
    FastingBS: 1,
    RestingECG: "ST",
    MaxHR: 108,
    ExerciseAngina: "Y",
    Oldpeak: 2.8,
    ST_Slope: "Flat"
  }
};

function loadPreset(presetKey) {
  const data = PRESETS[presetKey];
  if (!data) return;

  for (const [key, value] of Object.entries(data)) {
    const el = document.getElementById(key);
    if (el) {
      el.value = value;
    }
  }

  // Trigger quick subtle highlight
  const formCard = document.getElementById("patientFormCard");
  if (formCard) {
    formCard.style.boxShadow = "0 0 25px rgba(56, 189, 248, 0.4)";
    setTimeout(() => {
      formCard.style.boxShadow = "";
    }, 400);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predictionForm");
  const resultCard = document.getElementById("resultCard");
  const placeholder = document.getElementById("resultPlaceholder");
  const resultContent = document.getElementById("resultContent");
  const errorAlert = document.getElementById("errorAlert");
  const submitBtn = document.getElementById("submitBtn");

  if (form) {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();

      // Clear previous error
      if (errorAlert) {
        errorAlert.style.display = "none";
        errorAlert.innerHTML = "";
      }

      // Collect data
      const formData = new FormData(form);
      const payload = {};
      formData.forEach((value, key) => {
        payload[key] = value;
      });

      // UI Loading state
      submitBtn.disabled = true;
      submitBtn.innerHTML = `
        <svg class="spinner" width="20" height="20" viewBox="0 0 50 50" style="animation: spin 1s linear infinite;">
          <circle cx="25" cy="25" r="20" fill="none" stroke="currentColor" stroke-width="5" stroke-dasharray="31.4 31.4"></circle>
        </svg>
        Analyzing Cardiovascular Biomarkers...
      `;

      try {
        const response = await fetch("/api/predict", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
          throw new Error(data.errors ? data.errors.join("<br>") : (data.error || "Prediction failed"));
        }

        renderPredictionResult(data.data);
      } catch (err) {
        if (errorAlert) {
          errorAlert.style.display = "block";
          errorAlert.innerHTML = `<strong>Validation Error:</strong><br>${err.message}`;
        }
      } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = `
          <svg width="20" height="20" fill="currentColor" viewBox="0 0 20 20">
            <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-8.707l-3-3a1 1 0 00-1.414 1.414L10.586 9H7a1 1 0 100 2h3.586l-1.293 1.293a1 1 0 101.414 1.414l3-3a1 1 0 000-1.414z" clip-rule="evenodd"></path>
          </svg>
          Run Diagnostic Assessment
        `;
      }
    });
  }

  // Batch CSV upload handler
  const batchForm = document.getElementById("batchUploadForm");
  if (batchForm) {
    batchForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const fileInput = document.getElementById("batchFile");
      if (!fileInput.files.length) {
        alert("Please select a CSV file first.");
        return;
      }

      const fd = new FormData();
      fd.append("file", fileInput.files[0]);

      const batchStatus = document.getElementById("batchStatus");
      const batchTableContainer = document.getElementById("batchResultsContainer");
      const batchTbody = document.getElementById("batchResultsBody");

      batchStatus.textContent = "Processing CSV batch...";
      batchStatus.style.display = "block";

      try {
        const res = await fetch("/api/batch-predict", {
          method: "POST",
          body: fd
        });
        const result = await res.json();
        if (!res.ok || !result.success) {
          throw new Error(result.error || "Batch processing failed.");
        }

        batchStatus.textContent = `Completed! Processed ${result.total_processed} patients. (${result.successful_predictions} successful)`;
        batchTbody.innerHTML = "";

        result.results.slice(0, 50).forEach((r) => {
          const tr = document.createElement("tr");
          if (r.success) {
            const badgeClass = r.risk_tier === "HIGH" ? "badge-high" : (r.risk_tier === "MODERATE" ? "badge-mod" : "badge-low");
            tr.innerHTML = `
              <td>#${r.index + 1}</td>
              <td><strong>${r.prediction_label}</strong></td>
              <td>${r.probability_percent}%</td>
              <td><span class="risk-badge ${badgeClass}" style="font-size:0.75rem; padding:0.2rem 0.5rem;">${r.risk_tier}</span></td>
              <td style="font-size:0.8rem; color:#94a3b8;">${r.summary}</td>
            `;
          } else {
            tr.innerHTML = `
              <td>#${r.index + 1}</td>
              <td colspan="4" style="color:#ef4444; font-size:0.8rem;">Validation Failed: ${r.errors.join(", ")}</td>
            `;
          }
          batchTbody.appendChild(tr);
        });

        batchTableContainer.style.display = "block";
      } catch (e) {
        batchStatus.textContent = "Error: " + e.message;
      }
    });
  }
});

function renderPredictionResult(res) {
  const placeholder = document.getElementById("resultPlaceholder");
  const resultContent = document.getElementById("resultContent");

  if (placeholder) placeholder.style.display = "none";
  if (resultContent) resultContent.style.display = "block";

  // Score display
  const scoreValue = document.getElementById("scoreValue");
  const scoreBadge = document.getElementById("scoreBadge");
  const summaryText = document.getElementById("clinicalSummaryText");
  const actionText = document.getElementById("actionPlanText");
  const riskList = document.getElementById("riskFactorsList");
  const protectList = document.getElementById("protectiveFactorsList");

  if (scoreValue) {
    scoreValue.textContent = `${res.probability_percent}%`;
    if (res.risk_tier === "HIGH") {
      scoreValue.style.color = "var(--risk-high)";
    } else if (res.risk_tier === "MODERATE") {
      scoreValue.style.color = "var(--risk-mod)";
    } else {
      scoreValue.style.color = "var(--risk-low)";
    }
  }

  if (scoreBadge) {
    scoreBadge.textContent = `${res.risk_tier} RISK TIER • ${res.prediction_label.toUpperCase()}`;
    scoreBadge.className = `risk-badge ${res.risk_tier === "HIGH" ? "badge-high" : (res.risk_tier === "MODERATE" ? "badge-mod" : "badge-low")}`;
  }

  if (summaryText) summaryText.textContent = res.clinical_summary;
  if (actionText) actionText.textContent = res.action_plan;

  // Render Risk Factors
  if (riskList) {
    riskList.innerHTML = "";
    if (res.risk_factors.length === 0) {
      riskList.innerHTML = `<li><span style="color:#10b981;">✓</span> No severe clinical warning flags identified.</li>`;
    } else {
      res.risk_factors.forEach((factor) => {
        const li = document.createElement("li");
        li.innerHTML = `<span style="color:#ef4444;">⚠️</span> <span>${factor}</span>`;
        riskList.appendChild(li);
      });
    }
  }

  // Render Protective Factors
  if (protectList) {
    protectList.innerHTML = "";
    if (res.protective_factors.length === 0) {
      protectList.innerHTML = `<li><span style="color:#f59e0b;">•</span> Few protective indicators observed.</li>`;
    } else {
      res.protective_factors.forEach((factor) => {
        const li = document.createElement("li");
        li.innerHTML = `<span style="color:#10b981;">🛡️</span> <span>${factor}</span>`;
        protectList.appendChild(li);
      });
    }
  }

  // Smooth scroll into view on mobile
  if (window.innerWidth < 960) {
    document.getElementById("resultCard").scrollIntoView({ behavior: "smooth" });
  }
}
