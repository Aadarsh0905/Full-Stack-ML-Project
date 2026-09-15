"""
Data validation module for heart disease prediction inputs.
"""

from typing import Dict, Any, Tuple, List

VALID_OPTIONS = {
    "Sex": ["M", "F"],
    "ChestPainType": ["ATA", "NAP", "ASY", "TA"],
    "RestingECG": ["Normal", "ST", "LVH"],
    "ExerciseAngina": ["N", "Y"],
    "ST_Slope": ["Up", "Flat", "Down"]
}

NUMERICAL_RANGES = {
    "Age": (18, 110, "Age must be between 18 and 110 years"),
    "RestingBP": (60, 250, "Resting Blood Pressure must be between 60 and 250 mm Hg"),
    "Cholesterol": (0, 700, "Cholesterol must be between 0 and 700 mm/dl (0 if unknown)"),
    "FastingBS": (0, 1, "Fasting Blood Sugar must be 0 (<120 mg/dl) or 1 (>120 mg/dl)"),
    "MaxHR": (50, 230, "Maximum Heart Rate must be between 50 and 230 bpm"),
    "Oldpeak": (-3.0, 7.0, "Oldpeak (ST depression) must be between -3.0 and 7.0")
}


def validate_patient_data(data: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validates patient data dictionary.
    Returns:
        (is_valid, errors_list, sanitized_data)
    """
    errors = []
    sanitized = {}

    # Numerical validation
    for field, (min_val, max_val, err_msg) in NUMERICAL_RANGES.items():
        if field not in data or data[field] is None or str(data[field]).strip() == "":
            errors.append(f"Missing required field: '{field}'")
            continue
        try:
            val = float(data[field])
            if field in ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR"]:
                val = int(round(val))
            if not (min_val <= val <= max_val):
                errors.append(f"Invalid '{field}': {val}. {err_msg}")
            sanitized[field] = val
        except (ValueError, TypeError):
            errors.append(f"'{field}' must be a numeric value, got '{data[field]}'")

    # Categorical validation
    for field, valid_choices in VALID_OPTIONS.items():
        if field not in data or data[field] is None:
            errors.append(f"Missing required field: '{field}'")
            continue
        val = str(data[field]).strip()
        if val not in valid_choices:
            errors.append(f"Invalid '{field}': '{val}'. Must be one of {valid_choices}")
        sanitized[field] = val

    is_valid = len(errors) == 0
    return is_valid, errors, sanitized
