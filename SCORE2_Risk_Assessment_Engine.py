import pandas as pd
import math
import os
import numpy as np


def cholesterol_mgdl_to_mmol(mgdl):
    """Convert cholesterol from mg/dL to mmol/L"""
    return mgdl / 38.67


def get_score_type(age, has_diabetes, cvd):
    """
    Determine applicable SCORE2 model type.
    Adjusted for age range 40-70.
    """
    if cvd:
        return "Established_CVD"
    elif age >= 70:
        return "SCORE2_OP"
    elif age >= 40 and age < 70:
        return "SCORE2"
    else:
        # Patients 39 and below are not applicable for standard SCORE2 models
        return None


def calculate_score2_with_diagnosis(
        sex,
        age,
        smoker,
        systolic_blood_pressure,
        total_cholesterol,
        hdl_cholesterol,
        region_risk,
        cvd,
):
    """
    Calculate SCORE2 Risk Score - with diagnostic info
    """
    diagnosis_info = []

    # Diagnostic info collection
    if cvd:
        diagnosis_info.append("CVD Patient, skipping calculation")
        return None, diagnosis_info

    # Age range validation
    if age < 40 or age > 70:
        diagnosis_info.append(f"Age {age} outside 40-70 range")
        return None, diagnosis_info

    if pd.isna(total_cholesterol) or total_cholesterol < 1.3:
        diagnosis_info.append(f"Abnormal Total Cholesterol: {total_cholesterol}")
        return None, diagnosis_info

    # Missing variable check
    missing_vars = []
    if pd.isna(sex): missing_vars.append("Sex")
    if pd.isna(age): missing_vars.append("Age")
    if pd.isna(systolic_blood_pressure): missing_vars.append("SBP")
    if pd.isna(total_cholesterol): missing_vars.append("Total_Chol")
    if pd.isna(hdl_cholesterol): missing_vars.append("HDL")

    if missing_vars:
        diagnosis_info.append(f"Missing variables: {', '.join(missing_vars)}")
        return None, diagnosis_info

    # Range validation (mmol/L)
    if total_cholesterol < 1.3 or total_cholesterol > 10.3:
        diagnosis_info.append(f"Total Cholesterol {total_cholesterol} out of range (1.3-10.3 mmol/L)")
        return None, diagnosis_info
    if hdl_cholesterol < 0.26 or hdl_cholesterol > 3.88:
        diagnosis_info.append(f"HDL {hdl_cholesterol} out of range (0.26-3.88 mmol/L)")
        return None, diagnosis_info
    if systolic_blood_pressure < 50 or systolic_blood_pressure > 300:
        diagnosis_info.append(f"SBP {systolic_blood_pressure} out of range (50-300 mmHg)")
        return None, diagnosis_info

    score_type = get_score_type(age, False, cvd)

    if pd.isna(score_type) or score_type not in ["SCORE2", "SCORE2_OP"]:
        diagnosis_info.append(f"No applicable model: {score_type}")
        return None, diagnosis_info

    # SCORE2 Coefficients (Moderate Risk Region)
    coefficients = {
        "SCORE2": {
            "Homem": {
                "Age": 0.3742, "Smoking": 0.6012, "SBP": 0.2777, "Total cholesterol": 0.1458,
                "HDL": -0.2698, "Smoking_Age": -0.0755, "SBP_age": -0.0255,
                "Total cholesterol_age": -0.0281, "HDL_age": 0.0426, "Baseline_survival": 0.9605,
            },
            "Mulher": {
                "Age": 0.4648, "Smoking": 0.7744, "SBP": 0.3131, "Total cholesterol": 0.1002,
                "HDL": -0.2606, "Smoking_Age": -0.1088, "SBP_age": -0.0277,
                "Total cholesterol_age": -0.0226, "HDL_age": 0.0613, "Baseline_survival": 0.9776,
            },
        },
        "SCORE2_OP": {
            "Homem": {
                "Age": 0.0634, "Diabetes": 0.4245, "Smoking": 0.3524, "SBP": 0.0094,
                "Total cholesterol": 0.0850, "HDL": -0.3564, "Diabetes_Age": -0.0174,
                "Smoking_Age": -0.0247, "SBP_age": -0.0005, "Total cholesterol_age": 0.0073,
                "HDL_age": 0.0091, "Baseline_survival": 0.7576, "Mean linear predictor": 0.0929,
            },
            "Mulher": {
                "Age": 0.0789, "Diabetes": 0.6010, "Smoking": 0.4921, "SBP": 0.0102,
                "Total cholesterol": 0.0605, "HDL": -0.3040, "Diabetes_Age": -0.0107,
                "Smoking_Age": -0.0255, "SBP_age": -0.0004, "Total cholesterol_age": -0.0009,
                "HDL_age": 0.0154, "Baseline_survival": 0.8082, "Mean linear predictor": 0.2290,
            },
        },
    }

    region_risk_coeficients = {
        "SCORE2": {
            "Homem": {
                "Low": {"Scale1": -0.5699, "Scale2": 0.7476},
                "Moderate": {"Scale1": -0.1565, "Scale2": 0.8009},
                "High": {"Scale1": 0.3207, "Scale2": 0.9360},
                "Very high": {"Scale1": 0.5836, "Scale2": 0.8294},
            },
            "Mulher": {
                "Low": {"Scale1": -0.7380, "Scale2": 0.7019},
                "Moderate": {"Scale1": -0.3143, "Scale2": 0.7701},
                "High": {"Scale1": 0.5710, "Scale2": 0.9369},
                "Very high": {"Scale1": 0.9369, "Scale2": 0.8329},
            },
        },
        "SCORE2_OP": {
            "Homem": {
                "Low": {"Scale1": -0.34, "Scale2": 1.19},
                "Moderate": {"Scale1": 0.01, "Scale2": 1.25},
                "High": {"Scale1": 0.08, "Scale2": 1.15},
                "Very high": {"Scale1": 0.05, "Scale2": 0.70},
            },
            "Mulher": {
                "Low": {"Scale1": -0.52, "Scale2": 1.01},
                "Moderate": {"Scale1": -0.10, "Scale2": 1.10},
                "High": {"Scale1": 0.38, "Scale2": 1.09},
                "Very high": {"Scale1": 0.38, "Scale2": 0.69},
            },
        },
    }

    ten_year_risk = None

    try:
        if score_type == "SCORE2":
            cage = (age - 60) / 5
            smoking = 1 if smoker else 0
            csbp = (systolic_blood_pressure - 120) / 20
            ctchol = total_cholesterol - 6
            chdl = (hdl_cholesterol - 1.3) / 0.5
            x = (
                    coefficients[score_type][sex]["Age"] * cage
                    + coefficients[score_type][sex]["Smoking"] * smoking
                    + coefficients[score_type][sex]["SBP"] * csbp
                    + coefficients[score_type][sex]["Total cholesterol"] * ctchol
                    + coefficients[score_type][sex]["HDL"] * chdl
                    + coefficients[score_type][sex]["Smoking_Age"] * (smoking * cage)
                    + coefficients[score_type][sex]["SBP_age"] * (csbp * cage)
                    + coefficients[score_type][sex]["Total cholesterol_age"] * (ctchol * cage)
                    + coefficients[score_type][sex]["HDL_age"] * (chdl * cage)
            )
            ten_year_risk = 1 - (coefficients[score_type][sex]["Baseline_survival"]) ** math.exp(x)

        elif score_type == "SCORE2_OP":
            cage = age - 73
            smoking = 1 if smoker else 0
            csbp = systolic_blood_pressure - 150
            ctchol = total_cholesterol - 6
            chdl = hdl_cholesterol - 1.4
            x = (
                    coefficients[score_type][sex]["Age"] * cage
                    + coefficients[score_type][sex]["Diabetes"] * 0
                    + coefficients[score_type][sex]["Smoking"] * smoking
                    + coefficients[score_type][sex]["SBP"] * csbp
                    + coefficients[score_type][sex]["Total cholesterol"] * ctchol
                    + coefficients[score_type][sex]["HDL"] * chdl
                    + coefficients[score_type][sex]["Diabetes_Age"] * (0 * cage)
                    + coefficients[score_type][sex]["Smoking_Age"] * (smoking * cage)
                    + coefficients[score_type][sex]["SBP_age"] * (csbp * cage)
                    + coefficients[score_type][sex]["Total cholesterol_age"] * (ctchol * cage)
                    + coefficients[score_type][sex]["HDL_age"] * (chdl * cage)
            )
            ten_year_risk = 1 - (coefficients[score_type][sex]["Baseline_survival"]) ** math.exp(
                x - coefficients[score_type][sex]["Mean linear predictor"])

        if pd.isna(ten_year_risk):
            diagnosis_info.append("Calculated 10-year risk is NaN")
            return None, diagnosis_info

        calibrated_risk = (1 - math.exp(-math.exp(
            region_risk_coeficients[score_type][sex][region_risk]["Scale1"]
            + region_risk_coeficients[score_type][sex][region_risk]["Scale2"]
            * math.log(-math.log(1 - ten_year_risk))
        ))) * 100

        diagnosis_info.append("Calculation successful")
        return round(calibrated_risk, 2), diagnosis_info

    except Exception as e:
        diagnosis_info.append(f"Error during calculation: {str(e)}")
        return None, diagnosis_info


def interpret_risk(cvd, score, score_type, age):
    """Risk stratification interpretation"""
    if cvd:
        return "Very High"
    if pd.isna(score):
        return None

    if score_type == "SCORE2":
        thresholds = [2.5, 7.5] if age < 50 else [5, 10]
        if score < thresholds[0]:
            return "Low to Moderate"
        elif score < thresholds[1]:
            return "High"
        else:
            return "Very High"
    elif score_type == "SCORE2_OP":
        if score < 7.5:
            return "Low to Moderate"
        elif score < 15:
            return "High"
        else:
            return "Very High"
    return None


def calculate_all_score2_models_with_diagnosis(df, region_risk="Low"):
    """
    Calculate SCORE2 models for all patients with diagnostic info.
    Defaulting to Low risk for UKB data.
    """
    df_result = df.copy()
    df_result["Sex_Converted"] = df_result["Sexo"].map({1: "Homem", 0: "Mulher"})
    df_result["Applied_Model"] = None
    df_result["SCORE2_Recalculated"] = None
    df_result["CV_Risk_Recalculated"] = None
    df_result["LDL_Target_Recalculated"] = None
    df_result["Calculation_Diagnosis"] = None

    stats = {"Success": 0, "Age_Range_Error": 0, "Missing_Var": 0, "Out_of_Range": 0, "Not_Applicable": 0,
             "Calc_Error": 0}

    for idx, patient in df_result.iterrows():
        score_type = get_score_type(patient["Idade"], False, False)
        df_result.at[idx, "Applied_Model"] = score_type

        if score_type in ["SCORE2", "SCORE2_OP"]:
            score, diag_info = calculate_score2_with_diagnosis(
                sex=patient["Sex_Converted"], age=patient["Idade"], smoker=patient["Fumador"] == 1,
                systolic_blood_pressure=patient["TAs"], total_cholesterol=patient["Colesterol_Total"],
                hdl_cholesterol=patient["HDL"], region_risk=region_risk, cvd=False
            )
            diag_text = "; ".join(diag_info)
            df_result.at[idx, "Calculation_Diagnosis"] = diag_text

            if "successful" in diag_text:
                stats["Success"] += 1
            elif "Age" in diag_text:
                stats["Age_Range_Error"] += 1
            elif "Missing" in diag_text:
                stats["Missing_Var"] += 1
            elif "range" in diag_text:
                stats["Out_of_Range"] += 1
            elif "Error" in diag_text:
                stats["Calc_Error"] += 1

            df_result.at[idx, "SCORE2_Recalculated"] = score
            if score is not None:
                risk = interpret_risk(False, score, score_type, patient["Idade"])
                df_result.at[idx, "CV_Risk_Recalculated"] = risk
                df_result.at[idx, "LDL_Target_Recalculated"] = {"Low to Moderate": 100, "High": 70,
                                                                "Very High": 55}.get(risk, 100)
        else:
            stats["Not_Applicable"] += 1
            df_result.at[idx, "Calculation_Diagnosis"] = "Not applicable for SCORE2 models"

    return df_result, stats


def batch_calculate_all_models(input_file, output_file, region_risk="Low"):
    """Batch calculation for SCORE2 models with full diagnostics"""
    try:
        print("Reading CSV with comma separator...")
        df = pd.read_csv(input_file, sep=',', encoding='utf-8')
        print(f"Loaded {len(df)} records. Columns: {list(df.columns)}")
    except Exception as e:
        print(f"Initial read failed: {e}. Attempting with automatic separator detection...")
        df = pd.read_csv(input_file, sep=None, engine='python', encoding='utf-8')

    required_columns = ['Participant ID', 'Idade', 'Sexo', 'HDL', 'Colesterol_Total', 'TAs', 'Fumador']
    # Mapping logic remains same to ensure robust column matching

    print("\nStarting SCORE2 risk calculation...")
    results, stats = calculate_all_score2_models_with_diagnosis(df, region_risk)

    print("\n" + "=" * 60)
    print("SCORE2 Calculation Diagnostic Statistics")
    print("=" * 60)
    for reason, count in stats.items():
        print(f"  {reason}: {count} patients ({(count / len(results) * 100):.2f}%)")

    results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to: {output_file}")
    return results


if __name__ == "__main__":
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    input_file = os.path.join(desktop_path, "SCORE2.csv")
    output_file = os.path.join(desktop_path, "SCORE2_Calculated_Results.csv")

    print("=== SCORE2 Cardiovascular Risk Calculator (Diagnostic Version) ===")
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
    else:
        batch_calculate_all_models(input_file, output_file)