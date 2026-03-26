import pandas as pd
import numpy as np
import os


# ----------------------
# Helper Functions: Unit Recognition & Conversion
# ----------------------
def mmolL_to_mgDL(mmolL_value):
    """Cholesterol unit conversion: mmol/L → mg/dL"""
    return mmolL_value * 38.67


def detect_cholesterol_unit(values):
    """Auto-detect cholesterol units (mmol/L or mg/dL)"""
    valid_values = values.dropna()
    if len(valid_values) == 0:
        return "unknown"
    median_val = valid_values.median()
    return "mg/dL" if median_val > 20 else "mmol/L"


# ----------------------
# Official Framingham Risk Calculation Function (Verified)
# ----------------------
def calculate_framingham_cvd_official(
        age,
        gender,  # 'male'/'female'
        total_cholesterol,  # mg/dL
        hdl_cholesterol,  # mg/dL
        systolic_bp,  # mmHg
        is_treated_bp,  # True/False
        is_smoker,  # True/False
        is_diabetic  # True/False
):
    """
    Calculates 10-year total CVD risk based on official Framingham regression coefficients.
    Coefficients are aligned with the official published data.
    """
    # Age range validation
    if not (30 <= age <= 79) or pd.isna(age):
        return np.nan
    # Core parameter null check
    if pd.isna(total_cholesterol) or pd.isna(hdl_cholesterol) or pd.isna(systolic_bp):
        return np.nan

    # Coefficients per gender (Official Data)
    if gender == 'male':
        # Male coefficients - Official Data
        coefficients = {
            'log_age': 3.06117,
            'log_total_chol': 1.12370,
            'log_hdl_chol': -0.93263,
            'log_sbp_untreated': 1.93303,  # Untreated SBP coefficient
            'log_sbp_treated': 1.99881,    # Treated SBP coefficient
            'smoker': 0.65451,
            'diabetes': 0.57367
        }
        baseline_survival = 0.88936  # 10-year baseline survival
        constant = 23.9802           # Constant term
    elif gender == 'female':
        # Female coefficients - Official Data
        coefficients = {
            'log_age': 2.32888,
            'log_total_chol': 1.20904,
            'log_hdl_chol': -0.70833,
            'log_sbp_untreated': 2.76157,  # Untreated SBP coefficient
            'log_sbp_treated': 2.82263,    # Treated SBP coefficient
            'smoker': 0.52873,
            'diabetes': 0.69154
        }
        baseline_survival = 0.95012  # 10-year baseline survival
        constant = 26.1931           # Constant term
    else:
        return np.nan

    # Ensure values are positive (avoid log errors)
    if age <= 0 or total_cholesterol <= 0 or hdl_cholesterol <= 0 or systolic_bp <= 0:
        return np.nan

    try:
        # Calculate Linear Predictor ΣßX - Official Formula
        linear_predictor = (
                coefficients['log_age'] * np.log(age) +
                coefficients['log_total_chol'] * np.log(total_cholesterol) +
                coefficients['log_hdl_chol'] * np.log(hdl_cholesterol) +
                (coefficients['log_sbp_treated'] if is_treated_bp else coefficients['log_sbp_untreated']) * np.log(
            systolic_bp) +
                coefficients['smoker'] * (1 if is_smoker else 0) +
                coefficients['diabetes'] * (1 if is_diabetic else 0)
        )

        # Calculate risk: 1 - (baseline_survival)^exp(ΣßX - constant) - Official Formula
        risk_score = 1 - (baseline_survival ** np.exp(linear_predictor - constant))
        return round(risk_score * 100, 2)
    except (ValueError, ZeroDivisionError, OverflowError):
        return np.nan


# ----------------------
# Validation Function: Testing Accuracy
# ----------------------
def validate_calculation():
    """Verify calculation accuracy"""
    print("\n🧪 Verifying calculation accuracy...")

    # Test Case 1: Typical Male Patient
    test_1 = calculate_framingham_cvd_official(
        age=55,
        gender='male',
        total_cholesterol=200,
        hdl_cholesterol=45,
        systolic_bp=140,
        is_treated_bp=False,
        is_smoker=True,
        is_diabetic=False
    )
    print(f"Test Case 1 - Male (55y, Smoker, Untreated): {test_1}%")

    # Test Case 2: Female with identical conditions
    test_2 = calculate_framingham_cvd_official(
        age=55,
        gender='female',
        total_cholesterol=200,
        hdl_cholesterol=45,
        systolic_bp=140,
        is_treated_bp=False,
        is_smoker=True,
        is_diabetic=False
    )
    print(f"Test Case 2 - Female (55y, Smoker, Untreated): {test_2}%")

    # Test Case 3: Treated vs. Untreated difference
    test_3_treated = calculate_framingham_cvd_official(
        age=55,
        gender='male',
        total_cholesterol=200,
        hdl_cholesterol=45,
        systolic_bp=140,
        is_treated_bp=True,  # Treated
        is_smoker=True,
        is_diabetic=False
    )
    print(f"Test Case 3 - Male (Treated): {test_3_treated}%")
    print(f"Difference (Treated vs Untreated): {test_3_treated - test_1:.2f}%")

    return test_1, test_2, test_3_treated


# ----------------------
# Data Loading and Batch Processing
# ----------------------
if __name__ == "__main__":
    # File Path
    username = 'Administrator'
    file_name = 'FRS.CSV'
    file_path = os.path.join(f'C:/Users/{username}/Desktop/', file_name)

    # Load data (handling multiple encodings + column stripping)
    try:
        for encoding in ['utf-8', 'gbk', 'utf-8-sig']:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"✅ Data imported successfully (Encoding: {encoding}), total {len(df)} rows, {len(df.columns)} columns")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Unable to recognize file encoding. Please check file format.")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        exit()

    # Display original column names
    print("📋 Original Column Names: ", df.columns.tolist())

    # Key Fix 1: Strip whitespace + normalize case
    df.columns = df.columns.str.strip().str.replace(' ', '').str.lower()
    print("📋 Processed Column Names (Stripped + Lowercase): ", df.columns.tolist())

    # Key Fix 2: Flexible column matching
    required_columns = {
        'age': ['age'],
        'sex': ['sex'],
        'hdl_c': ['hdl-c', 'hdl', 'hdlcholesterol'],
        'total_cholesterol': ['cholesterol', 'totalcholesterol'],
        'systolic_bp': ['systolic', 'systolicbloodpressure', 'bloodpressure'],
        'smoking_status': ['smoking', 'smoker', 'smokingstatus'],
        'diabetes': ['diabetes', 'diabetic'],
        'treated_for_htn': ['treatedforhtn', 'htn', 'hypertension']
    }

    # Match columns
    matched_columns = {}
    for standard_name, patterns in required_columns.items():
        matched_col = None
        for col in df.columns:
            for pattern in patterns:
                if pattern in col:
                    matched_col = col
                    break
            if matched_col:
                break
        if matched_col:
            matched_columns[standard_name] = matched_col
            print(f"✅ Matched column: {standard_name} -> {matched_col}")
        else:
            print(f"❌ Failed to find matching column for: {standard_name}")

    # Check if all required columns are found
    missing_columns = set(required_columns.keys()) - set(matched_columns.keys())
    if missing_columns:
        print(f"❌ Missing required columns: {missing_columns}")
        print("Please verify your data contains columns for: age, sex, hdl-c, total cholesterol, systolic bp, smoking, diabetes, and htn treatment.")
        exit()

    # Rename columns for internal processing
    for standard_name, original_name in matched_columns.items():
        df[standard_name] = df[original_name]

    print("✅ All required columns successfully matched")

    # Data Conversion
    # Gender mapping
    df['gender'] = df['sex'].map({
        'm': 'male', 'f': 'female',
        '男': 'male', '女': 'female',
        1: 'male', 0: 'female',
        'male': 'male', 'female': 'female'
    }).fillna(np.nan)

    # Boolean mapping
    bool_map = {
        1: True, 0: False,
        'yes': True, 'no': False,
        '是': True, '否': False,
        'true': True, 'false': False,
        'current': True, 'never': False, 'former': False,
        'yes,current': True, 'no,never': False
    }

    # Handle Smoking Status
    df['is_smoker'] = df['smoking_status'].map(bool_map).fillna(False)
    if df['is_smoker'].isna().any():
        df['is_smoker'] = df['smoking_status'].astype(str).str.contains('current|yes|is_smoker|smoking', case=False, na=False)

    df['is_diabetic'] = df['diabetes'].map(bool_map).fillna(False)
    df['is_treated_bp'] = df['treated_for_htn'].map(bool_map).fillna(False)

    # Automatic Cholesterol Unit Conversion
    total_chol = df['total_cholesterol']
    hdl_chol = df['hdl_c']
    total_chol_unit = detect_cholesterol_unit(total_chol)
    hdl_chol_unit = detect_cholesterol_unit(hdl_chol)
    print(f"📏 Auto-detection: Total Cholesterol unit={total_chol_unit}, HDL-C unit={hdl_chol_unit}")
    df['total_cholesterol_mgdl'] = mmolL_to_mgDL(total_chol) if total_chol_unit == 'mmol/L' else total_chol
    df['hdl_cholesterol_mgdl'] = mmolL_to_mgDL(hdl_chol) if hdl_chol_unit == 'mmol/L' else hdl_chol

    # Data Type Validation
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['systolic_bp'] = pd.to_numeric(df['systolic_bp'], errors='coerce')
    df['total_cholesterol_mgdl'] = pd.to_numeric(df['total_cholesterol_mgdl'], errors='coerce')
    df['hdl_cholesterol_mgdl'] = pd.to_numeric(df['hdl_cholesterol_mgdl'], errors='coerce')

    # Run validation test
    validate_calculation()

    # Batch calculation
    print("\n🔄 Starting 10-year total CVD risk calculation...")
    df['10_year_total_cvd_risk(%)'] = df.apply(
        lambda row: calculate_framingham_cvd_official(
            age=row['age'],
            gender=row['gender'],
            total_cholesterol=row['total_cholesterol_mgdl'],
            hdl_cholesterol=row['hdl_cholesterol_mgdl'],
            systolic_bp=row['systolic_bp'],
            is_treated_bp=row['is_treated_bp'],
            is_smoker=row['is_smoker'],
            is_diabetic=row['is_diabetic']
        ),
        axis=1
    )

    # Save Results
    output_path = os.path.join(f'C:/Users/{username}/Desktop/', 'FRS_Results_Verified.csv')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    # Output Statistics
    valid_risk = df['10_year_total_cvd_risk(%)'].dropna()
    print(f"\n✅ Processing Complete! Results saved to: {output_path}")
    print(f"📊 Summary Statistics:")
    print(f"   - Total samples: {len(df)}")
    print(f"   - Successfully calculated: {len(valid_risk)}")

    if len(valid_risk) > 0:
        print(f"   - Risk range: {valid_risk.min():.2f}% ~ {valid_risk.max():.2f}%")
        print(f"   - Average risk: {valid_risk.mean():.2f}%")
        print(f"   - Median risk: {valid_risk.median():.2f}%")

        # Risk Stratification Statistics
        low_risk = valid_risk[valid_risk < 10].count()
        medium_risk = valid_risk[(valid_risk >= 10) & (valid_risk < 20)].count()
        high_risk = valid_risk[valid_risk >= 20].count()

        print(f"   - Low Risk (<10%): {low_risk} ({low_risk / len(valid_risk) * 100:.1f}%)")
        print(f"   - Medium Risk (10-20%): {medium_risk} ({medium_risk / len(valid_risk) * 100:.1f}%)")
        print(f"   - High Risk (≥20%): {high_risk} ({high_risk / len(valid_risk) * 100:.1f}%)")
    else:
        print("   - No successful risk values calculated. Please verify input data.")

    # Preview results
    print("\n📈 Result Preview (First 5 rows):")
    preview_cols = ['age', 'gender', '10_year_total_cvd_risk(%)', 'is_smoker', 'is_diabetic', 'is_treated_bp']
    available_cols = [col for col in preview_cols if col in df.columns]
    print(df[available_cols].head())