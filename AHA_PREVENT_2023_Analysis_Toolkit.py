import pandas as pd
import numpy as np
import math
import warnings

warnings.filterwarnings('ignore')

# Attempt to import Multiple Imputation libraries
try:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.preprocessing import LabelEncoder

    sklearn_available = True
except ImportError:
    sklearn_available = False

try:
    import miceforest as mf

    miceforest_available = True
except ImportError:
    miceforest_available = False

# Coefficient tables - Total CVD and ASCVD
coefficients = {
    'Total CVD': {
        'Female': {
            'C0': 0.7939329, 'C1': 0, 'C2': 0.0305239, 'C3': -0.1606857, 'C4': -0.2394003,
            'C5': 0.3600781, 'C6': 0.8667604, 'C7': 0.5360739, 'C8': 0, 'C9': 0,
            'C10': 0.6045917, 'C11': 0.0433769, 'C12': 0.3151672, 'C13': -0.1477655,
            'C14': -0.0663612, 'C15': 0.1197879, 'C16': -0.0819715, 'C17': 0.0306769,
            'C18': -0.0946348, 'C19': -0.27057, 'C20': -0.078715, 'C21': 0, 'C22': -0.1637806,
            'C31': -3.307728
        },
        'Male': {
            'C0': 0.7688528, 'C1': 0, 'C2': 0.0736174, 'C3': -0.0954431, 'C4': -0.4347345,
            'C5': 0.3362658, 'C6': 0.7692857, 'C7': 0.4386871, 'C8': 0, 'C9': 0,
            'C10': 0.5378979, 'C11': 0.0164827, 'C12': 0.288879, 'C13': -0.1337349,
            'C14': -0.0475924, 'C15': 0.150273, 'C16': -0.0517874, 'C17': 0.0191169,
            'C18': -0.1049477, 'C19': -0.2251948, 'C20': -0.0895067, 'C21': 0, 'C22': -0.1543702,
            'C31': -3.031168
        }
    },
    'ASCVD': {
        'Female': {
            'C0': 0.719883, 'C1': 0, 'C2': 0.1176967, 'C3': -0.151185, 'C4': -0.0835358,
            'C5': 0.3592852, 'C6': 0.8348585, 'C7': 0.4831078, 'C8': 0, 'C9': 0,
            'C10': 0.4864619, 'C11': 0.0397779, 'C12': 0.2265309, 'C13': -0.0592374,
            'C14': -0.0395762, 'C15': 0.0844423, 'C16': -0.0567839, 'C17': 0.0325692,
            'C18': -0.1035985, 'C19': -0.2417542, 'C20': -0.0791142, 'C21': 0, 'C22': -0.1671492,
            'C31': -3.819975
        },
        'Male': {
            'C0': 0.7099847, 'C1': 0, 'C2': 0.1658663, 'C3': -0.1144285, 'C4': -0.2837212,
            'C5': 0.3239977, 'C6': 0.7189597, 'C7': 0.3956973, 'C8': 0, 'C9': 0,
            'C10': 0.3690075, 'C11': 0.0203619, 'C12': 0.2036522, 'C13': -0.0865581,
            'C14': -0.0322916, 'C15': 0.114563, 'C16': -0.0300005, 'C17': 0.0232747,
            'C18': -0.0927024, 'C19': -0.2018525, 'C20': -0.0970527, 'C21': 0, 'C22': -0.1217081,
            'C31': -3.500655
        }
    }
}


def calculate_egfr(creatinine, age, sex):
    """Calculate eGFR using CKD-EPI formula"""
    if pd.isna(creatinine) or pd.isna(age) or pd.isna(sex):
        return np.nan
    if sex in ['Male', 'M', 1]:
        kappa, alpha, sex_factor = 0.9, -0.302, 1.000
    else:  # Female
        kappa, alpha, sex_factor = 0.7, -0.241, 1.012
    scr_ratio = creatinine / kappa
    term1 = scr_ratio ** alpha if scr_ratio <= 1 else 1
    term2 = 1 if scr_ratio <= 1 else scr_ratio ** (-1.2)
    return 142 * term1 * term2 * (0.9938 ** age) * sex_factor


def ensure_numeric_types(df):
    """Ensure all numerical columns are correct data types"""
    df_processed = df.copy()
    numeric_columns = [
        'Age', 'Cholesterol | Instance 0', 'HDL cholesterol | Instance 0',
        'Systolic blood pressure, automated reading | Instance 0 | Array 0',
        'BMI', 'Creatinine', 'eGFR', 'HBA1C'
    ]
    binary_columns = [
        'Smoking status | Instance 0', 'Use statins',
        'Diabetes diagnosed by doctor | Instance 0', 'Treated for HTN'
    ]
    for col in numeric_columns:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    for col in binary_columns:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').apply(
                lambda x: 1 if x and x != 0 else 0)
    return df_processed


def perform_mice_imputation(df, features_to_impute, categorical_features, n_imputations=5):
    """Perform MICE imputation on numerical and categorical variables"""
    print("Performing MICE Imputation...")
    df_imputed = df.copy()

    if miceforest_available:
        print("Using miceforest for MICE imputation...")
        kernel = mf.ImputationKernel(df_imputed[features_to_impute], datasets=n_imputations, random_state=42)
        kernel.mice(iterations=10, verbose=False)
        df_imputed[features_to_impute] = kernel.complete_data(dataset=0)
    elif sklearn_available:
        print("Using sklearn IterativeImputer for imputation...")
        categorical_encoders = {}
        df_encoded = df_imputed.copy()
        for feature in categorical_features:
            if feature in features_to_impute and feature in df_encoded.columns:
                le = LabelEncoder()
                non_null_mask = df_encoded[feature].notna()
                if non_null_mask.any():
                    df_encoded.loc[non_null_mask, feature] = le.fit_transform(
                        df_encoded.loc[non_null_mask, feature].astype(str))
                    categorical_encoders[feature] = le
        imputer = IterativeImputer(max_iter=10, random_state=42)
        df_encoded[features_to_impute] = imputer.fit_transform(df_encoded[features_to_impute])
        for feature, encoder in categorical_encoders.items():
            encoded_values = df_encoded[feature].round().astype(int)
            valid_indices = encoded_values.isin(range(len(encoder.classes_)))
            df_imputed.loc[valid_indices, feature] = encoder.inverse_transform(encoded_values[valid_indices])
    else:
        print("Warning: No MICE libraries found, using simple imputation...")
        for feature in features_to_impute:
            if feature not in categorical_features:
                df_imputed[feature] = df_imputed[feature].fillna(df_imputed[feature].median())
            else:
                df_imputed[feature] = df_imputed[feature].fillna(df_imputed[feature].mode()[0])

    imputation_flags = {}
    for feature in features_to_impute:
        flag_col = f"{feature}_imputed"
        df_imputed[flag_col] = df[feature].isna()
        imputation_flags[feature] = flag_col
    return df_imputed, imputation_flags


def calculate_risk_logit(row, model, sex):
    """Calculate risk logit value with safety range warnings"""
    try:
        c = coefficients[model][sex]
        age, tc, hdl, sbp = float(row['Age']), float(row['Cholesterol | Instance 0']), float(
            row['HDL cholesterol | Instance 0']), float(
            row['Systolic blood pressure, automated reading | Instance 0 | Array 0'])
        smoking, bmi, egfr, statin = float(row['Smoking status | Instance 0']), float(row['BMI']), float(
            row['eGFR']), float(row['Use statins'])
        diabetes, htn_med = float(row['Diabetes diagnosed by doctor | Instance 0']), float(row['Treated for HTN'])

        age_term = (age - 55) / 10
        non_hdl_chol = (tc - hdl) * 0.02586 - 3.5
        hdl_chol = (hdl * 0.02586 - 1.3) / 0.3
        sbp_low, sbp_high = (min(sbp, 110) - 110) / 20, (max(sbp, 110) - 130) / 20
        bmi_low, bmi_high = (min(bmi, 30) - 25) / 5, (max(bmi, 30) - 30) / 5
        egfr_low, egfr_high = (min(egfr, 60) - 60) / -15, (max(egfr, 60) - 90) / -15

        logit = (c['C0'] * age_term + c['C1'] * age_term ** 2 + c['C2'] * non_hdl_chol + c['C3'] * hdl_chol +
                 c['C4'] * sbp_low + c['C5'] * sbp_high + c['C6'] * diabetes + c['C7'] * smoking +
                 c['C8'] * bmi_low + c['C9'] * bmi_high + c['C10'] * egfr_low + c['C11'] * egfr_high +
                 c['C12'] * htn_med + c['C13'] * statin + c['C14'] * htn_med * sbp_high +
                 c['C15'] * statin * non_hdl_chol + c['C16'] * age_term * non_hdl_chol +
                 c['C17'] * age_term * hdl_chol + c['C18'] * age_term * sbp_high +
                 c['C19'] * age_term * diabetes + c['C20'] * age_term * smoking +
                 c['C21'] * bmi_high + c['C22'] * age_term * egfr_low + c['C31'])
        return logit
    except Exception as e:
        print(f"Error calculating risk logit: {e}")
        return np.nan


def calculate_risk_percentage(logit):
    if pd.isna(logit): return np.nan
    return 100 * math.exp(logit) / (1 + math.exp(logit))


def main():
    file_path = r"C:\Users\Administrator\Desktop\PREVENT-2023.xlsx"
    try:
        df = pd.read_excel(file_path)
        print(f"File loaded successfully! Shape: {df.shape}")
    except Exception as e:
        print(f"Error reading file: {e}");
        return

    if df['Creatinine'].mean() > 100:
        print("Creatinine detected in μmol/L, converting to mg/dL...")
        df['Creatinine'] /= 88.4
    if df['Cholesterol | Instance 0'].mean() < 10:
        print("Cholesterol detected in mmol/L, converting to mg/dL...")
        df['Cholesterol | Instance 0'] *= 38.67
        df['HDL cholesterol | Instance 0'] *= 38.67

    if 'eGFR' not in df.columns:
        print("Calculating eGFR...")
        df['eGFR'] = df.apply(lambda row: calculate_egfr(row['Creatinine'], row['Age'], row['Sex']), axis=1)

    df_processed = ensure_numeric_types(df)
    df_processed['Sex'] = df_processed['Sex'].replace({'M': 'Male', 1: 'Male', 'F': 'Female', 0: 'Female'})

    numerical_feats = ['Age', 'Cholesterol | Instance 0', 'HDL cholesterol | Instance 0',
                       'Systolic blood pressure, automated reading | Instance 0 | Array 0', 'BMI', 'eGFR', 'HBA1C',
                       'Smoking status | Instance 0', 'Use statins', 'Diabetes diagnosed by doctor | Instance 0',
                       'Treated for HTN']
    categorical_feats = ['Sex']

    df_imputed, flags = perform_mice_imputation(df_processed, numerical_feats + categorical_feats, categorical_feats)
    df_imputed = ensure_numeric_types(df_imputed)

    for model in ['Total CVD', 'ASCVD']:
        print(f"Calculating {model} risk...")
        df_imputed[f'{model} Logit'] = df_imputed.apply(lambda row: calculate_risk_logit(row, model, row['Sex']),
                                                        axis=1)
        df_imputed[f'{model} 10-Year Risk (%)'] = df_imputed[f'{model} Logit'].apply(calculate_risk_percentage)

    output_file = r"C:\Users\Administrator\Desktop\PREVENT_Risk_MICE_Results.xlsx"
    df_imputed.to_excel(output_file, index=False)
    print(f"Analysis complete. Results saved to: {output_file}")


if __name__ == "__main__":
    main()