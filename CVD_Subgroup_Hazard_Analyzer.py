import pandas as pd
import numpy as np
import statsmodels.api as sm

# 1. Load Data
file_path = "baseline data table.csv"
df = pd.read_csv(file_path)


# Helper Function: Perform Cox Regression Analysis
def analyze_subgroup(data, name):
    try:
        # Prepare data
        time = data['Specific_Time_to_Event_Days'].values
        status = data['Specific_CVD_Status'].values
        score = data['protein risk score'].values

        # Standardize score (Z-score) to calculate HR per SD
        score_std = (score - np.mean(score)) / np.std(score)

        # Fit Cox Model (Univariate: Time ~ Score)
        # Note: statsmodels PHReg does not require a constant term
        mod = sm.PHReg(time, score_std, status=status)
        res = mod.fit()

        # Extract results
        hr = np.exp(res.params[0])
        conf = res.conf_int()
        lower = np.exp(conf[0][0])
        upper = np.exp(conf[0][1])
        pval = res.pvalues[0]

        return {
            "Subgroup": name,
            "Sample Size": len(data),
            "Number of Events": status.sum(),
            "HR (95% CI)": f"{hr:.2f} ({lower:.2f}-{upper:.2f})",
            "P-value": f"{pval:.2e}"
        }
    except Exception as e:
        return {"Subgroup": name, "Error": str(e)}


# 2. Define Subgroups
subgroups = [
    (df, "Overall"),
    (df[df['Age'] < 65], "Age < 65 years"),
    (df[df['Age'] >= 65], "Age ≥ 65 years"),
    (df[df['Sex'] == 0], "Female"),
    (df[df['Sex'] == 1], "Male"),
    # Lag-time analysis: Exclude samples with follow-up < 2 years (730 days)
    (df[df['Specific_Time_to_Event_Days'] >= 730], "2-year Lag-time Analysis")
]

# 3. Run Analysis
results = []
for data_sub, name in subgroups:
    results.append(analyze_subgroup(data_sub, name))

# 4. Display Results
results_df = pd.DataFrame(results)
print(results_df)