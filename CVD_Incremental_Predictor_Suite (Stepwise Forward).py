import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import warnings
import os
import textwrap
from sklearn.utils import resample
import matplotlib.ticker as mticker

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-white')
# Font settings
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
except:
    pass
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 1. Variable Mapping Configuration (Core: Precise Names)
# ============================================================================
FILE_PATH = "Various scores of CVD.csv"

# Precise mapping dictionary: Original Column Name -> Display Name
EXACT_VAR_MAP = {
    # --- Baseline Variables ---
    'SCORE2': 'SCORE2',
    'Total CVD 10-Year Risk (%)': 'PREVENT',

    # --- Candidate Variables ---
    'FRS10_year_total_cvd_risk(%)': 'FRS',
    'Standard PRS for cardiovascular disease (CVD)': 'Polygenic risk score',
    'C-reactive protein | Instance 0': 'CRP',
    'NTproBNP': 'NTproBNP',
    'protein_risk_score': 'Protein risk score'
}

# Lists of original column names for base and candidate variables
BASE_VAR_ORIGINS = ['SCORE2', 'Total CVD 10-Year Risk (%)']
CAND_VAR_ORIGINS = [
    'FRS10_year_total_cvd_risk(%)',
    'Standard PRS for cardiovascular disease (CVD)',
    'C-reactive protein | Instance 0',
    'NTproBNP',
    'protein_risk_score'
]


# ============================================================================
# 2. Data Loading and Cleaning
# ============================================================================
def load_and_clean_data(file_path):
    if not os.path.exists(file_path):
        csvs = [f for f in os.listdir('.') if f.endswith('.csv')]
        if csvs:
            file_path = csvs[0]
            print(f"Auto-using file: {file_path}")
        else:
            raise FileNotFoundError("CSV file not found")

    # Try reading with different encodings
    for encoding in ['utf-8', 'gbk', 'latin1']:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            break
        except:
            continue

    # Standardize column names
    col_map = {}
    for col in df.columns:
        c_lower = col.lower()
        if 'event' in c_lower and 'status' in c_lower:
            col_map[col] = 'event_status'
        elif 'survival' in c_lower and 'time' in c_lower:
            col_map[col] = 'survival_time'
    if col_map:
        df.rename(columns=col_map, inplace=True)

    df_clean = df.dropna().copy()
    # Cox model requires time > 0
    df_clean['survival_time'] = df_clean['survival_time'].apply(lambda x: max(x, 0.001))

    return df_clean


# ============================================================================
# 3. Statistical Functions
# ============================================================================
def compare_models_bootstrap(time, event, risk_score_base, risk_score_new, pred_proba_base, pred_proba_new,
                             n_boot=1000):
    """Calculate metrics and confidence intervals via Bootstrap"""
    n = len(time)
    indices = np.arange(n)

    # Point estimation
    c_base = concordance_index(time, -risk_score_base, event)
    c_new = concordance_index(time, -risk_score_new, event)
    delta_c = c_new - c_base

    def compute_nri_idi(y, p_old, p_new):
        event_idx = (y == 1)
        nonevent_idx = (y == 0)
        # IDI
        idi = (np.mean(p_new[event_idx]) - np.mean(p_old[event_idx])) - \
              (np.mean(p_new[nonevent_idx]) - np.mean(p_old[nonevent_idx]))
        # Continuous NRI
        up_event = np.mean(p_new[event_idx] > p_old[event_idx])
        down_event = np.mean(p_new[event_idx] < p_old[event_idx])
        up_nonevent = np.mean(p_new[nonevent_idx] > p_old[nonevent_idx])
        down_nonevent = np.mean(p_new[nonevent_idx] < p_old[nonevent_idx])
        nri = (up_event - down_event) - (up_nonevent - down_nonevent)
        return nri, idi

    nri_point, idi_point = compute_nri_idi(event, pred_proba_base, pred_proba_new)

    # Bootstrap
    boots = {'c_base': [], 'c_new': [], 'delta_c': [], 'nri': [], 'idi': []}

    for _ in range(n_boot):
        idx = resample(indices, replace=True, n_samples=n)
        t_b = time.iloc[idx] if hasattr(time, 'iloc') else time[idx]
        e_b = event.iloc[idx] if hasattr(event, 'iloc') else event[idx]

        if len(np.unique(e_b)) < 2: continue
        try:
            cb = concordance_index(t_b, -risk_score_base[idx], e_b)
            cn = concordance_index(t_b, -risk_score_new[idx], e_b)
            bnri, bidi = compute_nri_idi(e_b, pred_proba_base[idx], pred_proba_new[idx])

            boots['c_base'].append(cb)
            boots['c_new'].append(cn)
            boots['delta_c'].append(cn - cb)
            boots['nri'].append(bnri)
            boots['idi'].append(bidi)
        except:
            continue

    def get_ci(data):
        if not data: return (0, 0)
        return np.percentile(data, 2.5), np.percentile(data, 97.5)

    return {
        'c_base': c_base, 'c_base_ci': get_ci(boots['c_base']),
        'c_new': c_new, 'c_new_ci': get_ci(boots['c_new']),
        'delta_c': delta_c, 'delta_c_ci': get_ci(boots['delta_c']),
        'nri': nri_point, 'nri_ci': get_ci(boots['nri']),
        'idi': idi_point, 'idi_ci': get_ci(boots['idi'])
    }


# ============================================================================
# 4. Main Workflow
# ============================================================================
try:
    df_clean = load_and_clean_data(FILE_PATH)
except Exception as e:
    print(f"Error: {e}")
    exit()

# Verify columns exist
all_vars = BASE_VAR_ORIGINS + CAND_VAR_ORIGINS
missing = [v for v in all_vars if v not in df_clean.columns]
if missing:
    print(f"Error: Missing columns {missing}")
    exit()

# Standardization
scaler = StandardScaler()
df_scaled = df_clean.copy()
df_scaled[all_vars] = scaler.fit_transform(df_scaled[all_vars])


# Helper functions
def fit_model(vars_list):
    # Cox
    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(df_scaled[['survival_time', 'event_status'] + vars_list],
                duration_col='survival_time', event_col='event_status')
        risk = cph.predict_partial_hazard(df_scaled[vars_list]).values
        log_risk = np.log(risk)
    except:
        return None, None
    # Logistic
    try:
        lr = LogisticRegression(max_iter=1000)
        lr.fit(df_scaled[vars_list], df_scaled['event_status'])
        prob = lr.predict_proba(df_scaled[vars_list])[:, 1]
    except:
        return None, None
    return log_risk, prob


def get_name(vars_list):
    return " + ".join([EXACT_VAR_MAP[v] for v in vars_list])


# --- Best Path Search Logic ---
results = []

# 1. Baseline Model (Reference)
print("Step 1: Evaluating baseline model...")
base_risk, base_prob = fit_model(BASE_VAR_ORIGINS)
res_base = compare_models_bootstrap(
    df_scaled['survival_time'], df_scaled['event_status'],
    base_risk, base_risk, base_prob, base_prob, n_boot=500
)

results.append({
    'Step': 0,
    'Model_Vars_Full': get_name(BASE_VAR_ORIGINS),
    'Is_Reference': True,
    'C_index': res_base['c_base'], 'C_ci': res_base['c_base_ci'],
    'Delta_C': 0, 'Delta_ci': (0, 0),
    'NRI': 0, 'NRI_ci': (0, 0),
    'IDI': 0, 'IDI_ci': (0, 0)
})

# 2. Stepwise Forward Selection
current_vars = BASE_VAR_ORIGINS.copy()
remaining_vars = CAND_VAR_ORIGINS.copy()

step = 1
while remaining_vars:
    best_c = -1
    best_var = None
    best_risk = None
    best_prob = None

    print(f"\nStep {step + 1}: Finding best variable ({len(remaining_vars)} remaining)...")

    # Test each remaining variable
    for var in remaining_vars:
        temp_vars = current_vars + [var]
        t_risk, t_prob = fit_model(temp_vars)
        if t_risk is None: continue

        # Calculate C-index
        c_val = concordance_index(df_scaled['survival_time'], -t_risk, df_scaled['event_status'])

        # Record the best
        if c_val > best_c:
            best_c = c_val
            best_var = var
            best_risk = t_risk
            best_prob = t_prob

    if best_var:
        # Select the variable
        current_vars.append(best_var)
        remaining_vars.remove(best_var)
        full_name = get_name(current_vars)
        print(f"  -> Selected: {EXACT_VAR_MAP[best_var]} (C-index: {best_c:.4f})")

        # Calculate detailed metrics relative to [Baseline Model]
        stats = compare_models_bootstrap(
            df_scaled['survival_time'], df_scaled['event_status'],
            base_risk, best_risk, base_prob, best_prob, n_boot=500
        )

        results.append({
            'Step': step,
            'Model_Vars_Full': full_name,
            'Is_Reference': False,
            'C_index': stats['c_new'], 'C_ci': stats['c_new_ci'],
            'Delta_C': stats['delta_c'], 'Delta_ci': stats['delta_c_ci'],
            'NRI': stats['nri'], 'NRI_ci': stats['nri_ci'],
            'IDI': stats['idi'], 'IDI_ci': stats['idi_ci']
        })
        step += 1
    else:
        print("  -> No further improvement possible, stopping.")
        break

# ============================================================================
# 5. Result Summary and Saving
# ============================================================================
results_df = pd.DataFrame(results)


# Formatting output
def fmt(val, ci, is_ref=False):
    if is_ref: return "Reference"
    return f"{val:.3f} ({ci[0]:.3f}-{ci[1]:.3f})"


export_df = results_df.copy()
export_df['C-index (95% CI)'] = export_df.apply(lambda x: f"{x['C_index']:.3f} ({x['C_ci'][0]:.3f}-{x['C_ci'][1]:.3f})",
                                                axis=1)
export_df['Delta C (95% CI)'] = export_df.apply(lambda x: fmt(x['Delta_C'], x['Delta_ci'], x['Is_Reference']), axis=1)
export_df['NRI (95% CI)'] = export_df.apply(lambda x: fmt(x['NRI'], x['NRI_ci'], x['Is_Reference']), axis=1)
export_df['IDI (95% CI)'] = export_df.apply(lambda x: fmt(x['IDI'], x['IDI_ci'], x['Is_Reference']), axis=1)

cols_out = ['Model_Vars_Full', 'C-index (95% CI)', 'Delta C (95% CI)', 'NRI (95% CI)', 'IDI (95% CI)']
export_df[cols_out].to_csv('best_path_forest_data.csv', index=False, encoding='utf-8-sig')
print("\nData saved: best_path_forest_data.csv")

# ============================================================================
# 6. Visualization (Best Path Forest Plot)
# ============================================================================

print("\nGenerating plot...")
plot_data = results_df.iloc[::-1].reset_index(drop=True)
n_rows = len(plot_data)

fig, axes = plt.subplots(1, 6, figsize=(24, n_rows * 0.8 + 2), sharey=True,
                         gridspec_kw={'width_ratios': [5, 1.2, 2, 2, 2, 2]})
plt.subplots_adjust(wspace=0.05, left=0.02, right=0.98)

y_pos = np.arange(n_rows)

# 1. Model Name Column
ax = axes[0]
ax.set_title("Model Combination", fontweight='bold', fontsize=18, loc='left')
for y, name, is_ref in zip(y_pos, plot_data['Model_Vars_Full'], plot_data['Is_Reference']):
    fw = 'bold' if is_ref else 'normal'
    display_name = textwrap.fill(name, width=60)
    ax.text(0, y, display_name, va='center', ha='left', fontsize=16, fontweight=fw)
ax.axis('off')

# 2. C-index Text Column
ax = axes[1]
ax.set_title("C-index (95% CI)", fontweight='bold', fontsize=18)
for y, row in plot_data.iterrows():
    text = f"{row['C_index']:.3f}\n({row['C_ci'][0]:.3f}-{row['C_ci'][1]:.3f})"
    ax.text(0.5, y, text, va='center', ha='center', fontsize=16)
ax.axis('off')

# --- Shared Plotting Function ---
def plot_col(ax_idx, data_col, ci_col, title, ref_val=0, is_cindex=False, decimals=2):
    ax = axes[ax_idx]
    ax.set_title(title, fontweight='bold', fontsize=18)
    ax.tick_params(axis='x', labelsize=12)

    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter(f'%.{decimals}f'))

    if not is_cindex:
        ax.axvline(ref_val, color='gray', linestyle='--', alpha=0.5)

    for y, val, ci, is_ref in zip(y_pos, plot_data[data_col], plot_data[ci_col], plot_data['Is_Reference']):
        if is_ref and not is_cindex:
            continue
        err_l = val - ci[0]
        err_u = ci[1] - val
        ax.errorbar(val, y, xerr=[[err_l], [err_u]], fmt='o', color='#c0392b', ecolor='#c0392b', capsize=3)

    ax.get_yaxis().set_visible(False)
    ax.grid(True, axis='x', linestyle=':', alpha=0.6)

# --- Subplot Rendering ---
plot_col(2, 'C_index', 'C_ci', 'C-index', is_cindex=True, decimals=3)
plot_col(3, 'Delta_C', 'Delta_ci', 'C-index Increase', decimals=3)
plot_col(4, 'NRI', 'NRI_ci', 'NRI', decimals=1)
plot_col(5, 'IDI', 'IDI_ci', 'IDI', decimals=2)

plt.suptitle("Incremental Predictive Value (Best Stepwise Path)", fontsize=20, y=0.98)
plt.savefig("best_path_forest_plot.png", dpi=300, bbox_inches='tight')
plt.show()