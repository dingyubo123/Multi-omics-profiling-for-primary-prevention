import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import warnings
import os
from itertools import combinations
from sklearn.utils import resample
import textwrap

warnings.filterwarnings('ignore')

# Set plotting style (Seaborn-whitegrid is ideal for academic publications)
plt.style.use('seaborn-v0_8-whitegrid')
try:
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']  # Arial is standard for SCI journals
except:
    pass
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

# ============================================================================
# 1. Key Configuration
# ============================================================================
FILE_PATH = "Various scores of CVD.csv"

# Variable Mapping
EXACT_VAR_MAP = {
    'SCORE2': 'SCORE2',
    'Total CVD 10-Year Risk (%)': 'PREVENT',
    'FRS10_year_total_cvd_risk(%)': 'FRS',
    'Standard PRS for cardiovascular disease (CVD)': 'PRS',
    'C-reactive protein | Instance 0': 'CRP',
    'NTproBNP': 'NTproBNP',
    'protein_risk_score': 'Protein Score'
}

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

    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except:
        try:
            df = pd.read_csv(file_path, encoding='gbk')
        except:
            df = pd.read_csv(file_path, encoding='latin1')

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
    df_clean['survival_time'] = df_clean['survival_time'].apply(lambda x: max(x, 0.001))
    return df_clean


# ============================================================================
# 3. Core Statistical Functions
# ============================================================================
def compare_models_bootstrap(time, event, risk_score_base, risk_score_new, pred_proba_base, pred_proba_new,
                             n_boot=1000):
    n = len(time)
    indices = np.arange(n)

    c_base = concordance_index(time, -risk_score_base, event)
    c_new = concordance_index(time, -risk_score_new, event)
    delta_c = c_new - c_base

    def compute_nri_idi(y, p_old, p_new):
        event_idx = (y == 1)
        nonevent_idx = (y == 0)
        idi = (np.mean(p_new[event_idx]) - np.mean(p_old[event_idx])) - \
              (np.mean(p_new[nonevent_idx]) - np.mean(p_old[nonevent_idx]))
        up_event = np.mean(p_new[event_idx] > p_old[event_idx])
        down_event = np.mean(p_new[event_idx] < p_old[event_idx])
        up_nonevent = np.mean(p_new[nonevent_idx] > p_old[nonevent_idx])
        down_nonevent = np.mean(p_new[nonevent_idx] < p_old[nonevent_idx])
        nri = (up_event - down_event) - (up_nonevent - down_nonevent)
        return nri, idi

    nri_point, idi_point = compute_nri_idi(event, pred_proba_base, pred_proba_new)

    boots = {'c_new': [], 'delta_c': [], 'nri': [], 'idi': [], 'c_base': []}
    for _ in range(n_boot):
        idx = resample(indices, replace=True, n_samples=n)
        t_b = time.iloc[idx] if hasattr(time, 'iloc') else time[idx]
        e_b = event.iloc[idx] if hasattr(event, 'iloc') else event[idx]
        if len(np.unique(e_b)) < 2: continue
        try:
            cb = concordance_index(t_b, -risk_score_base[idx], e_b)
            cn = concordance_index(t_b, -risk_score_new[idx], e_b)
            bnri, bidi = compute_nri_idi(e_b, pred_proba_base[idx], pred_proba_new[idx])
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
        'c_base': c_base,
        'c_new': c_new, 'c_new_ci': get_ci(boots['c_new']),
        'delta_c': delta_c, 'delta_c_ci': get_ci(boots['delta_c']),
        'nri': nri_point, 'nri_ci': get_ci(boots['nri']),
        'idi': idi_point, 'idi_ci': get_ci(boots['idi'])
    }


# ============================================================================
# 4. Analysis Execution
# ============================================================================
try:
    df_clean = load_and_clean_data(FILE_PATH)
except Exception as e:
    print(f"Error: {e}")
    exit()

scaler = StandardScaler()
model_cols = BASE_VAR_ORIGINS + CAND_VAR_ORIGINS
df_scaled = df_clean.copy()
df_scaled[model_cols] = scaler.fit_transform(df_scaled[model_cols])


def get_model_preds(vars_list_origin):
    cph = CoxPHFitter(penalizer=0.1)
    try:
        cph.fit(df_scaled[['survival_time', 'event_status'] + vars_list_origin],
                duration_col='survival_time', event_col='event_status')
        risk_scores = cph.predict_partial_hazard(df_scaled[vars_list_origin]).values
        log_risk_scores = np.log(risk_scores)
    except:
        return None, None
    lr = LogisticRegression(max_iter=1000)
    lr.fit(df_scaled[vars_list_origin], df_scaled['event_status'])
    pred_probs = lr.predict_proba(df_scaled[vars_list_origin])[:, 1]
    return log_risk_scores, pred_probs


# 1. Base Model Evaluation
print("Evaluating Base Model...")
base_risk, base_prob = get_model_preds(BASE_VAR_ORIGINS)
res_base = compare_models_bootstrap(
    df_scaled['survival_time'], df_scaled['event_status'],
    base_risk, base_risk, base_prob, base_prob, n_boot=200
)

results = []
results.append({
    'Display_Name': "Base Model (SCORE2 + PREVENT)",
    'Is_Reference': True,
    'C_index': res_base['c_base'], 'C_ci': (res_base['c_base'], res_base['c_base']),
    'Delta_C': 0, 'Delta_ci': (0, 0),
    'NRI': 0, 'NRI_ci': (0, 0),
    'IDI': 0, 'IDI_ci': (0, 0)
})

# 2. Loop Combinations
print("Evaluating Variable Combinations...")
for r in range(1, len(CAND_VAR_ORIGINS) + 1):
    for combo_origins in combinations(CAND_VAR_ORIGINS, r):
        current_vars_origin = BASE_VAR_ORIGINS + list(combo_origins)

        added_vars_names = [EXACT_VAR_MAP[col] for col in combo_origins]
        short_display_name = "+ " + " + ".join(added_vars_names)

        curr_risk, curr_prob = get_model_preds(current_vars_origin)
        if curr_risk is None: continue

        stats = compare_models_bootstrap(
            df_scaled['survival_time'], df_scaled['event_status'],
            base_risk, curr_risk, base_prob, curr_prob, n_boot=200
        )

        if stats['c_new'] > stats['c_base']:
            results.append({
                'Display_Name': short_display_name,
                'Is_Reference': False,
                'C_index': stats['c_new'], 'C_ci': stats['c_new_ci'],
                'Delta_C': stats['delta_c'], 'Delta_ci': stats['delta_c_ci'],
                'NRI': stats['nri'], 'NRI_ci': stats['nri_ci'],
                'IDI': stats['idi'], 'IDI_ci': stats['idi_ci']
            })

results_df = pd.DataFrame(results)

# ============================================================================
# 5. Smart Sorting
# ============================================================================
# Logic:
# 1. Separate Reference row
# 2. Sort the remaining models by C-index descending
ref_row = results_df[results_df['Is_Reference'] == True]
others = results_df[results_df['Is_Reference'] == False]
others_sorted = others.sort_values(by='C_index', ascending=False)

final_plot_data = pd.concat([ref_row, others_sorted]).reset_index(drop=True)

# Identify the best-performing model index (excluding Reference)
best_model_idx = 1

# Reverse data for plotting (Matplotlib plots from bottom to top)
plot_data_reversed = final_plot_data.iloc[::-1].reset_index(drop=True)
n_rows = len(plot_data_reversed)

# Recalculate best model index for the reversed data
best_model_plot_idx = n_rows - 1 - best_model_idx

# ============================================================================
# 6. Visualization (Publication Ready)
# ============================================================================
print("Generating Charts...")

fig, axes = plt.subplots(1, 6, figsize=(20, n_rows * 0.6 + 2), sharey=True,
                         gridspec_kw={'width_ratios': [3.5, 1.2, 2.5, 2.5, 2.5, 2.5]})
plt.subplots_adjust(wspace=0.1, left=0.02, right=0.98)

y_pos = np.arange(n_rows)

# Define Colors
COLOR_REF = 'black'
COLOR_BEST = '#D32F2F'
COLOR_NORMAL = '#455A64'

# --- Col 1: Model Names ---
ax = axes[0]
ax.set_title("Model Combination", fontweight='bold', loc='left', fontsize=12)
for i, (y, name, is_ref) in enumerate(
        zip(y_pos, plot_data_reversed['Display_Name'], plot_data_reversed['Is_Reference'])):
    fw = 'bold' if (is_ref or i == best_model_plot_idx) else 'normal'
    col = 'black'
    if i == best_model_plot_idx: col = COLOR_BEST
    ax.text(0, y, name, va='center', ha='left', fontsize=11, fontweight=fw, color=col)

ax.axis('off')
ax.set_ylim(-0.5, n_rows - 0.5)

# --- Col 2: C-index Values ---
ax = axes[1]
ax.set_title("C-index", fontweight='bold', fontsize=12)
for i, (y, row) in enumerate(zip(y_pos, plot_data_reversed.itertuples())):
    val_str = f"{row.C_index:.3f}"
    if not row.Is_Reference:
        text = f"{val_str}\n({row.C_ci[0]:.3f}-{row.C_ci[1]:.3f})"
    else:
        text = val_str + "\n(Ref)"

    fw = 'bold' if i == best_model_plot_idx else 'normal'
    col = COLOR_BEST if i == best_model_plot_idx else 'black'
    ax.text(0.5, y, text, va='center', ha='center', fontsize=9, fontweight=fw, color=col)
ax.axis('off')


# --- Plotting Function ---
def plot_column(ax_idx, col_val, col_ci, title, ref_line=0):
    ax = axes[ax_idx]
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.axvline(ref_line, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    for i, (y, val, ci, is_ref) in enumerate(zip(y_pos,
                                                 plot_data_reversed[col_val],
                                                 plot_data_reversed[col_ci],
                                                 plot_data_reversed['Is_Reference'])):
        if is_ref: continue
        color = COLOR_BEST if i == best_model_plot_idx else COLOR_NORMAL
        err_l = val - ci[0]
        err_u = ci[1] - val
        ax.errorbar(val, y, xerr=[[err_l], [err_u]], fmt='o', color=color, ecolor=color, capsize=3, markersize=5)

    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(True, axis='x', linestyle=':', alpha=0.6)


# --- Col 3: C-index Plot ---
ax = axes[2]
ax.set_title("C-index", fontweight='bold', fontsize=12)
for i, (y, val, ci, is_ref) in enumerate(
        zip(y_pos, plot_data_reversed['C_index'], plot_data_reversed['C_ci'], plot_data_reversed['Is_Reference'])):
    color = 'black' if is_ref else (COLOR_BEST if i == best_model_plot_idx else COLOR_NORMAL)
    fmt = 'D' if is_ref else 'o'

    if is_ref:
        ax.plot(val, y, fmt, color=color, markersize=6)
    else:
        err_l = val - ci[0]
        err_u = ci[1] - val
        ax.errorbar(val, y, xerr=[[err_l], [err_u]], fmt=fmt, color=color, ecolor=color, capsize=3, markersize=5)

ax.axvline(results_df[results_df['Is_Reference']]['C_index'].values[0], color='gray', linestyle='--', alpha=0.5)
ax.get_yaxis().set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, axis='x', linestyle=':', alpha=0.6)

# --- Col 4, 5, 6: Delta, NRI, IDI ---
plot_column(3, 'Delta_C', 'Delta_ci', "C-index Increase")
plot_column(4, 'NRI', 'NRI_ci', "NRI")
plot_column(5, 'IDI', 'IDI_ci', "IDI")

plt.suptitle("Evaluation of Added Predictive Value", fontsize=16, fontweight='bold', y=0.99)
plt.savefig("Publication_Forest_Plot.png", dpi=300, bbox_inches='tight')
plt.show()

print("Optimization Complete! The best model is highlighted.")