import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
import os
import numpy as np

# ==========================================
# 1. Configuration and File Loading
# ==========================================
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
file_name = 'Cumulative incidence rate curve.csv'  # Original file name kept
file_path = os.path.join(desktop_path, file_name)

print(f"🔹 Reading file: {file_path}")

try:
    if file_name.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    print("✅ Data loaded successfully!")
except Exception as e:
    print(f"❌ Loading error: {e}")
    df = pd.DataFrame()

# ==========================================
# 2. Data Cleaning and Model Definition
# ==========================================
if not df.empty:
    # Core Column Names
    time_col_days = 'survival_time'
    time_col_years = 'time_years'
    event_col = 'event_status'

    # Original Variable Names
    var_prot = 'protein risk score'
    var_score2 = 'SCORE2'
    var_prs = 'polygenic risk score'
    var_prevent = 'PREVENT'

    required_cols = [time_col_days, event_col, var_prot, var_score2, var_prs, var_prevent]

    if any(c not in df.columns for c in required_cols):
        print(f"❌ Error: Data missing columns {required_cols}")
    else:
        df[time_col_years] = df[time_col_days] / 365.25
        df = df.dropna(subset=required_cols)

        # ---------------------------------------------------------
        # Define 4 Models (Using full, unabbreviated names)
        # ---------------------------------------------------------
        models_config = {
            # Pairwise combinations
            "SCORE2 + polygenic risk score": [var_score2, var_prs],
            "SCORE2 + PREVENT": [var_score2, var_prevent],
            "polygenic risk score + PREVENT": [var_prs, var_prevent],

            # Full integrated model
            "SCORE2 + polygenic risk score + PREVENT + protein risk score": [var_score2, var_prs, var_prevent, var_prot]
        }

        # Parameters
        EVAL_TIME = 10     # 10-year risk
        RISK_THRESHOLD = 0.1  # 10% threshold


        # ==========================================
        # 3. Core Calculation Function (Net Benefit)
        # ==========================================
        def calculate_net_benefit_survival(df, risk_scores, time_point, thresholds):
            net_benefits = []
            kmf_all = KaplanMeierFitter()
            kmf_all.fit(df[time_col_years], df[event_col])
            prevalence = 1 - kmf_all.predict(time_point)
            N = len(df)

            for thresh in thresholds:
                high_risk_mask = risk_scores >= thresh
                n_high_risk = high_risk_mask.sum()

                if n_high_risk == 0:
                    net_benefits.append(0)
                    continue

                try:
                    kmf_sub = KaplanMeierFitter()
                    kmf_sub.fit(df.loc[high_risk_mask, time_col_years],
                                df.loc[high_risk_mask, event_col])
                    risk_sub = 1 - kmf_sub.predict(time_point)
                except:
                    risk_sub = 0

                tp_rate = risk_sub * (n_high_risk / N)
                fp_rate = (1 - risk_sub) * (n_high_risk / N)
                nb = tp_rate - fp_rate * (thresh / (1 - thresh))
                net_benefits.append(nb)
            return np.array(net_benefits)


        # ==========================================
        # 4. Model Training and Prediction
        # ==========================================
        print("⚙️ Fitting Cox Models...")
        predictions = {}

        for name, cols in models_config.items():
            cph = CoxPHFitter()
            subset = df[[time_col_years, event_col] + cols]
            cph.fit(subset, duration_col=time_col_years, event_col=event_col)

            surv_func = cph.predict_survival_function(df, times=[EVAL_TIME])
            predictions[name] = 1 - surv_func.iloc[0].values

        # ==========================================
        # 5. Plot DCA Curves (Figure 5B Style)
        # ==========================================
        print("📊 Plotting DCA Curves...")
        thresholds = np.linspace(0.01, 0.50, 50)

        fig, ax = plt.subplots(figsize=(12, 9))  # Canvas sized for long labels

        # Plot None
        ax.plot(thresholds * 100, np.zeros_like(thresholds), color='gray', linestyle='-', lw=1, label='None')

        # Plot All
        nb_all = calculate_net_benefit_survival(df, np.ones(len(df)), EVAL_TIME, thresholds)
        ax.plot(thresholds * 100, nb_all, color='black', linestyle='--', lw=1.5, label='All')

        # Plot Models
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red

        for i, (name, probs) in enumerate(predictions.items()):
            nb = calculate_net_benefit_survival(df, probs, EVAL_TIME, thresholds)

            # Smooth curve
            from scipy.ndimage import gaussian_filter1d
            nb_smooth = gaussian_filter1d(nb, sigma=1)

            # Bold red line for the full model containing protein score
            if "protein risk score" in name:
                lw = 3.0
                ls = '-'
                color = '#d62728'  # Red
            else:
                lw = 2.0
                ls = '--'
                color = colors[i % 3]

            ax.plot(thresholds * 100, nb_smooth, color=color, linestyle=ls, linewidth=lw, label=name)

        ax.set_title(f"Decision Curve Analysis ({EVAL_TIME}-year Risk)", fontsize=16, fontweight='bold')
        ax.set_xlabel("Risk Threshold (%)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Net Benefit", fontsize=12, fontweight='bold')
        ax.set_xlim(0, 50)

        # Fine-tune Y-axis range
        y_max = np.max(
            [calculate_net_benefit_survival(df, predictions[m], EVAL_TIME, [0.01])[0] for m in predictions]) * 1.2
        ax.set_ylim(-0.005, y_max)

        # Legend settings
        ax.legend(fontsize=12, loc='upper right', frameon=False)
        ax.grid(alpha=0.2, linestyle='--')

        save_dca = os.path.join(desktop_path, "Figure_5B_DCA_FullNames.png")
        plt.savefig(save_dca, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ DCA saved: {save_dca}")

        # ==========================================
        # 6. Generate Reclassification Table (Figure 5A Style)
        # ==========================================
        print("📋 Generating Reclassification Table...")

        # Defining comparisons
        name_ref = "SCORE2 + PREVENT"
        name_new = "SCORE2 + polygenic risk score + PREVENT + protein risk score"

        if name_ref not in predictions or name_new not in predictions:
            keys = list(predictions.keys())
            name_ref = keys[0]
            name_new = keys[-1]
            print(f"⚠️ Note: Defaulting comparison to '{name_ref}' vs '{name_new}'")

        prob_ref = predictions[name_ref]
        prob_new = predictions[name_new]

        # Data filtering (Cases within window or Non-cases beyond window)
        valid_mask = (df[time_col_years] > EVAL_TIME) | (df[event_col] == 1)
        valid_mask = valid_mask & ~((df[event_col] == 0) & (df[time_col_years] < EVAL_TIME))

        df_sub = df[valid_mask].copy()
        p_ref_sub = prob_ref[valid_mask]
        p_new_sub = prob_new[valid_mask]

        # Categorize (0: Low Risk, 1: High Risk)
        cat_ref = (p_ref_sub > RISK_THRESHOLD).astype(int)
        cat_new = (p_new_sub > RISK_THRESHOLD).astype(int)

        is_case = (df_sub[event_col] == 1) & (df_sub[time_col_years] <= EVAL_TIME)
        is_ncase = ~is_case

        # Statistical counts
        def get_n(r, c, mask):
            return ((cat_ref == r) & (cat_new == c) & mask).sum()

        # Cases
        c00 = get_n(0, 0, is_case)
        c01 = get_n(0, 1, is_case)
        c10 = get_n(1, 0, is_case)
        c11 = get_n(1, 1, is_case)
        c_tot = is_case.sum()

        # Non-cases
        n00 = get_n(0, 0, is_ncase)
        n01 = get_n(0, 1, is_ncase)
        n10 = get_n(1, 0, is_ncase)
        n11 = get_n(1, 1, is_ncase)
        n_tot = is_ncase.sum()

        # NRI/IDI Calculation
        nri_c = (c01 - c10) / c_tot
        nri_nc = (n10 - n01) / n_tot
        nri = nri_c + nri_nc

        idi = (p_new_sub[is_case].mean() - p_ref_sub[is_case].mean()) - \
              (p_new_sub[is_ncase].mean() - p_ref_sub[is_ncase].mean())

        # --- Table Plotting ---
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('off')

        def fmt(n, t):
            return f"{n}\n({n / t * 100:.1f}%)"

        # Break long names for table headers
        header_new = name_new.replace(" + ", " +\n")
        header_ref = name_ref.replace(" + ", " +\n")

        table_data = [
            ['', header_new, '', ''],
            ['', f'≤{RISK_THRESHOLD * 100:.0f}%', f'>{RISK_THRESHOLD * 100:.0f}%', 'Total No. (%)'],
            [f'{header_ref}\n≤{RISK_THRESHOLD * 100:.0f}%', c00, c01, fmt(c00 + c01, c_tot)],
            [f'{header_ref}\n>{RISK_THRESHOLD * 100:.0f}%', c10, c11, fmt(c10 + c11, c_tot)],
            ['Total Cases', c00 + c10, c01 + c11, f"{c_tot} (100.0)"],
            ['', '', '', ''], # Spacer
            [f'{header_ref}\n≤{RISK_THRESHOLD * 100:.0f}%', n00, n01, fmt(n00 + n01, n_tot)],
            [f'{header_ref}\n>{RISK_THRESHOLD * 100:.0f}%', n10, n11, fmt(n10 + n11, n_tot)],
            ['Total Non-cases', n00 + n10, n01 + n11, f"{n_tot} (100.0)"]
        ]

        t = ax.table(cellText=table_data, loc='center', cellLoc='center', bbox=[0.15, 0.2, 0.8, 0.75])
        t.auto_set_font_size(False)
        t.set_fontsize(9)

        # Style loops
        for (row, col), cell in t.get_celld().items():
            cell.set_edgecolor('black')
            if row == 0 and col == 1:
                cell.set_facecolor('#f0f0f0')
                cell.set_text_props(weight='bold')
            if row == 1 and col in [1, 2]: cell.set_facecolor('#fff2cc')
            if col == 0 and row > 1: cell.set_text_props(weight='bold')
            if row == 5: cell.set_visible(False)
            if row in [2, 3, 6, 7]: cell.set_height(0.12)
            elif row == 0: cell.set_height(0.1)

        plt.text(0.10, 0.75, "Cases", rotation=90, weight='bold', fontsize=12, va='center')
        plt.text(0.10, 0.40, "Non-cases", rotation=90, weight='bold', fontsize=12, va='center')

        # Footer stats
        stats_text = (f"Comparison: [{name_ref}] vs [{name_new}]\n"
                      f"---------------------------------------------------\n"
                      f"NRI: {nri * 100:.1f}%\n"
                      f"IDI: {idi:.4f}")

        plt.text(0.15, 0.05, stats_text, fontsize=11, family='monospace', va='bottom')

        save_tbl = os.path.join(desktop_path, "Figure_5A_Reclassification_FullNames.png")
        plt.savefig(save_tbl, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ NRI Table saved: {save_tbl}")

        print("\n🎉 Success! Two images generated on Desktop. Please check.")
else:
    print("❌ Failed: DataFrame is empty.")