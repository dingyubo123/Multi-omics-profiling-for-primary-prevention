import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
import os
import numpy as np
import matplotlib.ticker as mtick

# ==========================================
# 1. Path Configuration and Data Loading
# ==========================================
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
# Keep original filename for compatibility
file_name = 'Cumulative incidence rate curve.csv'
file_path = os.path.join(desktop_path, file_name)

print(f"Reading file: {file_path}")

try:
    if file_name.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    print("✅ Data loaded successfully!")
except FileNotFoundError:
    print(f"❌ Error: File '{file_name}' not found on Desktop.")
    df = pd.DataFrame()
except Exception as e:
    print(f"❌ Reading error: {e}")
    df = pd.DataFrame()

# ==========================================
# 2. Data Preparation
# ==========================================
if not df.empty:
    time_col_days = 'survival_time'
    time_col_years = 'time_years'
    event_col = 'event_status'
    main_score_col = 'protein risk score'
    risk_factors = ['SCORE2', 'polygenic risk score', 'PREVENT']

    required_cols = [time_col_days, event_col, main_score_col] + risk_factors

    # Check for missing columns
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"❌ Error: Missing columns: {missing_cols}")
    else:
        # Convert days to years
        df[time_col_years] = df[time_col_days] / 365.25
        df = df.dropna(subset=required_cols)

        # Categorize into Quintiles
        try:
            df['Quintile'] = pd.qcut(df[main_score_col], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        except ValueError:
            df['Quintile'] = pd.qcut(df[main_score_col].rank(method='first'), 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

        # Professional color palette
        colors = {
            'Q5': '#d62728', 'Q4': '#1f77b4', 'Q3': '#ff7f0e', 'Q2': '#2ca02c', 'Q1': '#9467bd'
        }

        # ==========================================
        # 3. Core Plotting Function
        # ==========================================
        def create_single_plot(data, title, filename_suffix):
            if data.empty:
                print(f"⚠️ Skipping: {title} (No data available)")
                return

            fig, ax = plt.subplots(figsize=(12, 9))
            kmfs = []
            groups = ['Q5', 'Q4', 'Q3', 'Q2', 'Q1']

            max_time = data[time_col_years].max()
            start_x = 0  # Starting from time zero
            end_x = max_time + 0.2

            # Generate standard x-ticks every 2 years
            custom_ticks = np.arange(int(start_x), int(max_time) + 2, 2)

            has_data = False
            for group in groups:
                mask = data['Quintile'] == group
                if mask.sum() == 0: continue
                has_data = True

                kmf = KaplanMeierFitter()
                kmf.fit(data.loc[mask, time_col_years],
                        event_observed=data.loc[mask, event_col],
                        label=group)

                # Plot cumulative density (Incidence)
                kmf.plot_cumulative_density(ax=ax, color=colors[group], linewidth=3.5,
                                            show_censors=False, ci_show=False)
                kmfs.append(kmf)

            if not has_data:
                plt.close(fig)
                return

            # --- Chart Styling ---
            ax.set_title(title, loc='left', fontsize=18, fontweight='bold', pad=25)
            ax.set_xlabel("Time (years)", fontsize=16, fontweight='bold')
            ax.set_ylabel("Cumulative event rate (%)", fontsize=16, fontweight='bold')

            # Format Y-axis as percentage
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))

            # Spine formatting
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.2)
            ax.spines['bottom'].set_linewidth(1.2)

            ax.set_xlim(left=start_x, right=end_x)
            ax.set_xticks(custom_ticks)
            ax.tick_params(axis='both', labelsize=14)
            ax.grid(axis='y', linestyle='--', alpha=0.3)

            # Legend
            handles, _ = ax.get_legend_handles_labels()
            long_labels = [f"{g} Protein risk score" for g in groups if any(k._label == g for k in kmfs)]
            ax.legend(handles, long_labels, loc='upper left', frameon=False, fontsize=16)

            # At-Risk Table
            if kmfs:
                add_at_risk_counts(*kmfs, ax=ax, rows_to_show=['At risk'])
                risk_ax = plt.gcf().axes[-1]
                for label in risk_ax.get_yticklabels():
                    text = label.get_text()
                    if text in colors:
                        label.set_color(colors[text])
                        label.set_fontweight('bold')
                        label.set_fontsize(16)
                risk_ax.tick_params(axis='x', labelsize=16)
                risk_ax.set_ylabel("Number at risk", fontsize=16, fontweight='bold', color='black', labelpad=20)

            plt.tight_layout()
            save_name = f"Incidence_Plot_{filename_suffix}_Final.png"
            save_path = os.path.join(desktop_path, save_name)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✅ Saved: {save_name}")


        # ==========================================
        # 4. Stratified Analysis Execution
        # ==========================================
        # Threshold for high risk (top 20th percentile)
        threshold = 0.8
        cutoff_SCORE2 = df['SCORE2'].quantile(threshold)
        cutoff_PRS = df['polygenic risk score'].quantile(threshold)
        cutoff_PREVENT = df['PREVENT'].quantile(threshold)

        print("Generating final publication-quality charts...")

        # Overall population
        create_single_plot(df, "Overall Population", "1_Overall")

        # Subgroup analysis
        mask2 = (df['SCORE2'] > cutoff_SCORE2) & (df['polygenic risk score'] > cutoff_PRS)
        create_single_plot(df[mask2], "High SCORE2 and High Polygenic risk score", "2_High_SCORE2_PRS")

        mask3 = (df['SCORE2'] > cutoff_SCORE2) & (df['PREVENT'] > cutoff_PREVENT)
        create_single_plot(df[mask3], "High SCORE2 and High PREVENT", "3_High_SCORE2_PREVENT")

        mask4 = (df['polygenic risk score'] > cutoff_PRS) & (df['PREVENT'] > cutoff_PREVENT)
        create_single_plot(df[mask4], "High Polygenic risk score and High PREVENT", "4_High_PRS_PREVENT")

        mask5 = (df['SCORE2'] > cutoff_SCORE2) & (df['polygenic risk score'] > cutoff_PRS) & (
                df['PREVENT'] > cutoff_PREVENT)
        create_single_plot(df[mask5], "High SCORE2, Polygenic risk score, and PREVENT", "5_High_All_Three")

        print(f"\n🎉 All charts generated! Check your Desktop for '_Final' files.")
else:
    print("❌ Critical Error: Input DataFrame is empty.")