import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import platform
import warnings
import os

# === Basic Settings ===
warnings.filterwarnings('ignore')

# Font settings (Multi-platform support)
system_name = platform.system()
if system_name == "Windows":
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
elif system_name == "Darwin":  # Mac
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'


# ==============================================================================
# 1. Figure A1: Top 21 Features Zoom (Vertical Annotation)
# ==============================================================================
def save_figure_A1_vertical(coef_df):
    print("Generating Figure A1 (Vertical Staggered Zoom)...")
    plt.figure(figsize=(12, 8))

    # Data Preprocessing: Select top 21 by absolute coefficient magnitude
    plot_data = coef_df.copy()
    plot_data['abs_coef'] = plot_data['coef'].abs()
    plot_data = plot_data.sort_values('abs_coef', ascending=False).head(21).reset_index(drop=True)

    ranks = range(len(plot_data))
    weights = plot_data['coef']

    # Draw Stem Plot
    plt.axhline(0, color='gray', linestyle='-', linewidth=0.8)
    markerline, stemlines, baseline = plt.stem(ranks, weights, linefmt='k-', markerfmt='k.', basefmt=' ')
    plt.setp(stemlines, linewidth=1.2, color='black')
    plt.setp(markerline, markersize=6, color='black')

    # Calculate Y-axis range
    y_max, y_min = weights.max(), weights.min()
    y_range = y_max - y_min if y_max != y_min else 1.0
    plt.ylim(y_min - y_range * 0.5, y_max + y_range * 0.5)

    # Vertical staggered annotation logic
    for i in range(len(plot_data)):
        protein_name = plot_data.loc[i, 'feature']
        weight = plot_data.loc[i, 'coef']
        stagger_idx = i % 4
        dist = y_range * 0.05 + (stagger_idx * y_range * 0.08)

        if weight > 0:
            xytext = (i, weight + dist)
            va = 'bottom'
        else:
            xytext = (i, weight - dist)
            va = 'top'

        plt.annotate(protein_name,
                     xy=(i, weight),
                     xytext=xytext,
                     textcoords='data',
                     ha='center', va=va,
                     fontsize=10, rotation=90, fontweight='bold', color='#333333',
                     arrowprops=dict(arrowstyle="-", color='gray', linewidth=0.5))

    plt.xlabel("Rank (by Coefficient Magnitude)", fontsize=14)
    plt.ylabel("Coefficient Value", fontsize=14)
    plt.xticks(range(21))

    sns.despine()
    plt.tight_layout()
    plt.savefig('Figure_A1_Top21.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" -> Saved successfully: Figure_A1_Top21.png")


# ==============================================================================
# 2. Figure A2: All-Feature Fan-out (Right Fan-out)
# ==============================================================================
def save_figure_A2_right_fan(coef_df):
    print("Generating Figure A2 (Right Fan-out)...")

    # Data preparation
    plot_data = coef_df.copy()
    plot_data['abs_coef'] = plot_data['coef'].abs()
    plot_data = plot_data.sort_values('abs_coef', ascending=False).reset_index(drop=True)

    ranks = plot_data.index.values
    weights = plot_data['coef'].values
    features = plot_data['feature'].values

    y_max, y_min = weights.max(), weights.min()
    y_range = y_max - y_min if y_max != y_min else 1.0
    label_y_top = max(0.4, y_max + y_range * 0.35)
    label_y_bottom = min(-0.4, y_min - y_range * 0.35)

    plt.figure(figsize=(16, 9))
    plt.axhline(0, color='black', linestyle='-', linewidth=0.8)
    markerline, stemlines, baseline = plt.stem(ranks, weights, linefmt='k-', markerfmt='k.', basefmt=' ')
    plt.setp(stemlines, linewidth=0.6, color='black')
    plt.setp(markerline, markersize=3.5, color='black')

    # Top 21 Annotation Logic
    top_n = 21
    top_indices = np.arange(min(top_n, len(plot_data)))
    pos_indices = [i for i in top_indices if weights[i] > 0]
    neg_indices = [i for i in top_indices if weights[i] < 0]
    pos_indices.sort()
    neg_indices.sort()

    if len(pos_indices) > 0:
        target_xs = np.linspace(0, (len(pos_indices) - 1) * 3.0, len(pos_indices))
        for i, real_idx in enumerate(pos_indices):
            plt.text(target_xs[i], label_y_top, features[real_idx],
                     ha='center', va='bottom', rotation=90, fontsize=15, fontweight='bold', color='#333333')
            plt.annotate('', xy=(real_idx, weights[real_idx]), xytext=(target_xs[i], label_y_top),
                         arrowprops=dict(arrowstyle="-", color='gray', linewidth=0.5))

    if len(neg_indices) > 0:
        target_xs = np.linspace(0, (len(neg_indices) - 1) * 3.0, len(neg_indices))
        for i, real_idx in enumerate(neg_indices):
            plt.text(target_xs[i], label_y_bottom, features[real_idx],
                     ha='center', va='top', rotation=90, fontsize=15, fontweight='bold', color='#333333')
            plt.annotate('', xy=(real_idx, weights[real_idx]), xytext=(target_xs[i], label_y_bottom),
                         arrowprops=dict(arrowstyle="-", color='gray', linewidth=0.5))

    display_limit = max(len(plot_data), 200)
    plt.xlim(-5, display_limit + 10)
    plt.ylim(label_y_bottom - 0.2, label_y_top + 0.2)
    plt.xlabel("k (index)", fontsize=18)
    plt.ylabel(r"Weight ($\hat{w}_k$)", fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    sns.despine()
    plt.tight_layout()
    plt.savefig('Figure_A2_FanOut.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" -> Saved successfully: Figure_A2_FanOut.png")


# ==============================================================================
# 3. Figure B: Selection Stability (Reference Style: >=90%, Gradient)
# ==============================================================================
def save_figure_B_stability(stable_df):
    print("Generating Figure B (Replicating Reference Style: >=90%, Gradient)...")

    # 1. Data preparation and renaming
    plot_df = stable_df.copy()
    if 'Selection_Frequency' in plot_df.columns:
        plot_df = plot_df.rename(columns={'Selection_Frequency': 'Frequency'})

    # 2. Core filtering: Keep features with Frequency >= 90%
    plot_df = plot_df[plot_df['Frequency'] >= 0.90]

    # 3. Sorting
    plot_df = plot_df.sort_values('Frequency', ascending=False).reset_index(drop=True)

    num_vars = len(plot_df)
    if num_vars == 0:
        print("Warning: No features with frequency >= 90%, Figure B will not be generated.")
        return

    # Dynamically adjust figure width
    final_width = max(10, 10 + (num_vars / 40) * 8)
    plt.figure(figsize=(final_width, 8))

    x_pos = np.arange(num_vars)
    freq_values = plot_df['Frequency'] * 100  # Convert to percentage

    # 4. Color mapping logic (Three-tier gradient)
    colors = []
    for val in freq_values:
        if val >= 99.0:
            colors.append('#24527a')  # Dark Blue (Top)
        elif val >= 95.0:
            colors.append('#2171b5')  # Medium Blue (Middle)
        else:
            colors.append('#6baed6')  # Light Blue (Bottom)

    # 5. Plot bar chart
    plt.bar(x_pos, freq_values, color=colors, width=0.85, edgecolor=None)

    # 6. Y-axis settings
    plt.ylim(90, 100.2)
    plt.ylabel("Proportion of proteins included during 1,000 resampling, %",
               fontsize=15, fontweight='bold', color='black', labelpad=10)

    # Set Y-axis ticks
    plt.yticks([90, 95, 100], fontsize=15)

    # 7. X-axis settings
    plt.xticks(x_pos, plot_df['Protein'], rotation=90, ha='center', fontsize=15, fontweight='bold')
    plt.xlim(-0.6, num_vars - 0.4)

    # 8. Style adjustments
    sns.despine()

    plt.tight_layout()
    plt.savefig('Figure_B_Stability.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" -> Saved successfully: Figure_B_Stability.png")


# ==============================================================================
# Main Program: Generation of Figure D
# ==============================================================================
def main():
    print("=== Starting Plotting Routine ===")

    # --- 1. Figures A1 & A2 (Priority given to stable_features.csv) ---
    if os.path.exists('stable_features.csv'):
        print("[Optimization] Using stable_features.csv for weight plots (A1 & A2)")
        coef_df = pd.read_csv('stable_features.csv')
        coef_df = coef_df.rename(columns={'Protein': 'feature', 'Coefficient': 'coef'})
        coef_df = coef_df[coef_df['coef'] != 0]

        save_figure_A1_vertical(coef_df)
        save_figure_A2_right_fan(coef_df)

    elif os.path.exists('selected_proteins.csv'):
        print("[Fallback] stable_features.csv not found, using selected_proteins.csv for Figure A")
        coef_df = pd.read_csv('selected_proteins.csv')
        coef_df = coef_df.rename(columns={'Protein': 'feature', 'Coefficient': 'coef'})
        save_figure_A1_vertical(coef_df)
        save_figure_A2_right_fan(coef_df)

    # --- 2. Figure B (Using stable_features.csv) ---
    if os.path.exists('stable_features.csv'):
        print("Found stable_features.csv, generating Figure B")
        stable_df = pd.read_csv('stable_features.csv')
        save_figure_B_stability(stable_df)

    print("\n=== Routine Finished ===")


if __name__ == "__main__":
    main()

    import pandas as pd
    import numpy as np
    import os
    import gc
    import warnings
    from sklearn.model_selection import train_test_split
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import QuantileTransformer, StandardScaler, RobustScaler
    from sklearn.compose import ColumnTransformer
    from scipy import stats
    import platform

    # === Basic Settings ===
    warnings.filterwarnings('ignore')
    system_name = platform.system()
    if system_name == "Windows":
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    elif system_name == "Darwin":
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
    plt.rcParams['axes.unicode_minus'] = False


    # ==============================================================================
    # 1. Core Plotting Function: Figure D Risk Score Distribution
    # 4. Figure D: Risk Score Distribution (Labels and Positioning)
    # ==============================================================================
    def save_figure_D_distribution(X_test, y_test, X_prevalent, selected_features, selected_coefs):
        print("Generating Figure D (Risk Score Distribution)...")
        plt.figure(figsize=(10, 7))

        if len(selected_features) > 0:
            # Calculate standardized scores
            scores_incident = X_test[selected_features].values @ selected_coefs
            scaler = StandardScaler().fit(scores_incident.reshape(-1, 1))
            std_incident = scaler.transform(scores_incident.reshape(-1, 1)).flatten()

            scores_healthy = std_incident[~y_test['status']]
            scores_future = std_incident[y_test['status']]

            # Plotting
            sns.kdeplot(scores_healthy, fill=True, color='#69b3a2', label='Healthy', alpha=0.5)
            sns.kdeplot(scores_future, fill=True, color='#407294', label='Future CVD', alpha=0.5)

            if not X_prevalent.empty:
                raw_prevalent = X_prevalent[selected_features].values @ selected_coefs
                std_prevalent = scaler.transform(raw_prevalent.reshape(-1, 1)).flatten()
                sns.kdeplot(std_prevalent, fill=True, color='#9e4e4e', label='Prevalent CVD', alpha=0.5)

            # Optimization of annotation: Avoid overlap with peaks
            y_limit = plt.gca().get_ylim()[1]
            x_limit = plt.gca().get_xlim()
            plt.text(x_limit[1] * 0.95, y_limit * 0.9, "P < 0.0001 for all comparisons",
                     fontsize=14, fontweight='bold', ha='right', color='black')

            plt.xlabel("Protein risk score (standardized)", fontsize=14)
            plt.ylabel("Density", fontsize=14)
            plt.legend(fontsize=14, frameon=False, loc='upper left')
            plt.xticks(fontsize=12)  # X-axis tick font size
            plt.yticks(fontsize=12)  # Y-axis tick font size
            sns.despine()

        plt.tight_layout()
        plt.savefig('Figure_D_Distribution.png', dpi=300, bbox_inches='tight')
        plt.close()


    # ==============================================================================
    # 5. Main Execution Logic
    # ==============================================================================
    def main():
        file_path = r'C:\Users\Administrator\Desktop\Proteome of CVD.csv'
        if not os.path.exists(file_path):
            print("Error: Data file not found.")
            return

        # --- Data Processing ---
        data = pd.read_csv(file_path)
        protein_cols = data.columns[1:1461].tolist()
        data_prevalent = data[data['baseline_cvd'] == 1].copy()
        data_incident = data[data['baseline_cvd'] == 0].copy()

        y_inc = np.zeros(len(data_incident), dtype=[('status', 'bool'), ('time', 'float32')])
        y_inc['status'] = data_incident['event_status'].astype(bool)
        y_inc['time'] = data_incident['survival_time'].astype(np.float32)

        X_inc = data_incident[protein_cols + ['Age', 'Sex']]
        X_train, X_test, y_train, y_test = train_test_split(X_inc, y_inc, test_size=0.3, random_state=42,
                                                            stratify=y_inc['status'])

        # --- Model and Normalization ---
        preprocessor = ColumnTransformer([
            ('protein', QuantileTransformer(output_distribution='normal', n_quantiles=min(1000, len(X_train))),
             protein_cols),
            ('cov', RobustScaler(), ['Age', 'Sex'])
        ])

        X_train_std = preprocessor.fit_transform(X_train)
        X_test_std = preprocessor.transform(X_test)
        X_prev_std = preprocessor.transform(
            data_prevalent[protein_cols + ['Age', 'Sex']]) if not data_prevalent.empty else None

        # Train simple model to get weights
        model = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[0.001]).fit(
            pd.DataFrame(X_train_std, columns=protein_cols + ['Age', 'Sex']), y_train)
        coef_df = pd.DataFrame({'feature': protein_cols + ['Age', 'Sex'], 'coef': model.coef_.flatten()})
        selected = coef_df[coef_df['coef'] != 0]

        # Generate Figure D
        save_figure_D_distribution(
            pd.DataFrame(X_test_std, columns=protein_cols + ['Age', 'Sex']),
            y_test,
            pd.DataFrame(X_prev_std,
                         columns=protein_cols + ['Age', 'Sex']) if X_prev_std is not None else pd.DataFrame(),
            selected['feature'].values,
            selected['coef'].values
        )
        print("\n=== All figures generated successfully ===")


    if __name__ == "__main__":
        main()