import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import chardet


def detect_encoding(file_path):
    """Detect file encoding"""
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)  # Read first 10000 bytes for encoding detection
        result = chardet.detect(raw_data)
        return result['encoding']


def safe_read_csv(file_path, **kwargs):
    """Safely read CSV file with automatic encoding handling"""
    # Attempt to detect encoding
    try:
        encoding = detect_encoding(file_path)
        print(f"Detected file encoding: {encoding}")
    except:
        encoding = 'utf-8'
        print("Using default encoding: utf-8")

    # Try reading with detected encoding
    try:
        data = pd.read_csv(file_path, encoding=encoding, **kwargs)
        return data
    except (UnicodeDecodeError, TypeError):
        # If failed, try common alternative encodings
        encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'gbk']

        for enc in encodings_to_try:
            try:
                print(f"Attempting encoding: {enc}")
                data = pd.read_csv(file_path, encoding=enc, **kwargs)
                print(f"Successfully used encoding: {enc}")
                return data
            except (UnicodeDecodeError, TypeError):
                continue

        # If all fail, use errors='ignore'
        print("All encoding attempts failed, using errors='ignore'")
        data = pd.read_csv(file_path, encoding='utf-8', errors='ignore', **kwargs)
        return data


def calculate_protein_risk_scores():
    """
    Calculates the protein risk score for each patient.
    """
    # 1. Load protein coefficient data
    print("Loading protein coefficient data...")

    # Prioritize stable features file if it exists
    if os.path.exists('stable_features.csv'):
        coef_df = pd.read_csv('stable_features.csv')
        print(f"Using stable features, total: {len(coef_df)} proteins")
    elif os.path.exists('selected_proteins.csv'):
        coef_df = pd.read_csv('selected_proteins.csv')
        print(f"Using initial selected features, total: {len(coef_df)} proteins")
    else:
        raise FileNotFoundError("Protein coefficient file not found (selected_proteins.csv or stable_features.csv)")

    # Create mapping dictionary from protein to coefficient
    protein_coef_dict = dict(zip(coef_df['Protein'], coef_df['Coefficient']))

    # 2. Load raw data
    print("Loading raw data...")
    file_path = r'C:\Users\Administrator\Desktop\Proteome of CVD.csv'

    # Determine columns to load (only protein columns and ID column)
    protein_columns = list(protein_coef_dict.keys())

    print("Detecting file encoding and reading column names...")

    # Safely read header information
    try:
        header_data = safe_read_csv(file_path, nrows=0)
        header = header_data.columns.tolist()
        print(f"Successfully read column names, total: {len(header)} columns")
    except Exception as e:
        print(f"Failed to read column names: {e}")
        # Binary mode fallback
        with open(file_path, 'rb') as f:
            first_line = f.readline()
            encodings = ['utf-8', 'gbk', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    header_line = first_line.decode(encoding).strip()
                    header = header_line.split(',')
                    print(f"Used encoding {encoding} to successfully read header")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                header_line = first_line.decode('utf-8', errors='replace').strip()
                header = header_line.split(',')
                print("Used utf-8 with errors='replace' to read header")

    # Try to find a patient ID column
    potential_id_cols = [col for col in header if any(x in col.lower() for x in ['participant', 'id', 'patient'])]

    if potential_id_cols:
        id_col = potential_id_cols[0]
        print(f"Using '{id_col}' as patient ID column")
    else:
        id_col = None
        print("No obvious ID column found, using row index as patient ID")

    # Determine columns to load
    use_cols = protein_columns.copy()
    if id_col and id_col not in use_cols:
        use_cols = [id_col] + protein_columns

    print(f"Preparing to load {len(use_cols)} data columns")

    # Load data safely
    data = safe_read_csv(file_path, usecols=use_cols, low_memory=False)

    # Process missing values (consistent with model training)
    data.replace([np.inf, -np.inf, 'NA', 'N/A', 'NaN', 'null', 'NULL', '?', '-'], np.nan, inplace=True)
    data.fillna(0, inplace=True)

    print(f"Original data shape: {data.shape}")

    # 3. Calculate risk score for each patient
    print("Calculating risk scores...")

    # Ensure all required protein columns exist
    missing_proteins = set(protein_columns) - set(data.columns)
    if missing_proteins:
        print(f"WARNING: Data is missing the following protein columns: {missing_proteins}")
        # Add missing columns with 0 values
        for protein in missing_proteins:
            data[protein] = 0

    # Calculate risk scores
    risk_scores = []
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Calculating risk scores"):
        score = 0
        for protein, coef in protein_coef_dict.items():
            if protein in row:
                score += row[protein] * coef
        risk_scores.append(score)

    # 4. Create result DataFrame
    result_df = pd.DataFrame()

    if id_col and id_col in data.columns:
        result_df['patient_id'] = data[id_col]
    else:
        result_df['patient_id'] = data.index

    result_df['protein_risk_score'] = risk_scores

    # 5. Save results
    output_file = 'patient_protein_risk_scores.csv'
    result_df.to_csv(output_file, index=False)
    print(f"Risk scores saved to {output_file}")

    # 6. Statistics
    print("\nRisk Score Statistics:")
    print(f"Patient Count: {len(result_df)}")
    print(f"Range: {result_df['protein_risk_score'].min():.4f} - {result_df['protein_risk_score'].max():.4f}")
    print(f"Mean: {result_df['protein_risk_score'].mean():.4f}")
    print(f"Std Dev: {result_df['protein_risk_score'].std():.4f}")

    # 7. Visualization
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.hist(result_df['protein_risk_score'], bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.title('Distribution of Protein Risk Scores')
        plt.xlabel('Risk Score')
        plt.ylabel('Number of Patients')
        plt.grid(True, alpha=0.3)
        plt.savefig('risk_score_distribution.png', dpi=150, bbox_inches='tight')
        print("Risk score distribution plot saved to risk_score_distribution.png")
    except ImportError:
        print("Matplotlib not available, skipping plot generation")

    return result_df


# Execution
if __name__ == "__main__":
    risk_scores_df = calculate_protein_risk_scores()

    # Display preview
    print("\nPreview of the first 10 patient risk scores:")
    print(risk_scores_df.head(10))