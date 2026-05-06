import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings('ignore')

DATA_DIR = os.path.join("..", "data")

AGGREGATE_FILES = {
    "Base Mistral": os.path.join(DATA_DIR, "Mistral_PubMedQA", "lobotomy_results_mistral_pubmedqa.csv"),
    "BioMistral": os.path.join(DATA_DIR, "BioMistral_PubMedQA", "lobotomy_results_biomistral_pubmedqa.csv")
}

MMLU_PREFIXES = {
    "Base Mistral": os.path.join(DATA_DIR, "Mistral_MMLU", "mmlu_mistralai_Mistral-7B-v0.1"),
    "BioMistral": os.path.join(DATA_DIR, "BioMistral_MMLU", "mmlu_BioMistral_BioMistral-7B")
}

LOGIC_SUBJECTS = ["formal_logic", "logical_fallacies", "global_facts"]
MEDICAL_SUBJECTS = ["college_medicine", "clinical_knowledge"]

SPARSITIES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

def calculate_domain_f1(df, subjects):
    subset = df[df['subject'].isin(subjects)]
    if subset.empty: return 0.0
    y_true = subset['true_answer'].tolist()
    y_pred = subset['parsed_guess'].tolist()
    return f1_score(y_true, y_pred, average="macro", zero_division=0)

def extract_mmlu_curve(prefix, subjects):
    f1s = []
    for s in SPARSITIES:
        file_path = f"{prefix}_{int(round(s*100))}_percent.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            f1s.append(calculate_domain_f1(df, subjects))
        else:
            f1s.append(0.0)
    return f1s

def calculate_auc(y_values):
    return np.trapezoid(y_values, SPARSITIES)

if __name__ == "__main__":
    print("-" * 50)
    print("ROBUSTNESS SCORES (Area Under the F1 Curve)")
    print("Higher = More resilient to structural damage")
    print("-" * 50)
    
    print("\n--- PubMedQA ---")
    for model_name, file_path in AGGREGATE_FILES.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df = df.sort_values(by='sparsity')
            auc = calculate_auc(df['f1'].tolist())
            print(f"{model_name}: {auc:.4f}")

    print("\n--- MMLU: Logic Domain ---")
    for model_name, prefix in MMLU_PREFIXES.items():
        f1s = extract_mmlu_curve(prefix, LOGIC_SUBJECTS)
        auc = calculate_auc(f1s)
        print(f"{model_name}: {auc:.4f}")

    print("\n--- MMLU: Medical Domain ---")
    for model_name, prefix in MMLU_PREFIXES.items():
        f1s = extract_mmlu_curve(prefix, MEDICAL_SUBJECTS)
        auc = calculate_auc(f1s)
        print(f"{model_name}: {auc:.4f}")
    
    print("-" * 50)