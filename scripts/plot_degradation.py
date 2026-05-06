import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings('ignore')

sns.set_theme(style="darkgrid")

DATA_DIR = os.path.join("..", "data")
FILES_DIR = os.path.join("..", "files")

os.makedirs(FILES_DIR, exist_ok=True)

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
    if subset.empty:
        return 0.0
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

def plot_pubmedqa():
    plt.figure(figsize=(10, 6))
    colors = {"Base Mistral": "#1f77b4", "BioMistral": "#d62728"} 
    
    for model_name, file_path in AGGREGATE_FILES.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            plt.plot(df['sparsity'] * 100, df['f1'], marker='o', linewidth=3, markersize=8, 
                     label=model_name, color=colors[model_name])
            
    plt.title('PubMedQA Degradation: Base vs. Fine-Tuned', fontsize=18, fontweight='bold')
    plt.xlabel('Pruning Sparsity (%)', fontsize=14)
    plt.ylabel('F1 Score', fontsize=14)
    plt.xlim(0, 95)
    plt.ylim(0, 1.0)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    out_path = os.path.join(FILES_DIR, "pubmedqa_degradation.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")
    plt.close()

def plot_mmlu_domain(subjects, domain_name, filename):
    plt.figure(figsize=(10, 6))
    colors = {"Base Mistral": "#1f77b4", "BioMistral": "#d62728"}
    
    for model_name, prefix in MMLU_PREFIXES.items():
        f1s = extract_mmlu_curve(prefix, subjects)
        plt.plot([s * 100 for s in SPARSITIES], f1s, marker='o', linewidth=3, markersize=8, 
                 label=model_name, color=colors[model_name])
        
    plt.title(f'MMLU {domain_name} Degradation: Base vs. Fine-Tuned', fontsize=18, fontweight='bold')
    plt.xlabel('Pruning Sparsity (%)', fontsize=14)
    plt.ylabel('F1 Score', fontsize=14)
    plt.xlim(0, 95)
    plt.ylim(0, 1.0)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    out_path = os.path.join(FILES_DIR, filename)
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")
    plt.close()

if __name__ == "__main__":
    print(f"Generating high-resolution graphics to {FILES_DIR}...")
    plot_pubmedqa()
    plot_mmlu_domain(LOGIC_SUBJECTS, "Logic", "mmlu_logic_degradation.png")
    plot_mmlu_domain(MEDICAL_SUBJECTS, "Medical", "mmlu_medical_degradation.png")
    print("Complete.")