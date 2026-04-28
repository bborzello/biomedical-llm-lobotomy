import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# --- PATH CONFIGURATION ---
DATA_DIR = os.path.join("..", "data")
FILES_DIR = os.path.join("..", "files")
os.makedirs(FILES_DIR, exist_ok=True)

# Exact filenames based on your directory structure
MISTRAL_0 = os.path.join(DATA_DIR, "Mistral_PubMedQA", "mistralai_Mistral-7B-v0.1_pubmedqa_0_percent.csv")
BIOMISTRAL_0 = os.path.join(DATA_DIR, "BioMistral_PubMedQA", "BioMistral_BioMistral-7B_pubmedqa_0_percent.csv")

def get_distribution(file_path, column_name):
    if not os.path.exists(file_path):
        print(f"Missing file: {file_path}")
        return {'yes': 0, 'no': 0, 'maybe': 0}
    
    df = pd.read_csv(file_path)
    # Count occurrences and normalize to percentages
    counts = df[column_name].value_counts(normalize=True).to_dict()
    
    # Ensure all keys exist
    return {
        'yes': counts.get('yes', 0.0) * 100,
        'no': counts.get('no', 0.0) * 100,
        'maybe': counts.get('maybe', 0.0) * 100,
        # Group 'X' (hallucinations) with 'maybe' or track separately if needed. 
        # For baseline, 'X' should be near 0.
        'invalid/X': counts.get('X', 0.0) * 100 
    }

def plot_affirmative_bias():
    print("Calculating baseline distributions...")
    
    # Ground truth is the same for both, just pull from Mistral's file
    truth_dist = get_distribution(MISTRAL_0, 'true_answer')
    mistral_dist = get_distribution(MISTRAL_0, 'parsed_guess')
    biomistral_dist = get_distribution(BIOMISTRAL_0, 'parsed_guess')

    labels = ['Ground Truth\n(Actual Dataset)', 'Base Mistral\n(Predictions)', 'BioMistral\n(Predictions)']
    
    yes_data = [truth_dist['yes'], mistral_dist['yes'], biomistral_dist['yes']]
    no_data = [truth_dist['no'], mistral_dist['no'], biomistral_dist['no']]
    maybe_data = [truth_dist['maybe'] + truth_dist.get('invalid/X', 0), 
                  mistral_dist['maybe'] + mistral_dist.get('invalid/X', 0), 
                  biomistral_dist['maybe'] + biomistral_dist.get('invalid/X', 0)]

    x = np.arange(len(labels))
    width = 0.6

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plotting stacked bars. Green for Yes to highlight the bias volume.
    ax.bar(x, yes_data, width, label='"Yes"', color='#2ca02c', edgecolor='white')
    ax.bar(x, no_data, width, bottom=yes_data, label='"No"', color='#d62728', edgecolor='white')
    ax.bar(x, maybe_data, width, bottom=np.array(yes_data)+np.array(no_data), label='"Maybe" / Invalid', color='#7f7f7f', edgecolor='white')

    ax.set_ylabel('Percentage of Responses (%)', fontsize=12, fontweight='bold')
    ax.set_title('The Zero-Shot Affirmative Bias on PubMedQA', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

    # Add text labels inside the "Yes" bars to hammer the point home
    for i, v in enumerate(yes_data):
        ax.text(i, v / 2, f"{v:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=12)

    plt.tight_layout()
    out_path = os.path.join(FILES_DIR, "pubmedqa_affirmative_bias.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved bias visualization to: {out_path}")
    plt.close()

if __name__ == "__main__":
    plot_affirmative_bias()