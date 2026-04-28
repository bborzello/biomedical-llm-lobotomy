import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings('ignore')

DATA_DIR = os.path.join("..", "data")
FILES_DIR = os.path.join("..", "files")

MISTRAL_0 = os.path.join(DATA_DIR, "Mistral_PubMedQA", "mistralai_Mistral-7B-v0.1_pubmedqa_0_percent.csv")
BIOMISTRAL_0 = os.path.join(DATA_DIR, "BioMistral_PubMedQA", "BioMistral_BioMistral-7B_pubmedqa_0_percent.csv")

def get_distribution(file_path, column_name):
    if not os.path.exists(file_path):
        return {'yes': 0, 'no': 0, 'maybe': 0, 'invalid/X': 0}
    
    df = pd.read_csv(file_path)
    counts = df[column_name].value_counts(normalize=True).to_dict()
    
    return {
        'yes': counts.get('yes', 0.0) * 100,
        'no': counts.get('no', 0.0) * 100,
        'maybe': counts.get('maybe', 0.0) * 100,
        'invalid/X': counts.get('X', 0.0) * 100 
    }

def plot_horizontal_bias():
    print("Generating horizontal baseline distribution...")
    
    truth_dist = get_distribution(MISTRAL_0, 'true_answer')
    mistral_dist = get_distribution(MISTRAL_0, 'parsed_guess')
    biomistral_dist = get_distribution(BIOMISTRAL_0, 'parsed_guess')

    labels = ['BioMistral', 'Base Mistral', 'Ground Truth']
    
    yes_data = [biomistral_dist['yes'], mistral_dist['yes'], truth_dist['yes']]
    no_data = [biomistral_dist['no'], mistral_dist['no'], truth_dist['no']]
    maybe_data = [
        biomistral_dist['maybe'] + biomistral_dist.get('invalid/X', 0), 
        mistral_dist['maybe'] + mistral_dist.get('invalid/X', 0), 
        truth_dist['maybe'] + truth_dist.get('invalid/X', 0)
    ]

    y = np.arange(len(labels))
    height = 0.6

    fig, ax = plt.subplots(figsize=(10, 4)) # Wide and short

    ax.barh(y, yes_data, height, label='"Yes"', color='#2ca02c', edgecolor='white')
    ax.barh(y, no_data, height, left=yes_data, label='"No"', color='#d62728', edgecolor='white')
    ax.barh(y, maybe_data, height, left=np.array(yes_data)+np.array(no_data), label='"Maybe" / Invalid', color='#7f7f7f', edgecolor='white')

    ax.set_xlabel('Percentage of Responses (%)', fontsize=12, fontweight='bold')
    ax.set_title('The Zero-Shot Affirmative Bias on PubMedQA', fontsize=14, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_xlim(0, 100)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=11)

    for i, v in enumerate(yes_data):
        ax.text(v / 2, i, f"{v:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=12)

    plt.tight_layout()
    out_path = os.path.join(FILES_DIR, "pubmedqa_affirmative_bias_horizontal.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved bias visualization to: {out_path}")
    plt.close()

if __name__ == "__main__":
    plot_horizontal_bias()