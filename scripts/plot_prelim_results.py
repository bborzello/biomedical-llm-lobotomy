import matplotlib.pyplot as plt
import csv
from pathlib import Path

sparsities, accuracy, precision, recall, f1_score = [], [], [], [], []

current_file = Path(__file__).resolve()
current_dir = current_file.parent
data_file = current_dir.parent / "data" / "lobotomy_results.csv"

with open(data_file, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        sparsities.append(int(row["Sparsity (%)"]))
        accuracy.append(float(row["Accuracy"]))
        precision.append(float(row["Precision"]))
        recall.append(float(row["Recall"]))
        f1_score.append(float(row["F1-Score"]))

plt.figure(figsize=(10, 6))
plt.style.use('seaborn-v0_8-whitegrid')

plt.plot(sparsities, accuracy, marker='o', linewidth=2, label='Accuracy', color='#1f77b4')
plt.plot(sparsities, precision, marker='s', linewidth=2, label='Precision', color='#ff7f0e')
plt.plot(sparsities, recall, marker='^', linewidth=2, label='Recall', color='#2ca02c')
plt.plot(sparsities, f1_score, marker='D', linewidth=2, label='F1-Score', color='#d62728', linestyle='--')

plt.title('TinyLlama-1.1B PubMedQA Resilience Curve (Layer-Wise L1 Pruning)', fontsize=14, pad=15)
plt.xlabel('Sparsity Threshold (%)', fontsize=12)
plt.ylabel('Metric Score', fontsize=12)
plt.xticks(sparsities)
plt.ylim(-0.05, 1.05)
plt.legend(loc='upper right', fontsize=10, frameon=True, shadow=True)

output_file = current_dir.parent / "files" / "results.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Graph successfully saved as {output_file}!")

plt.show()