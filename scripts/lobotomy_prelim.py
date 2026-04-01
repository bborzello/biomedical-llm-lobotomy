import torch
import torch.nn.utils.prune as prune
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
import os
import logging
import time
import csv

# Suppress warnings and logging
os.environ["HF_HUB_DISABLE_WARNINGS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
warnings.filterwarnings('ignore')

# Config
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SAMPLES = 100

print(f"Loading {MODEL_NAME} onto {DEVICE}...")


# Load Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)

# Load PubMedQA Dataset
print("Loading PubMedQA dataset...")
dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split=f"train[:{NUM_SAMPLES}]")

# Evaluation Function
def evaluate_model(current_model, dataset):
    current_model.eval()
    y_true = []
    y_pred = []

    print("Running inference loop...")
    with torch.no_grad():
        for i, item in enumerate(dataset):
            # Format the prompt for the model
            context = "".join(item['context']['contexts'])
            question = item['question']
            true_answer = item['final_decision'] # 'yes', 'no', or 'maybe'
            
            prompt = f"""<|system|>
                You are a highly accurate medical AI. You must answer the user's question with a single word: exactly 'yes', 'no', or 'maybe'. Do not explain your reasoning.</s>
                <|user|>
                Context: {context}
                Question: {question}</s>
                <|assistant|>
                """
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            
            # Generate a short answer
            outputs = current_model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
            decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the model's guess
            response = decoded_output[len(prompt):].strip().lower()
            
            # Classify the text output
            if "yes" in response:
                guess = "yes"
            elif "no" in response:
                guess = "no"
            else:
                guess = "maybe"

            y_true.append(true_answer)
            y_pred.append(guess)
            
            if (i + 1) % 25 == 0:
                print(f"  Processed {i + 1}/{NUM_SAMPLES} questions...")
            time.sleep(0.05)

    # Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, labels=["yes"], pos_label="yes", average="micro", zero_division=0)
    rec = recall_score(y_true, y_pred, labels=["yes"], pos_label="yes", average="micro", zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=["yes"], pos_label="yes", average="micro", zero_division=0)

    return acc, prec, rec, f1

# Layer-Wise L1 Unstructured Pruning Function
def apply_pruning(current_model, amount_to_prune, target_global_sparsity):
    torch.cuda.empty_cache()
    print(f"\nApplying Layer-Wise L1 Unstructured Pruning to reach {int(target_global_sparsity*100)}% sparsity...")
    print(f"  (Under the hood: pruning {amount_to_prune*100:.2f}% of the remaining active weights per layer)")
    
    for name, module in current_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(
                module, 
                name='weight', 
                amount=amount_to_prune
            )
            
    return current_model


# Main Execution Pipeline

log_sparsities = []
log_accuracy = []
log_precision = []
log_recall = []
log_f1 = []

# Baseline (0%)
print("\n--- BASELINE EVALUATION (0% Sparsity) ---")
acc_val, prec_val, rec_val, f1_val = evaluate_model(model, dataset)
print(f"Results -> Accuracy: {acc_val:.3f} | Precision: {prec_val:.3f} | Recall: {rec_val:.3f} | F1: {f1_val:.3f}")
log_sparsities.append(0)
log_accuracy.append(acc_val)
log_precision.append(prec_val)
log_recall.append(rec_val)
log_f1.append(f1_val)

# Iterative Pruning
target_sparsities = [0.20, 0.40, 0.60, 0.80]
current_sparsity = 0.0

for target in target_sparsities:
    amount_to_prune = (target - current_sparsity) / (1.0 - current_sparsity)
    
    model = apply_pruning(model, amount_to_prune, target)
    current_sparsity = target 
    
    print(f"\n--- EVALUATION ({int(target * 100)}% Sparsity) ---")
    acc_val, prec_val, rec_val, f1_val = evaluate_model(model, dataset)
    print(f"Results -> Accuracy: {acc_val:.3f} | Precision: {prec_val:.3f} | Recall: {rec_val:.3f} | F1: {f1_val:.3f}")
    
    log_sparsities.append(int(target * 100))
    log_accuracy.append(acc_val)
    log_precision.append(prec_val)
    log_recall.append(rec_val)
    log_f1.append(f1_val)

# Export to CSV
parent_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(parent_dir, "data")

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"Created directory: {data_dir}")

csv_filename = os.path.join(data_dir, "lobotomy_results.csv")

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Sparsity (%)", "Accuracy", "Precision", "Recall", "F1-Score"])
    for i in range(len(log_sparsities)):
        writer.writerow([log_sparsities[i], log_accuracy[i], log_precision[i], log_recall[i], log_f1[i]])

print(f"\nFull 20% interval lobotomy complete. Data saved to {csv_filename}!")