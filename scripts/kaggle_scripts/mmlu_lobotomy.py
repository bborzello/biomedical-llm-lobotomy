# mmlu_lobotomy.py
import os
import pandas as pd
import torch
import warnings
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, concatenate_datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re

warnings.filterwarnings('ignore')
os.environ["HF_HUB_DISABLE_WARNINGS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

try:
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    login(token=hf_token)
except Exception as e:
    raise RuntimeError("HF_TOKEN secret not found. Check Kaggle Add-ons > Secrets.")

RESULTS_FILE = "/kaggle/working/lobotomy_results_biomistral_mmlu.csv"
MODEL_NAME = "BioMistral/BioMistral-7B"
# RESULTS_FILE = "/kaggle/working/lobotomy_results_mistral_mmlu.csv"
# MODEL_NAME = "mistralai/Mistral-7B-v0.1" 

LOGIC_SUBJECTS = ["formal_logic", "logical_fallacies", "global_facts"]
MEDICAL_SUBJECTS = ["college_medicine", "clinical_knowledge"]
SAMPLES_PER_SUBJECT = 200
target_sparsities = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
FORCE_NEW_RUN = True 

if FORCE_NEW_RUN and os.path.exists(RESULTS_FILE):
    os.remove(RESULTS_FILE)
    print("FORCE_NEW_RUN is True. Wiped previous checkpoint. Starting fresh.")

print(f"Loading {MODEL_NAME} across Dual T4 GPUs...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float16,
    device_map="auto",
    max_memory={0: "10GiB", 1: "10GiB"}, 
    use_safetensors=False 
)

def load_mmlu_subsets():
    datasets = []
    all_subjects = LOGIC_SUBJECTS + MEDICAL_SUBJECTS
    for sub in all_subjects:
        print(f"Loading MMLU Subject: {sub}...")
        ds = load_dataset("cais/mmlu", sub, split=f"test[:{SAMPLES_PER_SUBJECT}]")
        datasets.append(ds)
    return concatenate_datasets(datasets)

print("Compiling MMLU dataset...")
dataset = load_mmlu_subsets()
total_samples = len(dataset)

def evaluate_model(current_model, dataset):
    current_model.eval()
    y_true, y_pred = [], []
    raw_prompts, raw_truths, raw_preds, raw_subjects, raw_guesses = [], [], [], [], []
    letters = ['A', 'B', 'C', 'D']
    
    with torch.no_grad():
        for i, item in enumerate(dataset):
            question = item['question']
            choices = item['choices']
            true_answer_idx = item['answer']
            true_answer = letters[true_answer_idx]
            subject = item['subject']
            
            prompt = f"Question: {question}\nOptions:\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = current_model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
            
            decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = decoded_output[len(prompt):].strip().upper()
            
            guess = "X"
            match = re.search(r'\b([ABCD1234])\b', response)
            
            if match:
                extracted = match.group(1)
                mapping = {'1': 'A', '2': 'B', '3': 'C', '4': 'D'}
                guess = mapping.get(extracted, extracted)
            else:
                for idx, choice in enumerate(choices):
                    clean_choice = str(choice).strip().upper()
                    if clean_choice and clean_choice in response:
                        guess = letters[idx]
                        break
                    
            y_true.append(true_answer)
            y_pred.append(guess)
            
            raw_prompts.append(prompt)
            raw_truths.append(true_answer)
            raw_preds.append(response)
            raw_subjects.append(subject)
            raw_guesses.append(guess)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{total_samples} questions...")

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    return acc, prec, rec, f1, raw_prompts, raw_truths, raw_preds, raw_subjects, raw_guesses

def apply_pruning(current_model, target_global_sparsity):
    torch.cuda.empty_cache()
    print(f"\nApplying Memory-Safe Layer-Wise Pruning to {int(target_global_sparsity*100)}% sparsity...")
    with torch.no_grad():
        for name, module in current_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data
                abs_W = torch.abs(W)
                flattened_abs_W = abs_W.view(-1)
                k = int(target_global_sparsity * flattened_abs_W.numel())
                if k > 0:
                    sorted_W, _ = torch.sort(flattened_abs_W)
                    threshold = sorted_W[k]
                    W[abs_W < threshold] = 0.0
    return current_model

def save_raw_logs(prompts, truths, preds, subjects, guesses, target):
    log_df = pd.DataFrame({
        'subject': subjects,
        'prompt': prompts, 
        'true_answer': truths, 
        'raw_response': preds,
        'parsed_guess': guesses
    })
    safe_model_name = MODEL_NAME.replace("/", "_")
    log_df.to_csv(f"/kaggle/working/mmlu_{safe_model_name}_{int(target*100)}_percent.csv", index=False)

completed_sparsities = []
if os.path.exists(RESULTS_FILE):
    df = pd.read_csv(RESULTS_FILE)
    completed_sparsities = df['sparsity'].tolist()
    print(f"Recovered state. Completed intervals: {completed_sparsities}")
    
    if completed_sparsities:
        max_completed = max(completed_sparsities)
        if max_completed > 0.0:
            print(f"Fast-forwarding directly to {int(max_completed*100)}% sparsity.")
            model = apply_pruning(model, max_completed)
else:
    print("No checkpoint found. Initializing Baseline run.")
    acc, prec, rec, f1, raw_p, raw_t, raw_pred, raw_sub, raw_g = evaluate_model(model, dataset)
    df = pd.DataFrame([{'sparsity': 0.0, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}])
    df.to_csv(RESULTS_FILE, index=False)
    save_raw_logs(raw_p, raw_t, raw_pred, raw_sub, raw_g, 0.0)
    completed_sparsities = [0.0]

for target in target_sparsities:
    if any(abs(target - c) < 1e-5 for c in completed_sparsities):
        continue
        
    print(f"\nExecuting target: {int(target*100)}%")
    model = apply_pruning(model, target)
    acc, prec, rec, f1, raw_p, raw_t, raw_pred, raw_sub, raw_g = evaluate_model(model, dataset)
    
    new_row = pd.DataFrame([{'sparsity': target, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(RESULTS_FILE, index=False)
    save_raw_logs(raw_p, raw_t, raw_pred, raw_sub, raw_g, target)
    print(f"Checkpoint secured for {int(target*100)}%.")

print("\nBatch Execution Complete.")