# pubmedqa_lobotomy.py
import os
import pandas as pd
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient

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
    raise RuntimeError("HF_TOKEN secret not found. You must configure this in Kaggle Add-ons > Secrets to download Mistral-7B.")

RESULTS_FILE = "/kaggle/working/lobotomy_results_biomistral_pubmedqa.csv"
MODEL_NAME = "BioMistral/BioMistral-7B"
# RESULTS_FILE = "/kaggle/working/lobotomy_results_mistral_pubmedqa.csv"
# MODEL_NAME = "mistralai/Mistral-7B-v0.1"

# Set to 10 for dry run, revert to 1000 for production
NUM_SAMPLES = 1000 
target_sparsities = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
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

print(f"Loading PubMedQA dataset (Samples: {NUM_SAMPLES})...")
dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split=f"train[:{NUM_SAMPLES}]")

def evaluate_model(current_model, dataset):
    current_model.eval()
    y_true, y_pred, raw_prompts, raw_truths, raw_preds, raw_guesses = [], [], [], [], [], []
    
    with torch.no_grad():
        for i, item in enumerate(dataset):
            context = "".join(item['context']['contexts'])
            question = item['question']
            true_answer = item['final_decision']
            
            prompt = f"Context: {context}\nQuestion: {question}\nAnswer (yes/no/maybe):"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = current_model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
            
            decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = decoded_output[len(prompt):].strip().lower()
            
            if "yes" in response: guess = "yes"
            elif "no" in response: guess = "no"
            else: guess = "maybe"

            y_true.append(true_answer)
            y_pred.append(guess)
            raw_prompts.append(prompt)
            raw_truths.append(true_answer)
            raw_preds.append(response)
            raw_guesses.append(guess)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(dataset)} questions...")

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, labels=["yes"], pos_label="yes", average="micro", zero_division=0)
    rec = recall_score(y_true, y_pred, labels=["yes"], pos_label="yes", average="micro", zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=["yes"], pos_label="yes", average="micro", zero_division=0)
    return acc, prec, rec, f1, raw_prompts, raw_truths, raw_preds, raw_guesses

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

def save_raw_logs(prompts, truths, preds, guesses, target):
    log_df = pd.DataFrame({
        'prompt': prompts, 
        'true_answer': truths, 
        'raw_response': preds,
        'parsed_guess': guesses
    })
    safe_name = MODEL_NAME.replace("/", "_")
    log_df.to_csv(f"/kaggle/working/{safe_name}_pubmedqa_{int(target*100)}_percent.csv", index=False)

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
    acc, prec, rec, f1, raw_p, raw_t, raw_pred, raw_g = evaluate_model(model, dataset)
    df = pd.DataFrame([{'sparsity': 0.0, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}])
    df.to_csv(RESULTS_FILE, index=False)
    save_raw_logs(raw_p, raw_t, raw_pred, raw_g, 0.0)
    completed_sparsities = [0.0]

for target in target_sparsities:
    if any(abs(target - c) < 1e-5 for c in completed_sparsities):
        continue
        
    print(f"\nExecuting target: {int(target*100)}%")
    model = apply_pruning(model, target)
    acc, prec, rec, f1, raw_p, raw_t, raw_pred, raw_g = evaluate_model(model, dataset)
    
    new_row = pd.DataFrame([{'sparsity': target, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(RESULTS_FILE, index=False)
    save_raw_logs(raw_p, raw_t, raw_pred, raw_g, target)
    print(f"Checkpoint secured for {int(target*100)}%.")

print("\nBatch Execution Complete.")