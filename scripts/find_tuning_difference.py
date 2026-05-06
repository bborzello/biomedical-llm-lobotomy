import pandas as pd
import os

# SPARSITY = 50
SPARSITY = 75

MISTRAL_FILE = f"../data/Mistral_MMLU/mmlu_mistralai_Mistral-7B-v0.1_{SPARSITY}_percent.csv"
BIOMISTRAL_FILE = f"../data/BioMistral_MMLU/mmlu_BioMistral_BioMistral-7B_{SPARSITY}_percent.csv"

def find_qualitative_examples():
    if not os.path.exists(MISTRAL_FILE) or not os.path.exists(BIOMISTRAL_FILE):
        print(f"Error: Missing raw generation files for {SPARSITY}% sparsity.")
        return

    # Load the generation logs
    df_mistral = pd.read_csv(MISTRAL_FILE)
    df_biomistral = pd.read_csv(BIOMISTRAL_FILE)

    df_merged = df_mistral.merge(
        df_biomistral, 
        on=['subject', 'prompt', 'true_answer'], 
        suffixes=('_mistral', '_bio')
    )

    medical_subjects = ["college_medicine", "clinical_knowledge"]
    df_merged = df_merged[df_merged['subject'].isin(medical_subjects)]
    condition = (df_merged['parsed_guess_mistral'] == 'X') & (df_merged['parsed_guess_bio'] == df_merged['true_answer'])
    
    examples = df_merged[condition]

    if examples.empty:
        print(f"No perfect examples at {SPARSITY}%. Try changing SPARSITY to 45 or 55.")
    else:
        print(f"Found {len(examples)} perfect examples!\n")
        ex = examples.iloc[0]
        print("="*80)
        print(f"SUBJECT: {ex['subject']} | TRUE ANSWER: {ex['true_answer']}")
        print("-" * 80)
        print(f"PROMPT:\n{ex['prompt'][:500]}...") # Truncated for readability
        print("-" * 80)
        print(f"BASE MISTRAL (FAILED PARSE): {ex['parsed_guess_mistral']}")
        print(f"MISTRAL RAW TEXT:\n{ex['raw_response_mistral']}")
        print("-" * 80)
        print(f"BIOMISTRAL (SUCCESS): {ex['parsed_guess_bio']}")
        print(f"BIOMISTRAL RAW TEXT:\n{ex['raw_response_bio']}")
        print("="*80)

if __name__ == "__main__":
    find_qualitative_examples()