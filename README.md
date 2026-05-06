# BioMedical LLM Lobotomy: The Structural Shield

**Authors:** Brandon Borzello & Shantanu Dalvi  
**Course:** CSCI 5541 NLP (Spring 2026) - University of Minnesota  

## Overview
This repository contains the code, data, and webpage deliverables for our evaluation of unstructured magnitude pruning on specialized medical LLMs. We demonstrate that fine-tuning does not make a model brittle; instead, domain specialization creates a "Structural Shield" that protects foundational logic circuits against severe architectural damage up to 60% sparsity.

## Repository Structure
```text
.
├── data/                       # Contains all raw generation logs and aggregate F1 scores
│   ├── BioMistral_MMLU/        
│   ├── BioMistral_PubMedQA/
│   ├── Mistral_MMLU/
│   └── Mistral_PubMedQA/
├── files/                      # CSS, images, and generated data visualizations
├── scripts/                    # Analytical and visualization scripts
│   ├── kaggle_scripts/         # The heavy-compute PyTorch ablation scripts
│   │   ├── mmlu_lobotomy.py
│   │   └── pubmedqa_lobotomy.py
│   ├── calculate_auc.py        # Calculates robustness scores (Area Under Curve)
│   ├── find_tuning_difference.py # Isolates qualitative divergence logs
│   ├── plot_bias.py            # Generates Affirmative Bias charts
│   └── plot_degradation.py     # Generates 5% interval F1 degradation curves
├── index.html                  # Interactive virtual poster frontend
└── README.md
```

## Hardware & Environment Requirements
* **Compute:** The pruning scripts are strictly configured for a **Dual NVIDIA T4 GPU** environment (16GB VRAM each). The scripts utilize `device_map="auto"` and restrict memory allocation to prevent sorting-overhead Out-Of-Memory (OOM) fatal errors.
* **Execution Environment:** The heavy compute scripts (`mmlu_lobotomy.py` and `pubmedqa_lobotomy.py`) were originally executed in a Kaggle Notebook environment. They utilize Kaggle's `UserSecretsClient` to fetch the HuggingFace token and save outputs to `/kaggle/working/`. **To run these locally**, you must replace the authentication block with standard `huggingface-cli login` and update the `RESULTS_FILE` target paths.
* **Dependencies:**
  ```bash
  pip install torch transformers datasets pandas scikit-learn huggingface_hub matplotlib seaborn
  ```

## Replication Protocol

### Step 1: Model Pruning & Evaluation
Due to the computational time required, the data is already generated and located in the `/data` directory. To replicate the raw compute process yourself:
1. Ensure your HuggingFace account has accepted the Mistral-7B license agreement.
2. Open `/scripts/kaggle_scripts/mmlu_lobotomy.py`.
3. Toggle the `MODEL_NAME` and `RESULTS_FILE` variables at the top of the script to switch between Base Mistral and BioMistral.
4. Execute the script. It will run through the 5% sparsity intervals, saving checkpoints automatically.
   ```bash
   python scripts/kaggle_scripts/mmlu_lobotomy.py
   python scripts/kaggle_scripts/pubmedqa_lobotomy.py
   ```

### Step 2: Generating AUC Metrics & Visualizations
To recalculate the Area Under the Curve (AUC) scores and regenerate the high-resolution `.png` charts used in the paper:
```bash
cd scripts
python calculate_auc.py
python plot_degradation.py
python plot_bias_horizontal.py
```
*(All visual outputs will be saved directly to the `/files` directory).*

### Step 3: Qualitative Error Analysis
To isolate the exact formatting failures and generative amnesia strings that define the "cognitive cliff" (as referenced in Section 3.3 of our report):
```bash
cd scripts
python find_tuning_difference.py
```

## Webpage Deployment
The frontend of this project is a static HTML page. To view the interactive virtual poster, simply clone this repository and open `index.html` in any modern web browser. No local web server is required.