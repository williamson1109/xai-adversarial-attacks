# XAI for Adversarial Attacks on Fake News Detection

> Master's thesis — NTNU, 2026  
> Martin Rustad & William Son Fagerstrøm  
> Supervisor: Özlem Özgöbek

This repository contains the full pipeline for our master's thesis *"Using XAI for Adversarial Attacks on Fake News Detection"*. We replicate and extend the SHAP-guided adversarial attack framework from [Kozik et al. (2024)](https://www.sciencedirect.com/science/article/pii/S0167404823005096) on the [LIAR dataset](https://huggingface.co/datasets/liar), and compare attack effectiveness across three model architectures: DistilBERT, RoBERTa, and TextCNN.

---

## Overview

The pipeline consists of four stages:

```
Raw LIAR data
     │
     ▼
1. Preprocessing    →  Binary FAKE/TRUE labels (8,090 samples)
     │
     ▼
2. Model Training   →  DistilBERT / RoBERTa / TextCNN
     │               (5×2 repeated k-fold CV)
     ▼
3. SHAP Analysis    →  Token-level importance scores
     │
     ▼
4. LLM Attack       →  Claude API guided by SHAP scores
                        → Adversarial examples + metrics
```

## Setup

This project runs on the NTNU IDUN HPC cluster (SLURM, GPU nodes). It should work on any Linux system with CUDA available.

```bash
# Clone the repository
git clone https://github.com/your-username/xai-adversarial-attacks.git
cd xai-adversarial-attacks

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

For the LLM attack, you need an [Anthropic API key](https://console.anthropic.com/):
```bash
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

---

## Usage

### 1. Preprocess Data

Downloads the LIAR dataset from HuggingFace, combines all splits, and creates an 80/20 stratified train/test split with binary labels (FAKE/TRUE). Half-true and barely-true labels are excluded as ambiguous.

```bash
python scripts/preprocess_liar.py \
  --from_hf \
  --split all \
  --out_dir data/processed/
```

### 2. Train Models

**DistilBERT** (frozen encoder, classification head only — replicates Kozik et al.):
```bash
python scripts/train.py \
  --model distilbert-base-uncased \
  --train_csv data/processed/liar_train.csv \
  --out_dir models/liar_model
```

**RoBERTa** (full fine-tuning):
```bash
python scripts/train.py \
  --model roberta-base \
  --train_csv data/processed/liar_train.csv \
  --out_dir models/roberta_model
```

**TextCNN** (Kim 2014, GloVe 100d embeddings):
```bash
# Download GloVe embeddings first
mkdir -p data/glove && cd data/glove
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip glove.6B.100d.txt
cd ../..

python scripts/train_textcnn.py \
  --train_csv data/processed/liar_train.csv \
  --glove_path data/glove/glove.6B.100d.txt \
  --out_dir models/textcnn_model
```

All models use 5×2 repeated k-fold cross-validation (n_splits=2, n_repeats=5, random_state=42) and report accuracy, balanced accuracy, F1, precision, recall, and G-mean.

### 3. Evaluate on Test Set

**Transformer models (DistilBERT / RoBERTa):**
```bash
python scripts/test.py \
  --model_dir models/liar_model/best_model \
  --test_csv data/processed/liar_test.csv

python scripts/test.py \
  --model_dir models/roberta_model/best_model \
  --test_csv data/processed/liar_test.csv
```

**TextCNN:**
```bash
python scripts/test_textcnn.py \
  --model_dir models/textcnn_model/best_model \
  --test_csv data/processed/liar_test.csv
```

### 4. Interactive SHAP Inspection

Inspect SHAP token importance scores for individual samples. Useful for understanding which tokens drive predictions before running the full attack.

```bash
python scripts/inspect_tokens.py \
  --model_dir models/liar_model/best_model \
  --test_csv data/processed/liar_test.csv \
  --sample_idx 0 \
  --html_out results/shap_inspection.html
```

### 5. Rule-Based SHAP Attack (Kozik et al. replication)

Replicates the four rule-based attack strategies from Kozik et al. (2024): SWR, SWI, SS, and BT.

```bash
python scripts/xai_attack_shap.py \
  --test_csv data/processed/liar_test.csv \
  --model_dir models/liar_model/best_model \
  --n_samples 50 \
  --out_path results/shap_attack_results.csv
```

### 6. LLM-Guided Adversarial Attack

The main contribution of this thesis. Uses SHAP token importance scores to guide Claude API in generating minimal, semantically-preserving adversarial modifications. Iteratively attacks from the best text found so far and stops on successful label flip.

```bash
export $(cat .env | xargs)

python scripts/llm_attack.py \
  --model_dir models/liar_model/best_model \
  --test_csv data/processed/liar_test.csv \
  --out_dir results/llm_attack/ \
  --n_samples 50 \
  --max_iter 10 \
  --budget_limit 2.00
```

To attack the same articles across models (for fair comparison):
```bash
python scripts/llm_attack.py \
  --model_dir models/roberta_model/best_model \
  --test_csv data/processed/liar_test.csv \
  --out_dir results/llm_attack_roberta/ \
  --reference_log results/llm_attack/llm_experiment_log.csv \
  --max_iter 10 \
  --budget_limit 2.00
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--n_samples` | 50 | Number of samples to attack |
| `--max_iter` | 10 | Max LLM iterations per sample |
| `--budget_limit` | 2.00 | Max USD to spend on API calls |
| `--reference_log` | None | Attack same articles as a previous run |
| `--seed` | 42 | Random seed for sample selection |

Outputs two CSVs per run: `llm_experiment_log_<model>.csv` and `llm_modification_detail_<model>.csv`.

---

## Model Performance

### Cross-Validation Results (5×2 repeated k-fold)

| Metric | DistilBERT | TextCNN | RoBERTa |
|--------|-----------|---------|---------|
| Accuracy | 0.625 ± 0.008 | 0.645 ± 0.008 | 0.667 ± 0.014 |
| Balanced Accuracy | 0.603 ± 0.009 | 0.637 ± 0.009 | 0.657 ± 0.012 |
| F1 (macro) | 0.597 ± 0.014 | 0.637 ± 0.009 | 0.657 ± 0.012 |
| G-mean | 0.572 ± 0.024 | 0.632 ± 0.011 | 0.649 ± 0.014 |

### Test Set Results (1,618 samples)

| Metric | DistilBERT | TextCNN | RoBERTa |
|--------|-----------|---------|---------|
| Accuracy | 0.6286 | 0.6465 | 0.6693 |
| Balanced Accuracy | 0.6195 | 0.6382 | 0.6650 |
| F1 (macro) | 0.6200 | 0.6388 | 0.6649 |
| G-mean | 0.6148 | 0.6344 | 0.6641 |

### Adversarial Attack Effectiveness (AE)

| Model | Samples | Flips | AE (%) |
|-------|---------|-------|--------|
| DistilBERT | 389 | 84 | 21.6% |
| RoBERTa | 194 | 61 | 31.4% |
| TextCNN | TBD | TBD | TBD |

---

## Dataset

The [LIAR dataset](https://huggingface.co/datasets/liar) contains 12,836 short political statements from PolitiFact with six veracity labels. We collapse these into binary labels:

| Original Label | Binary Label |
|---------------|--------------|
| false, pants-fire | FAKE (0) |
| mostly-true, true | TRUE (1) |
| half-true, barely-true | Excluded |

Final dataset: **8,090 samples** (44% FAKE, 56% TRUE).

---

## Architecture Details

**DistilBERT** — `distilbert-base-uncased`, frozen encoder, classification head only. Replicates Kozik et al. (2024). max_length=250, batch_size=16, epochs=3.

**RoBERTa** — `roberta-base`, full fine-tuning (frozen encoder causes class collapse for RoBERTa). max_length=250, batch_size=16, epochs=3.

**TextCNN** — Kim (2014) architecture. Filter sizes [3,4,5], 100 filters each, GloVe 100d frozen embeddings, dropout=0.5, max_len=250, batch_size=64, epochs=10, lr=1e-3.

**SHAP** — `shap.Explainer(predict_fn, tokenizer)` with `fixed_context=1`. `predict_fn` returns `logit(P(TRUE))` so negative SHAP = pushes toward FAKE, positive SHAP = pushes toward TRUE.

---

## Citation

If you use this code, please cite:

```
Kozik, R. et al. (2024). When explainability turns into a threat — 
using XAI to fool a fake-news detector. Future Generation Computer Systems.
```