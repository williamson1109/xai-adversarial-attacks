# Minimal LIAR Fake News Pipeline

This repository provides a minimal, reproducible pipeline for binary fake news detection on the LIAR dataset, with SHAP-based explainability.

## Pipeline Scripts

- `scripts/preprocess_liar.py`: Preprocess raw LIAR data to binary CSV.
- `scripts/train.py`: Train a binary classifier (DistilBERT by default).
- `scripts/test.py`: Evaluate the trained model on the test set.
- `scripts/xai_attack_shap.py`: Run SHAP explanations on test samples.

## Setup (Tesla P100/cu118)

1. **Create a virtual environment** (optional but recommended):
	```bash
	python3 -m venv .venv
	source .venv/bin/activate
	```
2. **Install dependencies:**
	```bash
	pip install -r requirements.txt
	```

## Usage

### 1. Preprocess Data
```bash
python scripts/preprocess_liar.py --input data/LIAR/data/raw/train.tsv --out_dir data/processed/
```

### 2. Train Model
```bash
python scripts/preprocess_liar.py \
  --input data/raw/train.tsv \
  --out_dir data/processed/
```

### 3. Test Model
```bash
python test.py \
  --test_csv /cluster/home/williasf/xai-adversarial-attacks/data/processed/liar_binary.csv \
  --model_dir /cluster/home/williasf/xai-adversarial-attacks/models/liar_model \
  --batch_size 32
```

### 4. SHAP Explainability
```bash
python scripts/xai_attack_shap.py \
  --test_csv data/processed/liar_binary.csv \
  --model_dir models/liar_model/best_model \
  --n_samples 10 \
  --out_path results/shap_attack_results.csv
```

## Notes
- All scripts are minimal and reproducible.
- SHAP explanations use the official `shap.Explainer(f, tokenizer)` pattern.
- For GPU (Tesla P100) ensure CUDA 11.8 is available.

---

For questions, contact: [Your Name]