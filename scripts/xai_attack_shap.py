import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import shap
import pandas as pd
from tqdm import tqdm

def predict_fn(texts, model, tokenizer, device):
    inputs = tokenizer(texts, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    inputs = {{k: v.to(device) for k, v in inputs.items()}}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
    return probs.cpu().numpy()

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()
    df = pd.read_csv(args.test_csv)
    texts = df['statement'].tolist()
    labels = df['label'].tolist()
    # Use a small sample for SHAP explanation (slow)
    sample_texts = texts[:args.n_samples]
    f = lambda x: predict_fn(x, model, tokenizer, device)
    explainer = shap.Explainer(f, tokenizer)
    shap_values = explainer(sample_texts)
    for i, text in enumerate(sample_texts):
        print(f"\nText: {text}")
        shap.plots.text(shap_values[i])
    # Optionally, save SHAP values
    if args.out_path:
        shap.save(args.out_path, shap_values)
        print(f"SHAP values saved to {args.out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_csv', required=True, help='Path to processed LIAR test CSV')
    parser.add_argument('--model_dir', required=True, help='Directory with trained model and tokenizer')
    parser.add_argument('--n_samples', type=int, default=5, help='Number of test samples to explain')
    parser.add_argument('--out_path', default=None, help='Optional path to save SHAP values')
    args = parser.parse_args()
    main(args)
