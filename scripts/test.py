import argparse
import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    balanced_accuracy_score,
    confusion_matrix
)
from tqdm import tqdm


def load_data(csv_path):
    df = pd.read_csv(csv_path, usecols=['statement', 'label'])
    return df['statement'].tolist(), df['label'].tolist()


def batch_predict(texts, tokenizer, model, device, batch_size):
    preds = []
    for i in tqdm(range(0, len(texts), batch_size), desc='Testing'):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=250
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            batch_preds = torch.argmax(logits, dim=1).tolist()
            preds.extend(batch_preds)
    return preds


def g_mean(labels, preds):
    cm = confusion_matrix(labels, preds)
    per_class_recall = cm.diagonal() / cm.sum(axis=1)
    return np.prod(per_class_recall) ** (1.0 / len(per_class_recall))


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    # Load test data
    texts, labels = load_data(args.test_csv)
    print(f"Loaded {len(texts)} test samples")

    # Run batched inference
    preds = batch_predict(texts, tokenizer, model, device, args.batch_size)

    # Metrics
    acc      = accuracy_score(labels, preds)
    bal_acc  = balanced_accuracy_score(labels, preds)
    gmean    = g_mean(labels, preds)

    print(f"\n{'='*50}")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"G-mean:            {gmean:.4f}")
    print(f"{'='*50}\n")
    print(classification_report(labels, preds, target_names=['FAKE', 'TRUE'], digits=4))

    # Save predictions
    df = pd.read_csv(args.test_csv)
    df['predicted'] = preds
    out_path = os.path.join(args.model_dir, 'test_predictions.csv')
    df.to_csv(out_path, index=False)
    print(f"Predictions saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test_csv',
        default='/cluster/home/williasf/xai-adversarial-attacks/data/processed/liar_test.csv',
        help='Path to processed LIAR test CSV'
    )
    parser.add_argument('--model_dir',  required=True, help='Directory with trained model and tokenizer')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    args = parser.parse_args()
    main(args)
