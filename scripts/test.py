import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df['statement'].tolist(), df['label'].tolist()

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()
    texts, labels = load_data(args.test_csv)
    preds = []
    for text in tqdm(texts, desc='Testing'):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.argmax(logits, dim=1).item()
            preds.append(pred)
    acc = accuracy_score(labels, preds)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(labels, preds, digits=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_csv', required=True, help='Path to processed LIAR test CSV')
    parser.add_argument('--model_dir', required=True, help='Directory with trained model and tokenizer')
    args = parser.parse_args()
    main(args)
