import argparse
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df['statement'].tolist(), df['label'].tolist()

def tokenize_function(examples, tokenizer):
    return tokenizer(examples['statement'], truncation=True, padding='max_length', max_length=128)

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    texts, labels = load_data(args.train_csv)
    df = pd.DataFrame({'statement': texts, 'label': labels})
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=labels)
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    train_ds = train_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_ds = val_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(args.out_dir, 'logs'),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    trainer.train()
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Model and tokenizer saved to {args.out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', required=True, help='Path to processed LIAR CSV')
    parser.add_argument('--out_dir', required=True, help='Output directory for model')
    parser.add_argument('--model', default='distilbert-base-uncased', help='HuggingFace model name')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    main(args)
