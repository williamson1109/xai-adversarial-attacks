import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             balanced_accuracy_score)
from tqdm import tqdm
from typing import Dict, List
import json


class FakeNewsDataset(Dataset):
    """PyTorch Dataset for fake news detection."""

    def __init__(self, texts, labels, tokenizer, max_length=250):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class DistilBERTTrainer:
    """
    Trainer implementing 5x2-fold cross-validation.
    Matches paper methodology exactly.
    """

    def __init__(self, model_class, model_kwargs: Dict,
                 train_config: Dict, max_length: int = 250,
                 device: str = None):
        """
        Args:
            model_class: DistilBERTClassifier class (not instance)
            model_kwargs: kwargs to pass to model constructor
            train_config: Training configuration dict
            max_length: Max token length for tokenizer
            device: cuda or cpu
        """
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.train_config = train_config
        self.max_length = max_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Using device: {self.device}")
        if self.device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")

    def cross_validate(self, df: pd.DataFrame,
                       save_dir: str = None) -> Dict:
        """
        Run 5x2-fold cross-validation (paper methodology).

        Args:
            df: Full dataset DataFrame with 'text' and 'label' columns
            save_dir: Directory to save results and checkpoints

        Returns:
            Dictionary with all fold metrics and summary statistics
        """
        cv_config = self.train_config['cross_validation']
        n_repeats = cv_config['n_repeats']
        n_splits = cv_config['n_splits']
        seed = cv_config['seed']

        rkf = RepeatedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=seed
        )

        all_metrics = []
        fold_num = 0
        total_folds = n_repeats * n_splits

        texts = df['text'].values
        labels = df['label'].values

        print(f"\nStarting {n_repeats}x{n_splits}-fold Cross Validation")
        print(f"   Total folds: {total_folds}")

        for fold_idx, (train_idx, val_idx) in enumerate(rkf.split(texts)):
            fold_num += 1
            print(f"\n{'='*60}")
            print(f"Fold {fold_num}/{total_folds}")
            print(f"{'='*60}")

            # Split data for this fold
            train_texts, val_texts = texts[train_idx], texts[val_idx]
            train_labels, val_labels = labels[train_idx], labels[val_idx]

            # Fresh model for each fold (critical for valid CV)
            model = self.model_class(**self.model_kwargs)
            model.to(self.device)

            batch_size = self.train_config['training']['batch_size']

            train_loader = self._make_loader(
                train_texts, train_labels,
                model.tokenizer, self.max_length,
                batch_size, shuffle=True
            )
            val_loader = self._make_loader(
                val_texts, val_labels,
                model.tokenizer, self.max_length,
                batch_size, shuffle=False
            )

            # Train this fold
            self._train_fold(model, train_loader, val_loader, fold_num)

            # Evaluate this fold
            metrics = self._evaluate(model, val_loader)
            metrics['fold'] = fold_num
            all_metrics.append(metrics)

            print(f"\nFold {fold_num} Results:")
            print(f"   Accuracy:          {metrics['accuracy']:.4f}")
            print(f"   Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
            print(f"   Precision:         {metrics['precision']:.4f}")
            print(f"   Recall:            {metrics['recall']:.4f}")
            print(f"   F1:                {metrics['f1']:.4f}")

            # Save checkpoint for this fold
            if save_dir:
                self._save_model(model, save_dir, f"fold_{fold_num}")

            # Free memory between folds
            del model
            if self.device == 'cuda':
                torch.cuda.empty_cache()

        # Compute and print summary statistics
        summary = self._compute_summary(all_metrics)
        self._print_summary(summary)

        # Save results to disk
        if save_dir:
            results_path = os.path.join(save_dir, 'cv_results.json')
            os.makedirs(save_dir, exist_ok=True)
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump({'folds': all_metrics, 'summary': summary}, f, indent=2)
            print(f"\nCV results saved to {results_path}")

        return {'folds': all_metrics, 'summary': summary}

    def _make_loader(self, texts, labels, tokenizer,
                     max_length, batch_size, shuffle) -> DataLoader:
        """Create a DataLoader from raw arrays."""
        dataset = FakeNewsDataset(texts, labels, tokenizer, max_length)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0
        )

    def _train_fold(self, model, train_loader: DataLoader,
                    val_loader: DataLoader, fold_num: int):
        """
        Train model for one fold with fixed early stopping.

        Args:
            model: Fresh DistilBERTClassifier instance
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            fold_num: Current fold number (for logging)
        """
        config = self.train_config['training']
        epochs = config['epochs']
        lr = config['learning_rate']
        patience = config.get('patience', 3)
        min_delta = config.get('min_delta', 0.001)  # Fixed: use min_delta not patience

        # Only optimize trainable params (frozen base excluded)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=config['weight_decay']
        )

        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['warmup_steps'],
            num_training_steps=total_steps
        )

        best_val_f1 = 0.0
        patience_counter = 0

        for epoch in range(epochs):
            # --- Training phase ---
            model.train()
            total_loss = 0.0

            progress_bar = tqdm(
                train_loader,
                desc=f"Fold {fold_num} | Epoch {epoch+1}/{epochs}"
            )

            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask, labels)
                loss = outputs['loss']
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['max_grad_norm']
                )

                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

            avg_loss = total_loss / len(train_loader)

            # --- Validation phase ---
            val_metrics = self._evaluate(model, val_loader)
            val_f1 = val_metrics['f1']

            print(f"   Epoch {epoch+1}/{epochs}: "
                  f"train_loss={avg_loss:.4f}  "
                  f"val_f1={val_f1:.4f}  "
                  f"val_acc={val_metrics['accuracy']:.4f}")

            # --- Fixed early stopping ---
            # Bug was: val_f1 > best_val_f1 + patience (patience=3, impossible to exceed)
            # Fix is: val_f1 > best_val_f1 + min_delta (min_delta=0.001, sensible threshold)
            if val_f1 > best_val_f1 + min_delta:
                best_val_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"   Early stopping at epoch {epoch+1} "
                          f"(best val F1: {best_val_f1:.4f})")
                    break

    def _evaluate(self, model, data_loader: DataLoader) -> Dict:
        """
        Evaluate model on a DataLoader.

        Args:
            model: Trained model
            data_loader: DataLoader to evaluate on

        Returns:
            Dictionary with accuracy, balanced_accuracy, precision, recall, f1
        """
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = model(input_ids, attention_mask)
                preds = torch.argmax(outputs['logits'], dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds,
            average='binary',
            zero_division=0
        )

        return {
            'accuracy': float(accuracy_score(all_labels, all_preds)),
            'balanced_accuracy': float(balanced_accuracy_score(all_labels, all_preds)),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }

    def _compute_summary(self, all_metrics: List[Dict]) -> Dict:
        """
        Compute mean and std across all folds (matches paper table format).

        Args:
            all_metrics: List of metric dicts from each fold

        Returns:
            Summary dict with mean and std per metric
        """
        summary = {}
        metric_keys = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1']

        for key in metric_keys:
            values = [m[key] for m in all_metrics]
            summary[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }

        return summary

    def _print_summary(self, summary: Dict):
        """Print final summary in paper format (mean +/- std)."""
        print(f"\n{'='*60}")
        print("5x2-Fold Cross Validation Summary (Paper Format)")
        print(f"{'='*60}")
        print(f"{'Metric':<25} {'Mean +/- Std':>20}")
        print(f"{'-'*50}")
        for metric, values in summary.items():
            print(f"{metric:<25} {values['mean']:.3f} +/- {values['std']:.3f}")
        print(f"{'='*60}")

    def _save_model(self, model, save_dir: str, name: str):
        """Save model state dict to disk."""
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{name}.pt")
        torch.save(model.state_dict(), path)
        print(f"   Checkpoint saved: {name}.pt")