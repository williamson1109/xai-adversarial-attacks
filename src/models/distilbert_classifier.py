import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
from typing import Dict, Optional


class DistilBERTClassifier(nn.Module):
    """
    DistilBERT-based binary classifier for fake news detection.
    Partially unfreezes last N transformer layers for better adaptation
    on noisy datasets like LIAR.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased",
                 num_labels: int = 2,
                 dropout: float = 0.3,
                 freeze_base: bool = True,
                 unfreeze_last_n_layers: int = 2):
        """
        Args:
            model_name: HuggingFace model identifier
            num_labels: Number of output classes
            dropout: Dropout for classification head
            freeze_base: If True, freeze DistilBERT base first
            unfreeze_last_n_layers: How many transformer layers to unfreeze
                                    from the top (0 = fully frozen, 6 = fully unfrozen)
        """
        super(DistilBERTClassifier, self).__init__()

        self.model_name = model_name
        self.num_labels = num_labels

        # Load pretrained DistilBERT
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)

        # Step 1: Freeze everything
        if freeze_base:
            for param in self.distilbert.parameters():
                param.requires_grad = False

        # Step 2: Selectively unfreeze last N transformer layers
        # DistilBERT has 6 transformer layers (0-5)
        if unfreeze_last_n_layers > 0:
            total_layers = 6
            first_trainable = total_layers - unfreeze_last_n_layers

            for i in range(first_trainable, total_layers):
                for param in self.distilbert.transformer.layer[i].parameters():
                    param.requires_grad = True

            # Also unfreeze the final layer norm
            for param in self.distilbert.transformer.layer[-1].parameters():
                param.requires_grad = True

        # Classification head
        hidden_size = self.distilbert.config.hidden_size  # 768

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_labels)
        )

        self.loss_fn = nn.CrossEntropyLoss()

        # Print parameter summary
        counts = self.count_trainable_params()
        print(f"DistilBERT loaded | "
              f"Trainable: {counts['trainable']:,} | "
              f"Frozen: {counts['frozen']:,} | "
              f"Unfrozen layers: {unfreeze_last_n_layers}/6")

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Ground truth labels [batch_size] (optional)

        Returns:
            Dictionary with logits, loss (if labels given), pooled_output
        """
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Use [CLS] token as sequence representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]

        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {
            'logits': logits,
            'loss': loss,
            'pooled_output': cls_output
        }

    def predict(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """Get predicted class labels."""
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            return torch.argmax(outputs['logits'], dim=-1)

    def predict_proba(self, input_ids: torch.Tensor,
                      attention_mask: torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities."""
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            return torch.softmax(outputs['logits'], dim=-1)

    def count_trainable_params(self) -> Dict[str, int]:
        """Count trainable vs frozen parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {
            'trainable': trainable,
            'frozen': frozen,
            'total': trainable + frozen
        }