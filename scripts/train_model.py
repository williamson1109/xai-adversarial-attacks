import sys
import yaml
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.models.distilbert_classifier import DistilBERTClassifier
from src.models.trainer import DistilBERTTrainer


def main():
    with open('configs/liar_config.yaml', 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)

    with open('configs/training_config.yaml', 'r', encoding='utf-8') as f:
        train_config = yaml.safe_load(f)

    # Load processed data
    processed_dir = data_config['paths']['processed_data']
    train_df = pd.read_csv(f"{processed_dir}/train.csv")
    val_df = pd.read_csv(f"{processed_dir}/val.csv")
    test_df = pd.read_csv(f"{processed_dir}/test.csv")

    # Combine train + val for CV (test stays held out)
    full_df = pd.concat([train_df, val_df], ignore_index=True)

    print(f"Dataset: {data_config['dataset']['name']}")
    print(f"   CV data:   {len(full_df)} examples")
    print(f"   Test data: {len(test_df)} examples (held out)")
    print(f"   Real (0):  {(full_df['label'] == 0).sum()}")
    print(f"   Fake (1):  {(full_df['label'] == 1).sum()}")

    # Model kwargs
    model_kwargs = {
        'model_name': data_config['model']['name'],
        'num_labels': data_config['model']['num_labels'],
        'dropout': train_config['model']['dropout'],
        'freeze_base': data_config['model']['freeze_base'],
        'unfreeze_last_n_layers': data_config['model']['unfreeze_last_n_layers'],
    }

    max_length = data_config['model']['max_length']

    # Print parameter counts
    temp_model = DistilBERTClassifier(**model_kwargs)
    counts = temp_model.count_trainable_params()
    print(f"\nModel: {data_config['model']['name']}")
    print(f"   Total:     {counts['total']:,}")
    print(f"   Trainable: {counts['trainable']:,}")
    print(f"   Frozen:    {counts['frozen']:,}")
    del temp_model

    # Initialize trainer
    trainer = DistilBERTTrainer(
        model_class=DistilBERTClassifier,
        model_kwargs=model_kwargs,
        train_config=train_config,
        max_length=max_length
    )

    # Run 5x2-fold CV
    results = trainer.cross_validate(
        full_df,
        save_dir=data_config['paths']['checkpoints']
    )

    print("\nTraining complete!")


if __name__ == "__main__":
    main()