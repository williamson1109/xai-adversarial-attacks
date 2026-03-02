import pandas as pd
from datasets import load_dataset
from typing import Tuple, Dict
from .base_processor import BaseDatasetProcessor

class LIARProcessor(BaseDatasetProcessor):
    """
    Processor for the LIAR dataset.
    Converts 6-class labels to binary classification.
    Drops ambiguous 'half-true' samples by default.
    """

    def __init__(self, config: Dict, seed: int = 42):
        super().__init__(config, seed)
        self.hf_dataset_name = config.get('hf_dataset_name', 'liar')
        self.drop_ambiguous = config.get('drop_ambiguous', True)

    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load LIAR dataset from HuggingFace."""
        print(f"Loading {self.hf_dataset_name} dataset from HuggingFace...")
        dataset = load_dataset(self.hf_dataset_name)

        train_df = self.process_split(dataset['train'])
        val_df = self.process_split(dataset['validation'])
        test_df = self.process_split(dataset['test'])

        return train_df, val_df, test_df

    def process_split(self, split) -> pd.DataFrame:
        """
        Convert LIAR split to binary classification format.
        Drops samples where label_mapping is null (e.g. half-true).
        """
        data = []
        dropped = 0

        for example in split:
            text = example['statement']
            original_label = example['label']
            binary_label = self.label_mapping.get(original_label)

            # Drop ambiguous labels (null in config)
            if binary_label is None:
                dropped += 1
                continue

            data.append({
                'text': text,
                'label': binary_label,
                'original_label': original_label
            })

        df = pd.DataFrame(data)

        if dropped > 0:
            print(f"  Dropped {dropped} ambiguous samples (half-true)")

        if self.config.get('remove_duplicates', True):
            before = len(df)
            df = df.drop_duplicates(subset=['text'])
            removed = before - len(df)
            if removed > 0:
                print(f"  Removed {removed} duplicate entries")

        return df