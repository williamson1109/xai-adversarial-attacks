from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple, Dict

class BaseDatasetProcessor(ABC):
    """
    Abstract base class for dataset processors.
    All dataset-specific processors should inherit from this.
    """
    
    def __init__(self, config: Dict, seed: int = 42):
        self.config = config
        self.seed = seed
        self.label_mapping = config.get('label_mapping', {})
        
    @abstractmethod
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load raw dataset from source.
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        pass
    
    @abstractmethod
    def process_split(self, split) -> pd.DataFrame:
        """
        Process a single dataset split into standardized format.
        
        Args:
            split: Raw dataset split
            
        Returns:
            DataFrame with columns: ['text', 'label', 'original_label']
        """
        pass
    
    def print_statistics(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                        test_df: pd.DataFrame) -> None:
        """Print dataset statistics."""
        print(f"\n{'='*60}")
        print(f"Dataset: {self.config.get('name', 'Unknown')}")
        print(f"{'='*60}")
        print(f"Train size: {len(train_df)}")
        print(f"Validation size: {len(val_df)}")
        print(f"Test size: {len(test_df)}")
        print(f"\nLabel distribution (train):")
        print(f"  0 (Real): {(train_df['label'] == 0).sum()} "
              f"({(train_df['label'] == 0).sum() / len(train_df) * 100:.1f}%)")
        print(f"  1 (Fake): {(train_df['label'] == 1).sum()} "
              f"({(train_df['label'] == 1).sum() / len(train_df) * 100:.1f}%)")
        
    def show_samples(self, df: pd.DataFrame, n_samples: int = 3) -> None:
        """Display sample examples from dataset."""
        print(f"\n{'='*60}")
        print("Sample examples:")
        print(f"{'='*60}")
        for idx in range(min(n_samples, len(df))):
            print(f"\nExample {idx + 1}:")
            text = df.iloc[idx]['text']
            print(f"Text: {text[:150]}{'...' if len(text) > 150 else ''}")
            print(f"Label: {df.iloc[idx]['label']} "
                  f"(Original: {df.iloc[idx].get('original_label', 'N/A')})")