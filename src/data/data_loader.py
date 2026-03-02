import os
import yaml
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
from .preprocessors.liar_processor import LIARProcessor

class DataLoader:
    """
    Main data loading interface.
    Handles loading configs and routing to appropriate processor.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize DataLoader with config file.
        
        Args:
            config_path: Path to YAML config file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._create_directories()
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _create_directories(self) -> None:
        """Create necessary directories from config."""
        for path_key, path_value in self.config['paths'].items():
            Path(path_value).mkdir(parents=True, exist_ok=True)
    
    def load_and_process(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load and process dataset based on config.
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        dataset_name = self.config['dataset']['name']
        
        # Route to appropriate processor
        if dataset_name == "LIAR":
            processor = LIARProcessor(
                config=self.config['preprocessing'],
                seed=self.config['preprocessing']['seed']
            )
            processor.config['hf_dataset_name'] = self.config['dataset']['hf_dataset_name']
            processor.config['name'] = dataset_name
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Load and process
        train_df, val_df, test_df = processor.load_raw_data()
        
        # Print statistics
        processor.print_statistics(train_df, val_df, test_df)
        processor.show_samples(train_df, n_samples=3)
        
        return train_df, val_df, test_df
    
    def save_processed_data(self, train_df: pd.DataFrame, 
                           val_df: pd.DataFrame, 
                           test_df: pd.DataFrame) -> None:
        """
        Save processed dataframes to CSV.
        
        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data
        """
        processed_dir = self.config['paths']['processed_data']
        
        train_path = os.path.join(processed_dir, 'train.csv')
        val_path = os.path.join(processed_dir, 'val.csv')
        test_path = os.path.join(processed_dir, 'test.csv')
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"\n✅ Processed data saved to {processed_dir}/")
        print(f"   - train.csv: {len(train_df)} examples")
        print(f"   - val.csv: {len(val_df)} examples")
        print(f"   - test.csv: {len(test_df)} examples")