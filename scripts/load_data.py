import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_loader import DataLoader

def main():
    """Load and process LIAR dataset."""
    
    # Load data using config
    loader = DataLoader('configs/liar_config.yaml')
    train_df, val_df, test_df = loader.load_and_process()
    
    # Save processed data
    loader.save_processed_data(train_df, val_df, test_df)
    
    print("\n🎉 Data loading complete!")

if __name__ == "__main__":
    main()