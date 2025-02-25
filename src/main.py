from src.data_collection.synthetic_data_manager import SyntheticDataManager
from src.data_processing.dataset_creator import DatasetCreator
from pathlib import Path

def main():
    # Create necessary directories
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    
    try:
        # Generate synthetic data
        print("Initializing data manager...")
        data_manager = SyntheticDataManager()
        print("Generating synthetic data...")
        data = data_manager.generate_synthetic_data()
        data_manager.save_data()
        
        # Process data
        print("Processing data...")
        dataset_creator = DatasetCreator()
        processed_data = dataset_creator.load_data('data/processed/processed_features.csv')
        processed_data = dataset_creator.process_data()
        dataset_creator.save_processed_data()
        
        print("Data preparation completed successfully!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main() 