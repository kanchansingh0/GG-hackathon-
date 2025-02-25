import pandas as pd
import numpy as np
from pathlib import Path

class DatasetCreator:
    def __init__(self):
        self.data = None
        
    def load_data(self, filepath):
        """Load data from CSV file"""
        self.data = pd.read_csv(filepath)
        return self.data
        
    def process_data(self):
        """Process the loaded data"""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data first.")
            
        # Add any necessary processing steps here
        # For example, handling missing values, scaling, etc.
        return self.data
        
    def save_processed_data(self, filepath='data/processed/processed_features.csv'):
        """Save processed data"""
        if self.data is not None:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            self.data.to_csv(filepath, index=False)
            print(f"Processed data saved to {filepath}") 