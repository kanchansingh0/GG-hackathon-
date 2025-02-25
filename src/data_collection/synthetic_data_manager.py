import numpy as np
import pandas as pd
from pathlib import Path

class SyntheticDataManager:
    def __init__(self):
        self.data = None
        
    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic data for testing"""
        np.random.seed(42)
        
        data = {
            'setup_slack': np.random.uniform(5.0, 15.0, n_samples),
            'hold_slack': np.random.uniform(-1.0, 1.0, n_samples),
            'timing_violation': np.random.randint(0, 2, n_samples),
            'fanin_count': np.random.randint(1, 10, n_samples),
            'fanout_count': np.random.randint(1, 8, n_samples),
            'operation_complexity': np.random.uniform(1.0, 10.0, n_samples),
            'path_length': np.random.randint(0, 10, n_samples),
            'logic_levels': np.random.randint(0, 5, n_samples)
        }
        
        self.data = pd.DataFrame(data)
        return self.data
        
    def save_data(self, filepath='data/processed/processed_features.csv'):
        """Save generated data to CSV"""
        if self.data is not None:
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            self.data.to_csv(filepath, index=False)
            print(f"Data saved to {filepath}")
        else:
            raise ValueError("No data generated yet. Call generate_synthetic_data first.")
