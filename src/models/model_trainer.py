from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np

class ModelTrainer:
    def __init__(self, model, features: List[str], target: str, test_size: float = 0.2):
        self.model = model
        self.features = features
        self.target = target
        self.test_size = test_size
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare features and targets for training"""
        X = df[self.features].values
        y = df[self.target].values
        
        return train_test_split(X, y, test_size=self.test_size, random_state=42)
    
    def train_and_evaluate(self, df: pd.DataFrame) -> Dict:
        """Train model and return evaluation metrics"""
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        
        # Train model
        self.model.train(X_train, y_train)
        
        # Evaluate on both sets
        train_metrics = self.model.evaluate(X_train, y_train)
        test_metrics = self.model.evaluate(X_test, y_test)
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }
