from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from typing import Dict, Any
import numpy as np

class TimingPredictor:
    def __init__(self, model_type: str = 'random_forest', params: Dict[str, Any] = None):
        self.model_type = model_type
        self.params = params or {}
        self.model = self._create_model()
        
    def _create_model(self):
        if self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=self.params.get('n_estimators', 100),
                max_depth=self.params.get('max_depth', None),
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=self.params.get('n_estimators', 100),
                learning_rate=self.params.get('learning_rate', 0.1),
                random_state=42
            )
        elif self.model_type == 'neural_network':
            return MLPRegressor(
                hidden_layer_sizes=self.params.get('hidden_layer_sizes', (100, 50)),
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X, y):
        """Train the model on the given features and targets"""
        return self.model.fit(X, y)
    
    def predict(self, X):
        """Predict combinational depths"""
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        predictions = self.predict(X)
        mse = np.mean((predictions - y) ** 2)
        mae = np.mean(np.abs(predictions - y))
        return {
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse)
        }
