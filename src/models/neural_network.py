import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict
import joblib
from pathlib import Path

class NeuralNetworkModel:
    def __init__(self):
        self.model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),  # Deeper network
            activation='relu',
            max_iter=2000,  # More iterations
            learning_rate_init=0.001,  # Adjusted learning rate
            early_stopping=True,  # Add early stopping
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the neural network with scaled features"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate model performance"""
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        
        # Handle potential warnings
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        return {
            'mse': float(mse),
            'r2': float(r2),
            'predictions': y_pred.tolist()
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def save(self, path: str = 'models'):
        """Save the model and scaler to disk"""
        Path(path).mkdir(exist_ok=True)
        joblib.dump(self.model, Path(path) / 'neural_network.joblib')
        joblib.dump(self.scaler, Path(path) / 'scaler.joblib')
        
    def load(self, path: str = 'models'):
        """Load the model and scaler from disk"""
        model_path = Path(path) / 'neural_network.joblib'
        scaler_path = Path(path) / 'scaler.joblib'
        
        if model_path.exists() and scaler_path.exists():
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
        else:
            raise FileNotFoundError(f"Model or scaler not found in {path}") 