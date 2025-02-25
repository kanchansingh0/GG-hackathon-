import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict
import joblib
from pathlib import Path

class RFModel:
    def __init__(self):
        # Adjusted parameters to prevent overfitting
        self.model = RandomForestRegressor(
            n_estimators=50,            # Reduced number of trees
            max_depth=5,                # Limited tree depth
            min_samples_split=10,       # Increased minimum samples to split
            min_samples_leaf=5,         # Added minimum samples per leaf
            max_features='sqrt',        # Use sqrt of features
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the random forest with scaled features"""
        # Print data shapes
        print(f"\nTraining data shape: {X.shape}")
        print(f"Target data shape: {y.shape}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        # Print feature importances
        feature_imp = self.model.feature_importances_
        print("\nFeature Importances:")
        for i, imp in enumerate(feature_imp):
            print(f"Feature {i}: {imp:.4f}")
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate model performance"""
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        
        # Print prediction statistics
        print(f"\nPrediction statistics:")
        print(f"Mean predicted value: {np.mean(y_pred):.4f}")
        print(f"Std predicted value: {np.std(y_pred):.4f}")
        print(f"Mean actual value: {np.mean(y):.4f}")
        print(f"Std actual value: {np.std(y):.4f}")
        
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
        joblib.dump(self.model, Path(path) / 'rf_model.joblib')
        joblib.dump(self.scaler, Path(path) / 'rf_scaler.joblib')
        
    def load(self, path: str = 'models'):
        """Load the model and scaler from disk"""
        model_path = Path(path) / 'rf_model.joblib'
        scaler_path = Path(path) / 'rf_scaler.joblib'
        
        if model_path.exists() and scaler_path.exists():
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
        else:
            raise FileNotFoundError(f"Model or scaler not found in {path}") 