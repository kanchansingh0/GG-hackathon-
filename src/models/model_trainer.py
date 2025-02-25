from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np

class ModelTrainer:
    def __init__(self, model, features: List[str], target: str, test_size: float = 0.2):
        self.model = model
        self.features = features
        self.target = target
        self.test_size = test_size
        self.is_trained = False
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare features and targets for training"""
        X = df[self.features].values
        y = df[self.target].values
        
        return train_test_split(X, y, test_size=self.test_size, random_state=42)
    
    def train_and_evaluate(self, df: pd.DataFrame) -> Dict:
        """Train model and return evaluation metrics with validation"""
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        
        # Train model
        self.model.train(X_train, y_train)
        self.is_trained = True
        
        # Cross validation
        cv_scores = cross_val_score(self.model.model, X_train, y_train, cv=5)
        
        # Get predictions
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        train_metrics = self._calculate_detailed_metrics(y_train, train_pred)
        test_metrics = self._calculate_detailed_metrics(y_test, test_pred)
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_scores': {
                'mean': cv_scores.mean(),
                'std': cv_scores.std(),
                'scores': cv_scores.tolist()
            },
            'feature_importance': self.model.get_feature_importance()
        }
    
    def _calculate_detailed_metrics(self, y_true, y_pred) -> Dict:
        """Calculate detailed performance metrics"""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    def check_model_status(self) -> Dict:
        """Check model training status and basic statistics"""
        if not self.is_trained:
            return {'status': 'Not trained'}
        
        return {
            'status': 'Trained',
            'feature_count': len(self.features),
            'model_type': type(self.model).__name__
        }
