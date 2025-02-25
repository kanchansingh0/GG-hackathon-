from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from typing import Dict, Any
import numpy as np

class TimingRFModel:
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        self.model = RandomForestRegressor(**self.params)
        self.feature_columns = [
            'fanin_count',
            'fanout_count',
            'logic_depth',
            'operation_complexity',
            'path_length'
        ]
        self.threshold = 0.5  # Threshold for timing violation classification
        
    def train(self, X, y):
        """Train the model"""
        X = np.asarray(X)
        y = np.asarray(y)
        return self.model.fit(X, y)
    
    def predict(self, X):
        """Make predictions"""
        X = np.asarray(X)
        return self.model.predict(X)
    
    def evaluate(self, X, y) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            # Make predictions
            predictions = self.predict(X)
            
            # Calculate regression metrics
            mse = np.mean((y - predictions) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y - predictions))
            r2 = self.model.score(X, y)
            
            # Calculate classification metrics
            y_binary = y > self.threshold
            pred_binary = predictions > self.threshold
            
            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_binary, pred_binary).ravel()
            
            # Calculate accuracy
            acc = accuracy_score(y_binary, pred_binary)
            
            # Calculate precision and recall
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            return {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'accuracy': float(acc),
                'confusion_matrix': {
                    'true_negative': int(tn),
                    'false_positive': int(fp),
                    'false_negative': int(fn),
                    'true_positive': int(tp)
                },
                'precision': float(precision),
                'recall': float(recall)
            }
        except Exception as e:
            print(f"Error in evaluate method: {str(e)}")
            raise

    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance scores"""
        return {
            'importance': self.model.feature_importances_,
            'std': np.std([tree.feature_importances_ 
                          for tree in self.model.estimators_], axis=0)
        }