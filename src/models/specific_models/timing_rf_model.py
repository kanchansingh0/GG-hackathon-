from sklearn.ensemble import RandomForestRegressor
from typing import Dict, Any
import numpy as np

class TimingRFModel:
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }
        self.model = RandomForestRegressor(**self.params)
        
    def train(self, X, y):
        return self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_feature_importance(self):
        return {
            'importance': self.model.feature_importances_,
            'std': np.std([tree.feature_importances_ 
                          for tree in self.model.estimators_], axis=0)
        } 