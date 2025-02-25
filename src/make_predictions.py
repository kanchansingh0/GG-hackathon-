# Since the directory already exists at C:\Users\kanchan singh\Desktop\GG-hackathon-\src
# Save this code as make_predictions.py in that directory

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

class RTLPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            random_state=42
        )
        self.scaler = StandardScaler()
        
        # Create and fit scaler with sample data
        sample_data = np.array([
            [10, 4, 3, 5, 8, 2],
            [15, 6, 4, 7, 10, 3],
            [20, 8, 5, 9, 12, 4]
        ])
        self.scaler.fit(sample_data)
        
        # Train model with sample data
        sample_targets = np.array([1.0, 1.5, 2.0])  # Example target values
        self.model.fit(sample_data, sample_targets)
        
    def predict_complexity(self, features):
        try:
            # Ensure features is 2D array
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Transform features using fitted scaler
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            result = float(self.model.predict(features_scaled)[0])
            return result
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return None

def example_usage():
    try:
        # Create predictor instance
        predictor = RTLPredictor()
        
        # Test features
        test_features = np.array([10, 4, 3, 5, 8, 2])
        
        # Make prediction
        result = predictor.predict_complexity(test_features)
        
        if result is not None:
            print(f"Predicted operation complexity: {result:.4f}")
        else:
            print("Error: Could not make prediction")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    example_usage() 