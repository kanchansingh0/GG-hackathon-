import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

class LogicLevelPredictor:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.load_model()
        
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            self.model = joblib.load(self.model_dir / "logic_depth_model.joblib")
            self.scaler = joblib.load(self.model_dir / "feature_scaler.joblib")
            print("Model loaded successfully!")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def predict(self, features: Dict[str, float]) -> float:
        """Make prediction for a single RTL design"""
        try:
            # Convert features to DataFrame
            df = pd.DataFrame([features])
            
            # Scale features
            scaled_features = self.scaler.transform(df)
            
            # Make prediction
            prediction = self.model.predict(scaled_features)[0]
            
            return prediction
            
        except Exception as e:
            raise Exception(f"Error during prediction: {str(e)}")
    
    def predict_batch(self, features_list: List[Dict[str, float]]) -> List[float]:
        """Make predictions for multiple RTL designs"""
        try:
            # Convert features to DataFrame
            df = pd.DataFrame(features_list)
            
            # Scale features
            scaled_features = self.scaler.transform(df)
            
            # Make predictions
            predictions = self.model.predict(scaled_features)
            
            return predictions.tolist()
            
        except Exception as e:
            raise Exception(f"Error during batch prediction: {str(e)}")

def main():
    # Example usage
    predictor = LogicLevelPredictor()
    
    # Example features for a new RTL design
    example_features = {
        'setup_slack': 5.0,
        'hold_slack': 2.0,
        'timing_violation': 0,
        'fanin_count': 3,
        'fanout_count': 4,
        'operation_complexity': 2.5,
        'path_length': 6
    }
    
    try:
        # Make prediction
        prediction = predictor.predict(example_features)
        print(f"\nPredicted Logic Levels: {prediction:.2f}")
        
        # Example batch prediction
        batch_features = [
            example_features,
            {k: v * 1.2 for k, v in example_features.items()}  # Slightly modified features
        ]
        
        batch_predictions = predictor.predict_batch(batch_features)
        print("\nBatch Predictions:")
        for i, pred in enumerate(batch_predictions):
            print(f"Design {i+1}: {pred:.2f} logic levels")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 