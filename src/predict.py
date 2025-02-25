import pandas as pd
import numpy as np
from pathlib import Path
from models.neural_network import NeuralNetworkModel

def load_model() -> NeuralNetworkModel:
    """Load the trained model"""
    model = NeuralNetworkModel()
    try:
        model.load()
        print("Model loaded successfully!")
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

def predict_depth(model: NeuralNetworkModel, features: pd.DataFrame) -> np.ndarray:
    """Make predictions for new data"""
    required_features = ['setup_slack', 'hold_slack', 'timing_violation', 
                        'fanin_count', 'fanout_count', 'operation_complexity', 
                        'path_length']
    
    # Validate features
    missing_features = [f for f in required_features if f not in features.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Make prediction
    X = features[required_features].values
    return model.predict(X)

def main():
    try:
        # Load model
        model = load_model()
        
        # Example: Load new data
        new_data = pd.DataFrame({
            'setup_slack': [5.0, 4.0, 3.0],
            'hold_slack': [2.0, 1.5, 1.0],
            'timing_violation': [0, 1, 0],
            'fanin_count': [3, 4, 5],
            'fanout_count': [2, 3, 4],
            'operation_complexity': [2.5, 3.0, 3.5],
            'path_length': [6, 7, 8]
        })
        
        # Make predictions
        predictions = predict_depth(model, new_data)
        
        # Add predictions to dataframe
        new_data['predicted_logic_levels'] = predictions
        
        # Print results
        print("\nPredictions:")
        print(new_data)
        
        # Save results
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        new_data.to_csv(output_dir / "predictions.csv", index=False)
        print(f"\nResults saved to: {output_dir / 'predictions.csv'}")
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main() 