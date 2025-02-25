import joblib
import pandas as pd
import numpy as np

def predict_complexity(input_data):
    try:
        # Load the trained model
        model = joblib.load('models/improved_rf_model.joblib')
        
        # Create a DataFrame with the required features
        required_features = [
            'setup_slack', 'hold_slack', 'timing_violation',
            'fanin_count', 'fanout_count', 'path_length'
        ]
        
        # Make prediction
        prediction = model.predict(input_data)
        
        return prediction[0]
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

def example_usage():
    # Example input data
    sample_input = pd.DataFrame({
        'setup_slack': [10.0],
        'hold_slack': [0.5],
        'timing_violation': [0],
        'fanin_count': [4],
        'fanout_count': [3],
        'path_length': [5]
    })
    
    # Calculate derived features
    sample_input['slack_ratio'] = sample_input['setup_slack'] / (sample_input['hold_slack'].abs() + 1e-6)
    sample_input['complexity_score'] = sample_input['fanin_count'] * sample_input['fanout_count']
    sample_input['path_density'] = sample_input['path_length'] / (sample_input['fanout_count'] + 1)
    sample_input['timing_score'] = sample_input['setup_slack'] * (1 - sample_input['timing_violation'])
    
    # Make prediction
    result = predict_complexity(sample_input)
    print(f"\nPredicted operation complexity: {result:.4f}")

if __name__ == "__main__":
    example_usage() 