import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

def evaluate_model():
    try:
        # Load the test data
        print("Loading test data...")
        df = pd.read_csv('data/processed/processed_features.csv')
        
        # Print basic statistics
        print("\nData Statistics:")
        print(f"Total samples: {len(df)}")
        print("\nFeature Statistics:")
        for column in df.columns:
            print(f"\n{column}:")
            print(f"Mean: {df[column].mean():.4f}")
            print(f"Std: {df[column].std():.4f}")
            print(f"Min: {df[column].min():.4f}")
            print(f"Max: {df[column].max():.4f}")
        
        # Check feature correlations
        print("\nFeature Correlations with Operation Complexity:")
        correlations = df.corr()['operation_complexity'].sort_values(ascending=False)
        print(correlations)
        
        # Load model predictions if they exist
        predictions_path = Path('results/complexity_predictions.csv')
        if predictions_path.exists():
            predictions = pd.read_csv(predictions_path)
            print("\nModel Predictions Analysis:")
            mse = mean_squared_error(predictions['Actual'], predictions['Predicted'])
            r2 = r2_score(predictions['Actual'], predictions['Predicted'])
            print(f"Mean Squared Error: {mse:.4f}")
            print(f"R² Score: {r2:.4f}")
        
        print("\nEvaluation completed!")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")

if __name__ == "__main__":
    evaluate_model() 