import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from predict import LogicLevelPredictor

def evaluate_model():
    # Load test data
    test_data = pd.read_csv("data/processed/processed_features.csv")
    
    # Initialize predictor
    predictor = LogicLevelPredictor()
    
    # Get actual values
    y_true = test_data['logic_levels']
    
    # Get predictions
    features = test_data.drop('logic_levels', axis=1).to_dict('records')
    y_pred = predictor.predict_batch(features)
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print("\nModel Evaluation:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Logic Levels')
    plt.ylabel('Predicted Logic Levels')
    plt.title('Predicted vs Actual Logic Levels')
    
    # Save plot
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "prediction_performance.png")
    plt.close()

if __name__ == "__main__":
    evaluate_model() 