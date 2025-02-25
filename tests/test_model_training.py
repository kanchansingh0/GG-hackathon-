import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd
from src.models.specific_models.timing_rf_model import TimingRFModel
from src.models.model_trainer import ModelTrainer

def generate_realistic_rtl_data(n_samples: int) -> pd.DataFrame:
    """Generate realistic RTL timing data"""
    # Generate base features with realistic relationships
    fanin_count = np.random.randint(1, 10, n_samples)  # Typically 1-10 inputs
    fanout_count = np.random.randint(1, 8, n_samples)  # Typically 1-8 outputs
    
    # Logic depth should correlate with fanin
    base_depth = np.ceil(np.log2(fanin_count + 1))  # Basic depth based on fanin
    logic_depth = base_depth + np.random.randint(0, 3, n_samples)  # Add some variation
    
    # Operation complexity correlates with both fanin and logic depth
    operation_complexity = (0.3 * fanin_count + 0.5 * logic_depth + 
                          np.random.normal(0, 0.5, n_samples))
    operation_complexity = np.maximum(operation_complexity, 1)  # Ensure positive
    
    # Path length correlates with logic depth and complexity
    path_length = (0.6 * logic_depth + 0.4 * operation_complexity + 
                  np.random.normal(0, 0.5, n_samples))
    path_length = np.maximum(path_length, logic_depth)  # Path length ≥ logic depth
    
    # Target (timing) should depend on all features with some noise
    target = (0.2 * fanin_count + 
             0.1 * fanout_count + 
             0.3 * logic_depth + 
             0.2 * operation_complexity + 
             0.2 * path_length + 
             np.random.normal(0, 0.1, n_samples))
    
    # Create DataFrame
    df = pd.DataFrame({
        'fanin_count': fanin_count,
        'fanout_count': fanout_count,
        'logic_depth': logic_depth,
        'operation_complexity': operation_complexity,
        'path_length': path_length,
        'target': target
    })
    
    # Normalize features to 0-1 range for consistent training
    for col in df.columns:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    
    # Add timing violation threshold (e.g., clock period)
    clock_period = 0.5  # Normalized clock period
    df['timing_violation'] = df['target'] > clock_period
    
    return df

def test_model_training_pipeline():
    try:
        # Generate realistic synthetic data
        n_samples = 1000
        df = generate_realistic_rtl_data(n_samples)
        
        print(f"\nDataset shape: {df.shape}")
        print(f"Feature statistics:\n{df.describe()}")
        
        # Print feature correlations
        print("\nFeature correlations with target:")
        correlations = df.corr()['target'].sort_values(ascending=False)
        print(correlations)
        
        # Initialize model and trainer
        model = TimingRFModel()
        trainer = ModelTrainer(
            model=model,
            features=model.feature_columns,
            target='target',
            test_size=0.2
        )
        
        # Train and evaluate
        metrics = trainer.train_and_evaluate(df)
        
        # Print detailed metrics
        print("\nModel Performance:")
        print(f"Training R²: {metrics['train_metrics']['r2']:.4f}")
        print(f"Test R²: {metrics['test_metrics']['r2']:.4f}")
        
        # Print classification metrics
        print("\nTiming Violation Detection:")
        print(f"Accuracy: {metrics['test_metrics']['accuracy']:.4f}")
        
        # Print confusion matrix
        cm = metrics['test_metrics']['confusion_matrix']
        print("\nConfusion Matrix:")
        print("                 Predicted No    Predicted Yes")
        print(f"Actual No     {cm['true_negative']:14d} {cm['false_positive']:14d}")
        print(f"Actual Yes    {cm['false_negative']:14d} {cm['true_positive']:14d}")
        
        # Print precision and recall
        print("\nDetailed Metrics:")
        print(f"Precision: {metrics['test_metrics']['precision']:.4f}")
        print(f"Recall: {metrics['test_metrics']['recall']:.4f}")
        
        # Print feature importance
        importance = metrics['feature_importance']
        print("\nFeature Importance:")
        for feat, imp in zip(model.feature_columns, importance['importance']):
            print(f"{feat}: {imp:.4f}")
        
        print("\nAll tests passed successfully!")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    test_model_training_pipeline() 