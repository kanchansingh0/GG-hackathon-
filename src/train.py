import pandas as pd
import numpy as np
from models.model_trainer import ModelTrainer
from models.rf_model import RFModel
from sklearn.model_selection import cross_val_score

def validate_data(df: pd.DataFrame, features: list, target: str):
    """Validate the input data"""
    print("\nData Validation:")
    print(f"Total samples: {len(df)}")
    
    # Check value ranges
    print("\nValue ranges:")
    for col in features + [target]:
        print(f"\n{col}:")
        print(f"  Min: {df[col].min():.2f}")
        print(f"  Max: {df[col].max():.2f}")
        print(f"  Mean: {df[col].mean():.2f}")
        print(f"  Std: {df[col].std():.2f}")
        print(f"  Unique values: {df[col].nunique()}")
    
    # Check correlations with target
    correlations = df[features].corrwith(df[target])
    print("\nFeature correlations with target:")
    for feat, corr in correlations.items():
        print(f"{feat}: {corr:.4f}")
        
    # Check for constant or near-constant features
    print("\nFeature variance:")
    for feat in features:
        variance = df[feat].var()
        print(f"{feat}: {variance:.4f}")

def train_model():
    try:
        # Load enhanced data
        df = pd.read_csv('data/processed/enhanced_features.csv')
        print("Data loaded successfully!")
        
        features = ['setup_slack', 'hold_slack', 'timing_violation', 
                   'fanin_count', 'fanout_count', 'operation_complexity', 
                   'path_length']
        target = 'derived_target'  # Using the new derived target
        
        # Initialize model with adjusted parameters
        model = RFModel()
        trainer = ModelTrainer(model, features=features, target=target)
        
        # Train and evaluate
        print("\nTraining model...")
        metrics = trainer.train_and_evaluate(df)
        
        # Save the model
        model.save()
        print("\nModel saved successfully!")
        
        # Print results
        print("\nTraining Results:")
        print(f"Train MSE: {metrics['train_metrics']['mse']:.4f}")
        print(f"Train R²: {metrics['train_metrics']['r2']:.4f}")
        print("\nTest Results:")
        print(f"Test MSE: {metrics['test_metrics']['mse']:.4f}")
        print(f"Test R²: {metrics['test_metrics']['r2']:.4f}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")

if __name__ == "__main__":
    train_model() 