import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
from src.models.specific_models.timing_rf_model import TimingRFModel
from src.models.model_trainer import ModelTrainer

def debug_model():
    print("Starting debug process...")
    
    try:
        # Generate sample data
        n_samples = 100
        print(f"\nGenerating {n_samples} synthetic samples...")
        
        # Create synthetic features
        data = {
            'fanin_count': np.random.randint(1, 10, n_samples),
            'fanout_count': np.random.randint(1, 8, n_samples),
            'logic_depth': np.random.randint(1, 15, n_samples),
            'operation_complexity': np.random.uniform(0, 1, n_samples),
            'path_length': np.random.uniform(0, 1, n_samples)
        }
        
        # Create target variable with clear threshold relationship
        data['target'] = (0.2 * data['fanin_count'] + 
                         0.1 * data['fanout_count'] + 
                         0.3 * data['logic_depth'] + 
                         0.2 * data['operation_complexity'] + 
                         0.2 * data['path_length'] + 
                         np.random.normal(0, 0.1, n_samples))
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Normalize features
        for col in df.columns:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        
        print("\nDataset created successfully")
        print(f"Dataset shape: {df.shape}")
        print("\nFeature statistics:")
        print(df.describe())
        
        # Initialize model
        print("\nInitializing model...")
        model = TimingRFModel()
        trainer = ModelTrainer(
            model=model,
            features=model.feature_columns,
            target='target',
            test_size=0.2
        )
        
        # Train and evaluate
        print("\nTraining and evaluating model...")
        metrics = trainer.train_and_evaluate(df)
        
        # Print all available metrics
        print("\nModel Performance:")
        print("\nTraining Metrics:")
        for metric_name, value in metrics['train_metrics'].items():
            if isinstance(value, dict):
                print(f"\n{metric_name}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{metric_name}: {value:.4f}")
        
        print("\nTest Metrics:")
        for metric_name, value in metrics['test_metrics'].items():
            if isinstance(value, dict):
                print(f"\n{metric_name}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{metric_name}: {value:.4f}")
        
        # Print feature importance
        print("\nFeature Importance:")
        importance = metrics['feature_importance']
        for feat, imp in zip(model.feature_columns, importance['importance']):
            print(f"{feat}: {imp:.4f}")
            
        print("\nDebug completed successfully!")
        
    except Exception as e:
        print(f"\nError during debug: {str(e)}")
        import traceback
        print("\nTraceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    debug_model()