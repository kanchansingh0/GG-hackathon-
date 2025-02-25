import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. Skipping XGBoost comparison.")

def compare_ml_agents():
    print("Comparing different ML agents for combinational depth prediction...")
    
    try:
        # Generate sample data
        n_samples = 1000
        print(f"\nGenerating {n_samples} synthetic samples...")
        
        # Create synthetic features
        data = {
            'fanin_count': np.random.randint(1, 10, n_samples),
            'fanout_count': np.random.randint(1, 8, n_samples),
            'logic_depth': np.random.randint(1, 15, n_samples),
            'operation_complexity': np.random.uniform(0, 1, n_samples),
            'path_length': np.random.uniform(0, 1, n_samples)
        }
        
        # Create target variable with clear relationships
        data['target'] = (0.2 * data['fanin_count'] + 
                         0.1 * data['fanout_count'] + 
                         0.3 * data['logic_depth'] + 
                         0.2 * data['operation_complexity'] + 
                         0.2 * data['path_length'] + 
                         np.random.normal(0, 0.1, n_samples))
        
        df = pd.DataFrame(data)
        print("Dataset created successfully.")
        
        # Normalize features
        for col in df.columns:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        
        # Prepare data
        X = df[['fanin_count', 'fanout_count', 'logic_depth', 
                'operation_complexity', 'path_length']]
        y = df['target']
        
        # Define models to compare
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42),
            'Linear Regression': LinearRegression(),
            'SVR': SVR(kernel='rbf'),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), 
                                         max_iter=1000, random_state=42)
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = XGBRegressor(n_estimators=200, random_state=42)
        
        # Compare models
        print("\nComparing models using 5-fold cross-validation...")
        results = {}
        for name, model in models.items():
            print(f"\nEvaluating {name}...")
            try:
                # Perform cross-validation
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                results[name] = {
                    'mean_score': cv_scores.mean(),
                    'std_score': cv_scores.std()
                }
                print(f"Completed {name} evaluation.")
            except Exception as e:
                print(f"Error evaluating {name}: {str(e)}")
                continue
            
        # Print results
        print("\n=== Model Comparison Results ===")
        for name, metrics in results.items():
            print(f"\n{name}:")
            print(f"Mean R² Score: {metrics['mean_score']:.4f}")
            print(f"Standard Deviation: {metrics['std_score']:.4f}")
        
        # Visualize results
        plt.figure(figsize=(12, 6))
        names = list(results.keys())
        mean_scores = [results[name]['mean_score'] for name in names]
        std_scores = [results[name]['std_score'] for name in names]
        
        plt.bar(names, mean_scores, yerr=std_scores, capsize=5)
        plt.title('Comparison of ML Agents for Combinational Depth Prediction')
        plt.ylabel('R² Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('model_comparison.png')
        print("\nModel comparison visualization saved as 'model_comparison.png'")
        
    except Exception as e:
        print(f"\nError during comparison: {str(e)}")
        import traceback
        print("\nTraceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    compare_ml_agents() 