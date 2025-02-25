import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
from src.models.specific_models.timing_rf_model import TimingRFModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

def display_model_metrics():
    print("Calculating model metrics...")
    
    try:
        # Generate sample data
        n_samples = 1000  # Using more samples for better evaluation
        
        # Create synthetic features
        data = {
            'fanin_count': np.random.randint(1, 10, n_samples),
            'fanout_count': np.random.randint(1, 8, n_samples),
            'logic_depth': np.random.randint(1, 15, n_samples),
            'operation_complexity': np.random.uniform(0, 1, n_samples),
            'path_length': np.random.uniform(0, 1, n_samples)
        }
        
        # Create target variable
        data['target'] = (0.2 * data['fanin_count'] + 
                         0.1 * data['fanout_count'] + 
                         0.3 * data['logic_depth'] + 
                         0.2 * data['operation_complexity'] + 
                         0.2 * data['path_length'] + 
                         np.random.normal(0, 0.1, n_samples))
        
        df = pd.DataFrame(data)
        
        # Normalize features
        for col in df.columns:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        
        # Split data into train and test sets
        from sklearn.model_selection import train_test_split
        X = df[['fanin_count', 'fanout_count', 'logic_depth', 
                'operation_complexity', 'path_length']]
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                           random_state=42)
        
        # Initialize and train model
        model = TimingRFModel()
        model.train(X_train, y_train)
        
        # Get metrics
        train_metrics = model.evaluate(X_train, y_train)
        test_metrics = model.evaluate(X_test, y_test)
        
        # Display accuracy
        print("\n=== Model Accuracy ===")
        print(f"Training Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"Testing Accuracy: {test_metrics['accuracy']:.4f}")
        
        # Display confusion matrix
        print("\n=== Confusion Matrix (Test Set) ===")
        cm = test_metrics['confusion_matrix']
        print("\nPredicted Negative  Predicted Positive")
        print(f"Actual Negative  {cm['true_negative']:^16d} {cm['false_positive']:^16d}")
        print(f"Actual Positive  {cm['false_negative']:^16d} {cm['true_positive']:^16d}")
        
        # Calculate additional metrics
        print("\n=== Additional Metrics ===")
        precision = test_metrics['precision']
        recall = test_metrics['recall']
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}")
        
        # Visualize confusion matrix
        plt.figure(figsize=(10, 8))
        cm_values = np.array([
            [cm['true_negative'], cm['false_positive']],
            [cm['false_negative'], cm['true_positive']]
        ])
        
        sns.heatmap(cm_values, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Predicted Negative', 'Predicted Positive'],
                   yticklabels=['Actual Negative', 'Actual Positive'])
        plt.title('Confusion Matrix Heatmap')
        
        # Save the plot
        plt.savefig('confusion_matrix.png')
        print("\nConfusion matrix visualization saved as 'confusion_matrix.png'")
        
        # Feature importance
        print("\n=== Feature Importance ===")
        importance = model.get_feature_importance()
        for feat, imp in zip(model.feature_columns, importance['importance']):
            print(f"{feat}: {imp:.4f}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        print("\nTraceback:")
        print(traceback.format_exc())

def compare_ml_agents():
    print("Comparing different ML agents for combinational depth prediction...")
    
    try:
        # Generate sample data (using your existing data generation code)
        n_samples = 1000
        data = {
            'fanin_count': np.random.randint(1, 10, n_samples),
            'fanout_count': np.random.randint(1, 8, n_samples),
            'logic_depth': np.random.randint(1, 15, n_samples),
            'operation_complexity': np.random.uniform(0, 1, n_samples),
            'path_length': np.random.uniform(0, 1, n_samples)
        }
        
        # Create target variable
        data['target'] = (0.2 * data['fanin_count'] + 
                         0.1 * data['fanout_count'] + 
                         0.3 * data['logic_depth'] + 
                         0.2 * data['operation_complexity'] + 
                         0.2 * data['path_length'] + 
                         np.random.normal(0, 0.1, n_samples))
        
        df = pd.DataFrame(data)
        
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
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000),
            'XGBoost': XGBRegressor(n_estimators=200, random_state=42)
        }
        
        # Compare models
        results = {}
        for name, model in models.items():
            # Perform cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            results[name] = {
                'mean_score': cv_scores.mean(),
                'std_score': cv_scores.std()
            }
            
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
        print(f"\nError: {str(e)}")
        import traceback
        print("\nTraceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    display_model_metrics()
    compare_ml_agents() 