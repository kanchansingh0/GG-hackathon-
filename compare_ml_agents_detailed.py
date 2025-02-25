import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. Skipping XGBoost comparison.")

def evaluate_model(model, X_train, X_test, y_train, y_test, threshold=0.5):
    """Evaluate a single model and return metrics"""
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Convert to binary predictions using threshold
    y_train_binary = y_train > threshold
    y_test_binary = y_test > threshold
    y_train_pred_binary = y_train_pred > threshold
    y_test_pred_binary = y_test_pred > threshold
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train_binary, y_train_pred_binary)
    test_accuracy = accuracy_score(y_test_binary, y_test_pred_binary)
    
    # Calculate confusion matrices
    train_cm = confusion_matrix(y_train_binary, y_train_pred_binary)
    test_cm = confusion_matrix(y_test_binary, y_test_pred_binary)
    
    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_confusion_matrix': train_cm,
        'test_confusion_matrix': test_cm,
        'train_predictions': y_train_pred,
        'test_predictions': y_test_pred
    }

def plot_confusion_matrix(cm, title, save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Violation', 'Violation'],
                yticklabels=['No Violation', 'Violation'])
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def compare_ml_agents_detailed():
    print("Performing detailed comparison of ML agents...")
    
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
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                           random_state=42)
        
        # Define models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42),
            'Linear Regression': LinearRegression(),
            'SVR': SVR(kernel='rbf'),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), 
                                         max_iter=1000, random_state=42)
        }
        
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = XGBRegressor(n_estimators=200, random_state=42)
        
        # Compare models
        results = {}
        for name, model in models.items():
            print(f"\nEvaluating {name}...")
            try:
                metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
                results[name] = metrics
                
                # Print results
                print(f"\n=== {name} Results ===")
                print(f"Training Accuracy: {metrics['train_accuracy']:.4f}")
                print(f"Testing Accuracy: {metrics['test_accuracy']:.4f}")
                
                # Plot confusion matrices
                plot_confusion_matrix(
                    metrics['train_confusion_matrix'],
                    f'{name} - Training Confusion Matrix',
                    f'confusion_matrix_{name.lower().replace(" ", "_")}_train.png'
                )
                plot_confusion_matrix(
                    metrics['test_confusion_matrix'],
                    f'{name} - Testing Confusion Matrix',
                    f'confusion_matrix_{name.lower().replace(" ", "_")}_test.png'
                )
                
            except Exception as e:
                print(f"Error evaluating {name}: {str(e)}")
                continue
        
        # Create comparison plot
        plt.figure(figsize=(12, 6))
        names = list(results.keys())
        train_scores = [results[name]['train_accuracy'] for name in names]
        test_scores = [results[name]['test_accuracy'] for name in names]
        
        x = np.arange(len(names))
        width = 0.35
        
        plt.bar(x - width/2, train_scores, width, label='Training Accuracy')
        plt.bar(x + width/2, test_scores, width, label='Testing Accuracy')
        
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.title('Model Comparison - Training vs Testing Accuracy')
        plt.xticks(x, names, rotation=45)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig('model_comparison_accuracy.png')
        print("\nModel comparison plots have been saved.")
        
        # Save detailed results to CSV
        results_df = pd.DataFrame({
            'Model': names,
            'Training Accuracy': train_scores,
            'Testing Accuracy': test_scores
        })
        results_df.to_csv('model_comparison_results.csv', index=False)
        print("\nDetailed results have been saved to 'model_comparison_results.csv'")
        
    except Exception as e:
        print(f"\nError during comparison: {str(e)}")
        import traceback
        print("\nTraceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    compare_ml_agents_detailed() 