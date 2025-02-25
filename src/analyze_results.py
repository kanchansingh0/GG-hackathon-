import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_results():
    try:
        # Load predictions
        predictions = pd.read_csv('results/improved_predictions.csv')
        importance = pd.read_csv('results/feature_importance.csv')
        
        # Calculate metrics
        mse = ((predictions['Actual'] - predictions['Predicted']) ** 2).mean()
        r2 = 1 - (((predictions['Actual'] - predictions['Predicted']) ** 2).sum() / 
                  ((predictions['Actual'] - predictions['Actual'].mean()) ** 2).sum())
        
        print("\nModel Performance Metrics:")
        print(f"MSE: {mse:.4f}")
        print(f"R²: {r2:.4f}")
        
        # Create plots directory
        Path('results/plots').mkdir(parents=True, exist_ok=True)
        
        # Actual vs Predicted plot
        plt.figure(figsize=(10, 6))
        plt.scatter(predictions['Actual'], predictions['Predicted'], alpha=0.5)
        plt.plot([predictions['Actual'].min(), predictions['Actual'].max()], 
                [predictions['Actual'].min(), predictions['Actual'].max()], 
                'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.savefig('results/plots/actual_vs_predicted.png')
        plt.close()
        
        # Feature importance plot
        plt.figure(figsize=(12, 6))
        # Sort importance by value
        importance = importance.sort_values('Importance', ascending=True)
        # Plot horizontal bar chart
        plt.barh(range(len(importance.head(10))), importance.head(10)['Importance'])
        plt.yticks(range(len(importance.head(10))), importance.head(10)['Feature'])
        plt.xlabel('Importance Score')
        plt.title('Top 10 Feature Importance')
        plt.tight_layout()
        plt.savefig('results/plots/feature_importance.png')
        plt.close()
        
        # Print top 10 important features
        print("\nTop 10 Most Important Features:")
        print(importance.tail(10)[::-1][['Feature', 'Importance']].to_string())
        
        print("\nAnalysis completed! Check the 'results/plots' directory for visualizations.")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    analyze_results() 