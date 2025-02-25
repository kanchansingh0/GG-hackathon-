import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def train_complexity_model():
    # Load data
    df = pd.read_csv('data/processed/processed_features.csv')
    print("Data loaded successfully!")
    
    # Select features and target
    features = ['setup_slack', 'hold_slack', 'timing_violation', 
               'fanin_count', 'fanout_count', 'path_length']
    target = 'operation_complexity'
    
    # Print target distribution
    print("\nOperation Complexity Distribution:")
    print(df[target].describe())
    
    # Prepare data
    X = df[features]
    y = df[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("\nTraining model to predict operation complexity...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, train_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    # Print results
    print("\nTraining Results:")
    print(f"Train MSE: {train_mse:.4f}")
    print(f"Train R²: {train_r2:.4f}")
    print("\nTest Results:")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    
    # Print feature importances
    importances = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importances:")
    print(importances)
    
    # Save predictions
    df_test = pd.DataFrame({
        'Actual': y_test,
        'Predicted': test_pred
    })
    df_test.to_csv('results/complexity_predictions.csv', index=False)
    print("\nPredictions saved to 'results/complexity_predictions.csv'")

if __name__ == "__main__":
    train_complexity_model() 