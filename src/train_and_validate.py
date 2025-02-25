import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from pathlib import Path

def train_and_validate():
    try:
        # Load data
        print("Loading data...")
        df = pd.read_csv('data/processed/processed_features.csv')
        
        # Prepare features and target
        features = ['setup_slack', 'hold_slack', 'timing_violation', 
                   'fanin_count', 'fanout_count', 'path_length']
        target = 'operation_complexity'
        
        X = df[features]
        y = df[target]
        
        # Split data
        print("\nSplitting data into train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        print("\nTraining Random Forest model...")
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        print("\nCalculating metrics...")
        train_mse = mean_squared_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print("\nTraining Results:")
        print(f"MSE: {train_mse:.4f}")
        print(f"R²: {train_r2:.4f}")
        
        print("\nTest Results:")
        print(f"MSE: {test_mse:.4f}")
        print(f"R²: {test_r2:.4f}")
        
        # Save model and predictions
        print("\nSaving model and predictions...")
        Path('models').mkdir(exist_ok=True)
        Path('results').mkdir(exist_ok=True)
        
        joblib.dump(model, 'models/rf_model.joblib')
        joblib.dump(scaler, 'models/rf_scaler.joblib')
        
        # Save test predictions
        test_results = pd.DataFrame({
            'Actual': y_test,
            'Predicted': test_pred
        })
        test_results.to_csv('results/complexity_predictions.csv', index=False)
        
        # Feature importance
        print("\nFeature Importance:")
        importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        print(importance)
        
        print("\nTraining and validation completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")

if __name__ == "__main__":
    train_and_validate() 