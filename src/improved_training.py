import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

def create_interaction_features(df):
    """Create interaction features"""
    df = df.copy()
    
    # Create meaningful combinations
    df['slack_ratio'] = df['setup_slack'] / (df['hold_slack'].abs() + 1e-6)
    df['complexity_score'] = df['fanin_count'] * df['fanout_count']
    df['path_density'] = df['path_length'] / (df['fanout_count'] + 1)
    df['timing_score'] = df['setup_slack'] * (1 - df['timing_violation'])
    
    return df

def train_improved_model():
    try:
        # Load data
        print("Loading data...")
        df = pd.read_csv('data/processed/processed_features.csv')
        
        # Feature engineering
        print("\nPerforming feature engineering...")
        df = create_interaction_features(df)
        
        # Prepare features and target
        features = [
            'setup_slack', 'hold_slack', 'timing_violation',
            'fanin_count', 'fanout_count', 'path_length',
            'slack_ratio', 'complexity_score', 'path_density', 'timing_score'
        ]
        target = 'operation_complexity'
        
        X = df[features]
        y = df[target]
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('rf', RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ))
        ])
        
        # Perform cross-validation
        print("\nPerforming 5-fold cross-validation...")
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='r2')
        
        print("\nCross-validation R² scores:")
        print(f"Mean R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train final model
        print("\nTraining final model...")
        pipeline.fit(X, y)
        
        # Make predictions
        y_pred = pipeline.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        print("\nFinal Model Results:")
        print(f"MSE: {mse:.4f}")
        print(f"R²: {r2:.4f}")
        
        # Feature importance (for the RF part)
        rf_model = pipeline.named_steps['rf']
        poly = pipeline.named_steps['poly']
        feature_names = poly.get_feature_names_out(features)
        
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(importance.head(10))
        
        # Save model and predictions
        print("\nSaving model and results...")
        Path('models').mkdir(exist_ok=True)
        Path('results').mkdir(exist_ok=True)
        
        joblib.dump(pipeline, 'models/improved_rf_model.joblib')
        
        # Save predictions
        results = pd.DataFrame({
            'Actual': y,
            'Predicted': y_pred
        })
        results.to_csv('results/improved_predictions.csv', index=False)
        
        # Save feature importance
        importance.to_csv('results/feature_importance.csv', index=False)
        
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")

if __name__ == "__main__":
    train_improved_model() 