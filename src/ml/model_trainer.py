import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class LogicLevelPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def prepare_data(self):
        """Load and prepare data for training"""
        # Load processed features
        data_path = Path("data/processed/processed_features.csv")
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found at {data_path}")
            
        df = pd.read_csv(data_path)
        print("\nAvailable columns:", df.columns.tolist())
        
        # Use logic_levels as target instead of logic_depth
        target_col = 'logic_levels'
        if target_col not in df.columns:
            raise KeyError(f"'{target_col}' column not found in the data. Available columns: " + 
                         ", ".join(df.columns))
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col != target_col]
        
        print(f"\nTarget column: {target_col}")
        print(f"Feature columns: {feature_cols}")
        
        X = df[feature_cols]
        y = df[target_col]
        
        print("\nData shape:")
        print(f"Features (X): {X.shape}")
        print(f"Target (y): {y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols
    
    def train(self):
        """Train the model"""
        try:
            # Prepare data
            X_train, X_test, y_train, y_test, feature_cols = self.prepare_data()
            
            # Train model
            print("\nTraining model...")
            self.model.fit(X_train, y_train)
            
            # Evaluate
            train_pred = self.model.predict(X_train)
            test_pred = self.model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'train_mse': mean_squared_error(y_train, train_pred),
                'test_mse': mean_squared_error(y_test, test_pred),
                'train_r2': r2_score(y_train, train_pred),
                'test_r2': r2_score(y_test, test_pred)
            }
            
            # Feature importance
            importance = dict(zip(feature_cols, self.model.feature_importances_))
            
            print("\nModel Performance:")
            print(f"Train MSE: {metrics['train_mse']:.4f}")
            print(f"Test MSE: {metrics['test_mse']:.4f}")
            print(f"Train R²: {metrics['train_r2']:.4f}")
            print(f"Test R²: {metrics['test_r2']:.4f}")
            
            print("\nTop 5 Important Features:")
            sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5])
            for feature, imp in sorted_importance.items():
                print(f"{feature}: {imp:.4f}")
                
            return metrics, importance
            
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            return None, None
    
    def save_model(self, output_dir: str = "models"):
        """Save the trained model and scaler"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save model and scaler
        joblib.dump(self.model, output_path / "logic_depth_model.joblib")
        joblib.dump(self.scaler, output_path / "feature_scaler.joblib")
        print(f"\nModel saved to {output_path}")
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Make predictions for new data"""
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)

def main():
    try:
        # Initialize and train model
        predictor = LogicLevelPredictor()
        metrics, importance = predictor.train()
        
        if metrics and importance:
            # Save model
            predictor.save_model()
            
            # Save metrics and importance
            results = {
                'metrics': metrics,
                'feature_importance': {k: float(v) for k, v in importance.items()}
            }
            
            output_dir = Path("results")
            output_dir.mkdir(exist_ok=True)
            
            import json
            with open(output_dir / "training_results.json", 'w') as f:
                json.dump(results, f, indent=4)
                
    except Exception as e:
        print(f"\nError in main: {str(e)}")

if __name__ == "__main__":
    main()
