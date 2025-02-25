from pathlib import Path

def check_model_files():
    print("Checking model files...")
    
    # Define expected model files
    model_files = [
        'models/rf_model.joblib',
        'models/rf_scaler.joblib',
        'results/complexity_predictions.csv'
    ]
    
    # Check each file
    for file_path in model_files:
        path = Path(file_path)
        if path.exists():
            print(f"✓ Found {file_path}")
            print(f"  Size: {path.stat().st_size / 1024:.2f} KB")
            print(f"  Last modified: {path.stat().st_mtime}")
        else:
            print(f"✗ Missing {file_path}")
    
    print("\nCheck completed!")

if __name__ == "__main__":
    check_model_files() 