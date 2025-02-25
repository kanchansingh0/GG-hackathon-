import pandas as pd
from pathlib import Path

def validate_processed_data():
    """Validate processed features"""
    # Check if file exists
    data_path = Path("data/processed/processed_features.csv")
    if not data_path.exists():
        print(f"\nError: Data file not found at {data_path}")
        print("Please run the data processor first to generate the features.")
        return
    
    # Check if file is empty
    if data_path.stat().st_size == 0:
        print(f"\nError: Data file is empty at {data_path}")
        return
    
    try:
        # Load processed data
        df = pd.read_csv(data_path)
        
        if df.empty:
            print("Error: DataFrame is empty after loading")
            return
            
        # Print basic information
        print("\nData Overview:")
        print(f"Number of samples: {len(df)}")
        print(f"Number of features: {len(df.columns)}")
        print("\nFeatures:", df.columns.tolist())
        
        # Check for missing values
        missing_values = df.isnull().sum()
        
        # Check feature distributions
        feature_stats = df.describe()
        
        # Check for outliers
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        outliers = {}
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers[col] = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)].shape[0]
        
        # Print validation results
        print("\nData Validation Results:")
        print("\nMissing Values:")
        print(missing_values[missing_values > 0])
        
        print("\nOutliers per Feature:")
        for feature, count in outliers.items():
            print(f"{feature}: {count} outliers")
        
        return {
            'missing_values': missing_values,
            'feature_stats': feature_stats,
            'outliers': outliers
        }
        
    except Exception as e:
        print(f"\nError while processing data: {str(e)}")
        return None

if __name__ == "__main__":
    validate_processed_data() 