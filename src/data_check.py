import pandas as pd
import numpy as np

def analyze_data():
    # Load the data
    df = pd.read_csv('data/processed/processed_features.csv')
    
    print("Data Analysis Report")
    print("===================")
    
    # Basic info
    print("\n1. Basic Information:")
    print(f"Number of samples: {len(df)}")
    print("\nColumns:")
    print(df.info())
    
    # Check for constant values
    print("\n2. Checking for constant values:")
    for col in df.columns:
        unique_vals = df[col].nunique()
        print(f"{col}: {unique_vals} unique values")
    
    # Check for NaN or infinite values
    print("\n3. Checking for NaN or infinite values:")
    print(df.isna().sum())
    print("\nInfinite values:")
    print(np.isinf(df.select_dtypes(include=np.number)).sum())
    
    # Value ranges
    print("\n4. Value ranges:")
    print(df.describe())
    
    # Save sample data
    print("\n5. First few rows of data:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    analyze_data() 