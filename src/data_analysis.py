import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def analyze_and_prepare_data():
    # Load data
    df = pd.read_csv('data/processed/processed_features.csv')
    
    print("Data Analysis Report")
    print("===================")
    
    # Check target variable distribution
    print("\nTarget Variable (logic_levels) Distribution:")
    print(df['logic_levels'].value_counts())
    
    # Check feature correlations
    print("\nFeature Correlations:")
    correlations = df.corr()['operation_complexity'].sort_values(ascending=False)
    print(correlations)
    
    # Check if we can create a more meaningful target
    print("\nTrying to create a derived target variable...")
    
    # Example: Create a composite target from operation_complexity and path_length
    df['derived_target'] = (df['operation_complexity'] * df['path_length']).apply(np.ceil)
    
    print("\nDerived target distribution:")
    print(df['derived_target'].describe())
    
    # Save the modified dataset
    df.to_csv('data/processed/enhanced_features.csv', index=False)
    print("\nEnhanced dataset saved to 'data/processed/enhanced_features.csv'")
    
    return df

if __name__ == "__main__":
    analyze_and_prepare_data() 