import tkinter as tk
from tkinter import ttk
import joblib
import pandas as pd
import numpy as np

class PredictionInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Operation Complexity Predictor")
        
        # Load model
        self.model = joblib.load('models/improved_rf_model.joblib')
        
        # Create input fields
        self.create_input_fields()
        
    def create_input_fields(self):
        # Create labels and entry fields
        features = [
            'setup_slack', 'hold_slack', 'timing_violation',
            'fanin_count', 'fanout_count', 'path_length'
        ]
        
        self.entries = {}
        for i, feature in enumerate(features):
            ttk.Label(self.root, text=feature).grid(row=i, column=0, padx=5, pady=5)
            self.entries[feature] = ttk.Entry(self.root)
            self.entries[feature].grid(row=i, column=1, padx=5, pady=5)
            
        # Predict button
        ttk.Button(self.root, text="Predict", command=self.make_prediction).grid(
            row=len(features), column=0, columnspan=2, pady=20
        )
        
        # Result label
        self.result_var = tk.StringVar()
        ttk.Label(self.root, textvariable=self.result_var).grid(
            row=len(features)+1, column=0, columnspan=2
        )
        
    def make_prediction(self):
        try:
            # Gather input values
            input_data = {}
            for feature, entry in self.entries.items():
                value = float(entry.get())
                input_data[feature] = [value]
                
            # Create DataFrame
            df = pd.DataFrame(input_data)
            
            # Calculate derived features
            df['slack_ratio'] = df['setup_slack'] / (df['hold_slack'].abs() + 1e-6)
            df['complexity_score'] = df['fanin_count'] * df['fanout_count']
            df['path_density'] = df['path_length'] / (df['fanout_count'] + 1)
            df['timing_score'] = df['setup_slack'] * (1 - df['timing_violation'])
            
            # Make prediction
            prediction = self.model.predict(df)
            self.result_var.set(f"Predicted Complexity: {prediction[0]:.4f}")
            
        except Exception as e:
            self.result_var.set(f"Error: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PredictionInterface(root)
    root.mainloop() 