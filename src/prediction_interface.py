import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import os
import joblib

class RTLPredictor:
    def __init__(self):
        # Initialize model and scaler
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            random_state=42
        )
        self.scaler = StandardScaler()
        
        # Create and fit scaler with sample data
        self.sample_data = np.array([
            [0.01, 2.0, 0.04, 3, 2, 4],    # Your input values
            [0.02, 1.5, 0.03, 4, 3, 5],
            [0.03, 1.0, 0.02, 5, 4, 6],
            [0.04, 0.5, 0.01, 6, 5, 7]
        ])
        
        # Create target values based on a simple formula
        self.sample_targets = np.array([
            self._calculate_target(row) for row in self.sample_data
        ])
        
        # Fit scaler and model
        self.scaler.fit(self.sample_data)
        self.model.fit(self.sample_data, self.sample_targets)
    
    def _calculate_target(self, features):
        # Simple formula to generate target values
        setup_slack, hold_slack, timing_violation, fanin, fanout, path_length = features
        return (
            0.3 * abs(setup_slack) +
            0.2 * abs(hold_slack) +
            0.2 * timing_violation +
            0.1 * fanin +
            0.1 * fanout +
            0.1 * path_length
        )

    def predict_complexity(self, features):
        try:
            # Convert features to numpy array if not already
            features = np.array(features, dtype=float)
            
            # Reshape if needed
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            result = float(self.model.predict(features_scaled)[0])
            return result
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return None

    def get_detailed_recommendations(self, features):
        setup_slack = features[0]
        hold_slack = features[1]
        timing_violation = features[2]
        fanin = features[3]
        fanout = features[4]
        path_length = features[5]
        
        recommendations = []
        
        # Standard thresholds for RTL design
        STANDARDS = {
            'setup_slack_min': 0.5,
            'hold_slack_min': 0.2,
            'timing_violation_max': 0.1,
            'fanin_max': 4,
            'fanout_max': 5,
            'path_length_max': 7
        }
        
        # Timing violations check
        if timing_violation > STANDARDS['timing_violation_max']:
            recommendations.append({
                'issue': "High Timing Violations",
                'fixes': [
                    "- Insert pipeline registers to break long paths",
                    "- Adjust clock constraints",
                    "- Consider retiming optimization",
                    "- Review and optimize critical paths"
                ]
            })
        
        # Setup slack check
        if setup_slack < STANDARDS['setup_slack_min']:
            recommendations.append({
                'issue': "Insufficient Setup Slack",
                'fixes': [
                    "- Optimize critical path logic",
                    "- Add pipeline stages",
                    "- Consider clock skew optimization",
                    "- Review and adjust timing constraints"
                ]
            })
        
        # Hold slack check
        if hold_slack < STANDARDS['hold_slack_min']:
            recommendations.append({
                'issue': "Insufficient Hold Slack",
                'fixes': [
                    "- Insert delay buffers",
                    "- Review clock network",
                    "- Adjust min delay constraints",
                    "- Check for short paths"
                ]
            })
        
        # Fanin optimization
        if fanin > STANDARDS['fanin_max']:
            recommendations.append({
                'issue': "High Fan-in Count",
                'fixes': [
                    "- Split complex gates into simpler ones",
                    "- Use hierarchical design approach",
                    "- Implement logic sharing",
                    "- Consider using lookup tables for complex functions"
                ]
            })
        
        # Fanout optimization
        if fanout > STANDARDS['fanout_max']:
            recommendations.append({
                'issue': "High Fan-out Count",
                'fixes': [
                    "- Add buffer trees",
                    "- Implement register duplication",
                    "- Use clock gating where applicable",
                    "- Consider restructuring logic cone"
                ]
            })
        
        # Path length optimization
        if path_length > STANDARDS['path_length_max']:
            recommendations.append({
                'issue': "Excessive Path Length",
                'fixes': [
                    "- Add pipeline registers",
                    "- Implement logic restructuring",
                    "- Consider parallel processing",
                    "- Review and optimize logic levels"
                ]
            })
            
        return recommendations

    def get_circuit_status(self, features, prediction):
        recommendations = self.get_detailed_recommendations(features)
        
        if not recommendations:
            return "✅ Circuit meets all design standards", "No optimizations needed", "success"
        
        status = f"❌ Circuit needs optimization ({len(recommendations)} issues found)"
        
        detailed_msg = "Recommended Optimizations:\n\n"
        for rec in recommendations:
            detailed_msg += f"[{rec['issue']}]\n"
            detailed_msg += "\n".join(rec['fixes'])
            detailed_msg += "\n\n"
            
        return status, detailed_msg.strip(), "warning"

class PredictionInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("RTL Circuit Analyzer")
        
        # Initialize predictor
        self.predictor = RTLPredictor()
        
        # Create input fields
        self.create_input_fields()
        
    def create_input_fields(self):
        # Define features
        self.features = [
            'setup_slack',
            'hold_slack',
            'timing_violation',
            'fanin_count',
            'fanout_count',
            'path_length'
        ]
        
        # Create input fields
        self.entries = {}
        for i, feature in enumerate(self.features):
            # Label
            ttk.Label(self.root, text=feature.replace('_', ' ').title()).grid(
                row=i, column=0, padx=5, pady=5, sticky='e'
            )
            # Entry field
            self.entries[feature] = ttk.Entry(self.root)
            self.entries[feature].grid(row=i, column=1, padx=5, pady=5)
        
        # Predict button
        ttk.Button(
            self.root, 
            text="Predict Complexity",
            command=self.make_prediction
        ).grid(row=len(self.features), column=0, columnspan=2, pady=20)
        
        # Result labels
        self.result_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.recommendations_text = tk.Text(
            self.root,
            height=10,
            width=50,
            wrap=tk.WORD,
            font=('Arial', 10)
        )
        self.recommendations_text.grid(
            row=len(self.features)+2,
            column=0,
            columnspan=2,
            padx=10,
            pady=5
        )
        
    def make_prediction(self):
        try:
            # Collect input values
            feature_values = []
            for feature in self.features:
                value = float(self.entries[feature].get())
                feature_values.append(value)
            
            features = np.array(feature_values)
            prediction = self.predictor.predict_complexity(features)
            
            if prediction is not None:
                self.result_var.set(f"Circuit Complexity Score: {prediction:.4f}")
                
                # Get detailed status and recommendations
                status, recommendations, status_type = self.predictor.get_circuit_status(features, prediction)
                
                self.status_var.set(status)
                
                # Update recommendations text
                self.recommendations_text.delete('1.0', tk.END)
                self.recommendations_text.insert('1.0', recommendations)
                
                # Set text color based on status
                if status_type == "success":
                    self.recommendations_text.configure(fg="green")
                else:
                    self.recommendations_text.configure(fg="red")
                    
            else:
                self.result_var.set("Error: Could not analyze circuit")
                self.status_var.set("")
                self.recommendations_text.delete('1.0', tk.END)
                
        except ValueError as ve:
            self.result_var.set("Error: Please enter valid numbers")
            self.status_var.set("")
            self.recommendations_text.delete('1.0', tk.END)
        except Exception as e:
            self.result_var.set(f"Error: {str(e)}")
            self.status_var.set("")
            self.recommendations_text.delete('1.0', tk.END)

def main():
    root = tk.Tk()
    app = PredictionInterface(root)
    root.mainloop()

if __name__ == "__main__":
    main()

print(tk.TkVersion) 