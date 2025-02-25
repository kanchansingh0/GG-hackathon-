import pandas as pd
import joblib
import re
from pathlib import Path

class RTLPredictor:
    def __init__(self):
        try:
            self.model = joblib.load('models/improved_rf_model.joblib')
        except:
            raise Exception("Model not found. Please ensure the trained model exists in 'models/improved_rf_model.joblib'")

    def analyze_verilog(self, file_path):
        """Analyze a Verilog file and extract metrics"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            # Extract basic metrics
            metrics = self._extract_metrics(content)
            
            # Prepare features for prediction
            features = self._prepare_features(metrics)
            
            # Make prediction
            complexity = self.model.predict(features)[0]
            
            return self._generate_report(metrics, complexity, file_path)
            
        except Exception as e:
            return f"Error analyzing file: {str(e)}"

    def _extract_metrics(self, content):
        """Extract metrics from Verilog content"""
        metrics = {
            'fanin_count': len(re.findall(r'input\s+(?:wire|reg)?\s*\[?.*?\]?\s*\w+', content)),
            'fanout_count': len(re.findall(r'output\s+(?:wire|reg)?\s*\[?.*?\]?\s*\w+', content)),
            'path_length': len(re.findall(r'always\s*@', content)) + len(re.findall(r'assign\s+', content)),
            'timing_violation': 1 if 'always @(posedge clk' in content and 'always @(negedge clk' in content else 0,
        }
        
        # Estimate setup and hold slack based on circuit characteristics
        always_blocks = len(re.findall(r'always\s*@', content))
        assignments = len(re.findall(r'<=|=(?!=)', content))
        nested_ops = len(re.findall(r'[{(\[].+[})\]]', content))
        
        metrics['setup_slack'] = 10.0 - (nested_ops * 0.5) - (always_blocks * 0.3)
        metrics['hold_slack'] = 2.0 - (assignments * 0.1)
        
        return metrics

    def _prepare_features(self, metrics):
        """Prepare features for model prediction"""
        df = pd.DataFrame([metrics])
        
        # Calculate derived features
        df['slack_ratio'] = df['setup_slack'] / (df['hold_slack'].abs() + 1e-6)
        df['complexity_score'] = df['fanin_count'] * df['fanout_count']
        df['path_density'] = df['path_length'] / (df['fanout_count'] + 1)
        df['timing_score'] = df['setup_slack'] * (1 - df['timing_violation'])
        
        return df

    def _generate_report(self, metrics, complexity, file_path):
        """Generate analysis report"""
        report = f"\nRTL Analysis Report for {Path(file_path).name}"
        report += "\n" + "="*50 + "\n"
        
        report += "\nCircuit Metrics:\n"
        for metric, value in metrics.items():
            report += f"- {metric}: {value}\n"
        
        report += f"\nPredicted Complexity Score: {complexity:.4f}\n"
        
        report += "\nRecommendations:\n"
        if complexity > 7.0:
            report += "- HIGH COMPLEXITY: Consider major restructuring\n"
            report += "- Add pipeline stages to break long paths\n"
            report += "- Review and optimize timing paths\n"
            report += "- Consider splitting into smaller modules\n"
        elif complexity > 5.0:
            report += "- MODERATE COMPLEXITY: Some optimization needed\n"
            report += "- Review critical paths\n"
            report += "- Consider adding pipeline registers\n"
            report += "- Optimize combinational logic\n"
        else:
            report += "- LOW COMPLEXITY: Design is well structured\n"
            report += "- Monitor timing during implementation\n"
            report += "- Regular maintenance should suffice\n"
            
        return report

def main():
    predictor = RTLPredictor()
    
    print("RTL Complexity Predictor")
    print("Enter path to Verilog file (or 'q' to quit):")
    
    while True:
        file_path = input("\nFile path: ").strip()
        
        if file_path.lower() == 'q':
            break
            
        if not file_path.endswith('.v'):
            print("Please provide a Verilog (.v) file")
            continue
            
        if not Path(file_path).exists():
            print("File not found")
            continue
            
        result = predictor.analyze_verilog(file_path)
        print(result)

if __name__ == "__main__":
    main() 