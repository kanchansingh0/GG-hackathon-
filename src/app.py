from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import re
from pathlib import Path

app = Flask(__name__)

# Load the trained model
model = joblib.load('models/improved_rf_model.joblib')

def analyze_verilog_file(content):
    """Analyze Verilog file content and extract metrics"""
    
    # Count fanin and fanout
    input_count = len(re.findall(r'input\s+(?:wire|reg)?\s*\[?.*?\]?\s*\w+', content))
    output_count = len(re.findall(r'output\s+(?:wire|reg)?\s*\[?.*?\]?\s*\w+', content))
    
    # Count always blocks and assignments
    always_blocks = len(re.findall(r'always\s*@', content))
    assignments = len(re.findall(r'<=|=(?!=)', content))
    
    # Estimate path length based on nested operations
    nested_ops = len(re.findall(r'[{(\[].+[})\]]', content))
    
    # Check for potential timing violations
    potential_violations = 0
    if 'posedge' in content and 'negedge' in content:
        potential_violations += 1
    if re.search(r'@\s*\(\*\)', content):  # Combinational always blocks
        potential_violations += 1
        
    metrics = {
        'setup_slack': 10.0 - (nested_ops * 0.5),  # Estimate setup slack based on complexity
        'hold_slack': 1.0 - (potential_violations * 0.2),
        'timing_violation': 1 if potential_violations > 0 else 0,
        'fanin_count': input_count,
        'fanout_count': output_count,
        'path_length': nested_ops + always_blocks
    }
    
    return metrics

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
            
        if not file.filename.endswith('.v'):
            return jsonify({'error': 'Please upload a Verilog (.v) file'})
            
        # Read file content
        content = file.read().decode('utf-8')
        
        # Analyze RTL
        metrics = analyze_verilog_file(content)
        
        # Prepare input data for model
        input_data = pd.DataFrame([metrics])
        
        # Calculate derived features
        input_data['slack_ratio'] = input_data['setup_slack'] / (input_data['hold_slack'].abs() + 1e-6)
        input_data['complexity_score'] = input_data['fanin_count'] * input_data['fanout_count']
        input_data['path_density'] = input_data['path_length'] / (input_data['fanout_count'] + 1)
        input_data['timing_score'] = input_data['setup_slack'] * (1 - input_data['timing_violation'])
        
        # Make prediction
        complexity = model.predict(input_data)[0]
        
        # Generate recommendations
        recommendations = []
        if complexity > 7.0:
            recommendations = [
                "Consider pipelining the logic",
                "Add registers to break long combinational paths",
                "Review clock constraints and timing requirements"
            ]
        elif complexity > 5.0:
            recommendations = [
                "Optimize complex logic",
                "Review operation implementations",
                "Consider adding pipeline registers"
            ]
        else:
            recommendations = [
                "Circuit complexity is manageable",
                "Monitor timing margins during implementation"
            ]
        
        return jsonify({
            'complexity': float(complexity),
            'metrics': metrics,
            'recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 