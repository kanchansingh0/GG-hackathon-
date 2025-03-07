from flask import Flask, request, jsonify, send_file
import numpy as np
import logging
import tkinter as tk
from src.prediction_interface import RTLPredictor, PredictionInterface

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize the predictor only (without GUI)
predictor = RTLPredictor()

@app.route('/')
def home():
    """Serve the main HTML page"""
    return send_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        # Extract features in the correct order
        features = np.array([
            float(data['setup_slack']),
            float(data['hold_slack']),
            float(data['timing_violation']),
            float(data['fanin_count']),
            float(data['fanout_count']),
            float(data['path_length'])
        ])
        
        # Validate input ranges
        if not (0 <= data['timing_violation'] <= 200):
            return jsonify({'error': 'Timing violation must be between 0 and 200 ps'}), 400
        if not (1 <= data['fanin_count'] <= 20):
            return jsonify({'error': 'Fan-in count must be between 1 and 20'}), 400
        if not (1 <= data['fanout_count'] <= 20):
            return jsonify({'error': 'Fan-out count must be between 1 and 20'}), 400
        if not (1 <= data['path_length'] <= 15):
            return jsonify({'error': 'Path length must be between 1 and 15'}), 400
        
        # Make prediction using the existing predictor
        prediction = predictor.predict_complexity(features)
        
        if prediction is None:
            return jsonify({'error': 'Failed to make prediction'}), 400
            
        # Get circuit status and recommendations using existing methods
        status, recommendations, status_type = predictor.get_circuit_status(features, prediction)
        
        return jsonify({
            'prediction': float(prediction),
            'status': status,
            'recommendations': recommendations,
            'status_type': status_type
        })
        
    except KeyError as e:
        return jsonify({'error': f'Missing required field: {str(e)}'}), 400
    except ValueError as e:
        return jsonify({'error': f'Invalid input value: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    # Create a hidden root window for tkinter (required for backend processing)
    root = tk.Tk()
    root.withdraw()  # Hide the window
    
    # Start the Flask app
    app.run(debug=True, port=5000) 