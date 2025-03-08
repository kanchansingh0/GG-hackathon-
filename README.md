# RTL Timing Violation Predictor

AI algorithm to predict combinational complexity/depth of signals to quickly identify timing violations in RTL designs using synthetic datasets.

## To install this project on your local IDE

### 1. System Requirements
- Python 3.8 or higher
- Git
- Any Python IDE (PyCharm, VS Code, etc.)
- 4GB RAM minimum
- 2GB free disk space

### 2. IDE Setup

#### For VS Code:
1. Install VS Code Extensions:
   - Python extension
   - Pylance
   - Jupyter
   - Python Test Explorer

#### For PyCharm:
1. Enable Python Scientific Mode
2. Install Python Scientific packages

### 3. Project Setup

1. *Clone the Repository*
bash
git clone <repository-url>
cd GG-hackathon-


2. *Create Virtual Environment*

For Windows:
bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate


For Linux/Mac:
bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate


3. *Install Required Packages*
bash
pip install -r requirements.txt


4. *Set the PYTHONPATH environment variable:*

    - On Windows:
    sh
    set PYTHONPATH=%cd%\src
    

    - On macOS/Linux:
    sh
    export PYTHONPATH=$(pwd)/src
    

5. *Run the project:*
sh
python -m src.predictor_interface


### 4. Required Packages

txt
# Core ML packages
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=0.24.2

# Visualization
matplotlib>=3.4.2
seaborn>=0.11.1

# ML framework
xgboost>=1.4.2

# Development tools
jupyter>=1.0.0
ipython>=7.24.1


### 5. Dataset Generation

The synthetic dataset uses these features:
- fanin_count: Input gate count
- fanout_count: Output count
- logic_depth: Gate levels
- operation_complexity: Operation complexity score
- path_length: Signal path length

To generate dataset:
bash
python compare_ml_agents_detailed.py


# RTL Timing Violation Predictor

## Overview
An AI-based approach to predict timing violations in RTL designs by analyzing combinational complexity and logic depth of signals before synthesis, reducing design iteration time and improving RTL code quality.

## Problem Statement
Timing analysis in complex IP/SoC design traditionally requires synthesis, which is time-consuming. This project predicts combinational logic depth from RTL code to identify potential timing violations during the RTL design phase.

## Implemented Approaches

### 1. Machine Learning Models

#### Primary Model: Random Forest Regressor
- Implementation: sklearn.ensemble.RandomForestRegressor
- Configuration:
  python
  params = {
      'n_estimators': 200,
      'max_depth': 15,
      'min_samples_split': 5,
      'min_samples_leaf': 2
  }
  
- Advantages:
  - Handles non-linear relationships in timing paths
  - Provides feature importance analysis
  - Robust to overfitting

### 2. Feature Engineering

#### Implemented Features
1. *Fan-in Count* (1-10 gates)
   - Measures input gate connections
   - Impacts combinational depth

2. *Fan-out Count* (1-8 outputs)
   - Measures output connections
   - Affects signal propagation

3. *Logic Depth* (1-15 levels)
   - Represents gate levels in path
   - Key timing violation indicator

4. *Operation Complexity* (0-1 normalized)
   - Complexity score of operations
   - Weighted impact on timing

5. *Path Length* (0-1 normalized)
   - Physical path characteristics
   - Routing complexity measure

### 3. Evaluation Metrics

#### Implemented Metrics
1. *Accuracy Metrics*
   - R² Score
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)

2. *Classification Metrics*
   - Binary accuracy (threshold-based)
   - Confusion matrix
   - Precision and recall

### 4. Visualization Methods

1. *Confusion Matrix*
   - Uses seaborn heatmap
   - Shows true/false positives/negatives
   - Helps identify prediction patterns

2. *Feature Importance*
   - Bar plots of feature weights
   - Identifies key timing factors
   - Guides design optimization

## Results Analysis

The model provides:
- Timing violation predictions
- Feature importance ranking
- Performance metrics
- Confusion matrix analysis

## Implementation Details

### Core Components
1. *Model Training*
   - Train-test split: 80-20
   - Cross-validation: 5-fold
   - Normalized features

2. *Prediction Pipeline*
   - Feature extraction
   - Model prediction
   - Threshold-based classification
   - Performance evaluation

### Key Files
- timing_rf_model.py: Main model implementation
- compare_ml_agents_detailed.py: Model evaluation
- model_metrics.py: Performance metrics

This implementation focuses on practical RTL timing violation prediction using Random Forest as the primary model, with comprehensive feature engineering and evaluation metrics.