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

1. **Clone the Repository**
```bash
git clone <repository-url>
cd RTL-Timing-Predictor
```

2. **Create Virtual Environment**

For Windows:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate
```

For Linux/Mac:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate
```

3. **Install Required Packages**
```bash
pip install -r requirements.txt
```

### 4. Required Packages

```txt
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
```

# RTL Timing Violation Predictor

AI algorithm to predict combinational complexity/depth of signals to quickly identify timing violations in RTL designs using synthetic datasets.

## Problem Overview

Timing analysis is crucial in complex IP/SoC design, but timing reports are only generated after synthesis, which is time-consuming. This project creates an AI algorithm to predict combinational logic depth of signals in behavioral RTL to identify potential timing violations early in the design process.

## Key Concepts

- **Combinational Complexity/Logic-depth**: Number of basic gates (AND/OR/NOT/NAND etc.) required to generate a signal following the longest path
- **Timing Violation**: Indicates when combinational logic depth exceeds what's supported at a given frequency

## Features

- Synthetic RTL dataset generation
- Feature extraction from RTL
- Timing violation prediction
- Model training and evaluation

## Project Structure