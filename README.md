# RTL Timing Violation Predictor

## AI algorithm to predict combinational complexity/depth of signals to quickly identify timing violations in RTL designs using synthetic datasets.

---

## 📌 Installation Guide

### 1. System Requirements
- Python 3.8 or higher
- Git
- Any Python IDE (PyCharm, VS Code, etc.)
- 4GB RAM minimum
- 2GB free disk space

### 2. Install Virtual Environment
```bash
python -m venv venv
```


### 3. Activate the Virtual Environment
#### For Windows
```bash
venv\Scripts\activate
```

#### For MacOS/Linux
```bash             
source venv/bin/activate
```


### 4. Install Required Packages
```bash
pip install -r requirements.txt
```


### 5. Set the PYTHONPATH environment variable
#### On Windows
```sh
set PYTHONPATH=%cd%\src```

#### On macOS/Linux
```sh
export PYTHONPATH=$(pwd)/src```


### 6. Run the project
```sh
python -m src.predictor_interface```


---

## 📌 Dataset Generation

### Generate Synthetic Dataset
```bash
python compare_ml_agents_detailed.py
```


---

## 📌 Overview
An AI-based approach to predict timing violations in RTL designs by analyzing combinational complexity and logic depth of signals before synthesis, reducing design iteration time and improving RTL code quality.

## 📌 Problem Statement
Timing analysis in complex IP/SoC design traditionally requires synthesis, which is time-consuming. This project predicts combinational logic depth from RTL code to identify potential timing violations during the RTL design phase.

---

## 📌 Implemented Approaches

### 1️⃣ Machine Learning Models

#### 🔹 Primary Model: Random Forest Regressor
- *Implementation*: sklearn.ensemble.RandomForestRegressor
- *Configuration:*
  python
  params = {
      'n_estimators': 200,
      'max_depth': 15,
      'min_samples_split': 5,
      'min_samples_leaf': 2
  }
  
- *Advantages:*
  - Handles non-linear relationships in timing paths
  - Provides feature importance analysis
  - Robust to overfitting

### 2️⃣ Feature Engineering

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

---

### 3️⃣ Evaluation Metrics

#### Implemented Metrics

1. *Accuracy Metrics*
   - R² Score
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)

2. *Classification Metrics*
   - Binary accuracy (threshold-based)
   - Confusion matrix
   - Precision and recall

---

### 4️⃣ Visualization Methods

#### *Confusion Matrix*
python
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Sample confusion matrix
y_true = [1, 0, 1, 1, 0, 0, 1]
y_pred = [1, 0, 1, 0, 0, 1, 1]
cm = confusion_matrix(y_true, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


#### *Feature Importance*
python
import matplotlib.pyplot as plt
import numpy as np

# Sample feature importance
features = ["Fan-in", "Fan-out", "Logic Depth", "Operation Complexity", "Path Length"]
importances = np.array([0.3, 0.2, 0.25, 0.15, 0.1])

plt.barh(features, importances, color='skyblue')
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance Ranking")
plt.show()


---

## 📌 Results Analysis

The model provides:
- Timing violation predictions
- Feature importance ranking
- Performance metrics
- Confusion matrix analysis

---

## 📌 Implementation Details

### 🔹 Core Components
#### *Model Training*
python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#### *Prediction Pipeline*
python
from sklearn.ensemble import RandomForestRegressor

# Train model
model = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)


---

### 🔹 Key Files
- timing_rf_model.py: Main model implementation
- compare_ml_agents_detailed.py: Model evaluation
- model_metrics.py: Performance metrics

---

## 📌 Summary
This implementation focuses on practical RTL timing violation prediction using Random Forest as the primary model, with comprehensive feature engineering and evaluation metrics.
