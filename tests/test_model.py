import pytest
import pandas as pd
import numpy as np
from src.models.rf_model import RFModel

def test_model_training():
    # Create sample data
    X = np.random.rand(100, 6)
    y = np.random.rand(100)
    
    model = RFModel()
    model.train(X, y)
    
    # Test predictions
    predictions = model.predict(X)
    assert len(predictions) == len(y)
