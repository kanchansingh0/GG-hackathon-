import pytest
import pandas as pd
from src.data_collection.synthetic_data_manager import SyntheticDataManager

def test_synthetic_data_generation():
    manager = SyntheticDataManager()
    data = manager.generate_synthetic_data(n_samples=100)
    
    assert isinstance(data, pd.DataFrame)
    assert len(data) == 100
    assert all(col in data.columns for col in [
        'setup_slack', 'hold_slack', 'timing_violation',
        'fanin_count', 'fanout_count', 'operation_complexity',
        'path_length', 'logic_levels'
    ])
