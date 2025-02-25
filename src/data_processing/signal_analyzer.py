from typing import Dict, List
import numpy as np
import yaml

class SignalAnalyzer:
    def __init__(self, config_path: str = "configs/data_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.timing_constraints = self.config['dataset']['synthetic']['timing_constraints']
    
    def analyze_timing(self, signal_info: Dict) -> Dict:
        """Analyze timing characteristics of a signal"""
        clock_period = self.timing_constraints['clock_period']
        setup_margin = self.timing_constraints['setup_margin']
        
        # Calculate delays and slack
        path_delay = self._calculate_path_delay(signal_info)
        setup_slack = clock_period - path_delay - setup_margin
        
        return {
            'path_delay': path_delay,
            'setup_slack': setup_slack,
            'has_violation': setup_slack < 0,
            'violation_margin': abs(setup_slack) if setup_slack < 0 else 0
        }
    
    def _calculate_path_delay(self, signal_info: Dict) -> float:
        """Calculate total path delay based on logic levels and complexity"""
        base_delay = 0.1  # Base gate delay in ns
        complexity_factor = signal_info.get('operation_complexity', 1.0)
        levels = signal_info.get('logic_levels', 1)
        
        return base_delay * complexity_factor * levels
