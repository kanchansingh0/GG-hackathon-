from typing import Dict, List
import numpy as np
from pathlib import Path

class RTLFeatureExtractor:
    def __init__(self):
        # Replace YAML config with Python dictionary
        self.config = {
            'dataset': {
                'synthetic': {
                    'timing_constraints': {
                        'clock_period': 10,
                        'hold_margin': 0.1
                    }
                }
            }
        }
            
    def extract_timing_features(self, signal_info: Dict) -> Dict:
        """Extract timing-specific features from signal"""
        clock_period = self.config['dataset']['synthetic']['timing_constraints']['clock_period']
        
        features = {
            'setup_slack': clock_period - signal_info['delay'],
            'hold_slack': signal_info['delay'] - self.config['dataset']['synthetic']['timing_constraints']['hold_margin'],
            'timing_violation': int(signal_info['delay'] > clock_period)
        }
        return features
    
    def extract_rtl_features(self, signal_info: Dict) -> Dict:
        """Extract RTL-specific features"""
        features = {
            'fanin_count': len(signal_info['fanin']),
            'fanout_count': len(signal_info.get('fanout', [])),
            'logic_levels': self._calculate_logic_levels(signal_info),
            'operation_complexity': self._estimate_operation_complexity(signal_info),
            'path_length': signal_info.get('path_length', 0)
        }
        return features
    
    def _calculate_logic_levels(self, signal_info: Dict) -> int:
        """Calculate number of logic levels in path"""
        if not signal_info['fanin']:
            return 0
        return max(signal_info.get('logic_levels', 0), 1)
    
    def _estimate_operation_complexity(self, signal_info: Dict) -> float:
        """Estimate computational complexity of operations"""
        ops = signal_info.get('operations', [])
        complexity_weights = {
            'and': 1.0,
            'or': 1.0,
            'xor': 1.2,
            'add': 2.0,
            'mult': 4.0
        }
        return sum(complexity_weights.get(op, 1.0) for op in ops)

    def extract_features(self, signal_info: Dict) -> Dict:
        """Extract features from signal information"""
        features = {
            'logic_depth': signal_info['logic_depth'],
            'fanin_count': len(signal_info['fanin']),
            'is_sequential': signal_info['type'] == 'reg',
            'width': self._parse_width(signal_info['width'])
        }
        
        # Add operation complexity
        features.update(self._analyze_operations(signal_info))
        
        return features
    
    def _parse_width(self, width_str: str) -> int:
        """Parse signal width from string format"""
        if not width_str or width_str == '[0:0]':
            return 1
        # Parse [x:y] format
        try:
            high, low = map(int, width_str.strip('[]').split(':'))
            return abs(high - low) + 1
        except:
            return 1
    
    def _analyze_operations(self, signal_info: Dict) -> Dict:
        """Analyze operation complexity"""
        features = {
            'has_arithmetic': 0,
            'has_logical': 0,
            'operation_count': 0
        }
        
        if 'operations' in signal_info:
            ops = signal_info['operations']
            features['has_arithmetic'] = int(any(op in ops for op in ['+', '-', '*', '/']))
            features['has_logical'] = int(any(op in ops for op in ['&', '|', '^']))
            features['operation_count'] = len(ops)
            
        return features
