from typing import Dict, List
import pandas as pd
import numpy as np

class RTLFeatureExtractor:
    def __init__(self):
        self.features = [
            'fanin_count',
            'fanout_count',
            'logic_depth',
            'operation_complexity',
            'path_length'
        ]
        
    def extract_features(self, rtl_info: Dict) -> Dict:
        """Extract features from RTL information"""
        features = {
            'fanin_count': len(rtl_info.get('fanin', [])),
            'fanout_count': len(rtl_info.get('fanout', [])),
            'logic_depth': rtl_info.get('logic_depth', 0),
            'operation_complexity': self._calculate_operation_complexity(rtl_info),
            'path_length': self._calculate_path_length(rtl_info)
        }
        return features
    
    def _calculate_operation_complexity(self, rtl_info: Dict) -> float:
        """Calculate operation complexity score"""
        ops = rtl_info.get('operations', [])
        weights = {
            'and': 1.0,
            'or': 1.0,
            'xor': 1.2,
            'add': 2.0,
            'mult': 4.0
        }
        return sum(weights.get(op, 1.0) for op in ops)
    
    def _calculate_path_length(self, rtl_info: Dict) -> int:
        """Calculate path length"""
        return rtl_info.get('path_length', len(rtl_info.get('fanin', [])))
    
    def process_features(self, features_list: List[Dict]) -> pd.DataFrame:
        """Process list of features into DataFrame"""
        df = pd.DataFrame(features_list)
        
        # Normalize features
        for feature in self.features:
            if feature in df.columns:
                df[feature] = (df[feature] - df[feature].mean()) / df[feature].std()
                
        return df
