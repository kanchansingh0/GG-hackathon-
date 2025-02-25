from pathlib import Path
import pandas as pd
from typing import Dict, List
from src.data_processing.feature_extractor import RTLFeatureExtractor

class DatasetCreator:
    def __init__(self, raw_dir: str, processed_dir: str):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.feature_extractor = RTLFeatureExtractor()
        
    def create_dataset(self) -> pd.DataFrame:
        """Create dataset from RTL files"""
        features_list = []
        
        # Process all RTL files
        for rtl_file in self.raw_dir.glob('**/*.v'):
            module_info = self._process_rtl_file(rtl_file)
            features = self.feature_extractor.extract_features(module_info)
            features['file'] = rtl_file.name
            features_list.append(features)
            
        # Create DataFrame
        if features_list:
            df = self.feature_extractor.process_features(features_list)
            
            # Save processed dataset
            output_path = self.processed_dir / 'features.csv'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            
            return df
            
        return pd.DataFrame()
    
    def _process_rtl_file(self, rtl_file: Path) -> Dict:
        """Process single RTL file"""
        # Read RTL content
        with open(rtl_file, 'r') as f:
            rtl_content = f.read()
        
        # Read timing report if exists
        timing_file = self.raw_dir / 'timing_reports' / f"{rtl_file.stem}_timing.json"
        timing_data = {}
        if timing_file.exists():
            import json
            with open(timing_file, 'r') as f:
                timing_data = json.load(f)
        
        # Extract basic RTL info
        module_info = {
            'file': str(rtl_file),
            'timing_data': timing_data,
            'fanin': [],
            'fanout': [],
            'operations': [],
            'logic_depth': timing_data.get('critical_path_depth', 0),
            'path_length': timing_data.get('path_length', 0)
        }
        
        # Parse RTL content for additional info
        # Add RTL parsing logic here
        
        return module_info