from pathlib import Path
import json
from typing import Dict, List
from feature_extractor import RTLFeatureExtractor
import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self):
        self.feature_extractor = RTLFeatureExtractor()
        self.data_dir = Path("data/raw/synthetic")
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def process_all_modules(self) -> List[Dict]:
        """Process all synthetic modules and extract features"""
        processed_data = []
        
        # Process RTL modules
        rtl_modules = self.data_dir / "rtl_modules"
        timing_reports = self.data_dir / "timing_reports"
        
        try:
            for rtl_file in rtl_modules.glob("*.v"):
                module_name = rtl_file.stem
                timing_file = timing_reports / f"{module_name}_timing.json"
                
                if timing_file.exists():
                    # Load timing data
                    with open(timing_file) as f:
                        timing_data = json.load(f)
                    
                    # Extract features
                    features = self._extract_module_features(rtl_file, timing_data)
                    
                    # Ensure numeric values
                    features = self._sanitize_features(features)
                    processed_data.append(features)
                    
            return processed_data
            
        except Exception as e:
            print(f"Error processing modules: {str(e)}")
            return []
    
    def _sanitize_features(self, features: Dict) -> Dict:
        """Ensure all feature values are numeric"""
        sanitized = {}
        for key, value in features.items():
            try:
                if isinstance(value, (str, bool)):
                    # Convert boolean to int
                    if isinstance(value, bool):
                        sanitized[key] = int(value)
                    # Try to convert string to float if it looks like a number
                    elif value.replace('.', '').isdigit():
                        sanitized[key] = float(value)
                    else:
                        # Skip non-numeric strings
                        continue
                else:
                    # Keep numeric values as is
                    sanitized[key] = value
            except ValueError:
                # Skip values that can't be converted
                continue
                
        return sanitized
    
    def _extract_module_features(self, rtl_file: Path, timing_data: Dict) -> Dict:
        """Extract features for a single module"""
        # Read RTL content
        with open(rtl_file) as f:
            rtl_content = f.read()
            
        # Prepare signal info dictionary
        signal_info = {
            'module_name': rtl_file.stem,
            'rtl_content': rtl_content,
            'fanin': timing_data.get('fanin', []),
            'fanout': timing_data.get('fanout', []),
            'delay': float(timing_data.get('delay', 0)),
            'logic_depth': int(timing_data.get('critical_path_depth', 0)),
            'operations': self._extract_operations(rtl_content),
            'type': 'comb' if 'always' not in rtl_content else 'reg',
            'width': self._extract_signal_width(rtl_content)
        }
        
        # Extract all features
        features = {}
        features.update(self.feature_extractor.extract_timing_features(signal_info))
        features.update(self.feature_extractor.extract_rtl_features(signal_info))
        
        return features
    
    def _extract_operations(self, rtl_content: str) -> List[str]:
        """Extract operations from RTL content"""
        operations = []
        if '+' in rtl_content: operations.append('add')
        if '*' in rtl_content: operations.append('mult')
        if '&' in rtl_content: operations.append('and')
        if '|' in rtl_content: operations.append('or')
        if '^' in rtl_content: operations.append('xor')
        return operations
    
    def _extract_signal_width(self, rtl_content: str) -> str:
        """Extract signal width from RTL content"""
        import re
        width_match = re.search(r'\[(\d+):(\d+)\]', rtl_content)
        if width_match:
            return f"[{width_match.group(1)}:{width_match.group(2)}]"
        return '[0:0]'
    
    def save_processed_data(self, processed_data: List[Dict]):
        """Save processed features to file"""
        if not processed_data:
            print("No data to save!")
            return
            
        try:
            # Convert to DataFrame
            df = pd.DataFrame(processed_data)
            
            # Ensure all columns are numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop any columns that couldn't be converted to numeric
            df = df.select_dtypes(include=[np.number])
            
            # Save to CSV
            output_file = self.processed_dir / "processed_features.csv"
            df.to_csv(output_file, index=False)
            print(f"\nProcessed data saved to: {output_file}")
            print(f"Number of features: {len(df.columns)}")
            print(f"Number of samples: {len(df)}")
            
        except Exception as e:
            print(f"Error saving data: {str(e)}")

def main():
    processor = DataProcessor()
    processed_data = processor.process_all_modules()
    processor.save_processed_data(processed_data)

if __name__ == "__main__":
    main()
