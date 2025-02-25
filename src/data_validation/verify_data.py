# Create a new file: src/data_validation/verify_data.py
import json
from pathlib import Path

def verify_dataset():
    data_dir = Path("data/raw/synthetic")
    
    # Check RTL files
    rtl_files = list((data_dir / "rtl_modules").glob("*.v"))
    print(f"\nFound {len(rtl_files)} RTL modules:")
    for rtl_file in rtl_files[:3]:  # Show first 3 as example
        print(f"- {rtl_file.name}")
    
    # Check timing reports
    timing_files = list((data_dir / "timing_reports").glob("*.json"))
    print(f"\nFound {len(timing_files)} timing reports:")
    
    # Read and display sample timing data
    if timing_files:
        with open(timing_files[0]) as f:
            sample_timing = json.load(f)
        print("\nSample timing data:")
        print(json.dumps(sample_timing, indent=2))

if __name__ == "__main__":
    verify_dataset()