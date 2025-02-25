from pathlib import Path

# Create directories
Path("src/data_collection").mkdir(parents=True, exist_ok=True)
Path("src/data_processing").mkdir(parents=True, exist_ok=True)

# Create __init__.py files
Path("src/__init__.py").touch(exist_ok=True)
Path("src/data_collection/__init__.py").touch(exist_ok=True)
Path("src/data_processing/__init__.py").touch(exist_ok=True)

print("Directory structure created successfully!") 