# main.py
import argparse
from src.data_collection import rtl_extractor, synthesis_runner
from src.data_processing import feature_extractor, dataset_creator
from src.models import model_selection, model_trainer
from src.evaluation import evaluator, visualize_results
from src.data_processing.dataset_creator import DatasetCreator
from pathlib import Path
from src.models.model_trainer import ModelTrainer
from data.dataset import RTLDataset
from data.synthesis_reports import SynthesisReports
from src.data_processing.feature_extractor import RTLFeatureExtractor
from src.data_processing.signal_analyzer import SignalAnalyzer
import yaml
from src.data_collection.synthetic_data_manager import SyntheticDataManager
from src.models.specific_models.timing_rf_model import TimingRFModel
from src.data_generation.signal_generator import SyntheticRTLGenerator
from src.utils.logger import Logger
import pandas as pd
from src.utils.visualize import plot_confusion_matrix, plot_feature_importance
import sys
sys.path.append(str(Path(__file__).parent))
import numpy as np

def main():
    try:
        # Generate sample data for testing
        n_samples = 1000
        print(f"Generating {n_samples} synthetic samples...")
        
        # Create synthetic features
        data = {
            'fanin_count': np.random.randint(1, 10, n_samples),
            'fanout_count': np.random.randint(1, 8, n_samples),
            'logic_depth': np.random.randint(1, 15, n_samples),
            'operation_complexity': np.random.uniform(0, 1, n_samples),
            'path_length': np.random.uniform(0, 1, n_samples)
        }
        
        # Create target variable
        data['target'] = (0.2 * data['fanin_count'] + 
                         0.1 * data['fanout_count'] + 
                         0.3 * data['logic_depth'] + 
                         0.2 * data['operation_complexity'] + 
                         0.2 * data['path_length'] + 
                         np.random.normal(0, 0.1, n_samples))
        
        df = pd.DataFrame(data)
        
        # Normalize features
        for col in df.columns:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        
        # Initialize and train model
        model = TimingRFModel()
        trainer = ModelTrainer(
            model=model,
            features=model.feature_columns,
            target='target',
            test_size=0.2
        )
        
        # Train and evaluate
        metrics = trainer.train_and_evaluate(df)
        
        # Print results
        print("\nModel Performance:")
        print(f"Training R²: {metrics['train_metrics']['r2']:.4f}")
        print(f"Test R²: {metrics['test_metrics']['r2']:.4f}")
        print(f"Accuracy: {metrics['test_metrics']['accuracy']:.4f}")
        
        # Save results
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Save metrics to CSV
        pd.DataFrame([metrics['train_metrics'], metrics['test_metrics']],
                    index=['train', 'test']).to_csv(
            results_dir / 'metrics.csv'
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        print("\nTraceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()