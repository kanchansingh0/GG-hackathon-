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
from src.data_generation.signal_generator import SignalGenerator
from src.utils.logger import Logger

def main():
    # Initialize logger
    logger = Logger("timing_predictor")
    
    # Setup directories
    raw_data_dir = Path("data/raw/synthetic")
    processed_data_dir = Path("data/processed/synthetic")
    
    # Ensure directories exist
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic dataset
    logger.info("Generating synthetic dataset...")
    data_manager = SyntheticDataManager()
    signal_generator = SignalGenerator()
    dataset = data_manager.generate_dataset(
        num_modules=100,
        complexity_range=(5, 20)
    )
    
    # Create and process dataset
    creator = DatasetCreator(str(raw_data_dir), str(processed_data_dir))
    logger.info("Processing RTL files...")
    features_df = creator.create_dataset()
    
    if len(features_df) > 0:
        logger.info(f"Generated dataset with {len(features_df)} samples")
        
        # Train model
        logger.info("Training model...")
        model = TimingRFModel()
        trainer = ModelTrainer(model)
        metrics = trainer.train_and_evaluate(features_df)
        
        logger.info("Model Performance:")
        logger.info(f"Training metrics: {metrics['train_metrics']}")
        logger.info(f"Test metrics: {metrics['test_metrics']}")
    else:
        logger.error("No RTL files found to process")

if __name__ == "__main__":
    main()