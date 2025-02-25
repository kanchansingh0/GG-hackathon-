import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
from src.models.specific_models.timing_rf_model import TimingRFModel
from src.models.model_trainer import ModelTrainer
from src.utils.logger import Logger

def debug_model_training():
    # Initialize logger
    logger = Logger("debug_model")
    
    try:
        # Create synthetic test data
        logger.info("Creating synthetic test data...")
        n_samples = 100
        n_features = 5
        X = np.random.rand(n_samples, n_features)
        y = np.random.rand(n_samples)
        
        # Create DataFrame
        feature_names = ['fanin_count', 'fanout_count', 'logic_depth', 
                        'operation_complexity', 'path_length']
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        logger.info(f"Created dataset with shape: {df.shape}")
        
        # Initialize model and trainer
        logger.info("Initializing model and trainer...")
        model = TimingRFModel()
        trainer = ModelTrainer(
            model=model,
            features=feature_names,
            target='target',
            test_size=0.2
        )
        
        # Check initial status
        initial_status = trainer.check_model_status()
        logger.info(f"Initial model status: {initial_status}")
        
        # Train and evaluate
        logger.info("Training model...")
        metrics = trainer.train_and_evaluate(df)
        
        # Log results
        logger.info("\nTraining Results:")
        logger.info(f"Training metrics: {metrics['train_metrics']}")
        logger.info(f"Test metrics: {metrics['test_metrics']}")
        logger.info(f"CV scores mean: {metrics['cv_scores']['mean']:.4f}")
        
        # Check final status
        final_status = trainer.check_model_status()
        logger.info(f"Final model status: {final_status}")
        
        return True, metrics
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        return False, None

if __name__ == "__main__":
    success, metrics = debug_model_training()
    if success:
        print("\nDebug completed successfully!")
    else:
        print("\nDebug failed!") 