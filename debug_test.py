import logging
logging.basicConfig(level=logging.INFO)

from tests.test_model_training import test_model_training_pipeline

def run_debug():
    try:
        print("\nStarting model training debug...")
        test_model_training_pipeline()
        print("\nDebug completed successfully!")
    except Exception as e:
        print(f"\nDebug failed with error: {str(e)}")
        print("\nPlease check the logs above for detailed information.")

if __name__ == "__main__":
    run_debug() 