#!/usr/bin/env python3

import torch
import logging
from training.self_play_trainer import SelfPlayTrainer, TrainingConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_training():
    """Test the training setup with minimal configuration"""
    try:
        # Create a minimal config for testing
        config = TrainingConfig(
            total_hands=100,  # Very small for testing
            max_epochs=2,     # Just 2 epochs
            parallel_games=2, # Minimal parallel games
            batch_size=32,    # Smaller batch size
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        logger.info(f"Testing with config: hands={config.total_hands}, epochs={config.max_epochs}")
        logger.info(f"Device: {config.device}")
        
        # Try to create the trainer
        logger.info("Creating SelfPlayTrainer...")
        trainer = SelfPlayTrainer(config)
        logger.info("SelfPlayTrainer created successfully!")
        
        # Try to start training
        logger.info("Starting training...")
        results = trainer.train()
        logger.info(f"Training completed! Results: {results}")
        
        return True
        
    except Exception as e:
        logger.error(f"Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training()
    if success:
        print("✅ Training test PASSED")
    else:
        print("❌ Training test FAILED")