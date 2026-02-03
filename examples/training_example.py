"""
Example usage of the Fashion JSON Encoder training system.

This script demonstrates how to use the training pipeline for both
standalone JSON encoder training and full contrastive learning.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
from data.fashion_dataset import FashionDataModule
from training.trainer import create_trainer_from_data_module
from utils.config import TrainingConfig


def main():
    """Demonstrate training system usage."""
    print("Fashion JSON Encoder Training Example")
    print("=" * 50)
    
    # Note: This example assumes you have a K-Fashion dataset
    dataset_path = "/path/to/kfashion/dataset"  # Update this path
    
    print(f"Dataset path: {dataset_path}")
    print("Note: Update the dataset_path variable to point to your K-Fashion dataset")
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Configuration
    config = TrainingConfig(
        batch_size=32,  # Smaller batch size for example
        learning_rate=1e-4,
        max_epochs=10,  # Fewer epochs for example
        embedding_dim=128,
        hidden_dim=256,
        target_categories=['상의', '하의', '아우터']
    )
    
    print(f"Training configuration:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Max epochs: {config.max_epochs}")
    print(f"  Embedding dim: {config.embedding_dim}")
    print(f"  Hidden dim: {config.hidden_dim}")
    
    # Setup data module
    print("\n1. Setting up data module...")
    data_module = FashionDataModule(
        dataset_path=dataset_path,
        target_categories=config.target_categories,
        batch_size=config.batch_size,
        num_workers=2,  # Fewer workers for example
        train_split=0.8,
        image_size=224,
        augment_prob=0.5
    )
    
    try:
        data_module.setup()
        print("✓ Data module setup completed")
        print(f"  Train samples: {len(data_module.train_dataset)}")
        print(f"  Val samples: {len(data_module.val_dataset)}")
        print(f"  Vocabulary sizes: {data_module.get_vocab_sizes()}")
    except FileNotFoundError:
        print("✗ Dataset not found. Please update the dataset_path variable.")
        print("  This example requires a real K-Fashion dataset to run.")
        return
    
    # Create trainer
    print("\n2. Creating trainer...")
    trainer = create_trainer_from_data_module(
        data_module=data_module,
        config=config,
        device=device,
        checkpoint_dir='example_checkpoints',
        log_dir='example_logs'
    )
    print("✓ Trainer created successfully")
    
    # Get data loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Phase 1: Standalone JSON encoder training
    print("\n3. Running standalone JSON encoder training...")
    print("This phase verifies that the JSON encoder can learn from metadata alone.")
    
    try:
        standalone_results = trainer.train_json_encoder_standalone(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=3  # Just a few epochs for example
        )
        
        print("✓ Standalone training completed")
        print(f"  Final validation loss: {standalone_results['val_losses'][-1]:.4f}")
        
        # Analyze results
        final_analysis = standalone_results['final_analysis']
        print(f"  Output analysis:")
        print(f"    Embedding dimension: {final_analysis['embedding_dim']}")
        print(f"    Is normalized: {final_analysis['is_normalized']}")
        print(f"    Mean norm: {final_analysis['norm_mean']:.4f}")
        
    except Exception as e:
        print(f"✗ Standalone training failed: {e}")
        return
    
    # Phase 2: Contrastive learning training
    print("\n4. Running contrastive learning training...")
    print("This phase aligns JSON and image embeddings using contrastive learning.")
    
    try:
        contrastive_results = trainer.train_contrastive_learning(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=5  # Just a few epochs for example
        )
        
        print("✓ Contrastive training completed")
        print(f"  Best validation loss: {contrastive_results['best_val_loss']:.4f}")
        
        # Analyze results
        final_metrics = contrastive_results['final_metrics']
        print(f"  Final metrics:")
        print(f"    Top-1 accuracy: {final_metrics['top1_accuracy']:.4f}")
        print(f"    Top-5 accuracy: {final_metrics['top5_accuracy']:.4f}")
        print(f"    Mean reciprocal rank: {final_metrics['mean_reciprocal_rank']:.4f}")
        print(f"    Avg positive similarity: {final_metrics['avg_positive_similarity']:.4f}")
        
    except Exception as e:
        print(f"✗ Contrastive training failed: {e}")
        return
    
    # Cleanup
    trainer.close()
    
    print("\n" + "=" * 50)
    print("TRAINING EXAMPLE COMPLETED SUCCESSFULLY")
    print("=" * 50)
    print("Checkpoints saved in: example_checkpoints/")
    print("Logs saved in: example_logs/")
    print("Use tensorboard to visualize training:")
    print("  tensorboard --logdir example_logs")
    
    print("\nTo run full training with your dataset:")
    print("  python training/train.py --dataset_path /path/to/your/dataset")


if __name__ == "__main__":
    main()