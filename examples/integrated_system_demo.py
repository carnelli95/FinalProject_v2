"""
Integrated System Demo

This script demonstrates how to use the integrated Fashion JSON Encoder system
for training, evaluation, and inference.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
from main import FashionEncoderSystem


def demo_training():
    """Demonstrate the training workflow."""
    print("=" * 60)
    print("DEMO: Training Workflow")
    print("=" * 60)
    
    # Note: This demo uses synthetic data since we don't have a real dataset path
    print("Initializing Fashion Encoder System...")
    
    # Initialize system with default configuration
    system = FashionEncoderSystem()
    
    print(f"System initialized on device: {system.device}")
    print(f"Configuration: {system.config}")
    
    # Run sanity check with synthetic data
    print("\nRunning sanity check with synthetic data...")
    try:
        results = system.sanity_check(dataset_path=None, num_epochs=2)
        
        # Check if sanity check passed
        validation = results.get('validation_results', {})
        all_passed = all(validation.get(check, False) for check in validation if check != 'errors')
        
        if all_passed:
            print("✓ Sanity check PASSED - System is working correctly!")
        else:
            print("⚠️  Sanity check had issues - Check the detailed results")
            
        # Show some results
        final_analysis = results.get('final_analysis', {})
        print(f"\nOutput Analysis:")
        print(f"  Embedding dimension: {final_analysis.get('embedding_dim', 'Unknown')}")
        print(f"  Is normalized: {final_analysis.get('is_normalized', 'Unknown')}")
        print(f"  Mean norm: {final_analysis.get('norm_mean', 0):.4f}")
        
    except Exception as e:
        print(f"Sanity check failed: {e}")
    
    # Cleanup
    system.cleanup()
    print("\nTraining demo completed.")


def demo_configuration():
    """Demonstrate configuration management."""
    print("=" * 60)
    print("DEMO: Configuration Management")
    print("=" * 60)
    
    # Create a custom configuration file
    config_path = "demo_config.json"
    
    print(f"Creating sample configuration file: {config_path}")
    from main import create_config_file
    create_config_file(config_path)
    
    # Load system with custom configuration
    print(f"Loading system with custom configuration...")
    system = FashionEncoderSystem(config_path=config_path)
    
    print(f"Loaded configuration:")
    print(f"  Batch size: {system.config.batch_size}")
    print(f"  Learning rate: {system.config.learning_rate}")
    print(f"  Embedding dim: {system.config.embedding_dim}")
    print(f"  Target categories: {system.config.target_categories}")
    
    # Cleanup
    system.cleanup()
    
    # Remove demo config file
    Path(config_path).unlink()
    print(f"\nConfiguration demo completed.")


def demo_model_components():
    """Demonstrate individual model components."""
    print("=" * 60)
    print("DEMO: Model Components")
    print("=" * 60)
    
    # Import model components
    from models.json_encoder import JSONEncoder
    from models.contrastive_learner import ContrastiveLearner
    from transformers import CLIPVisionModel
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create vocabulary sizes for demo
    vocab_sizes = {
        'category': 10,
        'style': 20,
        'silhouette': 15,
        'material': 25,
        'detail': 30
    }
    
    print(f"Vocabulary sizes: {vocab_sizes}")
    
    # Initialize JSON Encoder
    print("\n1. JSON Encoder Demo")
    json_encoder = JSONEncoder(
        vocab_sizes=vocab_sizes,
        embedding_dim=128,
        hidden_dim=256,
        output_dim=512
    ).to(device)
    
    print(f"JSON Encoder parameters: {sum(p.numel() for p in json_encoder.parameters()):,}")
    
    # Create sample batch
    batch_size = 4
    sample_batch = {
        'category': torch.randint(0, vocab_sizes['category'], (batch_size,)).to(device),
        'style': torch.randint(0, vocab_sizes['style'], (batch_size, 3)).to(device),
        'silhouette': torch.randint(0, vocab_sizes['silhouette'], (batch_size,)).to(device),
        'material': torch.randint(0, vocab_sizes['material'], (batch_size, 2)).to(device),
        'detail': torch.randint(0, vocab_sizes['detail'], (batch_size, 4)).to(device),
        'style_mask': torch.ones(batch_size, 3).to(device),
        'material_mask': torch.ones(batch_size, 2).to(device),
        'detail_mask': torch.ones(batch_size, 4).to(device)
    }
    
    # Forward pass
    with torch.no_grad():
        embeddings = json_encoder(sample_batch)
        print(f"Output shape: {embeddings.shape}")
        print(f"Output norm: {torch.norm(embeddings, dim=-1).mean().item():.4f}")
        print(f"Is normalized: {torch.allclose(torch.norm(embeddings, dim=-1), torch.ones(batch_size).to(device), atol=1e-3)}")
    
    # Initialize CLIP encoder
    print("\n2. CLIP Integration Demo")
    clip_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    print(f"CLIP Encoder parameters: {sum(p.numel() for p in clip_encoder.parameters()):,}")
    
    # Initialize Contrastive Learner
    print("\n3. Contrastive Learner Demo")
    contrastive_learner = ContrastiveLearner(
        json_encoder=json_encoder,
        clip_encoder=clip_encoder,
        temperature=0.07
    ).to(device)
    
    # Create sample images
    sample_images = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # Forward pass
    with torch.no_grad():
        embeddings_dict = contrastive_learner.get_embeddings(sample_images, sample_batch)
        
        print(f"Image embeddings shape: {embeddings_dict['image_embeddings'].shape}")
        print(f"JSON embeddings shape: {embeddings_dict['json_embeddings'].shape}")
        print(f"Similarity matrix shape: {embeddings_dict['similarity_matrix'].shape}")
        
        # Check similarity
        similarity_matrix = embeddings_dict['similarity_matrix']
        diagonal_similarities = torch.diag(similarity_matrix)
        print(f"Positive pair similarities: {diagonal_similarities.cpu().numpy()}")
        print(f"Mean positive similarity: {diagonal_similarities.mean().item():.4f}")
    
    print("\nModel components demo completed.")


def demo_data_processing():
    """Demonstrate data processing components."""
    print("=" * 60)
    print("DEMO: Data Processing")
    print("=" * 60)
    
    # Import data components
    from data.data_models import FashionItem, ProcessedBatch
    from data.processor import FashionDataProcessor
    
    print("1. Data Models Demo")
    
    # Create sample fashion item
    sample_item = FashionItem(
        image_path="sample.jpg",
        bbox=(10, 20, 100, 150),
        category="상의",
        style=["캐주얼", "스포티"],
        silhouette="오버핏",
        material=["면", "폴리에스터"],
        detail=["프린트", "포켓", "지퍼"]
    )
    
    print(f"Sample Fashion Item:")
    print(f"  Category: {sample_item.category}")
    print(f"  Style: {sample_item.style}")
    print(f"  Silhouette: {sample_item.silhouette}")
    print(f"  Material: {sample_item.material}")
    print(f"  Detail: {sample_item.detail}")
    
    print("\n2. Data Processor Demo")
    
    # Create processor (without real dataset)
    processor = FashionDataProcessor(
        dataset_path="dummy_path",
        target_categories=["상의", "하의", "아우터"]
    )
    
    # Demo polygon to bbox conversion
    sample_polygon = [(10, 20), (110, 20), (110, 170), (10, 170)]
    bbox = processor.polygon_to_bbox(sample_polygon)
    print(f"Polygon to BBox conversion:")
    print(f"  Polygon: {sample_polygon}")
    print(f"  BBox: {bbox}")
    
    # Demo vocabulary building (with sample data)
    sample_vocabularies = {
        'category': {'상의': 0, '하의': 1, '아우터': 2, '<UNK>': 3},
        'style': {'캐주얼': 0, '스포티': 1, '포멀': 2, '<UNK>': 3},
        'silhouette': {'오버핏': 0, '슬림핏': 1, '레귤러핏': 2, '<UNK>': 3},
        'material': {'면': 0, '폴리에스터': 1, '울': 2, '<UNK>': 3},
        'detail': {'프린트': 0, '포켓': 1, '지퍼': 2, '<UNK>': 3}
    }
    
    print(f"\nSample Vocabularies:")
    for field, vocab in sample_vocabularies.items():
        print(f"  {field}: {len(vocab)} items")
    
    print("\nData processing demo completed.")


def main():
    """Run all demos."""
    print("Fashion JSON Encoder - Integrated System Demo")
    print("=" * 80)
    
    try:
        # Run individual demos
        demo_configuration()
        print("\n")
        
        demo_model_components()
        print("\n")
        
        demo_data_processing()
        print("\n")
        
        demo_training()
        
        print("\n" + "=" * 80)
        print("ALL DEMOS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        print("\nNext steps:")
        print("1. Prepare your K-Fashion dataset")
        print("2. Run: python train.py --dataset_path /path/to/your/dataset")
        print("3. Monitor training with: tensorboard --logdir logs")
        print("4. Evaluate trained model with: python main.py evaluate --checkpoint_path checkpoints/best_model.pt")
        
    except Exception as e:
        print(f"\nDemo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()