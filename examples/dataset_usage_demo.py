"""
Demonstration of K-Fashion Dataset Loader Usage

This script shows how to use the KFashionDatasetLoader to process
K-Fashion dataset for training the JSON encoder.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.dataset_loader import KFashionDatasetLoader


def main():
    """Demonstrate K-Fashion dataset loading and processing."""
    
    print("🎯 K-Fashion Dataset Loader Demo")
    print("=" * 50)
    
    # Example dataset path (adjust this to your actual K-Fashion dataset location)
    dataset_path = "/path/to/k-fashion-dataset"
    
    # For demo purposes, we'll show the API usage
    print("\n📁 Initializing Dataset Loader...")
    loader = KFashionDatasetLoader(
        dataset_path=dataset_path,
        target_categories=['상의', '하의', '아우터'],  # Target categories
        image_size=(224, 224),                        # CLIP input size
        crop_padding=0.1                              # 10% padding around bbox
    )
    
    print(f"   Dataset path: {dataset_path}")
    print(f"   Target categories: {loader.target_categories}")
    print(f"   Image size: {loader.image_size}")
    print(f"   Crop padding: {loader.crop_padding}")
    
    # Note: The following would work with actual K-Fashion dataset
    print("\n📊 Dataset Loading Process:")
    print("   1. loader.load_dataset() - Load and filter JSON annotations")
    print("      - Parses JSON metadata files")
    print("      - Filters by target categories (상의, 하의, 아우터)")
    print("      - Converts polygons to bounding boxes")
    print("      - Validates image files exist")
    print("      - Filters out items without valid bounding boxes")
    
    print("\n   2. loader.build_vocabularies() - Build field vocabularies")
    print("      - Creates vocabulary for each JSON field")
    print("      - Adds <UNK> token for out-of-vocabulary items")
    print("      - Returns vocabulary mappings {token: index}")
    
    print("\n   3. Process individual items:")
    print("      - loader.get_cropped_image(item) - Crop and resize images")
    print("      - loader.get_processed_json(item) - Convert to vocab indices")
    
    # Show example usage code
    print("\n💻 Example Usage Code:")
    print("""
    # Load dataset
    fashion_items = loader.load_dataset()
    print(f"Loaded {len(fashion_items)} fashion items")
    
    # Build vocabularies
    vocabularies = loader.build_vocabularies()
    vocab_sizes = loader.get_vocab_sizes()
    print(f"Vocabulary sizes: {vocab_sizes}")
    
    # Process individual items
    for item in fashion_items[:5]:  # First 5 items
        # Get cropped image (224x224 RGB)
        image = loader.get_cropped_image(item)
        
        # Get processed JSON with vocabulary indices
        json_data = loader.get_processed_json(item)
        
        print(f"Item: {item.category}")
        print(f"  Image shape: {image.size}")
        print(f"  JSON fields: {list(json_data.keys())}")
        print(f"  Category ID: {json_data['category']}")
        print(f"  Style IDs: {json_data['style']}")
    """)
    
    print("\n📋 Key Features:")
    print("   ✅ Automatic polygon to bounding box conversion")
    print("   ✅ Category filtering (상의, 하의, 아우터)")
    print("   ✅ Vocabulary building with <UNK> token support")
    print("   ✅ Image cropping with configurable padding")
    print("   ✅ Automatic image resizing to 224x224 (CLIP compatible)")
    print("   ✅ Multi-categorical field support (style, material, detail)")
    print("   ✅ Robust error handling for missing files/invalid data")
    
    print("\n🔧 Integration with Training Pipeline:")
    print("   The processed data can be directly used with:")
    print("   - PyTorch DataLoader for batch processing")
    print("   - CLIP image encoder (224x224 RGB images)")
    print("   - JSON encoder (vocabulary indices)")
    print("   - Contrastive learning (image-JSON pairs)")
    
    print("\n✨ Ready for Task 2.1 Implementation!")


if __name__ == "__main__":
    main()