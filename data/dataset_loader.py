"""
K-Fashion Dataset Loader for Fashion JSON Encoder.

This module provides the KFashionDatasetLoader class that integrates all preprocessing
components to load and process K-Fashion dataset for training the JSON encoder.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterator
import PIL.Image
from PIL import Image

from .processor import FashionDataProcessor
from .data_models import FashionItem


class KFashionDatasetLoader:
    """
    K-Fashion dataset loader that handles JSON metadata parsing, filtering,
    polygon to bbox conversion, and image cropping.
    
    This class integrates all preprocessing components to provide a complete
    data loading pipeline for the Fashion JSON Encoder system.
    """
    
    def __init__(self, 
                 dataset_path: str,
                 target_categories: List[str] = None,
                 image_size: Tuple[int, int] = (224, 224),
                 crop_padding: float = 0.1):
        """
        Initialize the K-Fashion dataset loader.
        
        Args:
            dataset_path: Path to the K-Fashion dataset root directory
            target_categories: List of target categories to process (default: ['상의', '하의', '아우터'])
            image_size: Target image size for resizing cropped images
            crop_padding: Padding ratio to add around bounding boxes
        """
        self.dataset_path = Path(dataset_path)
        self.target_categories = target_categories or ['상의', '하의', '아우터']
        self.image_size = image_size
        self.crop_padding = crop_padding
        
        # Initialize data processor
        self.processor = FashionDataProcessor(
            dataset_path=str(self.dataset_path),
            target_categories=self.target_categories
        )
        
        # Cache for loaded data
        self._fashion_items: List[FashionItem] = []
        self._vocabularies_built = False
        
    def load_dataset(self, 
                    json_dir: str = "annotations",
                    image_dir: str = "images") -> List[FashionItem]:
        """
        Load the complete K-Fashion dataset.
        
        Args:
            json_dir: Subdirectory containing JSON annotation files
            image_dir: Subdirectory containing image files
            
        Returns:
            List of FashionItem objects with valid bounding boxes
        """
        json_path = self.dataset_path / json_dir
        image_path = self.dataset_path / image_dir
        
        if not json_path.exists():
            raise FileNotFoundError(f"JSON directory not found: {json_path}")
        if not image_path.exists():
            raise FileNotFoundError(f"Image directory not found: {image_path}")
        
        fashion_items = []
        
        # Process all JSON files
        json_files = list(json_path.glob("*.json"))
        
        for json_file in json_files:
            try:
                items = self._process_json_file(json_file, image_path)
                fashion_items.extend(items)
            except Exception as e:
                print(f"Warning: Failed to process {json_file}: {e}")
                continue
        
        # Filter items with valid bounding boxes
        valid_items = [item for item in fashion_items if self._has_valid_bbox(item)]
        
        self._fashion_items = valid_items
        return valid_items
    
    def _process_json_file(self, 
                          json_file: Path, 
                          image_path: Path) -> List[FashionItem]:
        """
        Process a single JSON annotation file.
        
        Args:
            json_file: Path to JSON annotation file
            image_path: Path to image directory
            
        Returns:
            List of FashionItem objects from this file
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        items = []
        
        # Handle different JSON structures (single item vs multiple items)
        if isinstance(data, list):
            annotations = data
        elif 'annotations' in data:
            annotations = data['annotations']
        else:
            # Single item format
            annotations = [data]
        
        for annotation in annotations:
            try:
                item = self._create_fashion_item(annotation, image_path)
                if item and item.category in self.target_categories:
                    items.append(item)
            except Exception as e:
                print(f"Warning: Failed to process annotation: {e}")
                continue
        
        return items
    
    def _create_fashion_item(self, 
                           annotation: Dict, 
                           image_path: Path) -> Optional[FashionItem]:
        """
        Create a FashionItem from JSON annotation.
        
        Args:
            annotation: Single item annotation dictionary
            image_path: Path to image directory
            
        Returns:
            FashionItem object or None if invalid
        """
        # Extract required fields
        image_filename = annotation.get('image_name') or annotation.get('file_name')
        if not image_filename:
            return None
        
        full_image_path = image_path / image_filename
        if not full_image_path.exists():
            return None
        
        # Extract polygon coordinates
        polygon_data = annotation.get('polygon') or annotation.get('segmentation')
        if not polygon_data:
            return None
        
        # Convert polygon to bbox
        try:
            if isinstance(polygon_data[0], list):
                # Multiple polygons - use the first one
                polygon = [(int(polygon_data[0][i]), int(polygon_data[0][i+1])) 
                          for i in range(0, len(polygon_data[0]), 2)]
            else:
                # Single polygon
                polygon = [(int(polygon_data[i]), int(polygon_data[i+1])) 
                          for i in range(0, len(polygon_data), 2)]
            
            bbox = self.processor.polygon_to_bbox(polygon)
        except (ValueError, IndexError, TypeError):
            return None
        
        # Extract metadata fields
        category = annotation.get('category', '')
        style = annotation.get('style', [])
        silhouette = annotation.get('silhouette', '')
        material = annotation.get('material', [])
        detail = annotation.get('detail', [])
        
        # Ensure list fields are actually lists
        if not isinstance(style, list):
            style = [style] if style else []
        if not isinstance(material, list):
            material = [material] if material else []
        if not isinstance(detail, list):
            detail = [detail] if detail else []
        
        return FashionItem(
            image_path=str(full_image_path),
            bbox=bbox,
            category=category,
            style=style,
            silhouette=silhouette,
            material=material,
            detail=detail
        )
    
    def _has_valid_bbox(self, item: FashionItem) -> bool:
        """
        Check if a FashionItem has a valid bounding box.
        
        Args:
            item: FashionItem to check
            
        Returns:
            True if bbox is valid, False otherwise
        """
        x, y, width, height = item.bbox
        return width > 0 and height > 0 and x >= 0 and y >= 0
    
    def build_vocabularies(self) -> Dict[str, Dict[str, int]]:
        """
        Build vocabularies from loaded fashion items.
        
        Returns:
            Dictionary mapping field names to vocabularies
        """
        if not self._fashion_items:
            raise ValueError("No fashion items loaded. Call load_dataset() first.")
        
        # Create temporary JSON files for vocabulary building
        json_data_list = []
        for item in self._fashion_items:
            json_data = {
                'category': item.category,
                'style': item.style,
                'silhouette': item.silhouette,
                'material': item.material,
                'detail': item.detail
            }
            json_data_list.append(json_data)
        
        # Build vocabularies using the processor
        vocabularies = {}
        field_tokens = {
            'category': set(),
            'style': set(),
            'silhouette': set(),
            'material': set(),
            'detail': set()
        }
        
        # Collect all unique tokens
        for json_data in json_data_list:
            field_tokens['category'].add(json_data.get('category', ''))
            field_tokens['silhouette'].add(json_data.get('silhouette', ''))
            
            # Multi-categorical fields
            for style in json_data.get('style', []):
                field_tokens['style'].add(style)
            for material in json_data.get('material', []):
                field_tokens['material'].add(material)
            for detail in json_data.get('detail', []):
                field_tokens['detail'].add(detail)
        
        # Build vocabularies with <UNK> token
        for field, tokens in field_tokens.items():
            vocab = {'<UNK>': 0}  # OOV token at index 0
            for i, token in enumerate(sorted(tokens), 1):
                if token:  # Skip empty strings
                    vocab[token] = i
            vocabularies[field] = vocab
        
        self.processor.vocabularies = vocabularies
        self._vocabularies_built = True
        return vocabularies
    
    def get_cropped_image(self, item: FashionItem) -> PIL.Image.Image:
        """
        Get cropped and resized image for a fashion item.
        
        Args:
            item: FashionItem to process
            
        Returns:
            Cropped and resized PIL Image
        """
        # Load original image
        image = Image.open(item.image_path).convert('RGB')
        
        # Crop using bbox with padding
        cropped = self.processor.crop_image_by_bbox(
            image, item.bbox, padding=self.crop_padding
        )
        
        # Resize to target size
        resized = cropped.resize(self.image_size, Image.Resampling.LANCZOS)
        
        return resized
    
    def get_processed_json(self, item: FashionItem) -> Dict[str, any]:
        """
        Get processed JSON data with vocabulary indices.
        
        Args:
            item: FashionItem to process
            
        Returns:
            Dictionary with fields converted to vocabulary indices
        """
        if not self._vocabularies_built:
            raise ValueError("Vocabularies not built. Call build_vocabularies() first.")
        
        json_data = {
            'category': item.category,
            'style': item.style,
            'silhouette': item.silhouette,
            'material': item.material,
            'detail': item.detail
        }
        
        return self.processor.process_json_fields(json_data)
    
    def get_vocab_sizes(self) -> Dict[str, int]:
        """
        Get vocabulary sizes for each field.
        
        Returns:
            Dictionary mapping field names to vocabulary sizes
        """
        return self.processor.get_vocab_sizes()
    
    def __len__(self) -> int:
        """Return number of loaded fashion items."""
        return len(self._fashion_items)
    
    def __getitem__(self, idx: int) -> FashionItem:
        """Get fashion item by index."""
        return self._fashion_items[idx]
    
    def __iter__(self) -> Iterator[FashionItem]:
        """Iterate over fashion items."""
        return iter(self._fashion_items)