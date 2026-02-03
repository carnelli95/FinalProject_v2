"""
Tests for validation utilities in the Fashion JSON Encoder system.

This module tests the InputValidator, ModelValidator, and TrainingErrorHandler classes
to ensure proper error detection and validation functionality.
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict

from utils.validators import InputValidator, ModelValidator, TrainingErrorHandler


class TestInputValidator:
    """Test cases for InputValidator class."""
    
    def test_validate_json_batch_valid_input(self):
        """Test validation with valid JSON batch input."""
        batch = {
            'category': torch.tensor([1, 2, 3], dtype=torch.long),
            'style': torch.tensor([[1, 2], [3, 0], [4, 5]], dtype=torch.long),
            'silhouette': torch.tensor([1, 2, 1], dtype=torch.long),
            'material': torch.tensor([[1, 0], [2, 3], [4, 0]], dtype=torch.long),
            'detail': torch.tensor([[1, 2], [0, 0], [3, 4]], dtype=torch.long),
            'style_mask': torch.tensor([[1, 1], [1, 0], [1, 1]], dtype=torch.long),
            'material_mask': torch.tensor([[1, 0], [1, 1], [1, 0]], dtype=torch.long),
            'detail_mask': torch.tensor([[1, 1], [0, 0], [1, 1]], dtype=torch.long),
        }
        
        # Should not raise any exception
        InputValidator.validate_json_batch(batch)
    
    def test_validate_json_batch_missing_field(self):
        """Test validation with missing required field."""
        batch = {
            'category': torch.tensor([1, 2, 3], dtype=torch.long),
            'style': torch.tensor([[1, 2], [3, 0], [4, 5]], dtype=torch.long),
            # Missing 'silhouette' field
            'material': torch.tensor([[1, 0], [2, 3], [4, 0]], dtype=torch.long),
            'detail': torch.tensor([[1, 2], [0, 0], [3, 4]], dtype=torch.long),
            'style_mask': torch.tensor([[1, 1], [1, 0], [1, 1]], dtype=torch.long),
            'material_mask': torch.tensor([[1, 0], [1, 1], [1, 0]], dtype=torch.long),
            'detail_mask': torch.tensor([[1, 1], [0, 0], [1, 1]], dtype=torch.long),
        }
        
        with pytest.raises(ValueError, match="Required field 'silhouette' missing from batch"):
            InputValidator.validate_json_batch(batch)
    
    def test_validate_json_batch_missing_mask(self):
        """Test validation with missing required mask."""
        batch = {
            'category': torch.tensor([1, 2, 3], dtype=torch.long),
            'style': torch.tensor([[1, 2], [3, 0], [4, 5]], dtype=torch.long),
            'silhouette': torch.tensor([1, 2, 1], dtype=torch.long),
            'material': torch.tensor([[1, 0], [2, 3], [4, 0]], dtype=torch.long),
            'detail': torch.tensor([[1, 2], [0, 0], [3, 4]], dtype=torch.long),
            'style_mask': torch.tensor([[1, 1], [1, 0], [1, 1]], dtype=torch.long),
            # Missing 'material_mask'
            'detail_mask': torch.tensor([[1, 1], [0, 0], [1, 1]], dtype=torch.long),
        }
        
        with pytest.raises(ValueError, match="Required mask 'material_mask' missing from batch"):
            InputValidator.validate_json_batch(batch)
    
    def test_validate_json_batch_batch_size_mismatch(self):
        """Test validation with inconsistent batch sizes."""
        batch = {
            'category': torch.tensor([1, 2, 3], dtype=torch.long),  # batch_size = 3
            'style': torch.tensor([[1, 2], [3, 0]], dtype=torch.long),  # batch_size = 2
            'silhouette': torch.tensor([1, 2, 1], dtype=torch.long),
            'material': torch.tensor([[1, 0], [2, 3], [4, 0]], dtype=torch.long),
            'detail': torch.tensor([[1, 2], [0, 0], [3, 4]], dtype=torch.long),
            'style_mask': torch.tensor([[1, 1], [1, 0]], dtype=torch.long),
            'material_mask': torch.tensor([[1, 0], [1, 1], [1, 0]], dtype=torch.long),
            'detail_mask': torch.tensor([[1, 1], [0, 0], [1, 1]], dtype=torch.long),
        }
        
        with pytest.raises(ValueError, match="Batch size mismatch in field 'style'"):
            InputValidator.validate_json_batch(batch)
    
    def test_validate_json_batch_wrong_dtype(self):
        """Test validation with wrong data type."""
        batch = {
            'category': torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),  # Wrong dtype
            'style': torch.tensor([[1, 2], [3, 0], [4, 5]], dtype=torch.long),
            'silhouette': torch.tensor([1, 2, 1], dtype=torch.long),
            'material': torch.tensor([[1, 0], [2, 3], [4, 0]], dtype=torch.long),
            'detail': torch.tensor([[1, 2], [0, 0], [3, 4]], dtype=torch.long),
            'style_mask': torch.tensor([[1, 1], [1, 0], [1, 1]], dtype=torch.long),
            'material_mask': torch.tensor([[1, 0], [1, 1], [1, 0]], dtype=torch.long),
            'detail_mask': torch.tensor([[1, 1], [0, 0], [1, 1]], dtype=torch.long),
        }
        
        with pytest.raises(ValueError, match="Field 'category' must be torch.long type"):
            InputValidator.validate_json_batch(batch)
    
    def test_validate_json_batch_wrong_dimensions(self):
        """Test validation with wrong tensor dimensions."""
        batch = {
            'category': torch.tensor([[1], [2], [3]], dtype=torch.long),  # Should be 1D
            'style': torch.tensor([[1, 2], [3, 0], [4, 5]], dtype=torch.long),
            'silhouette': torch.tensor([1, 2, 1], dtype=torch.long),
            'material': torch.tensor([[1, 0], [2, 3], [4, 0]], dtype=torch.long),
            'detail': torch.tensor([[1, 2], [0, 0], [3, 4]], dtype=torch.long),
            'style_mask': torch.tensor([[1, 1], [1, 0], [1, 1]], dtype=torch.long),
            'material_mask': torch.tensor([[1, 0], [1, 1], [1, 0]], dtype=torch.long),
            'detail_mask': torch.tensor([[1, 1], [0, 0], [1, 1]], dtype=torch.long),
        }
        
        with pytest.raises(ValueError, match="Category field must be 1D tensor"):
            InputValidator.validate_json_batch(batch)
    
    def test_validate_image_batch_valid_input(self):
        """Test validation with valid image batch input."""
        images = torch.randn(4, 3, 224, 224, dtype=torch.float32)
        
        # Should not raise any exception
        InputValidator.validate_image_batch(images)
    
    def test_validate_image_batch_wrong_dimensions(self):
        """Test validation with wrong image dimensions."""
        images = torch.randn(4, 3, 224, dtype=torch.float32)  # Missing height dimension
        
        with pytest.raises(ValueError, match="Images must be 4D tensor"):
            InputValidator.validate_image_batch(images)
    
    def test_validate_image_batch_wrong_channels(self):
        """Test validation with wrong number of channels."""
        images = torch.randn(4, 1, 224, 224, dtype=torch.float32)  # Grayscale instead of RGB
        
        with pytest.raises(ValueError, match="Images must have 3 channels"):
            InputValidator.validate_image_batch(images)
    
    def test_validate_image_batch_wrong_size(self):
        """Test validation with wrong image size."""
        images = torch.randn(4, 3, 256, 256, dtype=torch.float32)  # 256x256 instead of 224x224
        
        with pytest.raises(ValueError, match="Images must be 224x224 pixels"):
            InputValidator.validate_image_batch(images)
    
    def test_validate_image_batch_wrong_dtype(self):
        """Test validation with wrong data type."""
        images = torch.randint(0, 255, (4, 3, 224, 224), dtype=torch.uint8)  # uint8 instead of float32
        
        with pytest.raises(ValueError, match="Images must be float32 type"):
            InputValidator.validate_image_batch(images)


class TestModelValidator:
    """Test cases for ModelValidator class."""
    
    def test_validate_output_dimension_valid(self):
        """Test validation with valid output dimensions."""
        output = torch.randn(4, 512)  # Valid 512-dimensional output
        
        # Should not raise any exception
        ModelValidator.validate_output_dimension(output)
    
    def test_validate_output_dimension_wrong_dim(self):
        """Test validation with wrong output dimensions."""
        output = torch.randn(4, 256)  # Wrong dimension
        
        with pytest.raises(ValueError, match="Output dimension must be 512"):
            ModelValidator.validate_output_dimension(output)
    
    def test_validate_output_dimension_wrong_shape(self):
        """Test validation with wrong output shape."""
        output = torch.randn(4, 512, 1)  # 3D instead of 2D
        
        with pytest.raises(ValueError, match="Output must be 2D tensor"):
            ModelValidator.validate_output_dimension(output)
    
    def test_validate_normalization_valid(self):
        """Test validation with properly normalized embeddings."""
        embeddings = torch.randn(4, 512)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        
        # Should not raise any exception
        ModelValidator.validate_normalization(embeddings)
    
    def test_validate_normalization_invalid(self):
        """Test validation with non-normalized embeddings."""
        embeddings = torch.randn(4, 512) * 10  # Not normalized
        
        with pytest.raises(ValueError, match="Embeddings must be L2 normalized"):
            ModelValidator.validate_normalization(embeddings)
    
    def test_validate_clip_frozen_valid(self):
        """Test validation with unchanged CLIP parameters."""
        model = nn.Linear(10, 5)
        original_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # Should not raise any exception
        ModelValidator.validate_clip_frozen(model, original_params)
    
    def test_validate_clip_frozen_modified(self):
        """Test validation with modified CLIP parameters."""
        model = nn.Linear(10, 5)
        original_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # Modify parameters
        with torch.no_grad():
            for param in model.parameters():
                param.add_(0.1)
        
        with pytest.raises(ValueError, match="CLIP parameter .* has been modified"):
            ModelValidator.validate_clip_frozen(model, original_params)
    
    def test_validate_vocab_sizes_valid(self):
        """Test validation with valid vocabulary sizes."""
        vocab_sizes = {
            'category': 10,
            'style': 20,
            'silhouette': 15,
            'material': 25,
            'detail': 30
        }
        
        # Should not raise any exception
        ModelValidator.validate_vocab_sizes(vocab_sizes)
    
    def test_validate_vocab_sizes_missing_field(self):
        """Test validation with missing vocabulary field."""
        vocab_sizes = {
            'category': 10,
            'style': 20,
            # Missing 'silhouette'
            'material': 25,
            'detail': 30
        }
        
        with pytest.raises(ValueError, match="Required vocabulary field 'silhouette' missing"):
            ModelValidator.validate_vocab_sizes(vocab_sizes)
    
    def test_validate_vocab_sizes_invalid_size(self):
        """Test validation with invalid vocabulary size."""
        vocab_sizes = {
            'category': 10,
            'style': 0,  # Invalid size
            'silhouette': 15,
            'material': 25,
            'detail': 30
        }
        
        with pytest.raises(ValueError, match="Vocabulary size for 'style' must be positive integer"):
            ModelValidator.validate_vocab_sizes(vocab_sizes)


class TestTrainingErrorHandler:
    """Test cases for TrainingErrorHandler class."""
    
    def test_handle_loss_explosion_valid(self):
        """Test handling with normal loss values."""
        loss = torch.tensor(1.5)
        
        # Should not raise any exception
        TrainingErrorHandler.handle_loss_explosion(loss)
    
    def test_handle_loss_explosion_exploded(self):
        """Test handling with exploded loss values."""
        loss = torch.tensor(150.0)  # Above threshold
        
        with pytest.raises(RuntimeError, match="Loss exploded"):
            TrainingErrorHandler.handle_loss_explosion(loss)
    
    def test_handle_loss_explosion_infinite(self):
        """Test handling with infinite loss values."""
        loss = torch.tensor(float('inf'))
        
        with pytest.raises(RuntimeError, match="Loss is not finite"):
            TrainingErrorHandler.handle_loss_explosion(loss)
    
    def test_handle_gradient_issues_valid(self):
        """Test handling with normal gradient values."""
        model = nn.Linear(10, 5)
        
        # Create some controlled gradients
        input_data = torch.randn(2, 10) * 0.1  # Small input to keep gradients reasonable
        output = model(input_data)
        loss = output.sum() * 0.1  # Small loss to keep gradients reasonable
        loss.backward()
        
        # Should not raise any exception
        TrainingErrorHandler.handle_gradient_issues(model)
    
    def test_handle_gradient_issues_no_gradients(self):
        """Test handling with no gradients."""
        model = nn.Linear(10, 5)
        
        with pytest.raises(RuntimeError, match="No gradients found in model"):
            TrainingErrorHandler.handle_gradient_issues(model)
    
    def test_validate_batch_consistency_valid(self):
        """Test validation with consistent batch sizes."""
        images = torch.randn(4, 3, 224, 224)
        json_batch = {
            'category': torch.tensor([1, 2, 3, 4], dtype=torch.long),
            'style': torch.tensor([[1, 2], [3, 0], [4, 5], [6, 7]], dtype=torch.long),
        }
        
        # Should not raise any exception
        TrainingErrorHandler.validate_batch_consistency(images, json_batch)
    
    def test_validate_batch_consistency_mismatch(self):
        """Test validation with inconsistent batch sizes."""
        images = torch.randn(4, 3, 224, 224)  # batch_size = 4
        json_batch = {
            'category': torch.tensor([1, 2, 3], dtype=torch.long),  # batch_size = 3
        }
        
        with pytest.raises(ValueError, match="Batch size mismatch"):
            TrainingErrorHandler.validate_batch_consistency(images, json_batch)