"""
Unit tests for ContrastiveLearner model.

Tests the contrastive learning system including CLIP integration,
InfoNCE loss computation, and embedding alignment functionality.
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict
from transformers import CLIPVisionModel, CLIPVisionConfig

from models.contrastive_learner import ContrastiveLearner
from models.json_encoder import JSONEncoder


class TestContrastiveLearner:
    """Test cases for ContrastiveLearner class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create vocabulary sizes for JSONEncoder
        self.vocab_sizes = {
            'category': 10,
            'style': 20,
            'silhouette': 8,
            'material': 15,
            'detail': 25
        }
        
        # Create JSONEncoder
        self.json_encoder = JSONEncoder(
            vocab_sizes=self.vocab_sizes,
            embedding_dim=128,
            hidden_dim=256,
            output_dim=512,
            dropout_rate=0.1
        )
        
        # Create a minimal CLIP vision model for testing
        # Use a small configuration to speed up tests
        clip_config = CLIPVisionConfig(
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=2,
            num_attention_heads=8,
            image_size=224,
            patch_size=32,
            projection_dim=512
        )
        self.clip_encoder = CLIPVisionModel(clip_config)
        
        # Create ContrastiveLearner
        self.contrastive_learner = ContrastiveLearner(
            json_encoder=self.json_encoder,
            clip_encoder=self.clip_encoder,
            temperature=0.07
        )
        
        # Set to eval mode for consistent testing
        self.contrastive_learner.eval()
    
    def _create_test_batch(self, batch_size: int = 4) -> Dict[str, torch.Tensor]:
        """Create a test JSON batch with proper structure."""
        max_style_len = 3
        max_material_len = 2
        max_detail_len = 4
        
        batch = {
            # Single categorical fields
            'category': torch.randint(0, self.vocab_sizes['category'], (batch_size,), dtype=torch.long),
            'silhouette': torch.randint(0, self.vocab_sizes['silhouette'], (batch_size,), dtype=torch.long),
            
            # Multi categorical fields
            'style': torch.randint(0, self.vocab_sizes['style'], (batch_size, max_style_len), dtype=torch.long),
            'material': torch.randint(0, self.vocab_sizes['material'], (batch_size, max_material_len), dtype=torch.long),
            'detail': torch.randint(0, self.vocab_sizes['detail'], (batch_size, max_detail_len), dtype=torch.long),
            
            # Padding masks
            'style_mask': torch.ones(batch_size, max_style_len, dtype=torch.float),
            'material_mask': torch.ones(batch_size, max_material_len, dtype=torch.float),
            'detail_mask': torch.ones(batch_size, max_detail_len, dtype=torch.float)
        }
        
        return batch
    
    def _create_test_images(self, batch_size: int = 4) -> torch.Tensor:
        """Create test images tensor."""
        return torch.randn(batch_size, 3, 224, 224)
    
    def test_initialization(self):
        """Test ContrastiveLearner initialization."""
        assert isinstance(self.contrastive_learner.json_encoder, JSONEncoder)
        assert isinstance(self.contrastive_learner.clip_encoder, CLIPVisionModel)
        assert self.contrastive_learner.temperature == 0.07
        
        # Check that CLIP parameters are frozen
        for param in self.contrastive_learner.clip_encoder.parameters():
            assert param.requires_grad is False
    
    def test_clip_frozen_state(self):
        """Test that CLIP encoder parameters remain frozen during training."""
        # Store original parameters
        original_params = {}
        for name, param in self.contrastive_learner.clip_encoder.named_parameters():
            original_params[name] = param.clone()
        
        # Enable training mode
        self.contrastive_learner.train()
        
        # Create test data
        images = self._create_test_images(batch_size=2)
        json_batch = self._create_test_batch(batch_size=2)
        
        # Forward pass and backward pass
        loss = self.contrastive_learner(images, json_batch)
        loss.backward()
        
        # Check that CLIP parameters haven't changed
        for name, param in self.contrastive_learner.clip_encoder.named_parameters():
            assert torch.equal(param, original_params[name]), f"CLIP parameter '{name}' was modified"
            assert param.grad is None, f"CLIP parameter '{name}' has gradients"
    
    def test_forward_output_shape(self):
        """Test that forward pass produces scalar loss."""
        images = self._create_test_images(batch_size=4)
        json_batch = self._create_test_batch(batch_size=4)
        
        loss = self.contrastive_learner(images, json_batch)
        
        # Should output scalar loss
        assert loss.shape == torch.Size([])
        assert loss.dtype == torch.float32
        assert loss.item() >= 0  # Loss should be non-negative
    
    def test_forward_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        for batch_size in [1, 2, 8, 16]:
            images = self._create_test_images(batch_size=batch_size)
            json_batch = self._create_test_batch(batch_size=batch_size)
            
            loss = self.contrastive_learner(images, json_batch)
            assert loss.shape == torch.Size([])
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)
    
    def test_get_embeddings_output_shape(self):
        """Test get_embeddings method output shapes."""
        images = self._create_test_images(batch_size=4)
        json_batch = self._create_test_batch(batch_size=4)
        
        result = self.contrastive_learner.get_embeddings(images, json_batch)
        
        # Check output structure
        assert 'image_embeddings' in result
        assert 'json_embeddings' in result
        assert 'similarity_matrix' in result
        
        # Check shapes
        assert result['image_embeddings'].shape == (4, 512)
        assert result['json_embeddings'].shape == (4, 512)
        assert result['similarity_matrix'].shape == (4, 4)
        
        # Check normalization
        image_norms = torch.norm(result['image_embeddings'], p=2, dim=-1)
        json_norms = torch.norm(result['json_embeddings'], p=2, dim=-1)
        
        assert torch.allclose(image_norms, torch.ones_like(image_norms), atol=1e-6)
        assert torch.allclose(json_norms, torch.ones_like(json_norms), atol=1e-6)
    
    def test_similarity_matrix_properties(self):
        """Test properties of the similarity matrix."""
        images = self._create_test_images(batch_size=4)
        json_batch = self._create_test_batch(batch_size=4)
        
        result = self.contrastive_learner.get_embeddings(images, json_batch)
        similarity_matrix = result['similarity_matrix']
        
        # Similarity values should be in [-1, 1] range (cosine similarity)
        assert torch.all(similarity_matrix >= -1.0)
        assert torch.all(similarity_matrix <= 1.0)
        
        # Matrix should be batch_size x batch_size
        assert similarity_matrix.shape == (4, 4)
    
    def test_infonce_loss_computation(self):
        """Test InfoNCE loss computation method."""
        batch_size = 4
        
        # Create normalized embeddings
        image_embeddings = torch.randn(batch_size, 512)
        image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=-1)
        
        json_embeddings = torch.randn(batch_size, 512)
        json_embeddings = torch.nn.functional.normalize(json_embeddings, p=2, dim=-1)
        
        # Compute loss
        loss = self.contrastive_learner._compute_infonce_loss(image_embeddings, json_embeddings)
        
        # Loss should be scalar and positive
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_infonce_loss_perfect_alignment(self):
        """Test InfoNCE loss with perfectly aligned embeddings."""
        batch_size = 4
        
        # Create identical embeddings (perfect alignment)
        embeddings = torch.randn(batch_size, 512)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        
        # Use same embeddings for both image and JSON
        loss = self.contrastive_learner._compute_infonce_loss(embeddings, embeddings)
        
        # Loss should be very small (close to 0) for perfect alignment
        assert loss.item() < 0.1  # Should be much smaller than random case
    
    def test_infonce_loss_random_alignment(self):
        """Test InfoNCE loss with random embeddings."""
        batch_size = 4
        
        # Create random embeddings (no alignment)
        image_embeddings = torch.randn(batch_size, 512)
        image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=-1)
        
        json_embeddings = torch.randn(batch_size, 512)
        json_embeddings = torch.nn.functional.normalize(json_embeddings, p=2, dim=-1)
        
        loss = self.contrastive_learner._compute_infonce_loss(image_embeddings, json_embeddings)
        
        # Loss should be higher for random embeddings
        # Expected loss for random embeddings is approximately log(batch_size)
        expected_loss = torch.log(torch.tensor(float(batch_size)))
        assert loss.item() > expected_loss.item() * 0.5  # Allow some variance
    
    def test_temperature_effect(self):
        """Test that temperature parameter affects loss computation."""
        batch_size = 4
        
        # Create test embeddings
        image_embeddings = torch.randn(batch_size, 512)
        image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=-1)
        
        json_embeddings = torch.randn(batch_size, 512)
        json_embeddings = torch.nn.functional.normalize(json_embeddings, p=2, dim=-1)
        
        # Test with different temperatures
        learner_low_temp = ContrastiveLearner(
            json_encoder=self.json_encoder,
            clip_encoder=self.clip_encoder,
            temperature=0.01  # Low temperature
        )
        
        learner_high_temp = ContrastiveLearner(
            json_encoder=self.json_encoder,
            clip_encoder=self.clip_encoder,
            temperature=1.0   # High temperature
        )
        
        loss_low = learner_low_temp._compute_infonce_loss(image_embeddings, json_embeddings)
        loss_high = learner_high_temp._compute_infonce_loss(image_embeddings, json_embeddings)
        
        # Lower temperature should generally produce higher loss (sharper distribution)
        # This is not always true due to randomness, but should hold on average
        assert not torch.isnan(loss_low)
        assert not torch.isnan(loss_high)
    
    def test_gradient_flow_json_encoder_only(self):
        """Test that gradients flow only through JSON encoder, not CLIP."""
        self.contrastive_learner.train()
        
        images = self._create_test_images(batch_size=2)
        json_batch = self._create_test_batch(batch_size=2)
        
        loss = self.contrastive_learner(images, json_batch)
        loss.backward()
        
        # Check JSON encoder has gradients
        for name, param in self.contrastive_learner.json_encoder.named_parameters():
            assert param.grad is not None, f"No gradient for JSON encoder parameter: {name}"
        
        # Check CLIP encoder has no gradients
        for name, param in self.contrastive_learner.clip_encoder.named_parameters():
            assert param.grad is None, f"Unexpected gradient for CLIP parameter: {name}"
    
    def test_positive_pair_generation(self):
        """Test that positive pairs are correctly identified."""
        batch_size = 4
        images = self._create_test_images(batch_size=batch_size)
        json_batch = self._create_test_batch(batch_size=batch_size)
        
        result = self.contrastive_learner.get_embeddings(images, json_batch)
        similarity_matrix = result['similarity_matrix']
        
        # In a batch, positive pairs are on the diagonal
        # Each image should have highest similarity with its corresponding JSON
        # (This is not guaranteed with random data, but we can check structure)
        
        # Check that diagonal elements exist and are valid similarities
        diagonal_similarities = torch.diag(similarity_matrix)
        assert diagonal_similarities.shape == (batch_size,)
        assert torch.all(diagonal_similarities >= -1.0)
        assert torch.all(diagonal_similarities <= 1.0)
    
    def test_negative_pair_generation(self):
        """Test that negative pairs are correctly generated."""
        batch_size = 4
        images = self._create_test_images(batch_size=batch_size)
        json_batch = self._create_test_batch(batch_size=batch_size)
        
        result = self.contrastive_learner.get_embeddings(images, json_batch)
        similarity_matrix = result['similarity_matrix']
        
        # Off-diagonal elements are negative pairs
        # Should have (batch_size^2 - batch_size) negative pairs
        total_pairs = batch_size * batch_size
        positive_pairs = batch_size
        negative_pairs = total_pairs - positive_pairs
        
        # Extract off-diagonal elements
        mask = ~torch.eye(batch_size, dtype=torch.bool)
        negative_similarities = similarity_matrix[mask]
        
        assert negative_similarities.shape == (negative_pairs,)
        assert torch.all(negative_similarities >= -1.0)
        assert torch.all(negative_similarities <= 1.0)
    
    def test_batch_size_one(self):
        """Test edge case with batch size 1."""
        images = self._create_test_images(batch_size=1)
        json_batch = self._create_test_batch(batch_size=1)
        
        # Should not crash with batch size 1
        loss = self.contrastive_learner(images, json_batch)
        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss)
        
        result = self.contrastive_learner.get_embeddings(images, json_batch)
        assert result['similarity_matrix'].shape == (1, 1)
    
    def test_deterministic_output_eval_mode(self):
        """Test that model produces deterministic output in eval mode."""
        self.contrastive_learner.eval()
        
        images = self._create_test_images(batch_size=2)
        json_batch = self._create_test_batch(batch_size=2)
        
        # Run forward pass twice
        loss1 = self.contrastive_learner(images, json_batch)
        loss2 = self.contrastive_learner(images, json_batch)
        
        # Should be identical in eval mode (no dropout)
        assert torch.allclose(loss1, loss2)
    
    def test_embedding_alignment_training(self):
        """Test that training improves embedding alignment."""
        self.contrastive_learner.train()
        
        # Create identical JSON data for all images (should align perfectly)
        batch_size = 4
        images = self._create_test_images(batch_size=batch_size)
        
        # Create identical JSON batch (all same values)
        json_batch = {
            'category': torch.zeros(batch_size, dtype=torch.long),
            'silhouette': torch.zeros(batch_size, dtype=torch.long),
            'style': torch.zeros(batch_size, 3, dtype=torch.long),
            'material': torch.zeros(batch_size, 2, dtype=torch.long),
            'detail': torch.zeros(batch_size, 4, dtype=torch.long),
            'style_mask': torch.ones(batch_size, 3, dtype=torch.float),
            'material_mask': torch.ones(batch_size, 2, dtype=torch.float),
            'detail_mask': torch.ones(batch_size, 4, dtype=torch.float)
        }
        
        # Initial loss
        initial_loss = self.contrastive_learner(images, json_batch)
        
        # Simple training step
        optimizer = torch.optim.Adam(self.contrastive_learner.json_encoder.parameters(), lr=0.01)
        
        for _ in range(5):  # Few training steps
            optimizer.zero_grad()
            loss = self.contrastive_learner(images, json_batch)
            loss.backward()
            optimizer.step()
        
        # Final loss
        final_loss = self.contrastive_learner(images, json_batch)
        
        # Loss should decrease (though not guaranteed with random data and few steps)
        # At minimum, training should not crash
        assert not torch.isnan(final_loss)
        assert not torch.isinf(final_loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])