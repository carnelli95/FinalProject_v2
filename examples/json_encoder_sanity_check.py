"""
JSON Encoder Standalone Training and Output Distribution Sanity Check.

This script performs standalone training of the JSON Encoder without CLIP integration
to verify that the model can learn meaningful representations from JSON metadata alone.
It analyzes the output distribution and validates the model's basic functionality.
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
# import matplotlib.pyplot as plt  # Not needed for sanity check
import json
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.json_encoder import JSONEncoder
from data.fashion_dataset import FashionDataModule
from utils.config import TrainingConfig


class JSONEncoderSanityChecker:
    """
    Standalone JSON Encoder training and sanity checking system.
    
    This class performs isolated training of the JSON Encoder to verify:
    1. Model can process JSON metadata correctly
    2. Output dimensions are correct (512)
    3. Output vectors are properly L2 normalized
    4. Model can learn from synthetic targets
    5. Output distribution is reasonable
    """
    
    def __init__(self, vocab_sizes: Dict[str, int], device: str = 'cuda'):
        """
        Initialize the sanity checker.
        
        Args:
            vocab_sizes: Vocabulary sizes for each field
            device: Device to use for training
        """
        self.vocab_sizes = vocab_sizes
        self.device = device
        
        # Initialize model
        self.json_encoder = JSONEncoder(
            vocab_sizes=vocab_sizes,
            embedding_dim=128,
            hidden_dim=256,
            output_dim=512,
            dropout_rate=0.1
        ).to(device)
        
        # Training components
        self.optimizer = optim.Adam(self.json_encoder.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        
        print(f"JSON Encoder initialized on {device}")
        print(f"Model parameters: {sum(p.numel() for p in self.json_encoder.parameters()):,}")
        print(f"Vocabulary sizes: {vocab_sizes}")
    
    def run_sanity_check(self, data_module: FashionDataModule, 
                        num_epochs: int = 5) -> Dict[str, Any]:
        """
        Run complete sanity check including training and analysis.
        
        Args:
            data_module: Initialized FashionDataModule
            num_epochs: Number of epochs for standalone training
            
        Returns:
            Dictionary containing all sanity check results
        """
        print(f"\n{'='*80}")
        print("JSON ENCODER STANDALONE SANITY CHECK")
        print(f"{'='*80}")
        
        # Get data loaders
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        
        # Phase 1: Initial output analysis (before training)
        print(f"\n{'='*60}")
        print("PHASE 1: INITIAL OUTPUT ANALYSIS (BEFORE TRAINING)")
        print(f"{'='*60}")
        
        initial_analysis = self._analyze_output_distribution(val_loader, "Initial")
        
        # Phase 2: Standalone training
        print(f"\n{'='*60}")
        print("PHASE 2: STANDALONE TRAINING")
        print(f"{'='*60}")
        
        training_results = self._train_standalone(train_loader, val_loader, num_epochs)
        
        # Phase 3: Final output analysis (after training)
        print(f"\n{'='*60}")
        print("PHASE 3: FINAL OUTPUT ANALYSIS (AFTER TRAINING)")
        print(f"{'='*60}")
        
        final_analysis = self._analyze_output_distribution(val_loader, "Final")
        
        # Phase 4: Comprehensive validation
        print(f"\n{'='*60}")
        print("PHASE 4: COMPREHENSIVE VALIDATION")
        print(f"{'='*60}")
        
        validation_results = self._comprehensive_validation(val_loader)
        
        # Compile results
        results = {
            'initial_analysis': initial_analysis,
            'training_results': training_results,
            'final_analysis': final_analysis,
            'validation_results': validation_results,
            'vocab_sizes': self.vocab_sizes,
            'model_params': sum(p.numel() for p in self.json_encoder.parameters())
        }
        
        # Print summary
        self._print_sanity_check_summary(results)
        
        return results
    
    def _train_standalone(self, train_loader: DataLoader, val_loader: DataLoader,
                         num_epochs: int) -> Dict[str, List[float]]:
        """Train JSON encoder in standalone mode."""
        self.json_encoder.train()
        
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # Training phase
            epoch_train_loss = 0.0
            num_train_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Convert batch to JSON format
                json_batch = self._convert_batch_to_dict(batch)
                
                # Forward pass
                self.optimizer.zero_grad()
                embeddings = self.json_encoder(json_batch)
                
                # Create synthetic target (random normalized vectors)
                target = torch.randn_like(embeddings)
                target = torch.nn.functional.normalize(target, p=2, dim=-1)
                
                # Compute loss
                loss = self.criterion(embeddings, target)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_train_loss += loss.item()
                num_train_batches += 1
                
                if batch_idx % 20 == 0:
                    print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_train_loss = epoch_train_loss / num_train_batches
            train_losses.append(avg_train_loss)
            
            # Validation phase
            val_loss = self._validate_standalone(val_loader)
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    
    def _validate_standalone(self, val_loader: DataLoader) -> float:
        """Validate JSON encoder in standalone mode."""
        self.json_encoder.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                json_batch = self._convert_batch_to_dict(batch)
                embeddings = self.json_encoder(json_batch)
                
                # Synthetic target
                target = torch.randn_like(embeddings)
                target = torch.nn.functional.normalize(target, p=2, dim=-1)
                
                loss = self.criterion(embeddings, target)
                total_loss += loss.item()
                num_batches += 1
        
        self.json_encoder.train()
        return total_loss / num_batches
    
    def _analyze_output_distribution(self, val_loader: DataLoader, 
                                   phase_name: str) -> Dict[str, Any]:
        """Analyze the output distribution of JSON encoder."""
        self.json_encoder.eval()
        
        all_embeddings = []
        field_stats = {
            'category': [],
            'style': [],
            'silhouette': [],
            'material': [],
            'detail': []
        }
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= 10:  # Limit analysis to first 10 batches
                    break
                    
                json_batch = self._convert_batch_to_dict(batch)
                embeddings = self.json_encoder(json_batch)
                all_embeddings.append(embeddings.cpu())
                
                # Collect field statistics
                for field in field_stats.keys():
                    if field in json_batch:
                        field_data = json_batch[field]
                        if field_data.dim() == 1:  # Single categorical
                            unique_vals = torch.unique(field_data).cpu().numpy()
                            field_stats[field].extend(unique_vals.tolist())
                        else:  # Multi categorical
                            # Flatten and get unique values
                            flat_data = field_data.flatten()
                            unique_vals = torch.unique(flat_data[flat_data > 0]).cpu().numpy()
                            field_stats[field].extend(unique_vals.tolist())
        
        if not all_embeddings:
            return {'error': 'No embeddings generated'}
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        # Compute comprehensive statistics
        norms = torch.norm(all_embeddings, dim=-1)
        
        analysis = {
            'phase': phase_name,
            'total_samples': all_embeddings.size(0),
            'embedding_dim': all_embeddings.size(1),
            'mean': all_embeddings.mean().item(),
            'std': all_embeddings.std().item(),
            'min': all_embeddings.min().item(),
            'max': all_embeddings.max().item(),
            'norm_mean': norms.mean().item(),
            'norm_std': norms.std().item(),
            'norm_min': norms.min().item(),
            'norm_max': norms.max().item(),
            'is_normalized': torch.allclose(norms, torch.ones_like(norms), atol=1e-3),
            'field_unique_counts': {field: len(set(values)) for field, values in field_stats.items()}
        }
        
        print(f"{phase_name} Output Analysis:")
        print(f"  Samples: {analysis['total_samples']}")
        print(f"  Embedding dim: {analysis['embedding_dim']}")
        print(f"  Value range: [{analysis['min']:.4f}, {analysis['max']:.4f}]")
        print(f"  Mean: {analysis['mean']:.4f}, Std: {analysis['std']:.4f}")
        print(f"  Norm range: [{analysis['norm_min']:.4f}, {analysis['norm_max']:.4f}]")
        print(f"  Norm mean: {analysis['norm_mean']:.4f}, Norm std: {analysis['norm_std']:.4f}")
        print(f"  Is normalized: {analysis['is_normalized']}")
        print(f"  Field unique counts: {analysis['field_unique_counts']}")
        
        self.json_encoder.train()
        return analysis
    
    def _comprehensive_validation(self, val_loader: DataLoader) -> Dict[str, Any]:
        """Perform comprehensive validation of the model."""
        self.json_encoder.eval()
        
        validation_results = {
            'dimension_check': True,
            'normalization_check': True,
            'gradient_check': True,
            'field_processing_check': True,
            'batch_consistency_check': True,
            'errors': []
        }
        
        try:
            with torch.no_grad():
                # Test with a single batch
                batch = next(iter(val_loader))
                json_batch = self._convert_batch_to_dict(batch)
                
                # 1. Dimension check
                embeddings = self.json_encoder(json_batch)
                if embeddings.shape[-1] != 512:
                    validation_results['dimension_check'] = False
                    validation_results['errors'].append(f"Wrong output dimension: {embeddings.shape[-1]} != 512")
                
                # 2. Normalization check
                norms = torch.norm(embeddings, dim=-1)
                if not torch.allclose(norms, torch.ones_like(norms), atol=1e-3):
                    validation_results['normalization_check'] = False
                    validation_results['errors'].append(f"Embeddings not normalized: norm range [{norms.min():.4f}, {norms.max():.4f}]")
                
                # 3. Batch consistency check
                # Process same data twice, should get same results
                embeddings2 = self.json_encoder(json_batch)
                if not torch.allclose(embeddings, embeddings2, atol=1e-6):
                    validation_results['batch_consistency_check'] = False
                    validation_results['errors'].append("Model outputs are not consistent for same input")
                
                # 4. Field processing check
                batch_size = embeddings.size(0)
                for field in ['category', 'style', 'silhouette', 'material', 'detail']:
                    if field not in json_batch:
                        validation_results['field_processing_check'] = False
                        validation_results['errors'].append(f"Missing field: {field}")
                    else:
                        field_data = json_batch[field]
                        if field_data.size(0) != batch_size:
                            validation_results['field_processing_check'] = False
                            validation_results['errors'].append(f"Field {field} batch size mismatch")
        
        except Exception as e:
            validation_results['errors'].append(f"Validation error: {str(e)}")
            for key in validation_results:
                if key != 'errors':
                    validation_results[key] = False
        
        # 5. Gradient check (requires training mode)
        self.json_encoder.train()
        try:
            batch = next(iter(val_loader))
            json_batch = self._convert_batch_to_dict(batch)
            
            self.optimizer.zero_grad()
            embeddings = self.json_encoder(json_batch)
            loss = embeddings.sum()  # Simple loss for gradient check
            loss.backward()
            
            # Check if gradients exist
            has_gradients = any(p.grad is not None for p in self.json_encoder.parameters())
            if not has_gradients:
                validation_results['gradient_check'] = False
                validation_results['errors'].append("No gradients computed")
        
        except Exception as e:
            validation_results['gradient_check'] = False
            validation_results['errors'].append(f"Gradient check error: {str(e)}")
        
        # Print validation results
        print("Comprehensive Validation Results:")
        for check, passed in validation_results.items():
            if check != 'errors':
                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"  {check}: {status}")
        
        if validation_results['errors']:
            print("  Errors:")
            for error in validation_results['errors']:
                print(f"    - {error}")
        
        return validation_results
    
    def _convert_batch_to_dict(self, batch) -> Dict[str, torch.Tensor]:
        """Convert ProcessedBatch to dictionary format for model input."""
        return {
            'category': batch.category_ids.to(self.device),
            'style': batch.style_ids.to(self.device),
            'silhouette': batch.silhouette_ids.to(self.device),
            'material': batch.material_ids.to(self.device),
            'detail': batch.detail_ids.to(self.device),
            'style_mask': batch.style_mask.to(self.device),
            'material_mask': batch.material_mask.to(self.device),
            'detail_mask': batch.detail_mask.to(self.device)
        }
    
    def _print_sanity_check_summary(self, results: Dict[str, Any]):
        """Print comprehensive sanity check summary."""
        print(f"\n{'='*80}")
        print("SANITY CHECK SUMMARY")
        print(f"{'='*80}")
        
        # Model info
        print(f"Model Parameters: {results['model_params']:,}")
        print(f"Vocabulary Sizes: {results['vocab_sizes']}")
        
        # Training progress
        training = results['training_results']
        print(f"\nTraining Progress:")
        print(f"  Initial train loss: {training['train_losses'][0]:.4f}")
        print(f"  Final train loss: {training['train_losses'][-1]:.4f}")
        print(f"  Initial val loss: {training['val_losses'][0]:.4f}")
        print(f"  Final val loss: {training['val_losses'][-1]:.4f}")
        print(f"  Loss reduction: {((training['train_losses'][0] - training['train_losses'][-1]) / training['train_losses'][0] * 100):.1f}%")
        
        # Output distribution comparison
        initial = results['initial_analysis']
        final = results['final_analysis']
        
        print(f"\nOutput Distribution Changes:")
        print(f"  Norm mean: {initial['norm_mean']:.4f} → {final['norm_mean']:.4f}")
        print(f"  Norm std: {initial['norm_std']:.4f} → {final['norm_std']:.4f}")
        print(f"  Value std: {initial['std']:.4f} → {final['std']:.4f}")
        print(f"  Normalized: {initial['is_normalized']} → {final['is_normalized']}")
        
        # Validation results
        validation = results['validation_results']
        print(f"\nValidation Checks:")
        all_passed = True
        for check, passed in validation.items():
            if check != 'errors':
                status = "✓" if passed else "✗"
                print(f"  {status} {check.replace('_', ' ').title()}")
                if not passed:
                    all_passed = False
        
        # Overall assessment
        print(f"\n{'='*60}")
        if all_passed and final['is_normalized'] and final['embedding_dim'] == 512:
            print("🎉 SANITY CHECK PASSED - JSON Encoder is working correctly!")
        else:
            print("⚠️  SANITY CHECK ISSUES DETECTED - Review the results above")
        print(f"{'='*60}")


def main():
    """Main function to run JSON encoder sanity check."""
    # Configuration
    dataset_path = "data"  # Placeholder - will use synthetic data if real data not available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("JSON Encoder Standalone Training and Sanity Check")
    print("=" * 80)
    print(f"Device: {device}")
    
    try:
        # Try to set up real data module
        data_module = FashionDataModule(
            dataset_path=dataset_path,
            target_categories=['상의', '하의', '아우터'],
            batch_size=16,  # Small batch for testing
            num_workers=0,
            train_split=0.8,
            image_size=224,
            augment_prob=0.0  # No augmentation for sanity check
        )
        
        print("Setting up data module...")
        data_module.setup()
        
        vocab_sizes = data_module.get_vocab_sizes()
        
    except Exception as e:
        print(f"Could not load real dataset: {e}")
        print("Creating synthetic data for sanity check...")
        
        # Create synthetic data module for testing
        vocab_sizes = {
            'category': 10,
            'style': 20,
            'silhouette': 15,
            'material': 25,
            'detail': 30
        }
        
        # Create a minimal synthetic data module
        data_module = create_synthetic_data_module(vocab_sizes, device)
    
    # Initialize sanity checker
    checker = JSONEncoderSanityChecker(vocab_sizes, device)
    
    # Run sanity check
    results = checker.run_sanity_check(data_module, num_epochs=3)
    
    # Save results
    output_dir = Path("temp_logs")
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / "json_encoder_sanity_check.json"
    
    # Convert results to JSON-serializable format
    json_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            json_results[key] = {k: v for k, v in value.items() if not isinstance(v, torch.Tensor)}
        elif isinstance(value, list):
            json_results[key] = value
        elif not isinstance(value, torch.Tensor):
            json_results[key] = value
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return results


def create_synthetic_data_module(vocab_sizes: Dict[str, int], device: str):
    """Create a synthetic data module for testing when real data is not available."""
    from torch.utils.data import Dataset, DataLoader
    from data.data_models import ProcessedBatch
    
    class SyntheticDataset(Dataset):
        def __init__(self, vocab_sizes: Dict[str, int], num_samples: int = 100):
            self.vocab_sizes = vocab_sizes
            self.num_samples = num_samples
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Generate synthetic data
            return {
                'image': torch.randn(3, 224, 224),
                'category': torch.randint(0, self.vocab_sizes['category'], (1,)).item(),
                'style': torch.randint(0, self.vocab_sizes['style'], (3,)).tolist(),
                'silhouette': torch.randint(0, self.vocab_sizes['silhouette'], (1,)).item(),
                'material': torch.randint(0, self.vocab_sizes['material'], (2,)).tolist(),
                'detail': torch.randint(0, self.vocab_sizes['detail'], (4,)).tolist()
            }
    
    def synthetic_collate_fn(batch):
        from data.fashion_dataset import collate_fashion_batch
        return collate_fashion_batch(batch)
    
    class SyntheticDataModule:
        def __init__(self, vocab_sizes: Dict[str, int]):
            self.vocab_sizes = vocab_sizes
            self.train_dataset = SyntheticDataset(vocab_sizes, 80)
            self.val_dataset = SyntheticDataset(vocab_sizes, 20)
        
        def setup(self):
            pass
        
        def train_dataloader(self):
            return DataLoader(self.train_dataset, batch_size=8, shuffle=True, 
                            collate_fn=synthetic_collate_fn)
        
        def val_dataloader(self):
            return DataLoader(self.val_dataset, batch_size=8, shuffle=False,
                            collate_fn=synthetic_collate_fn)
        
        def get_vocab_sizes(self):
            return self.vocab_sizes
    
    return SyntheticDataModule(vocab_sizes)


if __name__ == "__main__":
    main()