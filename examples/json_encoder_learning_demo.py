"""
JSON Encoder Learning Demonstration.

This script demonstrates that the JSON Encoder can learn to differentiate between
different fashion metadata patterns and produce meaningful embeddings.
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.json_encoder import JSONEncoder
from data.data_models import ProcessedBatch


class JSONEncoderLearningDemo:
    """
    Demonstrates JSON Encoder's ability to learn meaningful patterns.
    
    This demo creates structured synthetic data with clear patterns and shows
    that the JSON encoder can learn to distinguish between different categories
    and produce consistent embeddings for similar items.
    """
    
    def __init__(self, device: str = 'cpu'):
        """Initialize the learning demo."""
        self.device = device
        
        # Define vocabulary sizes for structured synthetic data
        self.vocab_sizes = {
            'category': 5,      # 5 clear categories
            'style': 10,        # 10 style options
            'silhouette': 8,    # 8 silhouette types
            'material': 12,     # 12 material types
            'detail': 15        # 15 detail options
        }
        
        # Initialize model
        self.json_encoder = JSONEncoder(
            vocab_sizes=self.vocab_sizes,
            embedding_dim=128,
            hidden_dim=256,
            output_dim=512,
            dropout_rate=0.1
        ).to(device)
        
        # Training components
        self.optimizer = optim.Adam(self.json_encoder.parameters(), lr=1e-3)
        
        print(f"JSON Encoder Learning Demo initialized on {device}")
        print(f"Model parameters: {sum(p.numel() for p in self.json_encoder.parameters()):,}")
    
    def run_learning_demo(self, num_epochs: int = 10) -> Dict[str, Any]:
        """
        Run the learning demonstration.
        
        Args:
            num_epochs: Number of training epochs
            
        Returns:
            Dictionary containing demo results
        """
        print(f"\n{'='*80}")
        print("JSON ENCODER LEARNING DEMONSTRATION")
        print(f"{'='*80}")
        
        # Create structured synthetic data
        train_loader, val_loader = self._create_structured_data()
        
        # Phase 1: Analyze initial embeddings
        print(f"\n{'='*60}")
        print("PHASE 1: INITIAL EMBEDDING ANALYSIS")
        print(f"{'='*60}")
        
        initial_analysis = self._analyze_category_embeddings(val_loader, "Initial")
        
        # Phase 2: Train with contrastive-like objective
        print(f"\n{'='*60}")
        print("PHASE 2: CONTRASTIVE LEARNING SIMULATION")
        print(f"{'='*60}")
        
        training_results = self._train_with_contrastive_objective(train_loader, val_loader, num_epochs)
        
        # Phase 3: Analyze learned embeddings
        print(f"\n{'='*60}")
        print("PHASE 3: LEARNED EMBEDDING ANALYSIS")
        print(f"{'='*60}")
        
        final_analysis = self._analyze_category_embeddings(val_loader, "Final")
        
        # Phase 4: Test generalization
        print(f"\n{'='*60}")
        print("PHASE 4: GENERALIZATION TEST")
        print(f"{'='*60}")
        
        generalization_results = self._test_generalization()
        
        # Compile results
        results = {
            'initial_analysis': initial_analysis,
            'training_results': training_results,
            'final_analysis': final_analysis,
            'generalization_results': generalization_results,
            'vocab_sizes': self.vocab_sizes
        }
        
        # Print summary
        self._print_demo_summary(results)
        
        return results
    
    def _create_structured_data(self):
        """Create structured synthetic data with clear patterns."""
        from torch.utils.data import Dataset, DataLoader
        
        class StructuredDataset(Dataset):
            def __init__(self, num_samples: int = 200):
                self.num_samples = num_samples
                self.data = self._generate_structured_data()
            
            def _generate_structured_data(self):
                """Generate data with clear category-based patterns."""
                data = []
                
                for i in range(self.num_samples):
                    # Create category-based patterns
                    category = i % 5  # 5 categories
                    
                    # Each category has preferred styles, materials, etc.
                    if category == 0:  # Formal wear
                        style = [1, 2]  # Formal styles
                        silhouette = 1  # Fitted
                        material = [3, 4]  # Premium materials
                        detail = [1, 2, 3]  # Minimal details
                    elif category == 1:  # Casual wear
                        style = [5, 6]  # Casual styles
                        silhouette = 2  # Relaxed
                        material = [1, 2]  # Comfortable materials
                        detail = [4, 5]  # Casual details
                    elif category == 2:  # Sports wear
                        style = [7, 8]  # Athletic styles
                        silhouette = 3  # Athletic fit
                        material = [5, 6]  # Performance materials
                        detail = [6, 7, 8]  # Functional details
                    elif category == 3:  # Outerwear
                        style = [3, 4]  # Outerwear styles
                        silhouette = 4  # Structured
                        material = [7, 8, 9]  # Weather-resistant
                        detail = [9, 10]  # Protective details
                    else:  # category == 4, Accessories
                        style = [9]  # Accessory styles
                        silhouette = 5  # Various
                        material = [10, 11]  # Diverse materials
                        detail = [11, 12, 13, 14]  # Decorative details
                    
                    # Add some noise to make it realistic
                    if np.random.random() < 0.1:  # 10% noise
                        style = [np.random.randint(1, 10)]
                        material = [np.random.randint(1, 12)]
                    
                    data.append({
                        'category': category,
                        'style': style,
                        'silhouette': silhouette,
                        'material': material,
                        'detail': detail
                    })
                
                return data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                item = self.data[idx]
                return {
                    'image': torch.randn(3, 224, 224),  # Dummy image
                    'category': item['category'],
                    'style': item['style'],
                    'silhouette': item['silhouette'],
                    'material': item['material'],
                    'detail': item['detail']
                }
        
        def structured_collate_fn(batch):
            from data.fashion_dataset import collate_fashion_batch
            return collate_fashion_batch(batch)
        
        # Create datasets
        train_dataset = StructuredDataset(160)  # 80% for training
        val_dataset = StructuredDataset(40)     # 20% for validation
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, 
                                collate_fn=structured_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False,
                              collate_fn=structured_collate_fn)
        
        print(f"Created structured datasets:")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def _train_with_contrastive_objective(self, train_loader: DataLoader, 
                                        val_loader: DataLoader, num_epochs: int):
        """Train with a contrastive-like objective to learn category distinctions."""
        self.json_encoder.train()
        
        train_losses = []
        val_losses = []
        category_separations = []
        
        for epoch in range(num_epochs):
            # Training phase
            epoch_train_loss = 0.0
            num_train_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                json_batch = self._convert_batch_to_dict(batch)
                
                self.optimizer.zero_grad()
                embeddings = self.json_encoder(json_batch)
                
                # Contrastive-like loss: items of same category should be similar
                loss = self._compute_category_contrastive_loss(embeddings, json_batch['category'])
                
                loss.backward()
                self.optimizer.step()
                
                epoch_train_loss += loss.item()
                num_train_batches += 1
            
            avg_train_loss = epoch_train_loss / num_train_batches
            train_losses.append(avg_train_loss)
            
            # Validation phase
            val_loss, category_sep = self._validate_contrastive(val_loader)
            val_losses.append(val_loss)
            category_separations.append(category_sep)
            
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {avg_train_loss:.4f}, "
                  f"Val Loss = {val_loss:.4f}, Category Separation = {category_sep:.4f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'category_separations': category_separations
        }
    
    def _compute_category_contrastive_loss(self, embeddings: torch.Tensor, 
                                         categories: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss based on category similarity."""
        batch_size = embeddings.size(0)
        
        # Compute pairwise similarities
        similarities = torch.matmul(embeddings, embeddings.T)
        
        # Create positive/negative masks
        category_matrix = categories.unsqueeze(1) == categories.unsqueeze(0)
        positive_mask = category_matrix.float()
        negative_mask = 1.0 - positive_mask
        
        # Remove diagonal (self-similarity)
        eye = torch.eye(batch_size, device=embeddings.device)
        positive_mask = positive_mask - eye
        
        # Contrastive loss: maximize similarity for same category, minimize for different
        positive_loss = -similarities * positive_mask
        negative_loss = similarities * negative_mask
        
        # Average over valid pairs
        pos_count = positive_mask.sum()
        neg_count = negative_mask.sum()
        
        if pos_count > 0:
            positive_loss = positive_loss.sum() / pos_count
        else:
            positive_loss = torch.tensor(0.0, device=embeddings.device)
        
        if neg_count > 0:
            negative_loss = negative_loss.sum() / neg_count
        else:
            negative_loss = torch.tensor(0.0, device=embeddings.device)
        
        return positive_loss + negative_loss
    
    def _validate_contrastive(self, val_loader: DataLoader):
        """Validate contrastive learning progress."""
        self.json_encoder.eval()
        
        total_loss = 0.0
        all_embeddings = []
        all_categories = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                json_batch = self._convert_batch_to_dict(batch)
                embeddings = self.json_encoder(json_batch)
                
                loss = self._compute_category_contrastive_loss(embeddings, json_batch['category'])
                total_loss += loss.item()
                
                all_embeddings.append(embeddings.cpu())
                all_categories.append(json_batch['category'].cpu())
                num_batches += 1
        
        # Compute category separation metric
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_categories = torch.cat(all_categories, dim=0)
        
        category_separation = self._compute_category_separation(all_embeddings, all_categories)
        
        self.json_encoder.train()
        return total_loss / num_batches, category_separation
    
    def _compute_category_separation(self, embeddings: torch.Tensor, 
                                   categories: torch.Tensor) -> float:
        """Compute how well categories are separated in embedding space."""
        unique_categories = torch.unique(categories)
        
        intra_category_distances = []
        inter_category_distances = []
        
        for cat in unique_categories:
            cat_mask = categories == cat
            cat_embeddings = embeddings[cat_mask]
            
            if cat_embeddings.size(0) > 1:
                # Intra-category distances (should be small)
                cat_similarities = torch.matmul(cat_embeddings, cat_embeddings.T)
                cat_distances = 1.0 - cat_similarities
                # Remove diagonal
                mask = ~torch.eye(cat_embeddings.size(0), dtype=bool)
                intra_distances = cat_distances[mask]
                intra_category_distances.append(intra_distances.mean().item())
            
            # Inter-category distances (should be large)
            other_mask = categories != cat
            if other_mask.sum() > 0:
                other_embeddings = embeddings[other_mask]
                inter_similarities = torch.matmul(cat_embeddings, other_embeddings.T)
                inter_distances = 1.0 - inter_similarities
                inter_category_distances.append(inter_distances.mean().item())
        
        if intra_category_distances and inter_category_distances:
            avg_intra = np.mean(intra_category_distances)
            avg_inter = np.mean(inter_category_distances)
            # Separation score: higher is better (inter > intra)
            separation = avg_inter - avg_intra
            return separation
        
        return 0.0
    
    def _analyze_category_embeddings(self, val_loader: DataLoader, phase_name: str):
        """Analyze embeddings by category."""
        self.json_encoder.eval()
        
        category_embeddings = {i: [] for i in range(5)}
        
        with torch.no_grad():
            for batch in val_loader:
                json_batch = self._convert_batch_to_dict(batch)
                embeddings = self.json_encoder(json_batch)
                categories = json_batch['category']
                
                for i, cat in enumerate(categories):
                    category_embeddings[cat.item()].append(embeddings[i].cpu())
        
        # Compute category statistics
        category_stats = {}
        for cat, embs in category_embeddings.items():
            if embs:
                embs_tensor = torch.stack(embs)
                category_stats[cat] = {
                    'count': len(embs),
                    'mean_norm': torch.norm(embs_tensor, dim=-1).mean().item(),
                    'std_norm': torch.norm(embs_tensor, dim=-1).std().item(),
                    'centroid': embs_tensor.mean(dim=0)
                }
        
        # Compute inter-category similarities
        inter_similarities = {}
        for cat1 in category_stats:
            for cat2 in category_stats:
                if cat1 < cat2:  # Avoid duplicates
                    sim = torch.cosine_similarity(
                        category_stats[cat1]['centroid'].unsqueeze(0),
                        category_stats[cat2]['centroid'].unsqueeze(0)
                    ).item()
                    inter_similarities[f"{cat1}-{cat2}"] = sim
        
        analysis = {
            'phase': phase_name,
            'category_stats': {k: {sk: sv for sk, sv in v.items() if sk != 'centroid'} 
                             for k, v in category_stats.items()},
            'inter_similarities': inter_similarities,
            'avg_inter_similarity': np.mean(list(inter_similarities.values())) if inter_similarities else 0.0
        }
        
        print(f"{phase_name} Category Analysis:")
        for cat, stats in analysis['category_stats'].items():
            print(f"  Category {cat}: {stats['count']} samples, "
                  f"norm = {stats['mean_norm']:.4f} ± {stats['std_norm']:.4f}")
        print(f"  Average inter-category similarity: {analysis['avg_inter_similarity']:.4f}")
        
        self.json_encoder.train()
        return analysis
    
    def _test_generalization(self):
        """Test model's ability to generalize to new data patterns."""
        self.json_encoder.eval()
        
        # Create test cases with known patterns
        test_cases = [
            # Formal wear pattern
            {'category': 0, 'style': [1, 2], 'silhouette': 1, 'material': [3, 4], 'detail': [1, 2]},
            # Casual wear pattern  
            {'category': 1, 'style': [5, 6], 'silhouette': 2, 'material': [1, 2], 'detail': [4, 5]},
            # Mixed pattern (should be between categories)
            {'category': 0, 'style': [5, 6], 'silhouette': 1, 'material': [1, 2], 'detail': [1, 2]},
        ]
        
        embeddings = []
        with torch.no_grad():
            for case in test_cases:
                # Convert to batch format
                batch_data = {
                    'category': torch.tensor([case['category']], device=self.device),
                    'style': torch.tensor([case['style']], device=self.device),
                    'silhouette': torch.tensor([case['silhouette']], device=self.device),
                    'material': torch.tensor([case['material']], device=self.device),
                    'detail': torch.tensor([case['detail']], device=self.device),
                    'style_mask': torch.ones(1, len(case['style']), device=self.device),
                    'material_mask': torch.ones(1, len(case['material']), device=self.device),
                    'detail_mask': torch.ones(1, len(case['detail']), device=self.device)
                }
                
                embedding = self.json_encoder(batch_data)
                embeddings.append(embedding.cpu())
        
        # Compute similarities between test cases
        embeddings = torch.cat(embeddings, dim=0)
        similarities = torch.matmul(embeddings, embeddings.T)
        
        results = {
            'formal_casual_similarity': similarities[0, 1].item(),
            'formal_mixed_similarity': similarities[0, 2].item(),
            'casual_mixed_similarity': similarities[1, 2].item(),
            'embeddings_norm': torch.norm(embeddings, dim=-1).tolist()
        }
        
        print("Generalization Test Results:")
        print(f"  Formal-Casual similarity: {results['formal_casual_similarity']:.4f}")
        print(f"  Formal-Mixed similarity: {results['formal_mixed_similarity']:.4f}")
        print(f"  Casual-Mixed similarity: {results['casual_mixed_similarity']:.4f}")
        print(f"  All embeddings properly normalized: {all(abs(norm - 1.0) < 1e-3 for norm in results['embeddings_norm'])}")
        
        return results
    
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
    
    def _print_demo_summary(self, results: Dict[str, Any]):
        """Print comprehensive demo summary."""
        print(f"\n{'='*80}")
        print("LEARNING DEMONSTRATION SUMMARY")
        print(f"{'='*80}")
        
        # Training progress
        training = results['training_results']
        print(f"Training Progress:")
        print(f"  Initial train loss: {training['train_losses'][0]:.4f}")
        print(f"  Final train loss: {training['train_losses'][-1]:.4f}")
        print(f"  Loss improvement: {((training['train_losses'][0] - training['train_losses'][-1]) / training['train_losses'][0] * 100):.1f}%")
        print(f"  Category separation improvement: {training['category_separations'][0]:.4f} → {training['category_separations'][-1]:.4f}")
        
        # Category analysis comparison
        initial = results['initial_analysis']
        final = results['final_analysis']
        
        print(f"\nCategory Separation Analysis:")
        print(f"  Initial avg inter-category similarity: {initial['avg_inter_similarity']:.4f}")
        print(f"  Final avg inter-category similarity: {final['avg_inter_similarity']:.4f}")
        print(f"  Separation improvement: {initial['avg_inter_similarity'] - final['avg_inter_similarity']:.4f}")
        
        # Generalization results
        gen = results['generalization_results']
        print(f"\nGeneralization Test:")
        print(f"  Model can distinguish formal vs casual: {gen['formal_casual_similarity'] < 0.8}")
        print(f"  Mixed pattern behaves as expected: {gen['formal_mixed_similarity'] > gen['formal_casual_similarity']}")
        print(f"  All outputs properly normalized: {all(abs(norm - 1.0) < 1e-3 for norm in gen['embeddings_norm'])}")
        
        # Overall assessment
        print(f"\n{'='*60}")
        learning_success = (
            training['train_losses'][-1] < training['train_losses'][0] and
            final['avg_inter_similarity'] < initial['avg_inter_similarity'] and
            gen['formal_casual_similarity'] < 0.9
        )
        
        if learning_success:
            print("🎉 LEARNING DEMONSTRATION SUCCESSFUL!")
            print("   JSON Encoder can learn meaningful patterns from metadata!")
        else:
            print("⚠️  LEARNING DEMONSTRATION SHOWS MIXED RESULTS")
            print("   Review the detailed results above")
        print(f"{'='*60}")


def main():
    """Main function to run JSON encoder learning demo."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("JSON Encoder Learning Demonstration")
    print("=" * 80)
    print(f"Device: {device}")
    
    # Initialize demo
    demo = JSONEncoderLearningDemo(device)
    
    # Run demo
    results = demo.run_learning_demo(num_epochs=8)
    
    # Save results
    output_dir = Path("temp_logs")
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / "json_encoder_learning_demo.json"
    
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


if __name__ == "__main__":
    main()