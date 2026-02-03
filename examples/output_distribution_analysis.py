"""
JSON Encoder Output Distribution Analysis.

This script performs comprehensive analysis of the JSON Encoder's output distribution
to validate that it produces meaningful, well-distributed embeddings.
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import json
from typing import Dict, Any, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.json_encoder import JSONEncoder
from data.data_models import ProcessedBatch


class OutputDistributionAnalyzer:
    """
    Comprehensive analyzer for JSON Encoder output distributions.
    
    This analyzer performs various statistical tests and visualizations
    to validate that the JSON encoder produces meaningful embeddings.
    """
    
    def __init__(self, device: str = 'cpu'):
        """Initialize the output distribution analyzer."""
        self.device = device
        
        # Define vocabulary sizes
        self.vocab_sizes = {
            'category': 8,
            'style': 15,
            'silhouette': 10,
            'material': 20,
            'detail': 25
        }
        
        # Initialize model
        self.json_encoder = JSONEncoder(
            vocab_sizes=self.vocab_sizes,
            embedding_dim=128,
            hidden_dim=256,
            output_dim=512,
            dropout_rate=0.1
        ).to(device)
        
        print(f"Output Distribution Analyzer initialized on {device}")
        print(f"Model parameters: {sum(p.numel() for p in self.json_encoder.parameters()):,}")
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive output distribution analysis.
        
        Returns:
            Dictionary containing all analysis results
        """
        print(f"\n{'='*80}")
        print("COMPREHENSIVE OUTPUT DISTRIBUTION ANALYSIS")
        print(f"{'='*80}")
        
        # Phase 1: Basic distribution analysis
        print(f"\n{'='*60}")
        print("PHASE 1: BASIC DISTRIBUTION ANALYSIS")
        print(f"{'='*60}")
        
        basic_analysis = self._analyze_basic_distribution()
        
        # Phase 2: Field-specific analysis
        print(f"\n{'='*60}")
        print("PHASE 2: FIELD-SPECIFIC ANALYSIS")
        print(f"{'='*60}")
        
        field_analysis = self._analyze_field_specific_patterns()
        
        # Phase 3: Embedding space properties
        print(f"\n{'='*60}")
        print("PHASE 3: EMBEDDING SPACE PROPERTIES")
        print(f"{'='*60}")
        
        space_analysis = self._analyze_embedding_space_properties()
        
        # Phase 4: Consistency and stability
        print(f"\n{'='*60}")
        print("PHASE 4: CONSISTENCY AND STABILITY")
        print(f"{'='*60}")
        
        consistency_analysis = self._analyze_consistency_stability()
        
        # Phase 5: Dimensionality and coverage
        print(f"\n{'='*60}")
        print("PHASE 5: DIMENSIONALITY AND COVERAGE")
        print(f"{'='*60}")
        
        dimensionality_analysis = self._analyze_dimensionality_coverage()
        
        # Compile results
        results = {
            'basic_analysis': basic_analysis,
            'field_analysis': field_analysis,
            'space_analysis': space_analysis,
            'consistency_analysis': consistency_analysis,
            'dimensionality_analysis': dimensionality_analysis,
            'vocab_sizes': self.vocab_sizes,
            'model_info': {
                'parameters': sum(p.numel() for p in self.json_encoder.parameters()),
                'embedding_dim': 128,
                'hidden_dim': 256,
                'output_dim': 512
            }
        }
        
        # Print comprehensive summary
        self._print_comprehensive_summary(results)
        
        return results
    
    def _analyze_basic_distribution(self) -> Dict[str, Any]:
        """Analyze basic statistical properties of output distribution."""
        self.json_encoder.eval()
        
        # Generate diverse test data
        test_data = self._generate_diverse_test_data(1000)
        all_embeddings = []
        
        with torch.no_grad():
            for batch in test_data:
                embeddings = self.json_encoder(batch)
                all_embeddings.append(embeddings.cpu())
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        # Compute basic statistics
        norms = torch.norm(all_embeddings, dim=-1)
        
        analysis = {
            'num_samples': all_embeddings.size(0),
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
            'zero_dimensions': (all_embeddings == 0).all(dim=0).sum().item(),
            'active_dimensions': (all_embeddings.abs() > 1e-6).any(dim=0).sum().item()
        }
        
        # Compute percentiles
        flat_embeddings = all_embeddings.flatten()
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        analysis['percentiles'] = {
            f'p{p}': torch.quantile(flat_embeddings, p/100).item() 
            for p in percentiles
        }
        
        print(f"Basic Distribution Analysis:")
        print(f"  Samples: {analysis['num_samples']}")
        print(f"  Embedding dimension: {analysis['embedding_dim']}")
        print(f"  Value range: [{analysis['min']:.4f}, {analysis['max']:.4f}]")
        print(f"  Mean: {analysis['mean']:.4f}, Std: {analysis['std']:.4f}")
        print(f"  Norm range: [{analysis['norm_min']:.4f}, {analysis['norm_max']:.4f}]")
        print(f"  Properly normalized: {analysis['is_normalized']}")
        print(f"  Active dimensions: {analysis['active_dimensions']}/{analysis['embedding_dim']}")
        print(f"  Zero dimensions: {analysis['zero_dimensions']}")
        
        return analysis
    
    def _analyze_field_specific_patterns(self) -> Dict[str, Any]:
        """Analyze how different fields affect the output distribution."""
        self.json_encoder.eval()
        
        field_analysis = {}
        
        # Test each field's impact
        for field_name in ['category', 'style', 'silhouette', 'material', 'detail']:
            print(f"Analyzing field: {field_name}")
            
            # Generate data varying only this field
            field_data = self._generate_field_specific_data(field_name, 200)
            field_embeddings = []
            field_values = []
            
            with torch.no_grad():
                for batch, values in field_data:
                    embeddings = self.json_encoder(batch)
                    field_embeddings.append(embeddings.cpu())
                    field_values.extend(values)
            
            field_embeddings = torch.cat(field_embeddings, dim=0)
            
            # Analyze field-specific patterns
            unique_values = list(set(field_values))
            value_embeddings = {}
            
            for value in unique_values:
                value_mask = torch.tensor([v == value for v in field_values])
                if value_mask.sum() > 0:
                    value_embs = field_embeddings[value_mask]
                    value_embeddings[value] = {
                        'count': value_mask.sum().item(),
                        'mean_embedding': value_embs.mean(dim=0),
                        'std': value_embs.std().item(),
                        'norm_mean': torch.norm(value_embs, dim=-1).mean().item()
                    }
            
            # Compute inter-value similarities
            inter_similarities = []
            for i, val1 in enumerate(unique_values):
                for j, val2 in enumerate(unique_values):
                    if i < j and val1 in value_embeddings and val2 in value_embeddings:
                        sim = torch.cosine_similarity(
                            value_embeddings[val1]['mean_embedding'].unsqueeze(0),
                            value_embeddings[val2]['mean_embedding'].unsqueeze(0)
                        ).item()
                        inter_similarities.append(sim)
            
            field_analysis[field_name] = {
                'unique_values': len(unique_values),
                'avg_inter_similarity': np.mean(inter_similarities) if inter_similarities else 0.0,
                'std_inter_similarity': np.std(inter_similarities) if inter_similarities else 0.0,
                'value_stats': {k: {sk: sv for sk, sv in v.items() if sk != 'mean_embedding'} 
                              for k, v in value_embeddings.items()}
            }
            
            print(f"  Unique values: {field_analysis[field_name]['unique_values']}")
            print(f"  Avg inter-similarity: {field_analysis[field_name]['avg_inter_similarity']:.4f}")
        
        return field_analysis
    
    def _analyze_embedding_space_properties(self) -> Dict[str, Any]:
        """Analyze geometric properties of the embedding space."""
        self.json_encoder.eval()
        
        # Generate test data
        test_data = self._generate_diverse_test_data(500)
        all_embeddings = []
        
        with torch.no_grad():
            for batch in test_data:
                embeddings = self.json_encoder(batch)
                all_embeddings.append(embeddings.cpu())
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        # Compute pairwise similarities
        similarities = torch.matmul(all_embeddings, all_embeddings.T)
        
        # Remove diagonal (self-similarities)
        mask = ~torch.eye(similarities.size(0), dtype=bool)
        off_diagonal_sims = similarities[mask]
        
        # Compute space properties
        analysis = {
            'avg_pairwise_similarity': off_diagonal_sims.mean().item(),
            'std_pairwise_similarity': off_diagonal_sims.std().item(),
            'min_pairwise_similarity': off_diagonal_sims.min().item(),
            'max_pairwise_similarity': off_diagonal_sims.max().item(),
            'similarity_range': off_diagonal_sims.max().item() - off_diagonal_sims.min().item()
        }
        
        # Compute effective dimensionality (approximate)
        U, S, V = torch.svd(all_embeddings)
        total_variance = (S ** 2).sum()
        cumulative_variance = torch.cumsum(S ** 2, dim=0) / total_variance
        
        # Find dimensions needed for 90% and 95% variance
        dim_90 = (cumulative_variance >= 0.9).nonzero()[0].item() + 1
        dim_95 = (cumulative_variance >= 0.95).nonzero()[0].item() + 1
        
        analysis.update({
            'effective_dim_90': dim_90,
            'effective_dim_95': dim_95,
            'singular_values_top10': S[:10].tolist(),
            'variance_explained_top10': (S[:10] ** 2 / total_variance).tolist()
        })
        
        print(f"Embedding Space Properties:")
        print(f"  Avg pairwise similarity: {analysis['avg_pairwise_similarity']:.4f}")
        print(f"  Similarity range: [{analysis['min_pairwise_similarity']:.4f}, {analysis['max_pairwise_similarity']:.4f}]")
        print(f"  Effective dimensionality (90% var): {analysis['effective_dim_90']}")
        print(f"  Effective dimensionality (95% var): {analysis['effective_dim_95']}")
        
        return analysis
    
    def _analyze_consistency_stability(self) -> Dict[str, Any]:
        """Analyze consistency and stability of embeddings."""
        self.json_encoder.eval()
        
        # Test 1: Same input consistency
        test_batch = self._generate_diverse_test_data(1)[0]
        
        embeddings1 = self.json_encoder(test_batch)
        embeddings2 = self.json_encoder(test_batch)
        
        consistency_error = torch.norm(embeddings1 - embeddings2).item()
        
        # Test 2: Small perturbation stability
        # Create slightly perturbed version (change one field value)
        perturbed_batch = {k: v.clone() for k, v in test_batch.items()}
        if perturbed_batch['category'].size(0) > 0:
            # Change one category value
            perturbed_batch['category'][0] = (perturbed_batch['category'][0] + 1) % self.vocab_sizes['category']
        
        original_emb = self.json_encoder(test_batch)
        perturbed_emb = self.json_encoder(perturbed_batch)
        
        perturbation_distance = torch.norm(original_emb - perturbed_emb, dim=-1).mean().item()
        perturbation_similarity = torch.cosine_similarity(original_emb, perturbed_emb).mean().item()
        
        # Test 3: Dropout consistency (if model has dropout)
        self.json_encoder.train()  # Enable dropout
        dropout_embeddings = []
        for _ in range(5):
            emb = self.json_encoder(test_batch)
            dropout_embeddings.append(emb.cpu())
        
        dropout_embeddings = torch.stack(dropout_embeddings)
        dropout_std = dropout_embeddings.std(dim=0).mean().item()
        
        self.json_encoder.eval()  # Back to eval mode
        
        analysis = {
            'same_input_consistency_error': consistency_error,
            'perturbation_distance': perturbation_distance,
            'perturbation_similarity': perturbation_similarity,
            'dropout_variability': dropout_std,
            'is_deterministic': consistency_error < 1e-6,
            'is_stable_to_perturbation': perturbation_distance < 2.0  # Reasonable threshold
        }
        
        print(f"Consistency and Stability Analysis:")
        print(f"  Same input consistency error: {analysis['same_input_consistency_error']:.6f}")
        print(f"  Is deterministic: {analysis['is_deterministic']}")
        print(f"  Perturbation distance: {analysis['perturbation_distance']:.4f}")
        print(f"  Perturbation similarity: {analysis['perturbation_similarity']:.4f}")
        print(f"  Dropout variability: {analysis['dropout_variability']:.4f}")
        
        return analysis
    
    def _analyze_dimensionality_coverage(self) -> Dict[str, Any]:
        """Analyze how well the model uses the embedding space."""
        self.json_encoder.eval()
        
        # Generate diverse test data
        test_data = self._generate_diverse_test_data(1000)
        all_embeddings = []
        
        with torch.no_grad():
            for batch in test_data:
                embeddings = self.json_encoder(batch)
                all_embeddings.append(embeddings.cpu())
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        # Analyze dimension usage
        dim_means = all_embeddings.mean(dim=0)
        dim_stds = all_embeddings.std(dim=0)
        dim_ranges = all_embeddings.max(dim=0)[0] - all_embeddings.min(dim=0)[0]
        
        # Find unused dimensions
        unused_dims = (dim_stds < 1e-6).sum().item()
        low_variance_dims = (dim_stds < 0.01).sum().item()
        
        # Compute coverage metrics
        analysis = {
            'unused_dimensions': unused_dims,
            'low_variance_dimensions': low_variance_dims,
            'avg_dimension_std': dim_stds.mean().item(),
            'min_dimension_std': dim_stds.min().item(),
            'max_dimension_std': dim_stds.max().item(),
            'avg_dimension_range': dim_ranges.mean().item(),
            'dimension_usage_efficiency': (512 - unused_dims) / 512,
            'dimension_balance': 1.0 - (dim_stds.std() / dim_stds.mean()).item()  # Lower is more balanced
        }
        
        # Compute correlation between dimensions
        correlation_matrix = torch.corrcoef(all_embeddings.T)
        off_diagonal_corr = correlation_matrix[~torch.eye(512, dtype=bool)]
        
        analysis.update({
            'avg_dimension_correlation': off_diagonal_corr.mean().item(),
            'max_dimension_correlation': off_diagonal_corr.abs().max().item(),
            'high_correlation_pairs': (off_diagonal_corr.abs() > 0.8).sum().item()
        })
        
        print(f"Dimensionality and Coverage Analysis:")
        print(f"  Unused dimensions: {analysis['unused_dimensions']}/512")
        print(f"  Low variance dimensions: {analysis['low_variance_dimensions']}/512")
        print(f"  Dimension usage efficiency: {analysis['dimension_usage_efficiency']:.4f}")
        print(f"  Dimension balance: {analysis['dimension_balance']:.4f}")
        print(f"  Avg dimension correlation: {analysis['avg_dimension_correlation']:.4f}")
        print(f"  High correlation pairs: {analysis['high_correlation_pairs']}")
        
        return analysis
    
    def _generate_diverse_test_data(self, num_samples: int) -> List[Dict[str, torch.Tensor]]:
        """Generate diverse test data for analysis."""
        batches = []
        batch_size = 32
        
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            
            # Generate random data
            category = torch.randint(0, self.vocab_sizes['category'], (current_batch_size,))
            style = torch.randint(0, self.vocab_sizes['style'], (current_batch_size, 3))
            silhouette = torch.randint(0, self.vocab_sizes['silhouette'], (current_batch_size,))
            material = torch.randint(0, self.vocab_sizes['material'], (current_batch_size, 2))
            detail = torch.randint(0, self.vocab_sizes['detail'], (current_batch_size, 4))
            
            batch = {
                'category': category.to(self.device),
                'style': style.to(self.device),
                'silhouette': silhouette.to(self.device),
                'material': material.to(self.device),
                'detail': detail.to(self.device),
                'style_mask': torch.ones_like(style, dtype=torch.float).to(self.device),
                'material_mask': torch.ones_like(material, dtype=torch.float).to(self.device),
                'detail_mask': torch.ones_like(detail, dtype=torch.float).to(self.device)
            }
            
            batches.append(batch)
        
        return batches
    
    def _generate_field_specific_data(self, field_name: str, num_samples: int) -> List[Tuple[Dict[str, torch.Tensor], List]]:
        """Generate data varying only the specified field."""
        batches = []
        batch_size = 16
        
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            
            # Fixed values for other fields
            category = torch.zeros(current_batch_size, dtype=torch.long)
            style = torch.ones(current_batch_size, 2, dtype=torch.long)
            silhouette = torch.zeros(current_batch_size, dtype=torch.long)
            material = torch.ones(current_batch_size, 2, dtype=torch.long)
            detail = torch.ones(current_batch_size, 3, dtype=torch.long)
            
            # Vary the target field
            if field_name == 'category':
                category = torch.randint(0, self.vocab_sizes['category'], (current_batch_size,))
                field_values = category.tolist()
            elif field_name == 'style':
                style = torch.randint(0, self.vocab_sizes['style'], (current_batch_size, 2))
                field_values = [tuple(row.tolist()) for row in style]
            elif field_name == 'silhouette':
                silhouette = torch.randint(0, self.vocab_sizes['silhouette'], (current_batch_size,))
                field_values = silhouette.tolist()
            elif field_name == 'material':
                material = torch.randint(0, self.vocab_sizes['material'], (current_batch_size, 2))
                field_values = [tuple(row.tolist()) for row in material]
            elif field_name == 'detail':
                detail = torch.randint(0, self.vocab_sizes['detail'], (current_batch_size, 3))
                field_values = [tuple(row.tolist()) for row in detail]
            
            batch = {
                'category': category.to(self.device),
                'style': style.to(self.device),
                'silhouette': silhouette.to(self.device),
                'material': material.to(self.device),
                'detail': detail.to(self.device),
                'style_mask': torch.ones_like(style, dtype=torch.float).to(self.device),
                'material_mask': torch.ones_like(material, dtype=torch.float).to(self.device),
                'detail_mask': torch.ones_like(detail, dtype=torch.float).to(self.device)
            }
            
            batches.append((batch, field_values))
        
        return batches
    
    def _print_comprehensive_summary(self, results: Dict[str, Any]):
        """Print comprehensive analysis summary."""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE OUTPUT DISTRIBUTION ANALYSIS SUMMARY")
        print(f"{'='*80}")
        
        # Model info
        model_info = results['model_info']
        print(f"Model Information:")
        print(f"  Parameters: {model_info['parameters']:,}")
        print(f"  Architecture: {model_info['embedding_dim']} → {model_info['hidden_dim']} → {model_info['output_dim']}")
        
        # Basic distribution
        basic = results['basic_analysis']
        print(f"\nBasic Distribution:")
        print(f"  ✓ Correct output dimension: {basic['embedding_dim'] == 512}")
        print(f"  ✓ Properly normalized: {basic['is_normalized']}")
        print(f"  ✓ Active dimensions: {basic['active_dimensions']}/512 ({basic['active_dimensions']/512*100:.1f}%)")
        print(f"  ✓ Value distribution: mean={basic['mean']:.4f}, std={basic['std']:.4f}")
        
        # Field analysis
        field = results['field_analysis']
        print(f"\nField-Specific Analysis:")
        for field_name, field_data in field.items():
            print(f"  {field_name}: {field_data['unique_values']} values, "
                  f"avg similarity={field_data['avg_inter_similarity']:.4f}")
        
        # Space properties
        space = results['space_analysis']
        print(f"\nEmbedding Space Properties:")
        print(f"  ✓ Similarity range: [{space['min_pairwise_similarity']:.4f}, {space['max_pairwise_similarity']:.4f}]")
        print(f"  ✓ Effective dimensionality: {space['effective_dim_90']} (90% var), {space['effective_dim_95']} (95% var)")
        print(f"  ✓ Space utilization: {space['similarity_range']:.4f} range")
        
        # Consistency
        consistency = results['consistency_analysis']
        print(f"\nConsistency and Stability:")
        print(f"  ✓ Deterministic: {consistency['is_deterministic']}")
        print(f"  ✓ Stable to perturbation: {consistency['is_stable_to_perturbation']}")
        print(f"  ✓ Dropout variability: {consistency['dropout_variability']:.4f}")
        
        # Dimensionality
        dim = results['dimensionality_analysis']
        print(f"\nDimensionality and Coverage:")
        print(f"  ✓ Usage efficiency: {dim['dimension_usage_efficiency']:.4f}")
        print(f"  ✓ Dimension balance: {dim['dimension_balance']:.4f}")
        print(f"  ✓ Low correlation: {dim['avg_dimension_correlation']:.4f}")
        
        # Overall assessment
        print(f"\n{'='*60}")
        
        # Check key criteria
        criteria_passed = [
            basic['embedding_dim'] == 512,
            basic['is_normalized'],
            basic['active_dimensions'] > 400,  # At least 80% dimensions active
            consistency['is_deterministic'],
            dim['dimension_usage_efficiency'] > 0.8,
            abs(basic['mean']) < 0.1,  # Reasonable mean
            basic['std'] > 0.01  # Some variation
        ]
        
        if all(criteria_passed):
            print("🎉 OUTPUT DISTRIBUTION ANALYSIS PASSED!")
            print("   JSON Encoder produces well-distributed, meaningful embeddings!")
        else:
            print("⚠️  OUTPUT DISTRIBUTION ANALYSIS SHOWS SOME ISSUES")
            print("   Review the detailed results above")
        
        print(f"Criteria passed: {sum(criteria_passed)}/{len(criteria_passed)}")
        print(f"{'='*60}")


def main():
    """Main function to run output distribution analysis."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("JSON Encoder Output Distribution Analysis")
    print("=" * 80)
    print(f"Device: {device}")
    
    # Initialize analyzer
    analyzer = OutputDistributionAnalyzer(device)
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis()
    
    # Save results
    output_dir = Path("temp_logs")
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / "output_distribution_analysis.json"
    
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