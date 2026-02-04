#!/usr/bin/env python3
"""
Test improved training with enriched JSON fields

This script tests the impact of the JSON field recovery fix on model performance.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPVisionModel
from data.fashion_dataset import FashionDataModule
from models.json_encoder import JSONEncoder
from models.contrastive_learner import ContrastiveLearner
from utils.training_monitor import TrainingMonitor
from training.trainer import FashionTrainer
import json
from pathlib import Path


def test_improved_model():
    """Test model with improved JSON field extraction"""
    print("🧪 Testing improved model with enriched JSON fields...")
    
    # 데이터 설정 (배치 크기 맞춤)
    data_module = FashionDataModule(
        dataset_path="C:/sample/라벨링데이터",
        target_categories=['레트로', '로맨틱', '리조트'],
        batch_size=32  # config와 동일하게 설정
    )
    data_module.setup()
    
    # Print vocabulary sizes to confirm improvement
    vocab_sizes = data_module.dataset_loader.get_vocab_sizes()
    print(f"📚 Vocabulary sizes: {vocab_sizes}")
    
    # 설정 생성
    from utils.config import TrainingConfig
    config = TrainingConfig(
        learning_rate=3e-4,
        temperature=0.07,
        batch_size=32,  # 더 작은 배치 크기로 변경
        max_epochs=3,
        embedding_dim=128,
        hidden_dim=256,
        output_dim=512,
        dropout_rate=0.1
    )
    
    # Setup trainer
    trainer = FashionTrainer(
        config=config,
        vocab_sizes=vocab_sizes,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Initialize training monitor
    monitor = TrainingMonitor()
    
    print(f"🚀 Starting quick test training (3 epochs)...")
    print(f"📊 Expected improvement from enriched JSON fields:")
    print(f"   - Style vocab: 1 → {vocab_sizes['style']}")
    print(f"   - Silhouette vocab: 1 → {vocab_sizes['silhouette']}")
    print(f"   - Material vocab: 1 → {vocab_sizes['material']}")
    print(f"   - Detail vocab: 1 → {vocab_sizes['detail']}")
    
    # Quick training test
    monitor.start_training("JSON Field Recovery Test", 3)
    
    # Train using the trainer's contrastive learning method
    results = trainer.train_contrastive_learning(
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        num_epochs=3
    )
    
    # Extract best metrics from results (convert tensors to float)
    best_metrics = results.get('final_metrics', {})
    if 'best_val_metrics' in results:
        best_metrics = results['best_val_metrics']
    
    # Convert any tensor values to float for JSON serialization
    for key, value in best_metrics.items():
        if hasattr(value, 'item'):  # Check if it's a tensor
            best_metrics[key] = value.item()
        elif hasattr(value, 'tolist'):  # Check if it's a numpy array
            best_metrics[key] = value.tolist()
    
    monitor.finish_training()
    
    # Save test results
    test_results = {
        'vocab_sizes': vocab_sizes,
        'best_metrics': best_metrics,
        'improvement_summary': {
            'vocab_improvement': {
                'style': f"1 → {vocab_sizes['style']} ({vocab_sizes['style']}x)",
                'silhouette': f"1 → {vocab_sizes['silhouette']} ({vocab_sizes['silhouette']}x)",
                'material': f"1 → {vocab_sizes['material']} ({vocab_sizes['material']}x)",
                'detail': f"1 → {vocab_sizes['detail']} ({vocab_sizes['detail']}x)"
            },
            'expected_performance_boost': "Substantial improvement expected in Top-5 accuracy and MRR"
        }
    }
    
    results_path = Path("results/json_field_recovery_test.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ JSON Field Recovery Test Complete!")
    print(f"📄 Results saved: {results_path}")
    print(f"🎯 Best Top-5 Accuracy: {best_metrics['top5_accuracy']:.2f}%")
    print(f"🎯 Best MRR: {best_metrics['mrr']:.4f}")
    
    return test_results


if __name__ == "__main__":
    test_improved_model()