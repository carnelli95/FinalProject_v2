#!/usr/bin/env python3
"""
Baseline v2 생성 및 v1과 비교

현재 학습된 모델을 기준으로 Baseline v2를 생성하고 v1과 성능 비교
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import torch

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from main import FashionEncoderSystem
from utils.config import TrainingConfig


def evaluate_current_model() -> Dict[str, Any]:
    """현재 모델 성능 평가"""
    print("🔍 현재 모델 성능 평가 중...")
    
    # 데이터셋 경로
    dataset_path = "C:/sample/라벨링데이터"
    
    # 현재 설정 (Temperature 0.1 기준)
    config = TrainingConfig()
    config.temperature = 0.1
    config.batch_size = 16
    config.max_epochs = 8
    
    try:
        # 시스템 초기화
        system = FashionEncoderSystem()
        system.config = config
        
        # 데이터 설정
        print("📁 데이터 설정 중...")
        system.setup_data(dataset_path)
        
        # 트레이너 설정
        print("🏋️ 트레이너 설정 중...")
        system.setup_trainer()
        
        # 현재 최고 성능 체크포인트 로드
        checkpoint_path = "checkpoints/baseline_v1_best_model.pt"
        if Path(checkpoint_path).exists():
            print(f"📦 Baseline v1 체크포인트 로드: {checkpoint_path}")
            system.trainer.load_checkpoint(checkpoint_path)
        else:
            checkpoint_path = "checkpoints/best_model.pt"
            if Path(checkpoint_path).exists():
                print(f"📦 현재 체크포인트 로드: {checkpoint_path}")
                system.trainer.load_checkpoint(checkpoint_path)
            else:
                print("⚠️ 체크포인트가 없습니다. 현재 모델 상태로 평가합니다.")
        
        # 평가 수행
        print("📊 모델 평가 중...")
        metrics = system.trainer._final_evaluation(system.data_module.val_dataloader())
        
        # 결과 정리
        evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'model_name': 'Fashion JSON Encoder Baseline v2',
            'configuration': {
                'temperature': config.temperature,
                'batch_size': config.batch_size,
                'epochs': config.max_epochs,
                'learning_rate': config.learning_rate,
                'dataset': 'K-Fashion 2,172 items',
                'class_distribution': {
                    '레트로': len([item for item in system.data_module.train_dataset.fashion_items + system.data_module.val_dataset.fashion_items if item.category == '레트로']),
                    '로맨틱': len([item for item in system.data_module.train_dataset.fashion_items + system.data_module.val_dataset.fashion_items if item.category == '로맨틱']),
                    '리조트': len([item for item in system.data_module.train_dataset.fashion_items + system.data_module.val_dataset.fashion_items if item.category == '리조트'])
                }
            },
            'final_performance': {
                'top1_accuracy': metrics.get('top1_accuracy', 0.0),
                'top5_accuracy': metrics.get('top5_accuracy', 0.0),
                'mrr': metrics.get('mean_reciprocal_rank', 0.0),
                'validation_loss': metrics.get('val_loss', 0.0),
                'positive_similarity': metrics.get('avg_positive_similarity', 0.0),
                'negative_similarity': metrics.get('avg_negative_similarity', 0.0)
            },
            'additional_metrics': {
                'recall_at_3': metrics.get('recall_at_3', 0.0),
                'recall_at_10': metrics.get('recall_at_10', 0.0),
                'recall_at_20': metrics.get('recall_at_20', 0.0)
            }
        }
        
        # 정리
        system.cleanup()
        
        return evaluation_results
        
    except Exception as e:
        print(f"❌ 평가 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_baseline_v1_results() -> Dict[str, Any]:
    """Baseline v1 결과 로드"""
    v1_path = Path("results/baseline_v1_results.json")
    if v1_path.exists():
        with open(v1_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print("⚠️ Baseline v1 결과를 찾을 수 없습니다.")
        return None


def compare_baselines(v1_results: Dict[str, Any], v2_results: Dict[str, Any]) -> Dict[str, Any]:
    """Baseline v1과 v2 비교"""
    print("\n📊 Baseline v1 vs v2 비교 분석 중...")
    
    if not v1_results or not v2_results:
        print("❌ 비교할 결과가 없습니다.")
        return None
    
    # 주요 메트릭 추출
    v1_perf = v1_results['final_performance']
    v2_perf = v2_results['final_performance']
    
    # 성능 비교
    comparison = {
        'timestamp': datetime.now().isoformat(),
        'comparison_type': 'Baseline v1 vs v2',
        
        'v1_performance': {
            'top1_accuracy': v1_perf['top1_accuracy'] * 100,
            'top5_accuracy': v1_perf['top5_accuracy'] * 100,
            'mrr': v1_perf['mrr'],
            'validation_loss': v1_perf['validation_loss'],
            'positive_similarity': v1_perf['positive_similarity']
        },
        
        'v2_performance': {
            'top1_accuracy': v2_perf['top1_accuracy'] * 100,
            'top5_accuracy': v2_perf['top5_accuracy'] * 100,
            'mrr': v2_perf['mrr'],
            'validation_loss': v2_perf['validation_loss'],
            'positive_similarity': v2_perf['positive_similarity']
        },
        
        'improvements': {
            'top1_accuracy_diff': (v2_perf['top1_accuracy'] - v1_perf['top1_accuracy']) * 100,
            'top5_accuracy_diff': (v2_perf['top5_accuracy'] - v1_perf['top5_accuracy']) * 100,
            'mrr_diff': v2_perf['mrr'] - v1_perf['mrr'],
            'validation_loss_diff': v2_perf['validation_loss'] - v1_perf['validation_loss'],
            'positive_similarity_diff': v2_perf['positive_similarity'] - v1_perf['positive_similarity']
        },
        
        'relative_improvements': {
            'top1_accuracy_rel': ((v2_perf['top1_accuracy'] - v1_perf['top1_accuracy']) / v1_perf['top1_accuracy']) * 100 if v1_perf['top1_accuracy'] > 0 else 0,
            'top5_accuracy_rel': ((v2_perf['top5_accuracy'] - v1_perf['top5_accuracy']) / v1_perf['top5_accuracy']) * 100 if v1_perf['top5_accuracy'] > 0 else 0,
            'mrr_rel': ((v2_perf['mrr'] - v1_perf['mrr']) / v1_perf['mrr']) * 100 if v1_perf['mrr'] > 0 else 0
        },
        
        'configuration_comparison': {
            'v1_config': v1_results['configuration'],
            'v2_config': v2_results['configuration']
        }
    }
    
    return comparison


def print_comparison_summary(comparison: Dict[str, Any]):
    """비교 결과 요약 출력"""
    print(f"\n{'='*80}")
    print("📊 Baseline v1 vs v2 성능 비교 결과")
    print(f"{'='*80}")
    
    v1_perf = comparison['v1_performance']
    v2_perf = comparison['v2_performance']
    improvements = comparison['improvements']
    rel_improvements = comparison['relative_improvements']
    
    print(f"\n🎯 핵심 성능 지표:")
    print(f"{'메트릭':<20} {'v1':<15} {'v2':<15} {'절대 개선':<15} {'상대 개선':<15}")
    print(f"{'-'*80}")
    print(f"{'Top-1 정확도':<20} {v1_perf['top1_accuracy']:<15.1f}% {v2_perf['top1_accuracy']:<15.1f}% {improvements['top1_accuracy_diff']:<15.1f}%p {rel_improvements['top1_accuracy_rel']:<15.1f}%")
    print(f"{'Top-5 정확도':<20} {v1_perf['top5_accuracy']:<15.1f}% {v2_perf['top5_accuracy']:<15.1f}% {improvements['top5_accuracy_diff']:<15.1f}%p {rel_improvements['top5_accuracy_rel']:<15.1f}%")
    print(f"{'MRR':<20} {v1_perf['mrr']:<15.3f} {v2_perf['mrr']:<15.3f} {improvements['mrr_diff']:<15.3f} {rel_improvements['mrr_rel']:<15.1f}%")
    print(f"{'검증 손실':<20} {v1_perf['validation_loss']:<15.3f} {v2_perf['validation_loss']:<15.3f} {improvements['validation_loss_diff']:<15.3f} {'N/A':<15}")
    print(f"{'양성 유사도':<20} {v1_perf['positive_similarity']:<15.3f} {v2_perf['positive_similarity']:<15.3f} {improvements['positive_similarity_diff']:<15.3f} {'N/A':<15}")
    
    print(f"\n💡 주요 인사이트:")
    
    # Top-5 정확도 기준 분석
    if improvements['top5_accuracy_diff'] > 0:
        print(f"   ✅ Top-5 정확도 개선: {improvements['top5_accuracy_diff']:.1f}%p 향상")
    elif improvements['top5_accuracy_diff'] < 0:
        print(f"   ❌ Top-5 정확도 하락: {abs(improvements['top5_accuracy_diff']):.1f}%p 감소")
    else:
        print(f"   ➖ Top-5 정확도 동일")
    
    # MRR 기준 분석
    if improvements['mrr_diff'] > 0:
        print(f"   ✅ MRR 개선: {improvements['mrr_diff']:.3f} 향상")
    elif improvements['mrr_diff'] < 0:
        print(f"   ❌ MRR 하락: {abs(improvements['mrr_diff']):.3f} 감소")
    else:
        print(f"   ➖ MRR 동일")
    
    # 전체적인 성능 평가
    positive_changes = sum([
        1 if improvements['top1_accuracy_diff'] > 0 else 0,
        1 if improvements['top5_accuracy_diff'] > 0 else 0,
        1 if improvements['mrr_diff'] > 0 else 0,
        1 if improvements['validation_loss_diff'] < 0 else 0  # 손실은 낮을수록 좋음
    ])
    
    print(f"\n🏆 전체 평가:")
    if positive_changes >= 3:
        print(f"   🎉 Baseline v2가 v1 대비 전반적으로 우수한 성능을 보입니다!")
    elif positive_changes >= 2:
        print(f"   👍 Baseline v2가 v1 대비 일부 개선을 보입니다.")
    elif positive_changes >= 1:
        print(f"   🤔 Baseline v2가 v1 대비 미미한 개선을 보입니다.")
    else:
        print(f"   😔 Baseline v2가 v1 대비 성능 개선이 없거나 하락했습니다.")
    
    print(f"\n🔧 설정 비교:")
    v1_config = comparison['configuration_comparison']['v1_config']
    v2_config = comparison['configuration_comparison']['v2_config']
    
    print(f"   Temperature: v1={v1_config['temperature']} vs v2={v2_config['temperature']}")
    print(f"   Batch Size: v1={v1_config['batch_size']} vs v2={v2_config['batch_size']}")
    print(f"   Epochs: v1={v1_config['epochs']} vs v2={v2_config['epochs']}")


def create_baseline_v2():
    """Baseline v2 생성 및 비교"""
    print("🚀 Baseline v2 생성 및 v1과 비교")
    print("=" * 60)
    
    # STEP 1: 현재 모델 평가
    print("STEP 1: 현재 모델 성능 평가")
    v2_results = evaluate_current_model()
    
    if v2_results is None:
        print("❌ v2 평가 실패")
        return
    
    # STEP 2: v1 결과 로드
    print("\nSTEP 2: Baseline v1 결과 로드")
    v1_results = load_baseline_v1_results()
    
    if v1_results is None:
        print("❌ v1 결과 로드 실패")
        return
    
    # STEP 3: 비교 분석
    print("\nSTEP 3: v1 vs v2 비교 분석")
    comparison = compare_baselines(v1_results, v2_results)
    
    if comparison is None:
        print("❌ 비교 분석 실패")
        return
    
    # STEP 4: 결과 저장
    print("\nSTEP 4: 결과 저장")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # v2 결과 저장
    v2_file = results_dir / "baseline_v2_results.json"
    with open(v2_file, 'w', encoding='utf-8') as f:
        json.dump(v2_results, f, indent=2, ensure_ascii=False)
    print(f"💾 Baseline v2 결과 저장: {v2_file}")
    
    # 비교 결과 저장
    comparison_file = results_dir / "baseline_v1_vs_v2_comparison.json"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"💾 비교 결과 저장: {comparison_file}")
    
    # STEP 5: 요약 출력
    print("\nSTEP 5: 비교 결과 요약")
    print_comparison_summary(comparison)
    
    # 체크포인트 백업
    current_checkpoint = Path("checkpoints/best_model.pt")
    if current_checkpoint.exists():
        v2_checkpoint = Path("checkpoints/baseline_v2_best_model.pt")
        import shutil
        shutil.copy2(current_checkpoint, v2_checkpoint)
        print(f"\n📦 Baseline v2 체크포인트 백업: {v2_checkpoint}")
    
    print(f"\n✨ Baseline v2 생성 및 비교 완료!")
    
    return {
        'v1_results': v1_results,
        'v2_results': v2_results,
        'comparison': comparison
    }


if __name__ == "__main__":
    create_baseline_v2()