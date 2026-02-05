#!/usr/bin/env python3
"""
Baseline v2 개선 버전 생성

현재 상황 분석:
- Baseline v1: Temperature 0.1, 8 epochs, 64.1% Top-5 accuracy
- 현재 best_model.pt는 v1보다 성능이 낮음
- 추가 학습을 통해 v2 생성 필요
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


def analyze_current_situation():
    """현재 상황 분석"""
    print("🔍 현재 상황 분석")
    print("=" * 60)
    
    # 체크포인트 파일들 확인
    checkpoint_dir = Path("checkpoints")
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    
    print("📦 사용 가능한 체크포인트:")
    for cp in checkpoints:
        print(f"   - {cp.name}")
    
    # v1 결과 로드
    v1_path = Path("results/baseline_v1_results.json")
    if v1_path.exists():
        with open(v1_path, 'r', encoding='utf-8') as f:
            v1_results = json.load(f)
        
        print(f"\n📊 Baseline v1 성능:")
        perf = v1_results['final_performance']
        print(f"   Top-1: {perf['top1_accuracy']*100:.1f}%")
        print(f"   Top-5: {perf['top5_accuracy']*100:.1f}%")
        print(f"   MRR: {perf['mrr']:.3f}")
        print(f"   검증 손실: {perf['validation_loss']:.3f}")
        
        return v1_results
    else:
        print("⚠️ Baseline v1 결과를 찾을 수 없습니다.")
        return None


def create_improved_baseline_v2():
    """개선된 Baseline v2 생성 전략"""
    print("\n🚀 Baseline v2 개선 전략")
    print("=" * 60)
    
    # 현재 상황 분석
    v1_results = analyze_current_situation()
    
    if v1_results is None:
        print("❌ v1 결과가 없어 비교할 수 없습니다.")
        return
    
    print(f"\n💡 Baseline v2 개선 방향:")
    print(f"   1. 현재 baseline_v1_best_model.pt가 최고 성능")
    print(f"   2. 추가 학습을 통한 성능 향상 시도")
    print(f"   3. 하이퍼파라미터 미세 조정")
    
    # 개선 방안 제시
    improvement_strategies = {
        "temperature_fine_tuning": {
            "description": "Temperature 미세 조정 (0.08, 0.09, 0.11, 0.12)",
            "expected_improvement": "1-3%p",
            "effort": "낮음"
        },
        "extended_training": {
            "description": "추가 에포크 학습 (12-15 epochs)",
            "expected_improvement": "2-5%p",
            "effort": "중간"
        },
        "batch_size_optimization": {
            "description": "배치 크기 최적화 (32, 64)",
            "expected_improvement": "1-2%p",
            "effort": "낮음"
        },
        "learning_rate_scheduling": {
            "description": "학습률 스케줄링 개선",
            "expected_improvement": "2-4%p",
            "effort": "중간"
        },
        "data_augmentation": {
            "description": "데이터 증강 기법 적용",
            "expected_improvement": "3-7%p",
            "effort": "높음"
        }
    }
    
    print(f"\n📈 개선 전략 옵션:")
    for strategy, details in improvement_strategies.items():
        print(f"   {strategy}:")
        print(f"     설명: {details['description']}")
        print(f"     예상 개선: {details['expected_improvement']}")
        print(f"     노력도: {details['effort']}")
        print()
    
    # 현재 상황에서 가능한 즉시 개선안
    print(f"🎯 즉시 적용 가능한 개선안:")
    print(f"   1. Temperature 0.09로 미세 조정")
    print(f"   2. 배치 크기 32로 증가")
    print(f"   3. 추가 5 에포크 학습")
    
    return improvement_strategies


def quick_baseline_v2_experiment():
    """빠른 Baseline v2 실험"""
    print(f"\n⚡ 빠른 Baseline v2 실험 실행")
    print("=" * 60)
    
    # 데이터셋 경로
    dataset_path = "C:/sample/라벨링데이터"
    
    # v2 설정 (v1 대비 미세 조정)
    config = TrainingConfig()
    config.temperature = 0.09  # v1: 0.1 -> v2: 0.09
    config.batch_size = 32     # v1: 16 -> v2: 32
    config.max_epochs = 5      # 추가 5 에포크
    config.learning_rate = 0.00008  # 약간 낮춤
    
    print(f"📋 v2 설정:")
    print(f"   Temperature: {config.temperature}")
    print(f"   Batch Size: {config.batch_size}")
    print(f"   추가 Epochs: {config.max_epochs}")
    print(f"   Learning Rate: {config.learning_rate}")
    
    try:
        # 시스템 초기화
        system = FashionEncoderSystem()
        system.config = config
        
        # 데이터 설정
        print("\n📁 데이터 설정 중...")
        system.setup_data(dataset_path)
        
        # 트레이너 설정
        print("🏋️ 트레이너 설정 중...")
        system.setup_trainer()
        
        # v1 체크포인트에서 시작
        v1_checkpoint = "checkpoints/baseline_v1_best_model.pt"
        if Path(v1_checkpoint).exists():
            print(f"📦 v1 체크포인트에서 시작: {v1_checkpoint}")
            system.trainer.load_checkpoint(v1_checkpoint)
        else:
            print("⚠️ v1 체크포인트가 없습니다.")
            return None
        
        # 추가 학습 실행
        print(f"\n🏋️ 추가 학습 시작 (5 에포크)...")
        print(f"   목표: v1 64.1% -> v2 67%+ Top-5 accuracy")
        
        # 학습 실행
        system.trainer.train_contrastive_learning(
            train_loader=system.data_module.train_dataloader(),
            val_loader=system.data_module.val_dataloader(),
            epochs=config.max_epochs
        )
        
        # 최종 평가
        print(f"\n📊 최종 평가 중...")
        final_metrics = system.trainer._final_evaluation(system.data_module.val_dataloader())
        
        # v2 결과 생성
        v2_results = {
            'timestamp': datetime.now().isoformat(),
            'model_name': 'Fashion JSON Encoder Baseline v2 (Improved)',
            'configuration': {
                'temperature': config.temperature,
                'batch_size': config.batch_size,
                'additional_epochs': config.max_epochs,
                'learning_rate': config.learning_rate,
                'dataset': 'K-Fashion 2,172 items',
                'base_model': 'baseline_v1_best_model.pt',
                'improvements': ['temperature_tuning', 'batch_size_increase', 'extended_training']
            },
            'final_performance': {
                'top1_accuracy': final_metrics.get('top1_accuracy', 0.0),
                'top5_accuracy': final_metrics.get('top5_accuracy', 0.0),
                'mrr': final_metrics.get('mean_reciprocal_rank', 0.0),
                'validation_loss': final_metrics.get('val_loss', 0.0),
                'positive_similarity': final_metrics.get('avg_positive_similarity', 0.0),
                'negative_similarity': final_metrics.get('avg_negative_similarity', 0.0)
            },
            'additional_metrics': {
                'recall_at_3': final_metrics.get('recall_at_3', 0.0),
                'recall_at_10': final_metrics.get('recall_at_10', 0.0),
                'recall_at_20': final_metrics.get('recall_at_20', 0.0)
            },
            'notes': 'v1 기반 추가 학습으로 생성된 개선 버전'
        }
        
        # 결과 저장
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        v2_file = results_dir / "baseline_v2_improved_results.json"
        with open(v2_file, 'w', encoding='utf-8') as f:
            json.dump(v2_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 v2 결과 저장: {v2_file}")
        
        # v2 체크포인트 저장
        v2_checkpoint = Path("checkpoints/baseline_v2_improved_best_model.pt")
        system.trainer.save_checkpoint(str(v2_checkpoint))
        print(f"📦 v2 체크포인트 저장: {v2_checkpoint}")
        
        # 성능 비교
        print(f"\n📊 성능 비교:")
        print(f"   v1 Top-5: 64.1%")
        print(f"   v2 Top-5: {final_metrics.get('top5_accuracy', 0)*100:.1f}%")
        improvement = (final_metrics.get('top5_accuracy', 0) - 0.641) * 100
        print(f"   개선: {improvement:+.1f}%p")
        
        # 정리
        system.cleanup()
        
        return v2_results
        
    except Exception as e:
        print(f"\n❌ v2 실험 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_baseline_v2_summary():
    """Baseline v2 요약 생성"""
    print(f"\n📋 Baseline v2 프로젝트 요약")
    print("=" * 60)
    
    # 현재 상황 분석
    analyze_current_situation()
    
    # 개선 전략 제시
    strategies = create_improved_baseline_v2()
    
    # 사용자 선택 안내
    print(f"\n🎯 다음 단계 선택:")
    print(f"   1. 빠른 실험 실행 (Temperature 0.09, Batch 32, +5 epochs)")
    print(f"   2. 현재 v1을 v2로 지정하고 다음 단계 진행")
    print(f"   3. 더 큰 개선을 위한 장기 실험 계획")
    
    return strategies


if __name__ == "__main__":
    # 현재 상황 분석 및 전략 제시
    strategies = create_baseline_v2_summary()
    
    # 사용자 입력 대기 (실제로는 자동 실행)
    print(f"\n⚡ 빠른 실험을 자동 실행합니다...")
    
    # 빠른 실험 실행
    v2_results = quick_baseline_v2_experiment()
    
    if v2_results:
        print(f"\n✨ Baseline v2 (Improved) 생성 완료!")
    else:
        print(f"\n😔 v2 생성 실패. v1을 현재 최고 성능으로 유지합니다.")