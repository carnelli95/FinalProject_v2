#!/usr/bin/env python3
"""
빠른 Optuna 튜닝 테스트 (간소화된 버전)
"""

import optuna
import torch
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any

from data.fashion_dataset import FashionDataModule
from training.trainer import create_trainer_from_data_module
from utils.config import TrainingConfig


def quick_objective(trial: optuna.Trial) -> float:
    """간소화된 목적 함수"""
    
    # 하이퍼파라미터 제안
    config = TrainingConfig(
        learning_rate=trial.suggest_categorical('learning_rate', [1e-4, 3e-4, 5e-4]),
        temperature=trial.suggest_categorical('temperature', [0.05, 0.07, 0.1]),
        batch_size=trial.suggest_categorical('batch_size', [64, 96]),
        embedding_dim=128,  # 고정
        hidden_dim=256,     # 고정
        dropout_rate=0.1,   # 고정
        weight_decay=1e-4,  # 고정
        output_dim=512,
        max_epochs=3,       # 매우 짧게
    )
    
    print(f"\n🔍 Trial {trial.number + 1}")
    print(f"   학습률: {config.learning_rate:.6f}")
    print(f"   온도: {config.temperature:.3f}")
    print(f"   배치 사이즈: {config.batch_size}")
    
    try:
        # 데이터 모듈 준비
        data_module = FashionDataModule(
            dataset_path="C:/sample/라벨링데이터",
            target_categories=['레트로', '로맨틱', '리조트'],
            batch_size=config.batch_size
        )
        data_module.setup()
        
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        
        # 트레이너 생성
        trainer = create_trainer_from_data_module(
            data_module=data_module,
            config=config,
            device='cpu',
            checkpoint_dir=f'quick_tuning/trial_{trial.number}',
            log_dir=f'quick_tuning_logs/trial_{trial.number}'
        )
        
        # 짧은 학습 실행
        print(f"   🚀 학습 시작 (3 에포크)...")
        start_time = time.time()
        
        results = trainer.train_contrastive_learning(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.max_epochs
        )
        
        elapsed = time.time() - start_time
        
        # 간단한 메트릭 계산
        final_metrics = results.get('final_metrics', {})
        
        # Top-5 정확도를 목적 함수로 사용 (간단함)
        top5_accuracy = final_metrics.get('top5_accuracy', 0.0)
        mrr = final_metrics.get('mean_reciprocal_rank', 0.0)
        
        # 복합 목적 함수
        objective_value = 0.7 * top5_accuracy + 0.3 * mrr
        
        print(f"   ⏱️ 학습 완료: {elapsed:.1f}초")
        print(f"   📊 결과:")
        print(f"      ✅ Top-5 정확도: {top5_accuracy:.4f}")
        print(f"      🔍 MRR: {mrr:.4f}")
        print(f"      🏆 목적함수 값: {objective_value:.4f}")
        
        # 리소스 정리
        trainer.close()
        
        return objective_value
        
    except Exception as e:
        print(f"   ❌ Trial 실패: {e}")
        return 0.0  # 실패한 경우 최소값 반환


def main():
    """메인 함수"""
    print("Fashion JSON Encoder 빠른 튜닝 테스트")
    print("=" * 50)
    
    # Optuna 스터디 생성
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # 빠른 튜닝 실행 (5회만)
    start_time = time.time()
    study.optimize(quick_objective, n_trials=5)
    total_time = time.time() - start_time
    
    # 결과 분석
    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = best_trial.value
    
    print(f"\n🏆 튜닝 완료!")
    print(f"   총 소요 시간: {total_time/60:.1f}분")
    print(f"   최고 목적함수 값: {best_value:.4f}")
    print(f"\n📋 최적 하이퍼파라미터:")
    for key, value in best_params.items():
        print(f"   {key}: {value}")
    
    # 결과 저장
    results = {
        'best_params': best_params,
        'best_value': best_value,
        'n_trials': 5,
        'total_time': total_time,
        'all_trials': [
            {
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name
            }
            for trial in study.trials
        ]
    }
    
    # 결과 파일 저장
    results_dir = Path("quick_tuning_results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = int(time.time())
    results_file = results_dir / f"quick_tuning_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"📄 결과 저장: {results_file}")
    
    return results


if __name__ == "__main__":
    main()