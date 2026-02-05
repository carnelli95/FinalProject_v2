#!/usr/bin/env python3
"""
Temperature 0.15 Experiment

Temperature 튜닝 실험:
- Baseline v1: Temperature 0.1 (64.1% Top-5)
- 실험: Temperature 0.15 (예상 65-67% Top-5)
- 짧은 학습: 5 에포크로 빠른 검증

패션/텍스트 기반 contrastive learning에서 0.1~0.2 범위가 효과적
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from main import FashionEncoderSystem
from utils.config import TrainingConfig


def run_temperature_015_experiment():
    """Temperature 0.15 실험 실행"""
    print("🌡️ Temperature 0.15 Experiment")
    print("=" * 60)
    print("목표: Temperature 튜닝으로 성능 향상 검증")
    print("기준: Baseline v1 (64.1% Top-5, Temperature 0.1)")
    print("실험: Temperature 0.15 (5 에포크 빠른 검증)")
    print("=" * 60)
    
    # 실험 설정
    dataset_path = "C:/sample/라벨링데이터"
    
    # Temperature 0.15 설정
    config = TrainingConfig()
    config.temperature = 0.15  # 실험 온도
    config.batch_size = 16     # Baseline v1과 동일
    config.max_epochs = 5      # 빠른 검증을 위한 짧은 학습
    config.learning_rate = 1e-4
    
    print(f"📊 실험 설정:")
    print(f"   Temperature: {config.temperature} (vs Baseline 0.1)")
    print(f"   Batch Size: {config.batch_size}")
    print(f"   Epochs: {config.max_epochs} (빠른 검증)")
    print(f"   Learning Rate: {config.learning_rate}")
    print(f"   Dataset: {dataset_path}")
    
    try:
        # 시스템 초기화
        system = FashionEncoderSystem()
        system.config = config
        
        # 데이터 설정
        print("\n📁 데이터 설정 중...")
        system.setup_data(dataset_path)
        
        # 클래스 분포 확인
        train_dataset = system.data_module.train_dataset
        class_counts = {}
        for item in train_dataset.fashion_items:
            category = item.category
            class_counts[category] = class_counts.get(category, 0) + 1
        
        total_items = sum(class_counts.values())
        print(f"\n📈 클래스 분포 (학습 데이터):")
        for category, count in class_counts.items():
            percentage = count / total_items * 100
            print(f"   {category}: {count}개 ({percentage:.1f}%)")
        
        # 트레이너 설정
        print("\n🏋️ 트레이너 설정 중...")
        checkpoint_dir = "checkpoints"
        log_dir = "logs"
        system.setup_trainer(checkpoint_dir=checkpoint_dir, log_dir=log_dir)
        
        # 데이터로더 준비
        train_loader = system.data_module.train_dataloader()
        val_loader = system.data_module.val_dataloader()
        
        print(f"   학습 배치 수: {len(train_loader)}")
        print(f"   검증 배치 수: {len(val_loader)}")
        
        # 학습 실행
        print(f"\n🚀 Temperature 0.15 학습 시작...")
        print(f"   시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Contrastive Learning만 실행 (Stage 2)
        results = system.trainer.train_contrastive_learning(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.max_epochs
        )
        
        # 결과 분석
        print(f"\n📊 실험 결과:")
        final_metrics = results.get('final_metrics', {})
        
        top1_accuracy = final_metrics.get('top1_accuracy', 0) * 100
        top5_accuracy = final_metrics.get('top5_accuracy', 0) * 100
        mrr = final_metrics.get('mean_reciprocal_rank', 0)
        val_loss = results.get('best_val_loss', 0)
        
        print(f"   Top-1 정확도: {top1_accuracy:.1f}%")
        print(f"   Top-5 정확도: {top5_accuracy:.1f}%")
        print(f"   MRR: {mrr:.3f}")
        print(f"   검증 손실: {val_loss:.4f}")
        
        # Baseline v1과 비교
        baseline_top5 = 64.1
        baseline_top1 = 22.2
        baseline_mrr = 0.407
        baseline_temp = 0.1
        
        top5_improvement = top5_accuracy - baseline_top5
        top1_improvement = top1_accuracy - baseline_top1
        mrr_improvement = mrr - baseline_mrr
        
        print(f"\n📈 Baseline v1 (Temperature {baseline_temp}) 대비:")
        print(f"   Top-5: {top5_improvement:+.1f}% ({baseline_top5:.1f}% → {top5_accuracy:.1f}%)")
        print(f"   Top-1: {top1_improvement:+.1f}% ({baseline_top1:.1f}% → {top1_accuracy:.1f}%)")
        print(f"   MRR: {mrr_improvement:+.3f} ({baseline_mrr:.3f} → {mrr:.3f})")
        
        # Temperature 효과 분석
        temp_effect = {
            'temperature_change': config.temperature - baseline_temp,
            'performance_change': top5_improvement,
            'effectiveness': top5_improvement / (config.temperature - baseline_temp) if config.temperature != baseline_temp else 0
        }
        
        print(f"\n🌡️ Temperature 효과 분석:")
        print(f"   Temperature 변화: {baseline_temp} → {config.temperature} (+{temp_effect['temperature_change']:.2f})")
        print(f"   성능 변화: {top5_improvement:+.1f}%p")
        print(f"   효과성: {temp_effect['effectiveness']:.1f}%p per 0.01 temp")
        
        # 결과 저장
        experiment_results = {
            "experiment_name": "Temperature 0.15 Experiment",
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "temperature": config.temperature,
                "baseline_temperature": baseline_temp,
                "batch_size": config.batch_size,
                "epochs": config.max_epochs,
                "learning_rate": config.learning_rate,
                "dataset": dataset_path
            },
            "class_distribution": class_counts,
            "final_performance": {
                "top1_accuracy": final_metrics.get('top1_accuracy', 0),
                "top5_accuracy": final_metrics.get('top5_accuracy', 0),
                "mrr": final_metrics.get('mean_reciprocal_rank', 0),
                "validation_loss": val_loss
            },
            "baseline_comparison": {
                "baseline_v1_top5": baseline_top5,
                "baseline_v1_top1": baseline_top1,
                "baseline_v1_mrr": baseline_mrr,
                "baseline_v1_temperature": baseline_temp,
                "top5_improvement": top5_improvement,
                "top1_improvement": top1_improvement,
                "mrr_improvement": mrr_improvement
            },
            "temperature_analysis": temp_effect,
            "training_progression": results.get('train_losses', []),
            "validation_progression": results.get('val_losses', []),
            "notes": "Temperature 0.15로 5 에포크 빠른 검증 실험"
        }
        
        # 결과 파일 저장
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        results_file = results_dir / "temperature_015_experiment_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 결과 저장: {results_file}")
        
        # 실험 결론
        if top5_improvement >= 1.0:
            print(f"\n🎉 실험 성공!")
            print(f"   Temperature 0.15가 0.1보다 {top5_improvement:.1f}%p 우수")
            print(f"   추천: Temperature 0.15 사용")
        elif top5_improvement >= 0.5:
            print(f"\n✅ 실험 부분 성공!")
            print(f"   소폭 개선: {top5_improvement:.1f}%p")
            print(f"   고려: Temperature 0.15 사용 검토")
        elif top5_improvement >= -0.5:
            print(f"\n📊 실험 결과: 유사한 성능")
            print(f"   차이: {top5_improvement:.1f}%p (미미한 차이)")
            print(f"   결론: Temperature 0.1과 0.15 모두 적합")
        else:
            print(f"\n⚠️ 실험 결과: 성능 하락")
            print(f"   하락: {top5_improvement:.1f}%p")
            print(f"   추천: Temperature 0.1 유지")
        
        # 다음 단계 제안
        print(f"\n🔮 다음 단계 제안:")
        if top5_improvement > 0:
            print(f"   1. Temperature 0.15로 전체 8 에포크 학습")
            print(f"   2. Temperature 0.12, 0.18 추가 실험")
            print(f"   3. Query-aware 평가로 실제 성능 검증")
        else:
            print(f"   1. Temperature 0.1 유지 (Baseline v1)")
            print(f"   2. 다른 하이퍼파라미터 튜닝 고려")
            print(f"   3. Query-aware 평가로 현재 성능 분석")
        
        # 정리
        system.cleanup()
        
        print(f"\n✨ Temperature 0.15 실험 완료!")
        
        return experiment_results
        
    except Exception as e:
        print(f"\n❌ 실험 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    run_temperature_015_experiment()