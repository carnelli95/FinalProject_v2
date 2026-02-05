#!/usr/bin/env python3
"""
Class-Balanced Training Experiment

클래스 불균형 문제 해결을 위한 실험:
- 레트로 9% vs 로맨틱 46% vs 리조트 46% 불균형 해결
- ClassBalancedSampler를 사용한 학습
- Baseline v1 (Temperature 0.1) 대비 성능 개선 측정

예상 결과: Top-5 정확도 1-2% 향상 (64.1% → 65-66%)
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


def run_class_balanced_experiment():
    """클래스 균형 샘플링 실험 실행"""
    print("🔬 Class-Balanced Training Experiment")
    print("=" * 60)
    print("목표: 클래스 불균형 해결로 Top-5 정확도 1-2% 향상")
    print("기준: Baseline v1 (64.1% Top-5, Temperature 0.1)")
    print("=" * 60)
    
    # 실험 설정
    dataset_path = "C:/sample/라벨링데이터"  # 원래대로 복원
    
    # Baseline v1과 동일한 설정 사용
    config = TrainingConfig()
    config.temperature = 0.1  # Baseline v1 최적 온도
    config.batch_size = 16    # Baseline v1 배치 크기
    config.max_epochs = 8     # Baseline v1 에포크 수
    config.learning_rate = 1e-4
    
    print(f"📊 실험 설정:")
    print(f"   Temperature: {config.temperature}")
    print(f"   Batch Size: {config.batch_size}")
    print(f"   Epochs: {config.max_epochs}")
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
        
        # 클래스 균형 데이터로더 사용하도록 설정
        print("\n⚖️ 클래스 균형 샘플링 활성화...")
        train_loader = system.data_module.train_dataloader(use_class_balanced=True)
        val_loader = system.data_module.val_dataloader()
        
        print(f"   학습 배치 수: {len(train_loader)}")
        print(f"   검증 배치 수: {len(val_loader)}")
        
        # 학습 실행
        print(f"\n🚀 클래스 균형 학습 시작...")
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
        
        top5_improvement = top5_accuracy - baseline_top5
        top1_improvement = top1_accuracy - baseline_top1
        mrr_improvement = mrr - baseline_mrr
        
        print(f"\n📈 Baseline v1 대비 개선:")
        print(f"   Top-5: {top5_improvement:+.1f}% ({baseline_top5:.1f}% → {top5_accuracy:.1f}%)")
        print(f"   Top-1: {top1_improvement:+.1f}% ({baseline_top1:.1f}% → {top1_accuracy:.1f}%)")
        print(f"   MRR: {mrr_improvement:+.3f} ({baseline_mrr:.3f} → {mrr:.3f})")
        
        # 결과 저장
        experiment_results = {
            "experiment_name": "Class-Balanced Training",
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "temperature": config.temperature,
                "batch_size": config.batch_size,
                "epochs": config.max_epochs,
                "learning_rate": config.learning_rate,
                "class_balanced_sampling": True,
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
                "top5_improvement": top5_improvement,
                "top1_improvement": top1_improvement,
                "mrr_improvement": mrr_improvement
            },
            "training_progression": results.get('metrics_history', {}),
            "notes": "클래스 균형 샘플링을 통한 레트로 클래스 언더샘플링 문제 해결 실험"
        }
        
        # 결과 파일 저장
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        results_file = results_dir / "class_balanced_experiment_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 결과 저장: {results_file}")
        
        # 성공 여부 판단
        if top5_improvement >= 1.0:
            print(f"\n🎉 실험 성공!")
            print(f"   목표 달성: Top-5 정확도 {top5_improvement:.1f}% 향상")
        elif top5_improvement >= 0.5:
            print(f"\n✅ 실험 부분 성공!")
            print(f"   소폭 개선: Top-5 정확도 {top5_improvement:.1f}% 향상")
        else:
            print(f"\n⚠️ 실험 결과 분석 필요")
            print(f"   예상보다 낮은 개선: Top-5 정확도 {top5_improvement:.1f}% 변화")
        
        # 정리
        system.cleanup()
        
        print(f"\n✨ 클래스 균형 실험 완료!")
        
        return experiment_results
        
    except Exception as e:
        print(f"\n❌ 실험 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    run_class_balanced_experiment()