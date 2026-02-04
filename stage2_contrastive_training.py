#!/usr/bin/env python3
"""
Stage 2: Contrastive Learning 전체 학습
튜닝 없이 고정 파라미터로 바로 학습 실행
"""

import torch
import time
from pathlib import Path

from data.fashion_dataset import FashionDataModule
from training.trainer import create_trainer_from_data_module
from utils.config import TrainingConfig


def main():
    """Stage 2 Contrastive Learning 전체 학습"""
    
    print("🚀 Stage 2: Contrastive Learning 전체 학습 시작!")
    print("=" * 60)
    
    # 고정 파라미터 (튜닝에서 합리적인 값들)
    config = TrainingConfig(
        learning_rate=3e-4,     # 튜닝에서 좋았던 값
        temperature=0.05,       # 튜닝에서 좋았던 값  
        batch_size=4,           # 작은 배치로 수정 (합성 데이터용)
        embedding_dim=128,
        hidden_dim=256,
        dropout_rate=0.1,
        weight_decay=1e-4,
        output_dim=512,
        max_epochs=5,           # 테스트용으로 짧게
    )
    
    print(f"📊 학습 설정:")
    print(f"   학습률: {config.learning_rate}")
    print(f"   온도: {config.temperature}")
    print(f"   배치 사이즈: {config.batch_size}")
    print(f"   총 에포크: {config.max_epochs}")
    print()
    
    try:
        # 데이터 모듈 준비 (합성 데이터로 테스트)
        print("📁 데이터 로딩...")
        print("   ⚠️ 실제 데이터셋 경로를 확인할 수 없어 합성 데이터로 진행합니다.")
        
        # 합성 데이터로 테스트
        from examples.json_encoder_sanity_check import create_synthetic_data_module
        vocab_sizes = {
            'category': 10,
            'style': 20, 
            'silhouette': 15,
            'material': 25,
            'detail': 30
        }
        data_module = create_synthetic_data_module(vocab_sizes, 'cpu')
        data_module.setup()
        
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        
        print(f"   ✅ 합성 데이터로 학습 진행")
        print(f"   ✅ 학습 배치: {len(train_loader)}")
        print(f"   ✅ 검증 배치: {len(val_loader)}")
        print()
        
        # 트레이너 생성
        print("🔧 트레이너 설정...")
        trainer = create_trainer_from_data_module(
            data_module=data_module,
            config=config,
            device='cpu',  # GPU 있으면 'cuda'로 변경
            checkpoint_dir='stage2_checkpoints',
            log_dir='stage2_logs'
        )
        
        print(f"   ✅ 체크포인트 저장: stage2_checkpoints/")
        print(f"   ✅ 로그 저장: stage2_logs/")
        print()
        
        # Stage 2 Contrastive Learning 실행
        print("🔥 Stage 2: Contrastive Learning 학습 시작!")
        print(f"⏰ 시작 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 60)
        
        start_time = time.time()
        
        results = trainer.train_contrastive_learning(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.max_epochs
        )
        
        elapsed_time = time.time() - start_time
        
        # 결과 출력
        print("\n" + "=" * 60)
        print("🎉 Stage 2 학습 완료!")
        print("=" * 60)
        
        print(f"⏱️ 총 학습 시간: {elapsed_time/60:.1f}분")
        print(f"🏆 최고 검증 손실: {results['best_val_loss']:.4f}")
        print(f"📈 총 에포크: {results['total_epochs']}")
        
        # 최종 메트릭
        final_metrics = results.get('final_metrics', {})
        if final_metrics:
            print(f"\n📊 최종 성능:")
            print(f"   Top-1 정확도: {final_metrics.get('top1_accuracy', 0):.4f}")
            print(f"   Top-5 정확도: {final_metrics.get('top5_accuracy', 0):.4f}")
            print(f"   평균 역순위: {final_metrics.get('mean_reciprocal_rank', 0):.4f}")
            print(f"   평균 코사인 유사도: {final_metrics.get('mean_cosine_similarity', 0):.4f}")
        
        # 체크포인트 정보
        print(f"\n💾 저장된 파일:")
        print(f"   최고 모델: stage2_checkpoints/best_model.pt")
        print(f"   최종 모델: stage2_checkpoints/checkpoint_epoch_{config.max_epochs}.pt")
        print(f"   TensorBoard: tensorboard --logdir stage2_logs")
        
        # 다음 단계 안내
        print(f"\n🚀 다음 단계:")
        print(f"   1. 모델 평가: python main.py evaluate --checkpoint_path stage2_checkpoints/best_model.pt")
        print(f"   2. Stage 3: Downstream task 연결")
        print(f"   3. 추천 시스템 구축")
        
        # 리소스 정리
        trainer.close()
        
        return results
        
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 학습이 중단되었습니다.")
        return None
        
    except Exception as e:
        print(f"\n❌ 학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
    
    if results:
        print("\n✅ Stage 2 학습이 성공적으로 완료되었습니다!")
    else:
        print("\n❌ Stage 2 학습이 실패했습니다.")