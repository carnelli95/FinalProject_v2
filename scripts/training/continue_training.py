#!/usr/bin/env python3
"""
기존 체크포인트에서 학습 계속하기

현재 best_model.pt에서 학습을 이어서 50-100 에포크까지 진행합니다.
"""

import torch
import argparse
from pathlib import Path

from data.fashion_dataset import FashionDataModule
from training.trainer import create_trainer_from_data_module
from utils.config import TrainingConfig


def main():
    parser = argparse.ArgumentParser(description='기존 체크포인트에서 학습 계속하기')
    parser.add_argument('--epochs', type=int, default=50, help='추가로 학습할 에포크 수 (기본: 50)')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt', help='체크포인트 경로')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='디바이스')
    
    args = parser.parse_args()
    
    print(f"🚀 기존 체크포인트에서 학습 계속하기")
    print(f"📁 체크포인트: {args.checkpoint}")
    print(f"📊 추가 에포크: {args.epochs}")
    print(f"💻 디바이스: {args.device}")
    
    # 체크포인트 존재 확인
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ 체크포인트 파일이 없습니다: {checkpoint_path}")
        return
    
    # 기본 설정 (현재 설정 유지)
    config = TrainingConfig(
        batch_size=64,
        learning_rate=1e-4,
        temperature=0.07,
        max_epochs=args.epochs,  # 추가로 학습할 에포크 수
        embedding_dim=128,
        hidden_dim=256,
        output_dim=512,
        dropout_rate=0.1,
        weight_decay=1e-5
    )

    # 데이터 모듈 초기화
    print("\n📂 데이터 로딩...")
    data_module = FashionDataModule(
        dataset_path="C:/sample/라벨링데이터",
        target_categories=['레트로', '로맨틱', '리조트'],
        batch_size=config.batch_size
    )
    data_module.setup()
    
    print(f"\n⚙️ 학습 설정:")
    print(f"  배치 사이즈: {config.batch_size}")
    print(f"  학습률: {config.learning_rate}")
    print(f"  온도: {config.temperature}")
    print(f"  추가 에포크: {config.max_epochs}")
    
    # 트레이너 생성
    trainer = create_trainer_from_data_module(
        data_module=data_module,
        config=config,
        device=args.device
    )
    
    # 체크포인트 로드
    print(f"\n📥 체크포인트 로딩: {args.checkpoint}")
    checkpoint_info = trainer.load_checkpoint(args.checkpoint)
    
    current_epoch = checkpoint_info['epoch']
    print(f"✅ 체크포인트 로드 완료!")
    print(f"  이전 에포크: {current_epoch}")
    print(f"  이전 최고 검증 손실: {checkpoint_info['best_val_loss']:.4f}")
    
    # 데이터 로더 준비
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    print(f"\n📊 데이터셋 정보:")
    print(f"  학습 배치 수: {len(train_loader)}")
    print(f"  검증 배치 수: {len(val_loader)}")
    
    # 학습 계속하기
    print(f"\n🎯 Stage 2 Contrastive Learning 계속 진행...")
    print(f"   목표: {current_epoch + 1} → {current_epoch + args.epochs} 에포크")
    
    try:
        results = trainer.train_contrastive_learning(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs
        )
        
        print(f"\n✅ 학습 완료!")
        print(f"📈 최종 결과:")
        print(f"  최고 검증 손실: {results['best_val_loss']:.4f}")
        print(f"  총 학습 에포크: {current_epoch + results['total_epochs']}")
        
        # 최종 메트릭 출력
        final_metrics = results['final_metrics']
        print(f"\n📊 최종 성능 지표:")
        print(f"  Top-1 정확도: {final_metrics['top1_accuracy']:.4f}")
        print(f"  Top-5 정확도: {final_metrics['top5_accuracy']:.4f}")
        print(f"  MRR: {final_metrics['mean_reciprocal_rank']:.4f}")
        print(f"  Positive Similarity: {final_metrics['positive_similarity_mean']:.4f}")
        print(f"  Negative Similarity: {final_metrics['negative_similarity_mean']:.4f}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ 사용자에 의해 학습이 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 학습 중 오류 발생: {e}")
    finally:
        trainer.close()


if __name__ == "__main__":
    main()