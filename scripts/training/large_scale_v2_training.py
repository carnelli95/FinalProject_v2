#!/usr/bin/env python3
"""
대규모 데이터셋 (65GB) 고성능 GPU 학습 스크립트

최적화 포인트:
1. GPU 메모리 효율적 사용
2. 대용량 데이터 처리
3. 분산 학습 지원
4. 체크포인트 자동 저장
5. 메모리 누수 방지
"""

import os
import sys
import json
import time
import psutil
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from main import FashionEncoderSystem
from utils.config import TrainingConfig


class LargeScaleTrainingConfig(TrainingConfig):
    """대규모 학습을 위한 확장 설정"""
    
    def __init__(self):
        super().__init__()
        
        # GPU 최적화 설정
        self.mixed_precision = True
        self.gradient_accumulation_steps = 4
        self.max_grad_norm = 1.0
        
        # 대용량 데이터 처리
        self.batch_size = 64  # GPU 메모리에 따라 조정
        self.num_workers = 8  # CPU 코어 수에 따라 조정
        self.pin_memory = True
        self.persistent_workers = True
        
        # 학습 최적화
        self.learning_rate = 0.0001
        self.weight_decay = 1e-5
        self.temperature = 0.09  # v2 최적화된 값
        self.max_epochs = 15
        
        # 체크포인트 및 로깅
        self.save_every_n_epochs = 2
        self.eval_every_n_epochs = 1
        self.log_every_n_steps = 100
        
        # 메모리 관리
        self.empty_cache_every_n_steps = 500
        self.gc_collect_every_n_steps = 1000


def setup_gpu_environment():
    """GPU 환경 설정 및 최적화"""
    print("🔧 GPU 환경 설정 중...")
    
    # CUDA 사용 가능 확인
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA가 사용 불가능합니다. GPU 환경을 확인하세요.")
    
    # GPU 정보 출력
    gpu_count = torch.cuda.device_count()
    print(f"   사용 가능한 GPU: {gpu_count}개")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    # CUDA 최적화 설정
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # 메모리 관리 최적화
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    return gpu_count


def setup_distributed_training(rank, world_size):
    """분산 학습 설정"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 분산 프로세스 그룹 초기화
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """분산 학습 정리"""
    dist.destroy_process_group()


class MemoryMonitor:
    """메모리 사용량 모니터링"""
    
    def __init__(self):
        self.start_time = time.time()
        self.peak_memory = 0
        
    def log_memory_usage(self, step: int, prefix: str = ""):
        """메모리 사용량 로깅"""
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024**3
            max_memory = torch.cuda.max_memory_allocated() / 1024**3
            self.peak_memory = max(self.peak_memory, max_memory)
            
            if step % 100 == 0:  # 100 스텝마다 로깅
                print(f"   {prefix}Step {step}: GPU 메모리 {current_memory:.2f}GB / 최대 {max_memory:.2f}GB")
    
    def cleanup_memory(self):
        """메모리 정리"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


class LargeScaleTrainer:
    """대규모 데이터셋 학습을 위한 트레이너"""
    
    def __init__(self, config: LargeScaleTrainingConfig, rank: int = 0, world_size: int = 1):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
        
        self.memory_monitor = MemoryMonitor()
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        # 로깅 설정
        self.setup_logging()
        
    def setup_logging(self):
        """로깅 설정"""
        self.log_dir = Path("logs/large_scale_training")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = Path("checkpoints/large_scale")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 학습 로그 파일
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"training_log_{timestamp}.txt"
        
    def log_message(self, message: str):
        """메시지 로깅"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        
        if self.rank == 0:  # 메인 프로세스만 파일에 기록
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_msg + '\n')
    
    def setup_model_and_data(self, dataset_path: str):
        """모델 및 데이터 설정"""
        self.log_message("📁 대용량 데이터셋 로딩 중...")
        
        # 시스템 초기화
        self.system = FashionEncoderSystem()
        self.system.config = self.config
        
        # 데이터 설정
        self.system.setup_data(dataset_path)
        
        # 데이터 로더 최적화
        train_sampler = DistributedSampler(
            self.system.data_module.train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        ) if self.world_size > 1 else None
        
        self.train_loader = DataLoader(
            self.system.data_module.train_dataset,
            batch_size=self.config.batch_size,
            sampler=train_sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.system.data_module.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers
        )
        
        # 트레이너 설정
        self.system.setup_trainer()
        
        # 모델을 GPU로 이동
        self.system.trainer.contrastive_learner.to(self.device)
        
        # 분산 학습 설정
        if self.world_size > 1:
            self.system.trainer.contrastive_learner = DDP(
                self.system.trainer.contrastive_learner,
                device_ids=[self.rank],
                output_device=self.rank
            )
        
        # v2 체크포인트에서 시작
        v2_checkpoint = "checkpoints/baseline_v2_final_best_model.pt"
        if Path(v2_checkpoint).exists():
            self.log_message(f"📦 v2 체크포인트에서 시작: {v2_checkpoint}")
            self.system.trainer.load_checkpoint(v2_checkpoint)
        
        self.log_message(f"✅ 데이터셋 로딩 완료:")
        self.log_message(f"   학습 샘플: {len(self.system.data_module.train_dataset):,}")
        self.log_message(f"   검증 샘플: {len(self.system.data_module.val_dataset):,}")
        self.log_message(f"   배치 크기: {self.config.batch_size}")
        self.log_message(f"   총 배치 수: {len(self.train_loader):,}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """한 에포크 학습"""
        self.system.trainer.contrastive_learner.train()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        # 분산 학습 시 sampler 에포크 설정
        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)
        
        for step, batch in enumerate(self.train_loader):
            # 배치를 GPU로 이동
            batch = self.system.trainer._move_batch_to_device(batch)
            
            # Mixed Precision Training
            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                # Forward pass
                json_batch = self.system.trainer._convert_batch_to_dict(batch)
                embeddings = self.system.trainer.contrastive_learner.get_embeddings(
                    batch.images, json_batch
                )
                
                # Loss 계산
                loss = self.system.trainer.contrastive_learner.compute_contrastive_loss(
                    embeddings['image_embeddings'],
                    embeddings['json_embeddings'],
                    temperature=self.config.temperature
                )
                
                # Gradient Accumulation
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.system.trainer.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.system.trainer.contrastive_learner.parameters(),
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.system.trainer.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.system.trainer.contrastive_learner.parameters(),
                        self.config.max_grad_norm
                    )
                    self.system.trainer.optimizer.step()
                
                self.system.trainer.optimizer.zero_grad()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            
            # 로깅
            if step % self.config.log_every_n_steps == 0:
                avg_loss = total_loss / (step + 1)
                self.log_message(f"   Epoch {epoch}, Step {step}/{num_batches}, Loss: {avg_loss:.4f}")
                self.memory_monitor.log_memory_usage(step, f"Epoch {epoch} ")
            
            # 메모리 정리
            if step % self.config.empty_cache_every_n_steps == 0:
                torch.cuda.empty_cache()
            
            if step % self.config.gc_collect_every_n_steps == 0:
                gc.collect()
        
        return {'train_loss': total_loss / num_batches}
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """검증"""
        self.log_message(f"📊 Epoch {epoch} 검증 중...")
        
        metrics = self.system.trainer._final_evaluation(self.val_loader)
        
        self.log_message(f"   검증 결과:")
        self.log_message(f"     Top-1: {metrics.get('top1_accuracy', 0)*100:.1f}%")
        self.log_message(f"     Top-5: {metrics.get('top5_accuracy', 0)*100:.1f}%")
        self.log_message(f"     MRR: {metrics.get('mean_reciprocal_rank', 0):.3f}")
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """체크포인트 저장"""
        if self.rank != 0:  # 메인 프로세스만 저장
            return
        
        checkpoint_path = self.checkpoint_dir / f"large_scale_v2_epoch_{epoch}.pt"
        
        # 최고 성능 체크포인트 추적
        top5_accuracy = metrics.get('top5_accuracy', 0)
        
        # 현재 최고 성능 확인
        best_checkpoint = self.checkpoint_dir / "large_scale_v2_best.pt"
        current_best = 0.0
        
        if best_checkpoint.exists():
            try:
                checkpoint = torch.load(best_checkpoint, map_location='cpu')
                current_best = checkpoint.get('best_top5_accuracy', 0.0)
            except:
                pass
        
        # 체크포인트 저장
        save_dict = {
            'epoch': epoch,
            'model_state_dict': self.system.trainer.contrastive_learner.state_dict(),
            'optimizer_state_dict': self.system.trainer.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config.__dict__,
            'best_top5_accuracy': max(top5_accuracy, current_best)
        }
        
        torch.save(save_dict, checkpoint_path)
        self.log_message(f"💾 체크포인트 저장: {checkpoint_path}")
        
        # 최고 성능 업데이트
        if top5_accuracy > current_best:
            torch.save(save_dict, best_checkpoint)
            self.log_message(f"🏆 새로운 최고 성능! Top-5: {top5_accuracy*100:.1f}%")
    
    def train(self, dataset_path: str):
        """전체 학습 프로세스"""
        self.log_message("🚀 대규모 v2 학습 시작")
        self.log_message("=" * 80)
        
        # 모델 및 데이터 설정
        self.setup_model_and_data(dataset_path)
        
        # 학습 루프
        best_top5 = 0.0
        
        for epoch in range(1, self.config.max_epochs + 1):
            self.log_message(f"\n📚 Epoch {epoch}/{self.config.max_epochs} 시작")
            
            # 학습
            train_metrics = self.train_epoch(epoch)
            
            # 검증
            if epoch % self.config.eval_every_n_epochs == 0:
                val_metrics = self.validate(epoch)
                
                # 체크포인트 저장
                if epoch % self.config.save_every_n_epochs == 0:
                    self.save_checkpoint(epoch, val_metrics)
                
                # 최고 성능 추적
                current_top5 = val_metrics.get('top5_accuracy', 0)
                if current_top5 > best_top5:
                    best_top5 = current_top5
                    self.log_message(f"🎯 새로운 최고 성능: {best_top5*100:.1f}%")
            
            # 메모리 정리
            self.memory_monitor.cleanup_memory()
        
        # 최종 결과
        self.log_message(f"\n✨ 학습 완료!")
        self.log_message(f"   최고 Top-5 정확도: {best_top5*100:.1f}%")
        self.log_message(f"   총 학습 시간: {(time.time() - self.memory_monitor.start_time)/3600:.1f}시간")
        
        # 정리
        self.system.cleanup()


def run_distributed_training(rank, world_size, dataset_path: str):
    """분산 학습 실행"""
    try:
        # 분산 학습 설정
        setup_distributed_training(rank, world_size)
        
        # 설정 및 트레이너 생성
        config = LargeScaleTrainingConfig()
        trainer = LargeScaleTrainer(config, rank, world_size)
        
        # 학습 실행
        trainer.train(dataset_path)
        
    except Exception as e:
        print(f"❌ Rank {rank} 학습 실패: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_distributed()


def main():
    """메인 실행 함수"""
    print("🎯 대규모 v2 학습 시스템")
    print("=" * 80)
    
    # GPU 환경 설정
    gpu_count = setup_gpu_environment()
    
    # 데이터셋 경로 (65GB 데이터)
    dataset_path = input("📁 65GB 데이터셋 경로를 입력하세요: ").strip()
    if not dataset_path:
        dataset_path = "/path/to/your/65gb/dataset"  # 기본값
    
    if not Path(dataset_path).exists():
        print(f"❌ 데이터셋 경로가 존재하지 않습니다: {dataset_path}")
        return
    
    print(f"📊 설정 정보:")
    config = LargeScaleTrainingConfig()
    print(f"   배치 크기: {config.batch_size}")
    print(f"   최대 에포크: {config.max_epochs}")
    print(f"   학습률: {config.learning_rate}")
    print(f"   Temperature: {config.temperature}")
    print(f"   Mixed Precision: {config.mixed_precision}")
    print(f"   Gradient Accumulation: {config.gradient_accumulation_steps}")
    
    # 분산 학습 여부 결정
    if gpu_count > 1:
        print(f"\n🔥 {gpu_count}개 GPU로 분산 학습을 시작합니다...")
        mp.spawn(
            run_distributed_training,
            args=(gpu_count, dataset_path),
            nprocs=gpu_count,
            join=True
        )
    else:
        print(f"\n🔥 단일 GPU로 학습을 시작합니다...")
        trainer = LargeScaleTrainer(config)
        trainer.train(dataset_path)


if __name__ == "__main__":
    main()