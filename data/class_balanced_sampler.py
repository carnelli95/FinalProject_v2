"""
Class-Balanced Sampler for Fashion JSON Encoder

클래스 불균형 문제 해결을 위한 샘플러
레트로 9% 문제를 해결하여 Top-5 성능 1-2% 향상 목표
"""

import torch
from torch.utils.data import Sampler
from typing import List, Dict, Iterator
import numpy as np
from collections import Counter


class ClassBalancedSampler(Sampler):
    """
    클래스 균형을 맞춘 샘플러
    
    각 배치에서 모든 클래스가 균등하게 표현되도록 보장
    레트로(9%) 클래스의 언더샘플링 문제 해결
    """
    
    def __init__(self, 
                 dataset,
                 batch_size: int,
                 oversample_minority: bool = True,
                 min_samples_per_class: int = 2):
        """
        Args:
            dataset: FashionDataset 인스턴스
            batch_size: 배치 크기
            oversample_minority: 소수 클래스 오버샘플링 여부
            min_samples_per_class: 배치당 클래스별 최소 샘플 수
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.oversample_minority = oversample_minority
        self.min_samples_per_class = min_samples_per_class
        
        # 클래스별 인덱스 구축
        self.class_indices = self._build_class_indices()
        self.classes = list(self.class_indices.keys())
        self.num_classes = len(self.classes)
        
        # 클래스별 샘플 수 계산
        self.class_counts = {cls: len(indices) for cls, indices in self.class_indices.items()}
        
        print(f"📊 클래스 분포:")
        for cls, count in self.class_counts.items():
            print(f"   {cls}: {count}개 ({count/len(dataset)*100:.1f}%)")
        
        # 에포크당 총 샘플 수 계산
        if oversample_minority:
            # 가장 많은 클래스 기준으로 오버샘플링
            max_samples = max(self.class_counts.values())
            self.samples_per_epoch = max_samples * self.num_classes
        else:
            # 원본 데이터셋 크기 유지
            self.samples_per_epoch = len(dataset)
    
    def _build_class_indices(self) -> Dict[str, List[int]]:
        """클래스별 인덱스 딕셔너리 구축"""
        class_indices = {}
        
        for idx, item in enumerate(self.dataset.fashion_items):
            category = item.category
            if category not in class_indices:
                class_indices[category] = []
            class_indices[category].append(idx)
        
        return class_indices
    
    def __iter__(self) -> Iterator[int]:
        """균형 잡힌 배치 생성"""
        # 각 클래스별로 샘플 인덱스 준비
        class_iterators = {}
        for cls in self.classes:
            indices = self.class_indices[cls].copy()
            
            if self.oversample_minority:
                # 소수 클래스 오버샘플링
                max_samples = max(self.class_counts.values())
                current_samples = len(indices)
                
                if current_samples < max_samples:
                    # 반복 샘플링으로 확장
                    repeat_factor = (max_samples + current_samples - 1) // current_samples
                    indices = indices * repeat_factor
                    indices = indices[:max_samples]
            
            # 셔플
            np.random.shuffle(indices)
            class_iterators[cls] = iter(indices)
        
        # 배치 생성
        batch = []
        samples_generated = 0
        
        while samples_generated < self.samples_per_epoch:
            # 각 클래스에서 균등하게 샘플링
            for cls in self.classes:
                try:
                    # 클래스별 최소 샘플 수만큼 추가
                    for _ in range(self.min_samples_per_class):
                        if len(batch) < self.batch_size and samples_generated < self.samples_per_epoch:
                            idx = next(class_iterators[cls])
                            batch.append(idx)
                            samples_generated += 1
                except StopIteration:
                    # 해당 클래스 샘플이 부족하면 다시 셔플해서 재시작
                    indices = self.class_indices[cls].copy()
                    if self.oversample_minority:
                        max_samples = max(self.class_counts.values())
                        current_samples = len(indices)
                        if current_samples < max_samples:
                            repeat_factor = (max_samples + current_samples - 1) // current_samples
                            indices = indices * repeat_factor
                            indices = indices[:max_samples]
                    
                    np.random.shuffle(indices)
                    class_iterators[cls] = iter(indices)
                    
                    # 다시 시도
                    try:
                        for _ in range(self.min_samples_per_class):
                            if len(batch) < self.batch_size and samples_generated < self.samples_per_epoch:
                                idx = next(class_iterators[cls])
                                batch.append(idx)
                                samples_generated += 1
                    except StopIteration:
                        continue
            
            # 배치가 완성되면 yield
            if len(batch) >= self.batch_size:
                yield from batch[:self.batch_size]
                batch = batch[self.batch_size:]
        
        # 남은 샘플들 처리
        if batch:
            yield from batch
    
    def __len__(self) -> int:
        """에포크당 총 샘플 수"""
        return self.samples_per_epoch


class HardNegativeMiner:
    """
    레트로 클래스를 위한 Hard Negative Mining
    
    레트로와 가장 유사한 로맨틱/리조트 샘플을 찾아
    더 어려운 negative 샘플로 사용
    """
    
    def __init__(self, model, dataset, device='cpu'):
        self.model = model
        self.dataset = dataset
        self.device = device
        
        # 레트로 샘플 인덱스
        self.retro_indices = []
        self.non_retro_indices = []
        
        for idx, item in enumerate(dataset.fashion_items):
            if item.category == '레트로':
                self.retro_indices.append(idx)
            else:
                self.non_retro_indices.append(idx)
    
    def find_hard_negatives(self, retro_idx: int, k: int = 5) -> List[int]:
        """
        특정 레트로 샘플에 대한 hard negative 찾기
        
        Args:
            retro_idx: 레트로 샘플 인덱스
            k: 반환할 hard negative 수
            
        Returns:
            hard negative 인덱스 리스트
        """
        # 실제 구현에서는 모델을 사용해 유사도 계산
        # 여기서는 랜덤 샘플링으로 대체 (실제로는 임베딩 유사도 사용)
        hard_negatives = np.random.choice(self.non_retro_indices, size=k, replace=False)
        return hard_negatives.tolist()


def create_balanced_dataloader(dataset, 
                             batch_size: int = 16,
                             oversample_minority: bool = True,
                             min_samples_per_class: int = 2,
                             num_workers: int = 0) -> torch.utils.data.DataLoader:
    """
    클래스 균형 DataLoader 생성
    
    Args:
        dataset: FashionDataset
        batch_size: 배치 크기
        oversample_minority: 소수 클래스 오버샘플링 여부
        min_samples_per_class: 배치당 클래스별 최소 샘플 수
        num_workers: 워커 프로세스 수
        
    Returns:
        클래스 균형이 맞춰진 DataLoader
    """
    sampler = ClassBalancedSampler(
        dataset=dataset,
        batch_size=batch_size,
        oversample_minority=oversample_minority,
        min_samples_per_class=min_samples_per_class
    )
    
    from .fashion_dataset import collate_fashion_batch
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fashion_batch,
        num_workers=num_workers,
        pin_memory=True
    )