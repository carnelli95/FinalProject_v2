"""
Configuration classes for the Fashion JSON Encoder system.

This module defines the training configuration and other system settings.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainingConfig:
    """학습 하이퍼파라미터"""
    batch_size: int = 64
    learning_rate: float = 1e-4
    temperature: float = 0.07  # InfoNCE temperature (고정)
    embedding_dim: int = 128   # 필드별 embedding 차원
    hidden_dim: int = 256      # MLP hidden 차원
    output_dim: int = 512      # 최종 출력 차원 (고정)
    dropout_rate: float = 0.1
    weight_decay: float = 1e-5
    max_epochs: int = 100
    
    # 데이터 관련
    target_categories: List[str] = field(default_factory=lambda: ['상의', '하의', '아우터'])
    image_size: int = 224
    crop_padding: float = 0.1  # BBox 크롭 시 패딩 비율