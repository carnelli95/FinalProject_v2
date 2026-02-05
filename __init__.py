"""
Fashion JSON Encoder

패션 메타데이터를 512차원 임베딩으로 변환하여 
CLIP 이미지 임베딩과 정렬하는 PyTorch 기반 시스템
"""

__version__ = "0.1.0"
__author__ = "Fashion JSON Encoder Team"

# 핵심 모듈 임포트
try:
    from models import JSONEncoder, ContrastiveLearner
    from data import FashionItem, ProcessedBatch, EmbeddingOutput, FashionDataProcessor
    from utils import TrainingConfig, InputValidator, ModelValidator, TrainingErrorHandler
    
    __all__ = [
        'JSONEncoder',
        'ContrastiveLearner', 
        'FashionItem',
        'ProcessedBatch',
        'EmbeddingOutput',
        'FashionDataProcessor',
        'TrainingConfig',
        'InputValidator',
        'ModelValidator',
        'TrainingErrorHandler'
    ]
except ImportError:
    # 개발 환경에서 모듈이 아직 완전히 설정되지 않은 경우
    __all__ = []