"""
Fashion JSON Encoder

A PyTorch-based system for converting fashion metadata to 512-dimensional embeddings
aligned with CLIP image embeddings for fashion recommendation systems.
"""

__version__ = "0.1.0"
__author__ = "Fashion JSON Encoder Team"

# Core imports
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