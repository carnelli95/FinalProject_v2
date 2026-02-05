# 설계 문서

## 개요

패션 이미지 추천 시스템을 위한 고도화된 JSON Encoder 설계. K-Fashion 데이터셋의 JSON 메타데이터를 512차원 벡터로 변환하여 FashionCLIP 이미지 임베딩과 정렬되는 공통 임베딩 공간을 구축하며, **임베딩 중심성 기반 베스트셀러 Proxy** 혁신 기술을 포함한다.

### 핵심 설계 원칙

- **혁신성**: 임베딩 중심성 기반 베스트셀러 Proxy 시스템 구현
- **성능 최적화**: Temperature 0.1에서 Top-5 64.1% 달성
- **Query-Aware**: All Queries vs Anchor Queries 차별화 평가
- **확장성**: 다양한 평가 메트릭 및 분석 도구 지원

### 🎯 핵심 혁신: 임베딩 중심성 기반 베스트셀러 Proxy

**핵심 아이디어**: "베스트셀러를 판매 데이터 없이, 임베딩 공간의 중심성으로 근사(proxy)한다"

**개념 직관**: "중심에 가까울수록 대중적이다"

```
임베딩 공간에서 많은 상품과 비슷한 디자인 → 트렌드성 디자인 → 잘 팔릴 가능성 ↑
```

## 아키텍처

### 전체 시스템 구조 (고도화된 버전)

```
K-Fashion Dataset (2,172 items)
├── 이미지 (BBox crop)
└── JSON 메타데이터
    ↓
[전처리 파이프라인]
    ↓
학습 데이터: (crop_image, json_metadata) 쌍
    ↓
┌─────────────────┐    ┌─────────────────┐
│ FashionCLIP     │    │ JSON Encoder    │
│ Encoder         │    │ (최적화됨)      │
│ (Frozen)        │    │ T=0.1 최적      │
└─────────────────┘    └─────────────────┘
    ↓                      ↓
512차원 이미지 임베딩    512차원 JSON 임베딩
    ↓                      ↓
    └──────────────────────┘
              ↓
        InfoNCE Loss
    (Temperature τ=0.1)
              ↓
        [임베딩 중심성 분석]
              ↓
    ┌─────────────────────────────┐
    │ 베스트셀러 Proxy 시스템      │
    ├─────────────────────────────┤
    │ 1. 글로벌 중심 벡터 계산     │
    │ 2. 중심성 점수 계산         │
    │ 3. Anchor Set 선정 (상위10%)│
    │ 4. Query-Aware 평가        │
    └─────────────────────────────┘
              ↓
    ┌─────────────────────────────┐
    │ 성능 분석 및 평가           │
    ├─────────────────────────────┤
    │ • All Queries Recall@10     │
    │ • Anchor Queries Recall@10  │
    │ • 카테고리별 중심성 분석     │
    │ • Temperature 최적화        │
    └─────────────────────────────┘
```

### 임베딩 중심성 기반 베스트셀러 Proxy 아키텍처

```
STEP 1: 전체 임베딩 추출
┌─────────────────────────────────┐
│ 모든 이미지 → FashionCLIP       │
│ 결과: [N, 512] 임베딩 매트릭스   │
└─────────────────────────────────┘
              ↓
STEP 2: 글로벌 중심 벡터 계산
┌─────────────────────────────────┐
│ global_center = mean(embeddings)│
│ normalize(global_center)        │
└─────────────────────────────────┘
              ↓
STEP 3: 중심성 점수 계산
┌─────────────────────────────────┐
│ for each embedding:             │
│   score = cosine_sim(           │
│     embedding, global_center)   │
└─────────────────────────────────┘
              ↓
STEP 4: Anchor Set 생성
┌─────────────────────────────────┐
│ threshold = percentile(90)      │
│ anchor_indices = scores >= threshold│
│ 결과: 상위 10% = 베스트셀러 Proxy│
└─────────────────────────────────┘
```

### JSON Encoder 상세 구조

```
JSON 입력 스키마:
{
  "category": string,      # 단일 범주형
  "style": list[string],   # 다중 범주형
  "silhouette": string,    # 단일 범주형
  "material": list[string], # 다중 범주형
  "detail": list[string]   # 다중 범주형
}
    ↓
┌─────────────────────────────────────┐
│ Field-wise Embedding Layers        │
├─────────────────────────────────────┤
│ category_emb: Embedding(vocab_size) │
│ style_emb: Embedding(vocab_size)    │
│ silhouette_emb: Embedding(vocab_size)│
│ material_emb: Embedding(vocab_size) │
│ detail_emb: Embedding(vocab_size)   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Field Processing                    │
├─────────────────────────────────────┤
│ • 단일 범주형: embedding lookup     │
│ • 다중 범주형: padding mask를 적용하여
               유효 token에 대해서만 mean pooling을 수행한다│
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Concatenation                       │
│ [cat_emb, style_emb, sil_emb,      │
│  mat_emb, det_emb]                 │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ MLP Layers                          │
├─────────────────────────────────────┤
│ Linear(concat_dim → hidden_dim)     │
│ ReLU()                              │
│ Dropout(0.1)                        │
│ Linear(hidden_dim → 512)            │
│ L2 Normalization                    │
└─────────────────────────────────────┘
    ↓
512차원 정규화된 벡터
```

## 컴포넌트 및 인터페이스

### 1. JSONEncoder 클래스 (최적화됨)

```python
class JSONEncoder(nn.Module):
    def __init__(self, vocab_sizes: Dict[str, int], 
                 embedding_dim: int = 128,
                 hidden_dim: int = 256):
        """
        최적화된 JSON Encoder
        - Temperature 0.1에서 최적 성능
        - Top-5 정확도 64.1% 달성
        """
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            batch: JSON 배치 데이터
        Returns:
            torch.Tensor: [batch_size, 512] 정규화된 임베딩
        """
```

### 2. EmbeddingCentralityProxy 클래스 (핵심 혁신)

```python
class EmbeddingCentralityProxy:
    """임베딩 중심성 기반 베스트셀러 Proxy 시스템"""
    
    def __init__(self, system: FashionEncoderSystem):
        self.system = system
        self.global_center = None
        self.centrality_scores = None
        self.anchor_indices = None  # 베스트셀러 Proxy
        
    def extract_all_embeddings(self) -> Dict[str, Any]:
        """전체 이미지 임베딩 추출"""
        
    def compute_global_center(self) -> np.ndarray:
        """글로벌 중심 벡터 계산"""
        
    def compute_centrality_scores(self) -> np.ndarray:
        """중심성 점수 계산 (코사인 유사도)"""
        
    def create_anchor_and_tail_sets(self, anchor_percentile: int = 90) -> Dict[str, Any]:
        """Anchor Set (상위 10%) 생성"""
        
    def run_complete_analysis(self) -> Dict[str, Any]:
        """전체 중심성 분석 파이프라인"""
```

### 3. AnchorBasedEvaluator 클래스 (Query-Aware 평가)

```python
class AnchorBasedEvaluator:
    """Anchor Set 기반 Query-aware 평가 시스템"""
    
    def __init__(self, system: FashionEncoderSystem, 
                 anchor_indices: List[int], tail_indices: List[int]):
        self.anchor_indices = anchor_indices  # 베스트셀러 Proxy
        self.tail_indices = tail_indices
        
    def create_query_datasets(self) -> Dict[str, Any]:
        """쿼리 타입별 데이터셋 생성"""
        
    def evaluate_query_set(self, query_name: str, query_indices: List[int]) -> Dict[str, float]:
        """특정 쿼리 셋 평가 (Recall@K 포함)"""
        
    def run_anchor_based_evaluation(self) -> Dict[str, Any]:
        """Anchor 기반 포괄적 평가"""
```

### 4. 고도화된 ContrastiveLearner 클래스

```python
class ContrastiveLearner(nn.Module):
    def __init__(self, json_encoder: JSONEncoder, 
                 fashionclip_encoder: FashionCLIPVisionModel,
                 temperature: float = 0.1):  # 최적화된 temperature
        
    def forward(self, images: torch.Tensor, 
                json_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        최적화된 대조 학습
        - Temperature 0.1 사용
        - 향상된 성능 메트릭 계산
        """
        
    def get_embeddings(self, images: torch.Tensor, 
                      json_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """임베딩 추출 (중심성 분석용)"""
```

## 데이터 모델

### 입력 데이터 구조 (고도화됨)

```python
@dataclass
class FashionItem:
    """단일 패션 아이템 데이터 (중심성 정보 포함)"""
    image_path: str
    bbox: Tuple[int, int, int, int]
    category: str
    style: List[str]
    silhouette: str
    material: List[str]
    detail: List[str]
    
    # 중심성 분석 결과 (런타임 추가)
    centrality_score: Optional[float] = None
    is_anchor: Optional[bool] = None  # 베스트셀러 Proxy 여부
    
@dataclass
class CentralityAnalysisResult:
    """중심성 분석 결과"""
    global_center: np.ndarray  # [512] 글로벌 중심 벡터
    centrality_scores: np.ndarray  # [N] 각 아이템의 중심성 점수
    anchor_indices: np.ndarray  # 상위 10% 인덱스 (베스트셀러 Proxy)
    tail_indices: np.ndarray   # 하위 50% 인덱스
    
    # 통계 정보
    mean_centrality: float
    std_centrality: float
    anchor_threshold: float
    
    # 카테고리별 분석
    category_centrality: Dict[str, Dict[str, float]]
    
@dataclass
class QueryAwareEvaluationResult:
    """Query-Aware 평가 결과"""
    all_queries_metrics: Dict[str, float]
    anchor_queries_metrics: Dict[str, float]  # 베스트셀러 Proxy
    tail_queries_metrics: Dict[str, float]
    
    # 성능 개선 분석
    anchor_improvement: float  # Anchor vs All 개선폭
    goal_achievement: Dict[str, Any]  # 목표 달성 여부
```

### 고도화된 학습 설정

```python
@dataclass
class OptimizedTrainingConfig:
    """최적화된 학습 하이퍼파라미터"""
    # 최적화된 설정
    temperature: float = 0.1  # 최적 성능 확인됨
    batch_size: int = 32      # Recall@10 계산을 위해 증가
    learning_rate: float = 1e-4
    max_epochs: int = 8       # Baseline v1 설정
    
    # 모델 구조
    embedding_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 512     # 고정
    dropout_rate: float = 0.1
    
    # 중심성 분석 설정
    anchor_percentile: int = 90  # 상위 10%
    tail_percentile: int = 50    # 하위 50%
    
    # 평가 설정
    recall_k_values: List[int] = field(default_factory=lambda: [3, 5, 10, 20])
    evaluation_batch_size: int = 32
    
    # 성능 목표
    target_all_queries_recall_10: float = 0.75  # 75%
    target_anchor_queries_recall_10: float = 0.85  # 85%
```

### 성능 메트릭 데이터 모델

```python
@dataclass
class ComprehensiveMetrics:
    """포괄적 성능 메트릭"""
    # 기본 메트릭
    top1_accuracy: float
    top5_accuracy: float
    mean_reciprocal_rank: float
    
    # Recall@K 메트릭
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    
    # 유사도 분석
    avg_positive_similarity: float
    avg_negative_similarity: float
    similarity_gap: float  # positive - negative
    
    # 임베딩 품질
    embedding_norm_mean: float
    embedding_norm_std: float
    is_properly_normalized: bool
    
    # 카테고리별 성능
    category_performance: Dict[str, Dict[str, float]]
```

## 정확성 속성

*속성(Property)은 시스템의 모든 유효한 실행에서 참이어야 하는 특성이나 동작입니다. 속성은 인간이 읽을 수 있는 명세와 기계가 검증할 수 있는 정확성 보장 사이의 다리 역할을 합니다.*

### Property 1: 글로벌 중심 벡터 계산
*임의의* 이미지 임베딩 집합에 대해, 글로벌 중심 벡터는 모든 임베딩의 평균으로 계산되고 L2 정규화되어야 한다
**Validates: Requirements 17.1**

### Property 2: 중심성 점수 계산 정확성
*임의의* 임베딩과 글로벌 중심 벡터에 대해, 중심성 점수는 코사인 유사도로 올바르게 계산되어야 하며 [-1, 1] 범위에 있어야 한다
**Validates: Requirements 17.2**

### Property 3: Anchor Set 선정 정확성
*임의의* 중심성 점수 배열에 대해, 상위 10% 임계값으로 선정된 Anchor Set의 모든 원소는 임계값 이상이어야 하고, 전체의 약 10%를 차지해야 한다
**Validates: Requirements 17.3**

### Property 4: Anchor Set 중심성 우월성
*임의의* 중심성 점수 분포에서, Anchor Set (상위 10%)의 평균 중심성은 전체 평균 중심성보다 높아야 한다
**Validates: Requirements 17.5**

### Property 5: Query-Aware 평가 분리
*임의의* 데이터셋과 Anchor 인덱스에 대해, All Queries와 Anchor Queries가 올바르게 분리되어야 하고, Anchor Queries는 전체의 부분집합이어야 한다
**Validates: Requirements 18.1**

### Property 6: 포괄적 평가 메트릭 계산
*임의의* 유사도 매트릭스에 대해, Recall@K (K=3,5,10,20), Top-1 정확도, MRR이 올바르게 계산되어야 하고, 모든 값은 [0, 1] 범위에 있어야 한다
**Validates: Requirements 18.2, 20.1, 20.2**

### Property 7: Anchor Queries 성능 우월성
*임의의* 평가 결과에서, Anchor Queries의 Recall@10 성능은 All Queries의 성능보다 높거나 같아야 한다 (베스트셀러 Proxy 가설 검증)
**Validates: Requirements 18.3**

### Property 8: 카테고리별 분석 일관성
*임의의* 카테고리별 데이터에 대해, 각 카테고리의 중심성 통계와 성능 메트릭이 올바르게 계산되어야 하고, 모든 카테고리의 합이 전체와 일치해야 한다
**Validates: Requirements 17.4, 20.3**

### Property 9: 임베딩 품질 보장
*임의의* 생성된 임베딩에 대해, L2 norm이 1이어야 하고, 임베딩 차원이 512여야 하며, 분산이 양수여야 한다
**Validates: Requirements 20.4**

### Property 10: 설정 및 출력 형식 검증
*임의의* 평가 설정에서, 배치 크기가 32 이상이어야 하고, 평가 결과가 유효한 JSON 형태로 직렬화 가능해야 한다
**Validates: Requirements 18.4, 18.5**

### Property 11: 포괄적 보고서 생성
*임의의* 성능 데이터에 대해, temperature 비교 보고서, 최적 설정 권장사항, 성능 추이 시각화가 일관된 형태로 생성되어야 한다
**Validates: Requirements 19.4, 19.5, 20.5**

## 오류 처리

### 입력 데이터 검증

```python
class InputValidator:
    @staticmethod
    def validate_json_batch(batch: Dict[str, torch.Tensor]) -> None:
        """JSON 배치 데이터 유효성 검사"""
        required_fields = ['category', 'style', 'silhouette', 'material', 'detail']
        
        for field in required_fields:
            if field not in batch:
                raise ValueError(f"Required field '{field}' missing from batch")
                
        # 차원 검사
        batch_size = batch['category'].size(0)
        for field, tensor in batch.items():
            if tensor.size(0) != batch_size:
                raise ValueError(f"Batch size mismatch in field '{field}'")
                
        # 데이터 타입 검사
        for field, tensor in batch.items():
            if not tensor.dtype == torch.long:
                raise ValueError(f"Field '{field}' must be torch.long type")
    
    @staticmethod
    def validate_image_batch(images: torch.Tensor) -> None:
        """이미지 배치 데이터 유효성 검사"""
        if len(images.shape) != 4:
            raise ValueError("Images must be 4D tensor [batch, channels, height, width]")
            
        if images.shape[1] != 3:
            raise ValueError("Images must have 3 channels (RGB)")
            
        if images.shape[2] != 224 or images.shape[3] != 224:
            raise ValueError("Images must be 224x224 pixels")
```

### 모델 상태 검증

```python
class ModelValidator:
    @staticmethod
    def validate_output_dimension(output: torch.Tensor, expected_dim: int = 512) -> None:
        """출력 차원 검증"""
        if output.shape[-1] != expected_dim:
            raise ValueError(f"Output dimension must be {expected_dim}, got {output.shape[-1]}")
    
    @staticmethod
    def validate_normalization(embeddings: torch.Tensor, tolerance: float = 1e-6) -> None:
        """임베딩 정규화 검증"""
        norms = torch.norm(embeddings, dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=tolerance):
            raise ValueError("Embeddings must be L2 normalized")
    
    @staticmethod
    def validate_fashionclip_frozen(fashionclip_model: nn.Module, 
                           original_params: Dict[str, torch.Tensor]) -> None:
        """FashionCLIP 모델 고정 상태 검증"""
        for name, param in fashionclip_model.named_parameters():
            if not torch.equal(param, original_params[name]):
                raise ValueError(f"FashionCLIP parameter '{name}' has been modified during training")
```

### 학습 과정 오류 처리

```python
class TrainingErrorHandler:
    @staticmethod
    def handle_loss_explosion(loss: torch.Tensor, threshold: float = 100.0) -> None:
        """Loss 폭발 감지 및 처리"""
        if loss.item() > threshold:
            raise RuntimeError(f"Loss exploded: {loss.item():.4f} > {threshold}")
    
    @staticmethod
    def handle_gradient_issues(model: nn.Module) -> None:
        """Gradient 문제 감지"""
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        if total_norm > 10.0:
            raise RuntimeError(f"Gradient norm too large: {total_norm:.4f}")
        elif total_norm < 1e-8:
            raise RuntimeError(f"Gradient norm too small: {total_norm:.8f}")
```

## 테스트 전략

### 이중 테스트 접근법

본 시스템은 **단위 테스트**와 **속성 기반 테스트**를 모두 활용하여 포괄적인 검증을 수행합니다:

- **단위 테스트**: 특정 예제, 엣지 케이스, 오류 조건 검증
- **속성 테스트**: 모든 입력에 대한 범용 속성 검증  
- **통합**: 두 접근법이 상호 보완하여 완전한 커버리지 제공

### 속성 기반 테스트 설정

**라이브러리**: PyTorch와 호환되는 Hypothesis 사용
**설정**: 각 속성 테스트당 최소 100회 반복 실행
**태그 형식**: **Feature: fashion-json-encoder, Property {번호}: {속성 텍스트}**

### 고도화된 테스트 전략

**핵심 영역**:
- **임베딩 중심성 분석**: 글로벌 중심 계산, 중심성 점수, Anchor Set 선정
- **Query-Aware 평가**: All vs Anchor Queries 분리, Recall@K 계산
- **Temperature 최적화**: 다양한 temperature 값에서의 성능 비교
- **카테고리별 분석**: 중심성 및 성능의 카테고리별 차이
- **포괄적 메트릭**: MRR, Positive/Negative Similarity, 임베딩 품질

**엣지 케이스**:
- 단일 카테고리만 있는 경우의 중심성 계산
- 배치 크기가 10 미만인 경우의 Recall@10 처리
- 모든 임베딩이 동일한 경우의 중심성 분석
- 극단적 temperature 값 (0.01, 1.0)에서의 동작

**성능 검증**:
- Baseline v2 모델의 Top-5 64.1% 달성 확인
- Temperature 0.1 vs 0.15 성능 차이 (8.8%p) 검증
- Anchor Queries의 성능 우월성 확인

### 속성 테스트 상세 명세

각 정확성 속성은 다음과 같이 구현됩니다:

**Property 1 테스트**:
```python
@given(embeddings=generate_image_embeddings())
def test_global_center_calculation(embeddings):
    """Feature: fashion-json-encoder, Property 1: 글로벌 중심 벡터 계산"""
    global_center = compute_global_center(embeddings)
    expected_center = embeddings.mean(axis=0)
    expected_center = expected_center / np.linalg.norm(expected_center)
    assert np.allclose(global_center, expected_center)
    assert np.isclose(np.linalg.norm(global_center), 1.0)
```

**Property 2 테스트**:
```python
@given(embedding=generate_single_embedding(), center=generate_center_vector())
def test_centrality_score_calculation(embedding, center):
    """Feature: fashion-json-encoder, Property 2: 중심성 점수 계산 정확성"""
    score = compute_centrality_score(embedding, center)
    expected_score = np.dot(embedding, center) / (np.linalg.norm(embedding) * np.linalg.norm(center))
    assert np.isclose(score, expected_score)
    assert -1.0 <= score <= 1.0
```

**Property 7 테스트**:
```python
@given(evaluation_results=generate_evaluation_results())
def test_anchor_queries_performance_superiority(evaluation_results):
    """Feature: fashion-json-encoder, Property 7: Anchor Queries 성능 우월성"""
    all_recall_10 = evaluation_results['all_queries']['recall_at_10']
    anchor_recall_10 = evaluation_results['anchor_queries']['recall_at_10']
    assert anchor_recall_10 >= all_recall_10  # 베스트셀러 Proxy 가설 검증
```

### 통합 테스트

**전체 파이프라인 테스트**:
- K-Fashion 데이터셋 2,172개 아이템을 이용한 end-to-end 테스트
- 임베딩 중심성 분석 → Anchor Set 생성 → Query-Aware 평가 전 과정 검증
- Temperature 최적화 실험 재현성 확인

**성능 벤치마크**:
- Baseline v2 모델 성능 재현 (Top-5 64.1%)
- 다양한 배치 크기에서의 Recall@K 계산 안정성
- 대용량 데이터셋에서의 중심성 분석 확장성

**혁신 기능 검증**:
- 임베딩 중심성 기반 베스트셀러 Proxy의 유효성
- Query-Aware 평가 시스템의 차별화 능력
- 카테고리별 중심성 인사이트의 일관성