# 설계 문서

## 개요

패션 이미지 추천 시스템을 위한 JSON Encoder 설계. K-Fashion 데이터셋의 JSON 메타데이터를 512차원 벡터로 변환하여 FashionCLIP 이미지 임베딩과 정렬되는 공통 임베딩 공간을 구축한다.

### 핵심 설계 원칙

- **단순성**: Embedding + MLP 구조만 사용, 복잡한 아키텍처 배제
- **고정 출력**: 512차원 벡터 고정, 하이퍼파라미터 최소화
- **모듈성**: PyTorch Module 기반 구현으로 재사용성 확보
- **연구 중심**: 실험과 분석에 집중할 수 있는 명확한 구조

## 아키텍처

### 전체 시스템 구조

```
K-Fashion Dataset (64GB)
├── 이미지 (BBox crop)
└── JSON 메타데이터
    ↓
[전처리 파이프라인]
    ↓
학습 데이터: (crop_image, json_metadata) 쌍
    ↓
┌─────────────────┐    ┌─────────────────┐
│ FashionCLIP     │    │ JSON Encoder    │
│ Encoder         │    │ (학습 대상)      │
│ (Frozen)        │    │                 │
└─────────────────┘    └─────────────────┘
    ↓                      ↓
512차원 이미지 임베딩    512차원 JSON 임베딩
    ↓                      ↓
    └──────────────────────┘
              ↓
        InfoNCE Loss
    (Temperature τ=0.07)
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

### 1. JSONEncoder 클래스

```python
class JSONEncoder(nn.Module):
    def __init__(self, vocab_sizes: Dict[str, int], 
                 embedding_dim: int = 128,
                 hidden_dim: int = 256):
        """
        Args:
            vocab_sizes: 각 필드별 vocabulary 크기
            embedding_dim: 각 필드 embedding 차원
            hidden_dim: MLP hidden layer 차원
        """
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            batch: {
                'category': [batch_size],
                'style': [batch_size, max_style_len],
                'silhouette': [batch_size],
                'material': [batch_size, max_material_len],
                'detail': [batch_size, max_detail_len]
            }
        Returns:
            torch.Tensor: [batch_size, 512] 정규화된 임베딩
        """
```

### 2. ContrastiveLearner 클래스

```python
class ContrastiveLearner(nn.Module):
    def __init__(self, json_encoder: JSONEncoder, 
                 fashionclip_encoder: FashionCLIPVisionModel,
                 temperature: float = 0.07):
        
    def forward(self, images: torch.Tensor, 
                json_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            images: [batch_size, 3, 224, 224]
            json_data: JSON 배치 데이터
        Returns:
            InfoNCE loss 값
        """
```

### 3. 데이터 전처리 인터페이스

```python
class FashionDataProcessor:
    def __init__(self, dataset_path: str, 
                 target_categories: List[str] = ['레트로', '로맨틱', '리조트']):
        
    def polygon_to_bbox(self, polygon: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        """Polygon 좌표를 BBox로 변환"""
        
    def crop_image_by_bbox(self, image: PIL.Image, 
                          bbox: Tuple[int, int, int, int]) -> PIL.Image:
        """BBox 기준으로 이미지 크롭"""
        
    def build_vocabulary(self, json_files: List[str]) -> Dict[str, Dict[str, int]]:
        """각 필드별 vocabulary 구축"""
        
    def process_json_fields(self, json_data: Dict) -> Dict[str, Union[int, List[int]]]:
        """JSON 필드를 vocabulary index로 변환"""
```

## 데이터 모델

### 입력 데이터 구조

```python
@dataclass
class FashionItem:
    """단일 패션 아이템 데이터"""
    image_path: str
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    category: str
    style: List[str]
    silhouette: str
    material: List[str]
    detail: List[str]
    
@dataclass
class ProcessedBatch:
    """학습용 배치 데이터"""
    images: torch.Tensor  # [batch_size, 3, 224, 224]
    category_ids: torch.Tensor  # [batch_size]
    style_ids: torch.Tensor  # [batch_size, max_style_len]
    silhouette_ids: torch.Tensor  # [batch_size]
    material_ids: torch.Tensor  # [batch_size, max_material_len]
    detail_ids: torch.Tensor  # [batch_size, max_detail_len]
    
    # 패딩 마스크 (다중 범주형 필드용)
    style_mask: torch.Tensor  # [batch_size, max_style_len]
    material_mask: torch.Tensor  # [batch_size, max_material_len]
    detail_mask: torch.Tensor  # [batch_size, max_detail_len]
```

### 모델 출력 구조

```python
@dataclass
class EmbeddingOutput:
    """임베딩 출력 결과"""
    image_embeddings: torch.Tensor  # [batch_size, 512]
    json_embeddings: torch.Tensor   # [batch_size, 512]
    similarity_matrix: torch.Tensor  # [batch_size, batch_size]
    loss: torch.Tensor              # scalar
```

### 학습 설정

```python
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
    target_categories: List[str] = field(default_factory=lambda: ['레트로', '로맨틱', '리조트'])
    image_size: int = 224
    crop_padding: float = 0.1  # BBox 크롭 시 패딩 비율
```

## 정확성 속성

*속성(Property)은 시스템의 모든 유효한 실행에서 참이어야 하는 특성이나 동작입니다. 속성은 인간이 읽을 수 있는 명세와 기계가 검증할 수 있는 정확성 보장 사이의 다리 역할을 합니다.*

### Property 1: 고정 출력 차원
*임의의* JSON 메타데이터 입력에 대해, JSON_Encoder의 출력은 정확히 512차원이어야 한다
**Validates: Requirements 1.1**

### Property 2: 정규화된 출력 벡터
*임의의* JSON_Encoder 출력 벡터에 대해, L2 norm이 1이어야 하고 FashionCLIP 이미지 임베딩과 cosine similarity 계산이 가능해야 한다
**Validates: Requirements 1.2**

### Property 3: FashionCLIP 모델 고정 상태 유지
*임의의* 학습 과정에서, FashionCLIP Image Encoder의 파라미터는 학습 전후가 동일해야 한다
**Validates: Requirements 1.5**

### Property 4: 다중 범주형 필드 처리
*임의의* 다중 범주형 필드(style, material, detail)에 대해, 리스트 형태의 입력을 올바르게 처리하고 mean pooling을 통해 단일 임베딩으로 집계해야 한다
**Validates: Requirements 2.2, 2.4, 2.5**

### Property 5: Positive Pair 생성
*임의의* 학습 배치에서, 각 이미지 임베딩은 해당하는 JSON 임베딩과 positive pair를 형성해야 한다
**Validates: Requirements 3.1**

### Property 6: Negative Pair 생성
*임의의* 학습 배치에서, 각 이미지 임베딩은 다른 모든 JSON 임베딩과 negative pair를 형성해야 한다
**Validates: Requirements 3.2**

### Property 7: InfoNCE Loss 계산
*임의의* 학습 배치에 대해, InfoNCE loss가 올바르게 계산되어야 하며 temperature τ=0.07을 사용해야 한다
**Validates: Requirements 3.3**

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

### 단위 테스트 전략

**핵심 영역**:
- JSON 필드별 임베딩 처리 (category, style, silhouette, material, detail)
- 다중 범주형 필드의 mean pooling 동작
- MLP 레이어 통과 후 출력 차원 및 정규화
- InfoNCE loss 계산 정확성
- 배치 처리 및 패딩 마스크 적용

**엣지 케이스**:
- 빈 리스트를 가진 다중 범주형 필드
- 최대 길이를 초과하는 다중 범주형 필드
- 배치 크기가 1인 경우
- 모든 필드가 동일한 값을 가지는 경우

**오류 조건**:
- 잘못된 vocabulary index 입력
- 차원 불일치 상황
- 메모리 부족 상황 시뮬레이션

### 속성 테스트 상세 명세

각 정확성 속성은 다음과 같이 구현됩니다:

**Property 1 테스트**:
```python
@given(json_batch=generate_json_batch())
def test_fixed_output_dimension(json_batch):
    """Feature: fashion-json-encoder, Property 1: 고정 출력 차원"""
    output = json_encoder(json_batch)
    assert output.shape[-1] == 512
```

**Property 2 테스트**:
```python
@given(json_batch=generate_json_batch())
def test_normalized_output(json_batch):
    """Feature: fashion-json-encoder, Property 2: 정규화된 출력 벡터"""
    output = json_encoder(json_batch)
    norms = torch.norm(output, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
```

**Property 3 테스트**:
```python
@given(training_batch=generate_training_batch())
def test_fashionclip_frozen_state(training_batch):
    """Feature: fashion-json-encoder, Property 3: FashionCLIP 모델 고정 상태 유지"""
    original_params = {name: param.clone() for name, param in fashionclip_model.named_parameters()}
    # 학습 스텝 실행
    loss = contrastive_learner(training_batch['images'], training_batch['json'])
    loss.backward()
    optimizer.step()
    # FashionCLIP 파라미터 변경 여부 확인
    for name, param in fashionclip_model.named_parameters():
        assert torch.equal(param, original_params[name])
```

**Property 4 테스트**:
```python
@given(multi_categorical_data=generate_multi_categorical_data())
def test_multi_categorical_processing(multi_categorical_data):
    """Feature: fashion-json-encoder, Property 4: 다중 범주형 필드 처리"""
    for field in ['style', 'material', 'detail']:
        field_data = multi_categorical_data[field]
        # 리스트 형태 입력이 올바르게 처리되는지 확인
        embedding = json_encoder._process_multi_categorical(field_data, field)
        assert embedding.shape[-1] == json_encoder.embedding_dim
```

### 통합 테스트

**전체 파이프라인 테스트**:
- K-Fashion 데이터셋 샘플을 이용한 end-to-end 테스트
- 전처리 → 모델 학습 → 임베딩 생성 → 유사도 계산 전 과정 검증
- 메모리 사용량 및 학습 시간 모니터링

**성능 벤치마크**:
- 배치 크기별 처리 속도 측정
- GPU 메모리 사용량 프로파일링
- 대용량 데이터셋에서의 안정성 검증