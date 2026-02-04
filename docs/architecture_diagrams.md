# Fashion JSON Encoder - 시스템 아키텍처 및 데이터 흐름

## 전체 시스템 아키텍처

```mermaid
graph TB
    subgraph "Frontend (Nest.js)"
        UI[사용자 인터페이스]
        Upload[이미지 업로드]
        StyleForm[스타일 입력 폼]
        Results[추천 결과 표시]
    end
    
    subgraph "Backend API (FastAPI)"
        Router[API Router]
        ImageHandler[이미지 처리 핸들러]
        JSONHandler[JSON 처리 핸들러]
        RecommendationEngine[추천 엔진]
    end
    
    subgraph "ML Models"
        FashionCLIP[FashionCLIP Image Encoder<br/>🔒 Frozen]
        JSONEncoder[JSON Encoder<br/>📚 Trainable]
        SimilarityCalc[코사인 유사도 계산]
    end
    
    subgraph "Data Storage"
        ItemDB[(상품 데이터베이스)]
        EmbeddingCache[(임베딩 캐시)]
        ModelCheckpoints[(모델 체크포인트)]
    end
    
    subgraph "Monitoring & Analytics"
        KPIDashboard[KPI 대시보드]
        Metrics[성능 메트릭]
        Logs[로그 시스템]
    end
    
    %% Frontend to Backend
    UI --> Router
    Upload --> ImageHandler
    StyleForm --> JSONHandler
    
    %% Backend to ML Models
    ImageHandler --> FashionCLIP
    JSONHandler --> JSONEncoder
    FashionCLIP --> SimilarityCalc
    JSONEncoder --> SimilarityCalc
    
    %% ML Models to Recommendation
    SimilarityCalc --> RecommendationEngine
    RecommendationEngine --> Results
    
    %% Data Flow
    ItemDB --> RecommendationEngine
    EmbeddingCache --> SimilarityCalc
    ModelCheckpoints --> JSONEncoder
    
    %% Monitoring
    RecommendationEngine --> KPIDashboard
    Router --> Metrics
    Metrics --> Logs
    
    style FashionCLIP fill:#ffcccc
    style JSONEncoder fill:#ccffcc
    style KPIDashboard fill:#ccccff
```

## 프론트엔드 ↔ 백엔드 ↔ FastAPI 데이터 흐름

### 1. 이미지 기반 추천 흐름

```mermaid
sequenceDiagram
    participant F as Frontend (Nest.js)
    participant B as Backend API
    participant ML as ML Pipeline
    participant DB as Database
    
    F->>B: POST /api/recommend/image
    Note over F,B: {"input_type": "image", "file": FormData}
    
    B->>B: 이미지 검증 및 전처리
    B->>ML: 이미지 → FashionCLIP Encoder
    ML->>ML: 512차원 임베딩 생성
    
    ML->>DB: 임베딩 캐시에서 상품 임베딩 조회
    DB->>ML: 상품 임베딩 리스트 반환
    
    ML->>ML: 코사인 유사도 계산
    ML->>B: Top-K 유사 상품 ID 반환
    
    B->>DB: 상품 메타데이터 조회
    DB->>B: 상품 정보 반환
    
    B->>F: JSON 응답
    Note over B,F: {"recommendations": [...], "similarity_scores": [...]}
```

### 2. JSON 스타일 기반 추천 흐름

```mermaid
sequenceDiagram
    participant F as Frontend (Nest.js)
    participant B as Backend API
    participant ML as ML Pipeline
    participant DB as Database
    
    F->>B: POST /api/recommend/style
    Note over F,B: {"input_type": "json", "category": "상의", "style": ["레트로"]}
    
    B->>B: JSON 데이터 검증 및 전처리
    B->>ML: JSON → JSON Encoder
    ML->>ML: 512차원 임베딩 생성
    
    ML->>DB: 이미지 임베딩 캐시 조회
    DB->>ML: 이미지 임베딩 리스트 반환
    
    ML->>ML: 코사인 유사도 계산
    ML->>B: Top-K 유사 이미지 ID 반환
    
    B->>DB: 이미지 메타데이터 조회
    DB->>B: 이미지 정보 반환
    
    B->>F: JSON 응답
    Note over B,F: {"recommendations": [...], "similarity_scores": [...]}
```

## 상세 컴포넌트 아키텍처

### ML Pipeline 내부 구조

```mermaid
graph LR
    subgraph "Image Processing Pipeline"
        IMG[Input Image<br/>224x224x3] --> CLIP[FashionCLIP<br/>Vision Encoder]
        CLIP --> IMGEM[Image Embedding<br/>512차원, L2 정규화]
    end
    
    subgraph "JSON Processing Pipeline"
        JSON[JSON Input] --> VOCAB[Vocabulary<br/>Mapping]
        VOCAB --> EMB[Field Embeddings]
        EMB --> MLP[MLP Layers]
        MLP --> JSONEM[JSON Embedding<br/>512차원, L2 정규화]
    end
    
    subgraph "Similarity Calculation"
        IMGEM --> COS[Cosine Similarity<br/>Matrix]
        JSONEM --> COS
        COS --> TOPK[Top-K Selection]
        TOPK --> REC[Recommendations]
    end
    
    style CLIP fill:#ffcccc
    style EMB fill:#ccffcc
    style COS fill:#ffffcc
```

### JSON Encoder 상세 구조

```mermaid
graph TB
    subgraph "Input JSON Fields"
        CAT[category: string]
        STY[style: list[string]]
        SIL[silhouette: string]
        MAT[material: list[string]]
        DET[detail: list[string]]
    end
    
    subgraph "Embedding Layers"
        CATE[Category Embedding<br/>128차원]
        STYE[Style Embedding<br/>128차원]
        SILE[Silhouette Embedding<br/>128차원]
        MATE[Material Embedding<br/>128차원]
        DETE[Detail Embedding<br/>128차원]
    end
    
    subgraph "Processing Logic"
        SINGLE[단일 범주형<br/>Direct Lookup]
        MULTI[다중 범주형<br/>Mean Pooling]
    end
    
    subgraph "MLP Network"
        CONCAT[Concatenation<br/>640차원]
        LINEAR1[Linear Layer<br/>640 → 256]
        RELU[ReLU Activation]
        DROPOUT[Dropout 0.1]
        LINEAR2[Linear Layer<br/>256 → 512]
        L2NORM[L2 Normalization]
    end
    
    CAT --> CATE
    STY --> STYE
    SIL --> SILE
    MAT --> MATE
    DET --> DETE
    
    CATE --> SINGLE
    SILE --> SINGLE
    STYE --> MULTI
    MATE --> MULTI
    DETE --> MULTI
    
    SINGLE --> CONCAT
    MULTI --> CONCAT
    CONCAT --> LINEAR1
    LINEAR1 --> RELU
    RELU --> DROPOUT
    DROPOUT --> LINEAR2
    LINEAR2 --> L2NORM
    
    L2NORM --> OUTPUT[512차원 정규화된<br/>JSON 임베딩]
    
    style MULTI fill:#ccffcc
    style L2NORM fill:#ffcccc
```

## 데이터베이스 스키마

### 상품 데이터베이스 구조

```mermaid
erDiagram
    FASHION_ITEMS {
        string item_id PK
        string category
        string image_path
        json style_tags
        string silhouette
        json material_tags
        json detail_tags
        json bbox_coordinates
        timestamp created_at
        timestamp updated_at
    }
    
    EMBEDDINGS_CACHE {
        string item_id PK
        blob image_embedding
        blob json_embedding
        float embedding_norm
        timestamp computed_at
        string model_version
    }
    
    RECOMMENDATION_LOGS {
        string log_id PK
        string session_id
        string input_type
        json input_data
        json recommendations
        json similarity_scores
        timestamp request_time
        float response_time_ms
    }
    
    KPI_METRICS {
        string metric_id PK
        string metric_name
        float metric_value
        json metadata
        timestamp recorded_at
    }
    
    FASHION_ITEMS ||--|| EMBEDDINGS_CACHE : "item_id"
    FASHION_ITEMS ||--o{ RECOMMENDATION_LOGS : "recommended_items"
```

## KPI 대시보드 아키텍처

### 대시보드 컴포넌트 구조

```mermaid
graph TB
    subgraph "KPI Dashboard Frontend"
        HEADER[헤더 - 실시간 상태]
        CARDS[KPI 카드 영역]
        CHARTS[차트 시각화 영역]
        SEARCH[검색 결과 영역]
        PARAMS[하이퍼파라미터 영역]
        AUGMENT[데이터 증강 영역]
    end
    
    subgraph "KPI Cards"
        DATACNT[총 학습 데이터 수]
        EPOCH[현재 에포크/진행률]
        ACC[Top-1/Top-5 정확도]
        MRR[Mean Reciprocal Rank]
        SIM[Positive/Negative Similarity]
        NORM[임베딩 정규화 상태]
    end
    
    subgraph "Chart Visualizations"
        LOSS[Train/Validation Loss 곡선]
        METRICS[메트릭 변화 추이]
        LR[학습률 변화]
        EMBED[임베딩 분포 히스토그램]
    end
    
    subgraph "Search Results"
        TOPK[카테고리별 Top-K 이미지]
        SCORES[유사도 점수 표시]
        QUERY[JSON 쿼리 예시]
    end
    
    subgraph "Data Sources"
        TRAINING[학습 로그]
        CHECKPOINTS[체크포인트]
        CACHE[임베딩 캐시]
        REALTIME[실시간 메트릭]
    end
    
    CARDS --> DATACNT
    CARDS --> EPOCH
    CARDS --> ACC
    CARDS --> MRR
    CARDS --> SIM
    CARDS --> NORM
    
    CHARTS --> LOSS
    CHARTS --> METRICS
    CHARTS --> LR
    CHARTS --> EMBED
    
    SEARCH --> TOPK
    SEARCH --> SCORES
    SEARCH --> QUERY
    
    TRAINING --> CARDS
    CHECKPOINTS --> CHARTS
    CACHE --> SEARCH
    REALTIME --> HEADER
    
    style CARDS fill:#e1f5fe
    style CHARTS fill:#f3e5f5
    style SEARCH fill:#e8f5e8
```

## 배포 및 인프라 아키텍처

### 개발/스테이징/프로덕션 환경

```mermaid
graph TB
    subgraph "Development Environment"
        DEV_PC[개발 PC<br/>CPU/GTX 1660]
        DEV_DATA[샘플 데이터<br/>~3개 카테고리]
        DEV_TARGET[목표: 70% Top-K 유사도]
    end
    
    subgraph "Staging Environment"
        STAGE_SERVER[학교 서버<br/>≥24GB VRAM, 128GB RAM]
        STAGE_DATA[전체 데이터<br/>65GB, 23개 카테고리]
        STAGE_TARGET[목표: ≥70% Top-5 유사도]
    end
    
    subgraph "Production Environment"
        PROD_SERVER[고성능 서버<br/>24~48GB VRAM]
        PROD_DATA[상위 10% + 신상품<br/>전체 데이터]
        PROD_TARGET[목표: 70~90% Top-5 유사도]
    end
    
    subgraph "Shared Components"
        DOCKER[Docker Containers]
        NGINX[Load Balancer]
        REDIS[Caching Layer]
        POSTGRES[Database]
        MONITORING[모니터링 시스템]
    end
    
    DEV_PC --> STAGE_SERVER
    STAGE_SERVER --> PROD_SERVER
    
    DOCKER --> DEV_PC
    DOCKER --> STAGE_SERVER
    DOCKER --> PROD_SERVER
    
    NGINX --> PROD_SERVER
    REDIS --> PROD_SERVER
    POSTGRES --> PROD_SERVER
    MONITORING --> PROD_SERVER
    
    style DEV_PC fill:#fff3e0
    style STAGE_SERVER fill:#e8f5e8
    style PROD_SERVER fill:#e3f2fd
```

## 성능 최적화 전략

### 추론 최적화 파이프라인

```mermaid
graph LR
    subgraph "Input Processing"
        INPUT[사용자 입력] --> VALIDATE[입력 검증]
        VALIDATE --> PREPROCESS[전처리]
    end
    
    subgraph "Caching Layer"
        PREPROCESS --> CACHE_CHECK{캐시 확인}
        CACHE_CHECK -->|Hit| CACHE_RETURN[캐시된 결과 반환]
        CACHE_CHECK -->|Miss| MODEL_INFERENCE
    end
    
    subgraph "Model Inference"
        MODEL_INFERENCE[모델 추론] --> BATCH_PROCESS[배치 처리]
        BATCH_PROCESS --> GPU_COMPUTE[GPU 연산]
        GPU_COMPUTE --> EMBEDDING[임베딩 생성]
    end
    
    subgraph "Similarity Search"
        EMBEDDING --> FAISS[FAISS 인덱스 검색]
        FAISS --> TOPK_SELECT[Top-K 선택]
        TOPK_SELECT --> CACHE_STORE[결과 캐싱]
    end
    
    subgraph "Response"
        CACHE_STORE --> FORMAT[응답 포맷팅]
        CACHE_RETURN --> FORMAT
        FORMAT --> RESPONSE[최종 응답]
    end
    
    style CACHE_CHECK fill:#fff3e0
    style FAISS fill:#e8f5e8
    style GPU_COMPUTE fill:#e3f2fd
```

이 아키텍처 문서는 Fashion JSON Encoder 시스템의 전체적인 구조와 데이터 흐름을 시각적으로 보여줍니다. 각 컴포넌트 간의 상호작용과 데이터 변환 과정을 명확히 이해할 수 있도록 구성되었습니다.