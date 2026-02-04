# Fashion JSON Encoder - API 아키텍처 및 실행 가이드

## 📋 개요

이 문서는 Fashion JSON Encoder 시스템의 API 아키텍처, 데이터 흐름, 그리고 실행 방법을 설명합니다.

**나인오즈 비즈니스 요구사항 반영**: 이 시스템은 나인오즈의 실제 비즈니스 요구사항에 맞게 두 가지 별개의 추천 시스템을 제공합니다.

## 🏗️ 시스템 아키텍처

### 전체 구조
```
Frontend (Nest.js) ↔ Backend API (FastAPI) ↔ ML Models (PyTorch)
                                           ↕
                                    신상품 데이터베이스 & 캐시
```

### 핵심 컴포넌트
- **FastAPI 서버**: 나인오즈 비즈니스 로직 기반 RESTful API 제공
- **FashionCLIP**: 이미지 → 512차원 임베딩 (Frozen)
- **신상품 추천 엔진**: 코사인 유사도 기반 신상품 매칭
- **이중 추천 시스템**: 내부 전략용 + 고객 맞춤용

## 🎯 나인오즈 비즈니스 로직

### 1. 내부 전략용 추천 시스템
- **목적**: 트렌드 분석 및 신상품 기획 전략 수립
- **입력**: 나인오즈 상위 10% 판매 상품 이미지
- **출력**: 유사한 스타일의 신상품 추천
- **활용**: 어떤 신상품이 인기 상품과 유사한지 분석

### 2. 고객 맞춤용 추천 시스템
- **목적**: 개인화된 신상품 추천
- **입력**: 고객이 업로드하거나 클릭한 상품 이미지
- **출력**: 고객 취향에 맞는 신상품 추천
- **활용**: 고객에게 개인화된 신상품 노출

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 필요한 패키지 설치
pip install fastapi uvicorn torch torchvision pillow aiohttp

# 프로젝트 디렉토리로 이동
cd fashion-json-encoder
```

### 2. API 서버 실행
```bash
# 방법 1: 직접 실행
python start_api_server.py

# 방법 2: uvicorn 직접 사용
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. 시스템 검증
```bash
# 전체 시스템 통합 테스트
python test_system_integration.py

# API 데모 및 Sanity Check
python demo_api.py
```

## 📡 API 엔드포인트

### 기본 정보
- **Base URL**: `http://localhost:8000`
- **API 문서**: `http://localhost:8000/docs`
- **헬스 체크**: `http://localhost:8000/health`

### 주요 엔드포인트

#### 1. 상위 10% 상품 → 신상품 추천 (나인오즈 내부용)
```http
POST /api/recommend/top10_to_new
Content-Type: multipart/form-data

# 요청
{
  "file": "상위 10% 판매 상품 이미지",
  "top_k": 5,
  "similarity_threshold": 0.1
}

# 응답
{
  "status": "success",
  "input_info": {
    "input_type": "top10_product_image",
    "business_purpose": "internal_trend_analysis"
  },
  "recommendations": [
    {
      "item_id": "new_item_009",
      "category": "신상 하의",
      "style": ["트렌디", "모던"],
      "similarity_score": 0.8999,
      "metadata": {
        "is_new_product": true,
        "business_context": "internal_trend_analysis",
        "launch_date": "2026-02-01"
      }
    }
  ]
}
```

#### 2. 고객 입력 → 신상품 추천 (고객용)
```http
POST /api/recommend/customer_input
Content-Type: multipart/form-data

# 요청
{
  "file": "고객 업로드/클릭 상품 이미지",
  "top_k": 10,
  "similarity_threshold": 0.2
}

# 응답
{
  "status": "success",
  "input_info": {
    "input_type": "customer_input_image",
    "business_purpose": "personalized_customer_recommendation"
  },
  "recommendations": [
    {
      "item_id": "new_item_027",
      "category": "신상 아우터",
      "similarity_score": 0.9234,
      "metadata": {
        "is_new_product": true,
        "business_context": "personalized_recommendation"
      }
    }
  ]
}
```

#### 3. JSON 스타일 기반 추천 (레거시 - 호환성 유지)
```http
POST /api/recommend/style
Content-Type: application/json

# 참고: 레거시 엔드포인트로, 새로운 비즈니스 로직에서는 
# 위의 두 엔드포인트 사용을 권장합니다.
```

#### 4. KPI 대시보드 데이터
```http
GET /api/dashboard/kpi

# 응답
{
  "kpi_cards": {
    "training_data": {
      "total_items": 2172,
      "categories": {"레트로": 196, "로맨틱": 994, "리조트": 998}
    },
    "performance_metrics": {
      "top_5_accuracy": 0.1045,
      "mrr": 0.0543
    }
  }
}
```

## 🔄 나인오즈 비즈니스 데이터 흐름

### 신상품 추천 프로세스

#### 내부 전략용 (상위 10% → 신상품)
1. **입력**: 나인오즈 상위 10% 판매 상품 이미지
2. **임베딩 생성**: FashionCLIP으로 512차원 벡터 생성
3. **신상품 매칭**: 신상품 데이터베이스와 코사인 유사도 계산
4. **트렌드 분석**: 카테고리별 다양성 확보로 트렌드 인사이트 제공

#### 고객 맞춤용 (고객 입력 → 신상품)
1. **입력**: 고객 업로드 또는 클릭 상품 이미지
2. **임베딩 생성**: FashionCLIP으로 512차원 벡터 생성
3. **개인화 매칭**: 신상품 데이터베이스에서 개인 취향 반영
4. **추천 제공**: Top-K 신상품을 개인화 순서로 제공

### 신상품 데이터베이스 관리
```python
# 신상품 임베딩 사전 계산 및 캐시
new_products = load_new_products_from_database()
new_product_embeddings = []

for product in new_products:
    embedding = fashionclip_model.encode_image(product.image)
    normalized_embedding = F.normalize(embedding, p=2, dim=-1)
    new_product_embeddings.append({
        "item_id": product.id,
        "embedding": normalized_embedding,
        "launch_date": product.launch_date,
        "category": product.category
    })

# 신상품 임베딩 캐시 저장
save_to_cache(new_product_embeddings, "new_products_embeddings")
```

## 🧪 테스트 및 검증

### 나인오즈 비즈니스 로직 테스트
```bash
python demo_api.py
```

**검증 항목:**
- ✅ 상위 10% → 신상품 추천 엔드포인트 동작
- ✅ 고객 입력 → 신상품 추천 엔드포인트 동작
- ✅ 신상품 데이터베이스 쿼리 로직
- ✅ 비즈니스 컨텍스트별 추천 결과 차이
- ✅ 512차원 임베딩 및 L2 정규화

### 통합 테스트 실행
```bash
python test_system_integration.py
```

**테스트 단계:**
1. **Stage 1**: 나인오즈 API 엔드포인트 검증
2. **Stage 2**: 신상품 추천 로직 검증
3. **Stage 3**: 비즈니스 메트릭 추적 검증

## 📊 성능 모니터링

### 나인오즈 비즈니스 KPI
- **내부 전략 KPI**: 
  - 상위 10% 상품 대비 신상품 유사도 분포
  - 카테고리별 트렌드 매칭 정확도
  - 신상품 기획 인사이트 품질
- **고객 맞춤 KPI**:
  - 개인화 추천 클릭률 (CTR)
  - 신상품 구매 전환율
  - 고객 만족도 점수

### 기술적 메트릭
- **API 응답 시간**: 실시간 성능 측정
- **임베딩 품질**: 코사인 유사도 분포
- **캐시 효율성**: 신상품 임베딩 캐시 히트율

### 대시보드 구성
1. **비즈니스 KPI 카드**: 나인오즈 핵심 지표 실시간 표시
2. **추천 성능**: 내부용 vs 고객용 성능 비교
3. **신상품 분석**: 신상품별 추천 빈도 및 성과
4. **시스템 모니터링**: API 성능 및 리소스 사용량

## 🔧 개발 환경별 설정

### 1단계: 나인오즈 API 검증 (개발 PC)
- **환경**: CPU 또는 GTX 1660
- **데이터**: 신상품 샘플 데이터
- **목표**: 두 추천 시스템 엔드포인트 정상 동작
- **용도**: 비즈니스 로직 검증, API 연동 테스트

### 2단계: 신상품 데이터베이스 구축 (학교 서버)
- **환경**: ≥24GB VRAM, 128GB RAM
- **데이터**: 전체 신상품 데이터 + 상위 10% 상품 데이터
- **목표**: 실시간 추천 성능 확보
- **용도**: 본격적인 신상품 추천 서비스

### 3단계: 실전 서비스 배포 (고성능 서버)
- **환경**: 24~48GB VRAM
- **데이터**: 나인오즈 전체 상품 + 실시간 신상품 업데이트
- **목표**: 상용 서비스 수준 성능
- **용도**: 실제 고객 대상 서비스

## 🐛 문제 해결

### 나인오즈 비즈니스 로직 관련

1. **신상품 데이터베이스 연결 실패**
```bash
# 신상품 데이터 경로 확인
ls "신상품_데이터베이스/"

# 신상품 임베딩 캐시 확인
ls "cache/new_products_embeddings.pt"
```

2. **추천 결과 차이 없음**
```python
# 비즈니스 컨텍스트 확인
print(f"Business context: {recommendation.metadata.business_context}")

# 데이터베이스 타입 확인
print(f"Database type: {search_type}")
```

3. **상위 10% 상품 식별 실패**
```python
# 판매 데이터 연동 확인
top_products = get_top_selling_products(percentage=10)
print(f"Top 10% products count: {len(top_products)}")
```

### 일반적인 오류

1. **모델 로딩 실패**
```bash
# 체크포인트 파일 확인
ls checkpoints/best_model.pt

# 없으면 랜덤 초기화로 진행
```

2. **API 서버 시작 실패**
```bash
# 포트 충돌 확인
netstat -an | findstr :8000

# 다른 포트 사용
uvicorn api.main:app --port 8001
```

### 성능 최적화

1. **신상품 임베딩 캐시 최적화**
```python
# 신상품 임베딩 사전 계산 및 캐시
@lru_cache(maxsize=1000)
def get_new_product_embedding(product_id):
    return load_cached_embedding(product_id)

# 배치 임베딩 생성
def batch_generate_embeddings(new_products):
    with torch.no_grad():
        embeddings = fashionclip_model.encode_batch(new_products)
    return F.normalize(embeddings, p=2, dim=-1)
```

2. **추천 응답 속도 개선**
```python
# 비동기 처리
async def parallel_similarity_search(query_embedding, databases):
    tasks = [search_database(query_embedding, db) for db in databases]
    results = await asyncio.gather(*tasks)
    return combine_results(results)
```

## 📁 파일 구조

```
fashion-json-encoder/
├── api/
│   └── main.py                 # 나인오즈 비즈니스 로직 FastAPI 서버
├── docs/
│   ├── architecture_diagrams.md
│   └── json_data_flow.md       # 나인오즈 API 명세 포함
├── models/
│   ├── json_encoder.py         # 레거시 호환성 유지
│   └── contrastive_learner.py
├── data/
│   ├── dataset_loader.py
│   └── fashion_dataset.py
├── utils/
│   └── validators.py
├── cache/                      # 신상품 임베딩 캐시
│   └── new_products_embeddings.pt
├── temp_logs/                  # 테스트 결과 저장
├── start_api_server.py         # 서버 시작 스크립트
├── demo_api.py                 # 나인오즈 API 데모
├── test_system_integration.py  # 비즈니스 로직 통합 테스트
└── README_API_ARCHITECTURE.md  # 이 문서
```

## 🔗 관련 문서

- [아키텍처 다이어그램](docs/architecture_diagrams.md)
- [JSON 데이터 흐름](docs/json_data_flow.md)
- [요구사항 문서](.kiro/specs/fashion-json-encoder/requirements.md)
- [설계 문서](.kiro/specs/fashion-json-encoder/design.md)

## 📞 지원

문제가 발생하면 다음을 확인하세요:

1. **로그 파일**: `temp_logs/` 디렉토리의 결과 파일들
2. **API 문서**: `http://localhost:8000/docs`
3. **헬스 체크**: `http://localhost:8000/health`
4. **통합 테스트**: `python test_system_integration.py`

---

이 가이드를 통해 나인오즈의 비즈니스 요구사항에 맞는 Fashion JSON Encoder 시스템을 성공적으로 실행하고 테스트할 수 있습니다. 

**핵심 포인트:**
- 상위 10% 상품 → 신상품 추천 (내부 전략용)
- 고객 입력 → 신상품 추천 (고객 맞춤용)
- 신상품 데이터베이스 중심의 추천 시스템
- 실시간 임베딩 생성 및 유사도 계산

추가 질문이나 문제가 있으면 언제든 문의하세요!