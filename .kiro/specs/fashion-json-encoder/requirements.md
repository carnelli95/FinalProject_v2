# 요구사항 문서

## 소개

패션 이미지 추천 시스템 연구 프로젝트를 위한 JSON Encoder 구현. K-fashion 데이터셋의 JSON 메타데이터를 학습하여 FashionCLIP 이미지 임베딩과 정렬되는 512차원 Attribute Embedding을 생성한다.

## 프로젝트 목표

- **주요 목표**: K-Fashion 데이터셋 기반으로 JSON ↔ 이미지 유사도 검색 시스템 구축
- **현재 상태**: 2,172개 아이템으로 초기 학습 완료 (15 에포크)
- **향후 계획**: 추가 학습을 통한 성능 향상 및 실용적 검색 시스템 구축

## 데이터 스코프

- **데이터셋**: K-Fashion 데이터셋 (전체 64GB)
- **학습 대상 카테고리**: 레트로(196개), 로맨틱(994개), 리조트(998개) - 총 2,172개 아이템
- **데이터 구조**: C:/sample/라벨링데이터/{카테고리}/{파일번호}.json 형식
- **학습/검증 분할**: 80% / 20% (학습: 1,737개, 검증: 435개)
- **목표**: 이미지 ↔ JSON 멀티모달 임베딩 공간 정렬

## 전처리 요구사항

- **Polygon to BBox 변환**: 모든 polygon 어노테이션을 bounding box로 변환 필수
- **이미지 크롭**: BBox 기준으로 crop된 이미지 생성
- **BBox 없는 이미지**: 학습 데이터에서 제외
- **학습 단위**: (단일 의류 item crop 이미지, 해당 item의 JSON 메타데이터) 쌍

## JSON 입력 스키마 가정

JSON_Encoder는 전처리된 JSON을 입력으로 받으며, 각 필드는 다음과 같은 타입을 가진다:

- **category**: string (단일 값)
- **style**: list[string]
- **silhouette**: string (단일 값)
- **material**: list[string]
- **detail**: list[string]

모든 필드는 사전에 정의된 vocabulary index로 변환된 상태로 입력된다:

- **단일 범주형 필드**: embedding lookup 방식 사용
- **다중 범주형 필드**: 다중 범주형 필드는 embedding lookup 후 mean pooling 방식으로 집계한다

## 학습 목표

- **임베딩 공간 정렬**: 이미지 임베딩 ↔ JSON 임베딩 공간 정렬
- **손실 함수**: InfoNCE Loss 고정 사용
- **학습 방식**: Contrastive Learning (In-batch negative sampling)

## 출력물

- **JSON Encoder**: PyTorch 기반 신경망 모델
- **Image Encoder**: FashionCLIP 기반 (frozen 상태 유지)
- **공통 임베딩**: 512차원 벡터 공간

## 용어집

- **JSON_Encoder**: JSON 메타데이터를 512차원 벡터로 변환하는 신경망 모델
- **FashionCLIP_Image_Encoder**: 이미지를 512차원 벡터로 변환하는 사전 훈련된 모델 (학습 대상 아님)
- **Attribute_Embedding**: JSON Encoder가 출력하는 512차원 벡터
- **K_Fashion_Dataset**: 패션 이미지와 JSON 메타데이터 쌍으로 구성된 데이터셋
- **Contrastive_Learning**: positive/negative 쌍을 이용한 학습 방식

---

# 핵심 기능 요구사항

## 요구사항 1: JSON 임베딩 생성

**사용자 스토리:** 연구자로서, JSON 메타데이터를 512차원 벡터로 변환하고 싶다, 이미지 임베딩과 비교 가능한 공간에서 작업하기 위해서.

### 승인 기준

1. WHEN JSON 메타데이터가 입력되면, THE JSON_Encoder SHALL 정확히 512차원 벡터를 출력한다
2. WHEN 출력 벡터가 생성되면, THE JSON_Encoder SHALL FashionCLIP 이미지 임베딩과 cosine similarity 계산이 가능한 형태로 출력한다
3. THE JSON_Encoder SHALL PyTorch Module로 구현된다
4. THE JSON_Encoder SHALL Embedding과 MLP만을 사용한 단순한 구조로 구성된다
5. THE FashionCLIP_Image_Encoder SHALL NOT be updated or fine-tuned during training

## 요구사항 2: 패션 속성 처리

**사용자 스토리:** 연구자로서, 다양한 패션 속성을 처리하고 싶다, 각 속성의 특성에 맞는 인코딩을 위해서.

### 승인 기준

1. WHEN category 필드가 입력되면, THE JSON_Encoder SHALL 단일 범주형 데이터로 처리하며 embedding lookup 방식을 사용한다
2. WHEN style 필드가 입력되면, THE JSON_Encoder SHALL 다중 범주형 데이터로 처리한다
3. WHEN silhouette 필드가 입력되면, THE JSON_Encoder SHALL 단일 범주형 데이터로 처리한다
4. WHEN material 필드가 입력되면, THE JSON_Encoder SHALL 다중 범주형 데이터로 처리한다
5. WHEN detail 필드가 입력되면, THE JSON_Encoder SHALL 다중 범주형 데이터로 처리한다

## 요구사항 3: 대조 학습

**사용자 스토리:** 연구자로서, contrastive learning으로 모델을 학습하고 싶다, 이미지와 JSON 메타데이터 간의 의미적 정렬을 위해서.

### 승인 기준

1. WHEN 학습 데이터가 제공되면, THE 시스템 SHALL positive pair (이미지 임베딩, 해당 JSON)를 생성한다
2. WHEN 학습 데이터가 제공되면, THE 시스템 SHALL negative pair (이미지 임베딩, 다른 JSON들)를 생성한다
3. THE 시스템 SHALL InfoNCE loss 함수를 사용한다 (cosine similarity 기반 loss는 사용하지 않음)

## 요구사항 4: 모델 구조 문서화

**사용자 스토리:** 연구자로서, 모델 구조를 명확히 이해하고 싶다, 연구 목적에 맞는 구현을 위해서.

### 승인 기준

1. THE 시스템 SHALL JSON Encoder 모델 구조 다이어그램을 텍스트로 제공한다
2. THE 시스템 SHALL PyTorch Module 클래스 설계를 제공한다
3. THE 시스템 SHALL 입력 필드별 벡터 처리 흐름을 문서화한다
4. THE 시스템 SHALL 학습 시 데이터 흐름을 요약한다

## 요구사항 5: 데이터 로딩

**사용자 스토리:** 연구자로서, 새로운 카테고리별 폴더 구조에서 데이터를 로드하고 싶다, 레트로/로맨틱/리조트 카테고리별로 정리된 데이터를 효율적으로 처리하기 위해서.

### 승인 기준

1. THE 시스템 SHALL C:/sample/라벨링데이터/{카테고리}/ 경로에서 JSON 파일을 로드한다
2. WHEN load_dataset() 함수가 호출되면, THE 시스템 SHALL 레트로, 로맨틱, 리조트 폴더를 모두 스캔한다
3. WHEN JSON 파일이 발견되면, THE 시스템 SHALL 파일명에서 숫자 ID를 추출하여 매핑한다
4. THE 시스템 SHALL 로드된 데이터 개수와 카테고리별 분포를 출력한다
5. IF 데이터가 로드되지 않았다면, THE 시스템 SHALL 구체적인 경로 오류 메시지를 제공한다

## 요구사항 6: 모델 복잡도 제한

**사용자 스토리:** 연구자로서, 모델 복잡도를 제한하고 싶다, 연구 목적에 집중하기 위해서.

### 승인 기준

1. THE JSON_Encoder SHALL Transformer 구조를 사용하지 않는다
2. THE JSON_Encoder SHALL Attention 메커니즘을 사용하지 않는다
3. THE JSON_Encoder SHALL Graph 구조를 사용하지 않는다
4. THE JSON_Encoder SHALL 과도한 하이퍼파라미터 튜닝을 요구하지 않는다

## 요구사항 7: 유사도 검색 시스템

**사용자 스토리:** 연구자로서, 유사도 검색 시스템을 구현하고 싶다, JSON 쿼리로 유사한 패션 이미지를 찾기 위해서.

### 승인 기준

1. WHEN JSON 쿼리가 입력되면, THE 시스템 SHALL JSON을 512차원 임베딩으로 변환한다
2. WHEN 이미지 데이터베이스가 제공되면, THE 시스템 SHALL 모든 이미지를 512차원 임베딩으로 변환한다
3. THE 시스템 SHALL JSON 임베딩과 이미지 임베딩 간 코사인 유사도를 계산한다
4. THE 시스템 SHALL Top-K 유사한 이미지를 반환한다 (K는 사용자 지정 가능)
5. THE 시스템 SHALL 검색 결과를 시각화하여 저장한다

## 요구사항 8: 성능 평가

**사용자 스토리:** 연구자로서, 모델 성능을 정량적으로 평가하고 싶다, 연구 결과의 객관성을 확보하기 위해서.

### 승인 기준

1. THE 시스템 SHALL Precision@K 지표를 계산한다 (K=1,5,10)
2. THE 시스템 SHALL Recall@K 지표를 계산한다 (K=1,5,10)
3. THE 시스템 SHALL Mean Reciprocal Rank (MRR) 지표를 계산한다
4. THE 시스템 SHALL 카테고리별 검색 정확도를 분석한다
5. THE 시스템 SHALL 유사도 점수 분포를 시각화한다

## 요구사항 9: 성능 메트릭 추적

**사용자 스토리:** 연구자로서, 포괄적인 성능 메트릭을 추적하고 싶다, 모델 성능을 다각도로 평가하기 위해서.

### 승인 기준

1. THE 시스템 SHALL Top-1 정확도를 계산하고 기록한다
2. THE 시스템 SHALL Top-5 정확도를 계산하고 기록한다
3. THE 시스템 SHALL Mean Reciprocal Rank (MRR)를 계산한다
4. THE 시스템 SHALL Positive Similarity 평균을 추적한다
5. THE 시스템 SHALL Negative Similarity 평균을 추적한다
6. THE 시스템 SHALL 임베딩 정규화 상태를 확인한다 (L2 norm = 1)

## 요구사항 10: 결과물 관리

**사용자 스토리:** 연구자로서, 학습 결과물을 체계적으로 관리하고 싶다, 재현 가능한 연구를 위해서.

### 승인 기준

1. THE 시스템 SHALL 모델 체크포인트를 `checkpoints/best_model.pt`에 저장한다
2. THE 시스템 SHALL 학습 결과를 `results/training_results.json`에 저장한다
3. THE 시스템 SHALL 시각화 이미지를 `results/*.png`에 저장한다
4. THE 시스템 SHALL 유사도 검색 결과를 `results/similarity_search/`에 저장한다
5. THE 시스템 SHALL 모든 결과물에 타임스탬프를 포함한다

## 요구사항 11: 빠른 테스트 및 데모

**사용자 스토리:** 연구자로서, 테스트와 데모를 빠르게 실행하고 싶다, 개발 과정에서 빠른 피드백을 받기 위해서.

### 승인 기준

1. THE 시스템 SHALL 유사도 검색 데모에서 샘플 이미지 수를 조정 가능해야 한다 (기본값: 20개, 최대 100개)
2. THE 시스템 SHALL 단위 테스트에서 작은 배치 크기를 사용해야 한다 (테스트용 배치 크기: 2-4개)
3. THE 시스템 SHALL 테스트 모드에서 에포크 수를 줄여야 한다 (테스트용: 1-2 에포크)
4. THE 시스템 SHALL CPU 모드에서도 안정적으로 동작해야 한다
5. THE 시스템 SHALL 데모 스크립트에 `--fast` 또는 `--quick` 옵션을 제공해야 한다
6. THE 시스템 SHALL 테스트 실행 시간을 30초 이내로 유지해야 한다

## 요구사항 12: 학습 모니터링

**사용자 스토리:** 연구자 및 개발자로서, 학습 진행 상황을 간단하게 모니터링하고 싶다. tqdm 진행 바로 실시간 진행률을 확인하고, matplotlib으로 기본적인 학습 차트를 생성하여 학습 결과를 시각적으로 파악할 수 있어야 한다.

### 승인 기준

1. THE 시스템 SHALL tqdm을 사용하여 학습 진행 상황을 표시한다 (현재 에포크/전체 에포크, 경과 시간, 주요 메트릭)
2. THE 시스템 SHALL matplotlib을 사용하여 학습 완료 후 기본 차트를 생성한다 (2x2 서브플롯 구성)
3. THE 시스템 SHALL 학습 결과를 간단한 형태로 저장한다 (training_summary.json, training_charts.png)
4. THE 시스템 SHALL 최소한의 라이브러리만 사용한다 (tqdm, matplotlib)
5. THE 시스템 SHALL 콘솔에서 즉시 확인 가능한 진행률을 제공한다

## 요구사항 13: JSON Encoder 독립 검증

**사용자 스토리:** 연구자로서, JSON Encoder Standalone Training and Sanity Check를 수행하고 싶다, JSON 데이터 임베딩 구조와 학습 과정의 정상 동작을 확인하기 위해서.

### 승인 기준

1. THE 시스템 SHALL 실제 데이터가 없거나 테스트용 샘플 데이터일 때에도 synthetic data로 임베딩 및 추천 파이프라인을 점검할 수 있어야 한다
2. THE 시스템 SHALL 출력 임베딩이 512차원이며 정규화(Norm=1) 상태임을 확인해야 한다
3. THE 시스템 SHALL JSON 필드가 정상적으로 처리되는지 확인해야 한다
4. THE 시스템 SHALL Loss, Gradient, Batch Consistency, Field Processing 등을 종합 검증해야 한다
5. WHEN 모든 검증이 완료되면, THE 시스템 SHALL **SANITY CHECK PASS** 메시지를 출력해야 한다
6. THE 시스템 SHALL 검증 결과를 JSON 파일로 저장해야 한다 (`temp_logs/sanity_check_results.json`)

## 요구사항 14: 나인오즈 추천 시스템 API 구조

**사용자 스토리:** 나인오즈 개발팀으로서, 두 가지 별개의 추천 시스템을 구현하고 싶다, 내부 전략용과 고객 맞춤형 신상품 추천을 위해서.

### 승인 기준

#### 14.1 상위 10% 상품 → 신상품 추천 (내부용)
1. THE 시스템 SHALL `/api/recommend/top10_to_new` 엔드포인트를 제공해야 한다
2. THE 시스템 SHALL 나인오즈 판매량 상위 10% 상품 이미지를 입력으로 받아야 한다
3. THE 시스템 SHALL 상위 10% 상품 이미지를 FashionCLIP으로 512차원 임베딩 생성해야 한다
4. THE 시스템 SHALL 신상품 이미지 데이터베이스와 코사인 유사도를 계산해야 한다
5. THE 시스템 SHALL 높은 유사도 순으로 신상품을 매칭하여 반환해야 한다
6. THE 시스템 SHALL 신상품 트렌드 분석용 데이터를 제공해야 한다

#### 14.2 고객 입력 → 신상품 추천 (고객용)
7. THE 시스템 SHALL `/api/recommend/customer_input` 엔드포인트를 제공해야 한다
8. THE 시스템 SHALL 고객 업로드 이미지 또는 클릭 상품 이미지를 입력으로 받아야 한다
9. THE 시스템 SHALL 고객 입력을 FashionCLIP으로 512차원 임베딩 생성해야 한다
10. THE 시스템 SHALL 신상품 데이터베이스와 코사인 유사도를 계산해야 한다
11. THE 시스템 SHALL Top-K 유사 신상품을 개인화 추천으로 반환해야 한다

#### 14.3 공통 기능
12. THE 시스템 SHALL 신상품 이미지 데이터베이스를 관리해야 한다
13. THE 시스템 SHALL 실시간 임베딩 생성 및 유사도 계산을 수행해야 한다
14. THE 시스템 SHALL KPI 지표를 추적해야 한다 (추천 정확도, 응답 속도, 임베딩 품질)

## 요구사항 15: 단계별 개발 목표

**사용자 스토리:** 프로젝트 관리자로서, 추천 시스템 단계별 목표와 환경을 명확히 정의하고 싶다, 체계적인 개발 및 검증을 위해서.

### 승인 기준

#### 1단계: 샘플 검증
1. THE 시스템 SHALL 샘플 데이터 (~3개 카테고리)로 70% Top-k 유사도를 달성해야 한다
2. THE 시스템 SHALL 집 PC (CPU 또는 GTX 1660) 환경에서 동작해야 한다
3. THE 시스템 SHALL JSON Encoder 구조 검증을 완료해야 한다

#### 2단계: Stage2 모델 점검
4. THE 시스템 SHALL Stage2 학습 모델과 샘플 데이터로 Top-5 70%를 달성해야 한다
5. THE 시스템 SHALL 집 PC (CPU 가능, GPU 있으면 빠름) 환경에서 동작해야 한다
6. THE 시스템 SHALL Contrastive Learning 모델 임베딩 품질을 검증해야 한다

#### 3단계: 실전 추천 데모
7. THE 시스템 SHALL 실제 상위 10% 판매 상품 + 신상품 전체 데이터로 Top-5 70~90%를 달성해야 한다
8. THE 시스템 SHALL 고성능 서버 (GPU 권장, 24~48GB VRAM) 환경에서 동작해야 한다
9. THE 시스템 SHALL 평균 유사도, 추천 속도 KPI를 측정해야 한다

## 요구사항 16: 체크리스트 기반 검증

**사용자 스토리:** 품질 관리자로서, 체크리스트 기반 시스템 검증을 수행하고 싶다, 각 단계별 완료 상태를 명확히 추적하기 위해서.

### 승인 기준

#### Stage 점검 체크리스트
1. THE 시스템 SHALL JSON Encoder Standalone Sanity Check PASS 상태를 확인해야 한다
2. THE 시스템 SHALL Stage2 모델 임베딩 품질 점검 완료 상태를 확인해야 한다
3. THE 시스템 SHALL 추천 파이프라인 → 프론트/백 연동 테스트 완료 상태를 확인해야 한다

#### 실전 준비 체크리스트
4. THE 시스템 SHALL 상위 10% 상품 임베딩 생성 완료 상태를 확인해야 한다
5. THE 시스템 SHALL 신상품 전체 임베딩 생성 완료 상태를 확인해야 한다
6. THE 시스템 SHALL FastAPI에서 이미지/JSON 입력 → 임베딩 → Top-k 추천 흐름을 확인해야 한다

#### 검증 상태 추적
7. THE 시스템 SHALL 각 체크리스트 항목의 완료/미완료 상태를 추적해야 한다
8. THE 시스템 SHALL 검증 결과를 구조화된 보고서로 생성해야 한다
9. THE 시스템 SHALL 미완료 항목에 대한 구체적인 액션 아이템을 제공해야 한다

---

# 학습 파이프라인 사양

## 1단계: 독립 JSON 인코더 학습
- **에포크**: 5
- **배치 사이즈**: 64
- **임베딩 차원**: 512
- **출력 정규화**: L2 norm 유지
- **목적**: JSON 인코더 단독 학습으로 기본 표현 학습

## 2단계: 대조 학습 (Contrastive Learning)
- **에포크**: 10 (향후 50-100으로 확장 예정)
- **배치 사이즈**: 64-128 범위 권장
- **학습률**: 0.0001
- **온도(T)**: 0.07
- **목적**: 이미지-JSON 임베딩 공간 정렬

## 유사도 검색 사양
- **입력**: JSON 쿼리 (카테고리별)
- **처리**: JSON → 512차원 임베딩 변환
- **비교**: 이미지 임베딩과 코사인 유사도 계산
- **출력**: 카테고리별 Top-5 이미지 시각화 및 결과 저장
- **검증**: 초기 학습 단계에서도 의미 있는 유사도 결과 확인

## 성능 메트릭 정의

### 정량적 지표
- **Top-1 정확도**: 가장 유사한 이미지가 정답인 비율
- **Top-5 정확도**: 상위 5개 중 정답이 포함된 비율
- **Mean Reciprocal Rank (MRR)**: 정답의 평균 역순위
- **Positive Similarity**: 매칭 쌍의 평균 유사도
- **Negative Similarity**: 비매칭 쌍의 평균 유사도

### 정성적 지표
- **임베딩 정규화**: L2 norm = 1 확인
- **카테고리별 클러스터링**: 시각적 분석
- **검색 품질**: 사용자 관점 평가

---

# 하이퍼파라미터 튜닝 계획

## 지원 도구 및 프레임워크

### Optuna (베이지안 최적화)
- **장점**: 효율적인 하이퍼파라미터 탐색, 조기 종료 지원
- **적용 범위**: 소규모-중규모 실험 (단일 GPU)
- **최적화 알고리즘**: TPE (Tree-structured Parzen Estimator)

### Ray Tune (분산 튜닝)
- **장점**: 대규모 분산 실험, 다양한 스케줄러 지원
- **적용 범위**: 대규모 실험 (다중 GPU/노드)
- **스케줄러**: ASHA, PopulationBasedTraining

### W&B Sweep (실험 추적)
- **장점**: 실험 시각화, 협업 지원, 결과 분석
- **적용 범위**: 모든 규모의 실험
- **탐색 방법**: Grid, Random, Bayesian

## 튜닝 대상 하이퍼파라미터

### 우선순위 1: 학습 설정
- **에포크**: 50, 75, 100
- **배치 사이즈**: 64, 96, 128
- **학습률**: 0.0001, 0.0003, 0.0005, 0.001
- **온도(T)**: 0.05, 0.07, 0.1

### 우선순위 2: 모델 구조
- **임베딩 차원**: 256, 512, 768
- **은닉층 차원**: 256, 512, 1024
- **드롭아웃 비율**: 0.1, 0.2, 0.3

### 우선순위 3: 정규화 및 최적화
- **가중치 감쇠**: 1e-5, 1e-4, 1e-3
- **스케줄러**: CosineAnnealingLR, StepLR, ExponentialLR
- **그래디언트 클리핑**: 0.5, 1.0, 2.0

## 튜닝 전략

### 단계별 접근법
1. **1단계**: 기본 하이퍼파라미터 (에포크, 배치, 학습률) - 20회 실험
2. **2단계**: 모델 구조 (임베딩, 은닉층) - 15회 실험  
3. **3단계**: 세부 튜닝 (정규화, 스케줄러) - 10회 실험

### 평가 지표
- **주 지표**: Top-5 정확도
- **보조 지표**: MRR, 학습 안정성, 수렴 속도
- **조기 종료**: 검증 손실 기준 (patience=10)

---

# 대시보드 및 모니터링

## KPI 대시보드 구조

### 상단 KPI 카드 영역
- 총 학습 데이터 수, 카테고리별 아이템 수
- 현재 학습 에포크 및 진행률
- Top-1 / Top-5 정확도, MRR, Positive/Negative Similarity
- 임베딩 정규화 상태(L2 norm)

### 학습 손실 및 메트릭 시각화 영역
- Train / Validation Loss 곡선
- 임베딩 정규화 통계
- 학습률(LR) 변화

### 유사도 검색 시각화 영역
- 카테고리별 Top-K 유사 이미지
- JSON 쿼리 ↔ 이미지 코사인 유사도 점수 표시

### 하이퍼파라미터 / 튜닝 관리 영역
- 현재 batch size, epochs, learning rate, temperature 등 표시
- 자동 튜닝 결과 요약 (실험적)

### 데이터 증강 샘플 확인 영역
- 이미지 증강 예시: 회전, flip, color jitter
- JSON 증강 예시: Field Dropout, Synonym Replacement

## 데이터 제공 형식
- JSON Encoder 학습 및 유사도 검색 관련 데이터를 REST API 또는 JSON 파일 형태로 제공
- 각 영역별로 필요한 값과 구조를 포함: `{ kpi: {...}, loss_curve: [...], top_k_images: [...], hyperparams: {...}, augmented_samples: [...] }`
- Nest.js 기반 프론트와 연동 가능하도록 각 영역별 props/데이터 키 명시

---

# 데이터 증강 전략

## 이미지 데이터 증강

### 기본 증강 기법
- **회전**: ±15도 범위
- **수평 뒤집기**: 50% 확률
- **색상 조정**: brightness(±0.2), contrast(±0.2), saturation(±0.2)
- **크롭 및 리사이즈**: RandomResizedCrop (0.8-1.0 비율)

### 고급 증강 기법
- **MixUp**: 이미지 쌍 선형 결합
- **CutMix**: 이미지 영역 교체
- **AutoAugment**: 자동 증강 정책 학습

## JSON 데이터 증강

### 필드별 증강 전략
- **Field Dropout**: 20% 확률로 필드 마스킹
- **Synonym Replacement**: 스타일/소재 유의어 교체
- **Attribute Mixing**: 동일 카테고리 내 속성 혼합

### 카테고리별 증강
- **레트로**: 빈티지, 클래식 스타일 강화
- **로맨틱**: 페미닌, 우아함 속성 강화  
- **리조트**: 캐주얼, 편안함 속성 강화

---

# 고급 모델 구조 실험 계획

## Multi-head Attention 기반 JSON 인코더

### 구조 설계
```
JSON Fields → Field Embeddings → Multi-head Attention → Layer Norm → MLP → Output
```

### 설정 옵션
- **헤드 수**: 4, 8, 16
- **어텐션 차원**: 128, 256, 512
- **레이어 수**: 2, 4, 6

## 깊은 네트워크 구조

### 잔차 연결 (Residual Connection)
- **ResNet 스타일**: Skip connection 적용
- **레이어 수**: 3, 5, 7층
- **활성화 함수**: ReLU, GELU, Swish

### 정규화 기법
- **Batch Normalization**: 각 레이어 후 적용
- **Layer Normalization**: Transformer 스타일
- **Dropout**: 0.1, 0.2, 0.3 비율

---

# K-Fashion 단계별 개발 로드맵

## 1. 샘플 데이터 기반 검증 단계

**목적**: 샘플 데이터를 활용하여 모델 학습 파이프라인과 데이터 연동 구조를 검증

### 코드/JSON 구조 검증
- JSON 메타데이터 형식과 필드 일관성 확인
- 이미지 파일명, 식별자, 스타일/카테고리 라벨 존재 여부 확인
- 좌표 정보 (렉트/폴리곤) 형식 검증

### 프론트/백/JSON 연동 테스트
- FastAPI 기반 백엔드에서 JSON 데이터를 정상적으로 읽고 처리하는지 확인
- 프론트엔드 요청 시 JSON 응답 포맷 유지 및 임베딩 전달 검증
- 양방향 추천 시스템 시 이미지 → JSON, JSON → 이미지 추천 결과 확인

### 임베딩 품질 초기 확인
- 샘플 데이터로 학습한 임베딩으로 Top-5 유사도 확인 (목표 70%)
- 추천 시스템 구조 점검 및 시각화 가능 여부 확인

## 2. 전체 데이터 학습 단계 (65GB)

**목적**: 전체 데이터 기반으로 Contrastive Learning 학습을 수행하여 모델의 기본 추천 성능 확보

### 학습 목표
- Top-5 유사도 최소 70% 달성
- 전체 23개 카테고리 학습 완료
- 기타 폴더 제외 가능 (필요 시 포함)

### 환경 요구
- **GPU**: A100 / RTX 3090 이상
- **VRAM**: ≥24GB
- **RAM**: ≥128GB
- **CUDA + PyTorch Mixed Precision 지원**

### 검증 사항
- 학습 체크포인트 생성 및 Resume 기능 확인
- Stage2 모델 임베딩 품질 확인
- JSON → 이미지, 이미지 → JSON 추천 정상 작동 여부 확인

## 3. 하이퍼파라미터 튜닝 단계

**목적**: 전체 데이터 기반 학습 후 임베딩 품질을 극대화하고 Top-5/Top-1 유사도 90% 이상 달성

### 튜닝 항목
- learning rate, batch size, temperature, embedding dimension
- contrastive loss 관련 설정
- 데이터 증강 전략

### 환경 요구
- 학교 고성능 GPU/CPU 환경
- 장시간 학습 가능 (수십~수백 epoch)
- 체크포인트 관리 및 TensorBoard 로그 확인

### 검증 사항
- 임베딩 품질, Top-5/Top-1 유사도 향상 확인
- 프론트/백 연동 결과 재검증
- KPI 대시보드 업데이트 가능

## 4. 단계별 요구사항 요약 표

| 단계 | 데이터 범위 | 목표 Top-5 유사도 | 학습 목적 | 필요 환경 | 특징/비고 |
|------|-------------|-------------------|-----------|-----------|-----------|
| **1단계: 샘플 검증** | 샘플 데이터 (~3개 카테고리, 소량) | 70% | - 코드/JSON 구조 검증<br>- 프론트/백/JSON 연동 테스트<br>- 임베딩 품질 초기 확인 | 집 PC (CPU 또는 GTX 1660) | - 빠른 확인 가능<br>- 임베딩/추천 구조 점검용 |
| **2단계: 전체 데이터 학습** | 전체 데이터 (65GB, 23개 카테고리) | ≥70% | - 전체 데이터 기반 모델 학습<br>- Stage2 임베딩 품질 확보 | 학교 서버 GPU (≥24GB VRAM, 128GB RAM) | - 학습 체크포인트/Resume 필수<br>- 기타 폴더 포함 여부 선택 가능 |
| **3단계: 하이퍼파라미터 튜닝** | 전체 데이터 | 90% 이상 | - Contrastive learning 파라미터 최적화<br>- 임베딩 품질 최대화 | 학교 서버 GPU + CUDA, Mixed Precision | - Top-5/Top-1 유사도 개선 목표<br>- 장시간 학습 필요 |

---

# 현재 구현 상태 (2026-02-05 기준)

## ✅ 완료된 기능
- **전체 파이프라인**: JSON → 임베딩 → 이미지 임베딩 → 코사인 유사도 계산
- **데이터 로딩**: K-Fashion 데이터셋 2,172개 아이템 로드 (레트로: 196, 로맨틱: 994, 리조트: 998)
- **모델 학습**: 15 에포크 초기 학습 완료 (독립 학습 5 + 대조 학습 10)
- **유사도 검색**: Top-5 유사 이미지 검색 및 시각화 데모 구현
- **시스템 안정성**: 오류 없는 전체 파이프라인 실행
- **재사용성**: test_similarity_search.py 스크립트 제공

## 📊 현재 성능 지표
- **Top-5 정확도**: 1.04%
- **유사도 범위**: 0.0146~0.0340 (의미있는 값 생성)
- **평균 역순위**: 0.0543
- **임베딩 정규화**: L2 정규화 완료

## ⚠️ 현재 한계사항
- **학습 부족**: 15 에포크로 성능 제한
- **하이퍼파라미터**: 기본값 사용, 튜닝 미적용
- **데이터 증강**: 미적용 상태
- **모델 구조**: 단순한 MLP 기반 구조
- **평가 지표**: Top-K 유사도만 확인, 정량적 평가 부족

---

# 전역 제약 사항

1. JSON_Encoder의 출력 차원은 모든 경우에 512로 고정된다
2. 출력 차원 변경을 위한 실험 또는 대안 설계는 본 연구 범위에 포함되지 않는다
3. FashionCLIP 이미지 인코더는 frozen 상태를 유지한다
4. 모든 개선사항은 기존 파이프라인과의 호환성을 유지해야 한다