# 요구사항 문서

## 소개

패션 이미지 추천 시스템 연구 프로젝트를 위한 JSON Encoder 구현. K-fashion 데이터셋의 JSON 메타데이터를 학습하여 CLIP 이미지 임베딩과 정렬되는 512차원 Attribute Embedding을 생성한다.

## 데이터 스코프

- **데이터셋**: K-Fashion 데이터셋 (전체 64GB)
- **학습 대상 카테고리**: 상의, 하의, 아우터
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
- **Image Encoder**: CLIP 기반 (frozen 상태 유지)
- **공통 임베딩**: 512차원 벡터 공간

## 용어집

- **JSON_Encoder**: JSON 메타데이터를 512차원 벡터로 변환하는 신경망 모델
- **CLIP_Image_Encoder**: 이미지를 512차원 벡터로 변환하는 사전 훈련된 모델 (학습 대상 아님)
- **Attribute_Embedding**: JSON Encoder가 출력하는 512차원 벡터
- **K_Fashion_Dataset**: 패션 이미지와 JSON 메타데이터 쌍으로 구성된 데이터셋
- **Contrastive_Learning**: positive/negative 쌍을 이용한 학습 방식

## 요구사항

### 요구사항 1

**사용자 스토리:** 연구자로서, JSON 메타데이터를 512차원 벡터로 변환하고 싶다, 이미지 임베딩과 비교 가능한 공간에서 작업하기 위해서.

#### 승인 기준

1. WHEN JSON 메타데이터가 입력되면, THE JSON_Encoder SHALL 정확히 512차원 벡터를 출력한다
2. WHEN 출력 벡터가 생성되면, THE JSON_Encoder SHALL CLIP 이미지 임베딩과 cosine similarity 계산이 가능한 형태로 출력한다
3. THE JSON_Encoder SHALL PyTorch Module로 구현된다
4. THE JSON_Encoder SHALL Embedding과 MLP만을 사용한 단순한 구조로 구성된다
5. THE CLIP_Image_Encoder SHALL NOT be updated or fine-tuned during training


### 요구사항 2

**사용자 스토리:** 연구자로서, 다양한 패션 속성을 처리하고 싶다, 각 속성의 특성에 맞는 인코딩을 위해서.

#### 승인 기준

1. WHEN category 필드가 입력되면,
   THE JSON_Encoder SHALL 단일 범주형 데이터로 처리하며
   embedding lookup 방식을 사용한다
2. WHEN style 필드가 입력되면, THE JSON_Encoder SHALL 다중 범주형 데이터로 처리한다
3. WHEN silhouette 필드가 입력되면, THE JSON_Encoder SHALL 단일 범주형 데이터로 처리한다
4. WHEN material 필드가 입력되면, THE JSON_Encoder SHALL 다중 범주형 데이터로 처리한다
5. WHEN detail 필드가 입력되면, THE JSON_Encoder SHALL 다중 범주형 데이터로 처리한다

### 요구사항 3

**사용자 스토리:** 연구자로서, contrastive learning으로 모델을 학습하고 싶다, 이미지와 JSON 메타데이터 간의 의미적 정렬을 위해서.

#### 승인 기준

1. WHEN 학습 데이터가 제공되면, THE 시스템 SHALL positive pair (이미지 임베딩, 해당 JSON)를 생성한다
2. WHEN 학습 데이터가 제공되면, THE 시스템 SHALL negative pair (이미지 임베딩, 다른 JSON들)를 생성한다
3. THE 시스템 SHALL InfoNCE loss 함수를 사용한다 (cosine similarity 기반 loss는 사용하지 않음)

### 요구사항 4

**사용자 스토리:** 연구자로서, 모델 구조를 명확히 이해하고 싶다, 연구 목적에 맞는 구현을 위해서.

#### 승인 기준

1. THE 시스템 SHALL JSON Encoder 모델 구조 다이어그램을 텍스트로 제공한다
2. THE 시스템 SHALL PyTorch Module 클래스 설계를 제공한다
3. THE 시스템 SHALL 입력 필드별 벡터 처리 흐름을 문서화한다
4. THE 시스템 SHALL 학습 시 데이터 흐름을 요약한다

### 요구사항 5

**사용자 스토리:** 연구자로서, 모델 복잡도를 제한하고 싶다, 연구 목적에 집중하기 위해서.

#### 승인 기준

1. THE JSON_Encoder SHALL Transformer 구조를 사용하지 않는다
2. THE JSON_Encoder SHALL Attention 메커니즘을 사용하지 않는다
3. THE JSON_Encoder SHALL Graph 구조를 사용하지 않는다
4. THE JSON_Encoder SHALL 과도한 하이퍼파라미터 튜닝을 요구하지 않는다

## 전역 제약 사항

1. JSON_Encoder의 출력 차원은 모든 경우에 512로 고정된다
2. 출력 차원 변경을 위한 실험 또는 대안 설계는 본 연구 범위에 포함되지 않는다