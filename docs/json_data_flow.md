# JSON 데이터 흐름 및 API 명세

## 나인오즈 비즈니스 요구사항 반영

이 API는 나인오즈의 실제 비즈니스 요구사항에 맞게 설계된 두 가지 별개의 추천 시스템을 제공합니다:

1. **내부 전략용**: 상위 10% 상품 → 신상품 추천 (트렌드 분석)
2. **고객 맞춤용**: 고객 입력 → 신상품 추천 (개인화 추천)

## API 엔드포인트 명세

### 1. 상위 10% 상품 → 신상품 추천 API (나인오즈 내부용)

#### 요청 (Frontend → Backend)

**엔드포인트**: `POST /api/recommend/top10_to_new`

**Content-Type**: `multipart/form-data`

```json
{
  "input_type": "top10_product_image",
  "file": "FormData 객체 (상위 10% 판매 상품 이미지)",
  "top_k": 5,
  "similarity_threshold": 0.1
}
```

#### 응답 (Backend → Frontend)

**Content-Type**: `application/json`

```json
{
  "status": "success",
  "request_id": "req_20260205_001",
  "input_info": {
    "input_type": "top10_product_image",
    "file_name": "top_seller_001.jpg",
    "image_size": [224, 224],
    "processed_at": "2026-02-05T10:30:00Z",
    "business_purpose": "internal_trend_analysis"
  },
  "recommendations": [
    {
      "item_id": "new_item_009",
      "category": "신상 하의",
      "style": ["트렌디", "모던"],
      "silhouette": "스트레이트",
      "material": ["니트", "폴리에스터"],
      "detail": ["지퍼", "포켓"],
      "similarity_score": 0.8999,
      "image_url": "/images/new_products/new_item_009.jpg",
      "metadata": {
        "brand": "나인오즈 신상",
        "price": 98000,
        "color": "네이비",
        "is_new_product": true,
        "business_context": "internal_trend_analysis",
        "launch_date": "2026-02-01"
      }
    }
  ],
  "performance_metrics": {
    "embedding_time_ms": 45.2,
    "similarity_search_time_ms": 12.8,
    "total_response_time_ms": 78.5,
    "cache_hit": false
  }
}
```

### 2. 고객 입력 → 신상품 추천 API (고객용)

#### 요청 (Frontend → Backend)

**엔드포인트**: `POST /api/recommend/customer_input`

**Content-Type**: `multipart/form-data`

```json
{
  "input_type": "customer_input_image",
  "file": "FormData 객체 (고객 업로드 또는 클릭 상품 이미지)",
  "top_k": 10,
  "similarity_threshold": 0.2
}
```

#### 응답 (Backend → Frontend)

**Content-Type**: `application/json`

```json
{
  "status": "success",
  "request_id": "req_20260205_002",
  "input_info": {
    "input_type": "customer_input_image",
    "file_name": "customer_upload_001.jpg",
    "image_size": [224, 224],
    "processed_at": "2026-02-05T10:35:00Z",
    "business_purpose": "personalized_customer_recommendation"
  },
  "recommendations": [
    {
      "item_id": "new_item_027",
      "category": "신상 아우터",
      "style": ["심플", "미니멀"],
      "silhouette": "테일러드",
      "material": ["니트", "폴리에스터"],
      "detail": ["버튼", "라펠"],
      "similarity_score": 0.9234,
      "image_url": "/images/new_products/new_item_027.jpg",
      "metadata": {
        "brand": "트렌드 브랜드",
        "price": 134000,
        "color": "블랙",
        "is_new_product": true,
        "business_context": "personalized_recommendation",
        "launch_date": "2026-02-01"
      }
    }
  ],
  "performance_metrics": {
    "json_encoding_time_ms": 8.3,
    "similarity_search_time_ms": 15.7,
    "total_response_time_ms": 34.2,
    "cache_hit": true
  }
}
```

### 3. JSON 스타일 기반 추천 API (레거시 - 호환성 유지) <-- 이거는 필요 없을 것으로 보임 1번 2번만 참고!!

#### 요청 (Frontend → Backend)

**엔드포인트**: `POST /api/recommend/image`

**Content-Type**: `multipart/form-data`

```json
{
  "input_type": "image",
  "file": "FormData 객체",
  "options": {
    "top_k": 5,
    "category_filter": ["상의", "하의", "아우터"],
    "similarity_threshold": 0.1
  }
}
```

#### 응답 (Backend → Frontend)

**Content-Type**: `application/json`

```json
{
  "status": "success",
  "request_id": "req_20260205_001",
  "input_info": {
    "input_type": "image",
    "file_name": "user_upload_001.jpg",
    "image_size": [224, 224],
    "processed_at": "2026-02-05T10:30:00Z"
  },
  "recommendations": [
    {
      "item_id": "item_009",
      "category": "하의",
      "style": ["스포티", "포멀"],
      "silhouette": "스트레이트",
      "material": ["폴리에스터", "스판덱스"],
      "detail": ["지퍼", "포켓"],
      "similarity_score": 0.8999,
      "image_url": "/images/items/item_009.jpg",
      "metadata": {
        "brand": "Fashion Brand A",
        "price": 89000,
        "color": "네이비"
      }
    },
    {
      "item_id": "item_027",
      "category": "아우터",
      "style": ["포멀"],
      "silhouette": "테일러드",
      "material": ["울", "폴리에스터"],
      "detail": ["버튼", "라펠"],
      "similarity_score": 0.7504,
      "image_url": "/images/items/item_027.jpg",
      "metadata": {
        "brand": "Fashion Brand B",
        "price": 159000,
        "color": "블랙"
      }
    }
  ],
  "performance_metrics": {
    "embedding_time_ms": 45.2,
    "similarity_search_time_ms": 12.8,
    "total_response_time_ms": 78.5,
    "cache_hit": false
  }
}
```

#### 요청 (Frontend → Backend)

**엔드포인트**: `POST /api/recommend/style`

**Content-Type**: `application/json`

**참고**: 이 엔드포인트는 레거시 호환성을 위해 유지되며, 새로운 비즈니스 로직에서는 위의 두 엔드포인트를 사용하는 것을 권장합니다.

```json
{
  "input_type": "json",
  "style_description": {
    "category": "상의",
    "style": ["레트로", "캐주얼"],
    "silhouette": "오버사이즈",
    "material": ["니트", "폴리에스터"],
    "detail": ["라운드넥", "긴소매"]
  },
  "options": {
    "top_k": 10,
    "include_similar_categories": true,
    "similarity_threshold": 0.2
  }
}
```

#### 응답 (Backend → Frontend)

**Content-Type**: `application/json`

```json
{
  "status": "success",
  "request_id": "req_20260205_002",
  "input_info": {
    "input_type": "json_legacy",
    "processed_fields": {
      "category": "상의",
      "style": ["레트로", "캐주얼"],
      "silhouette": "오버사이즈",
      "material": ["니트", "폴리에스터"],
      "detail": ["라운드넥", "긴소매"]
    },
    "embedding_dimension": 512,
    "processed_at": "2026-02-05T10:35:00Z",
    "note": "레거시 엔드포인트 - 호환성 유지용"
  },
  "recommendations": [
    {
      "item_id": "img_001",
      "image_url": "/images/dataset/retro/001.jpg",
      "similarity_score": 0.9234,
      "matched_attributes": {
        "category": "상의",
        "style": ["레트로", "빈티지"],
        "silhouette": "오버사이즈",
        "material": ["니트"],
        "detail": ["라운드넥"]
      },
      "bbox": [45, 23, 180, 220],
      "metadata": {
        "source_category": "레트로",
        "file_path": "C:/sample/라벨링데이터/레트로/001.json"
      }
    },
    {
      "item_id": "img_045",
      "image_url": "/images/dataset/romantic/045.jpg",
      "similarity_score": 0.8567,
      "matched_attributes": {
        "category": "상의",
        "style": ["캐주얼", "로맨틱"],
        "silhouette": "루즈핏",
        "material": ["니트", "코튼"],
        "detail": ["라운드넥", "긴소매"]
      },
      "bbox": [32, 18, 195, 235],
      "metadata": {
        "source_category": "로맨틱",
        "file_path": "C:/sample/라벨링데이터/로맨틱/045.json"
      }
    }
  ],
  "performance_metrics": {
    "json_encoding_time_ms": 8.3,
    "similarity_search_time_ms": 15.7,
    "total_response_time_ms": 34.2,
    "cache_hit": true
  }
}
```

## 나인오즈 비즈니스 로직 구현

### 신상품 데이터베이스 관리

#### 신상품 임베딩 생성 과정
```python
# 1. 신상품 이미지 로드
new_product_images = load_new_products_from_database()

# 2. FashionCLIP으로 임베딩 생성
new_product_embeddings = []
for image in new_product_images:
    with torch.no_grad():
        embedding = fashionclip_model.encode_image(image)
        normalized_embedding = F.normalize(embedding, p=2, dim=-1)
        new_product_embeddings.append(normalized_embedding)

# 3. 신상품 임베딩 데이터베이스 저장
save_embeddings_to_cache(new_product_embeddings, "new_products_db")
```

#### 상위 10% 상품 관리
```python
# 나인오즈 판매 데이터에서 상위 10% 상품 식별
top_10_percent_products = get_top_selling_products(percentage=10)

# 상위 10% 상품 임베딩 사전 계산
top_products_embeddings = []
for product in top_10_percent_products:
    embedding = fashionclip_model.encode_image(product.image)
    top_products_embeddings.append(embedding)
```

### 추천 알고리즘 차이점

#### 1. 내부 전략용 (상위 10% → 신상품)
- **목적**: 트렌드 분석 및 신상품 기획 전략 수립
- **입력**: 나인오즈 상위 10% 판매 상품 이미지
- **출력**: 유사한 스타일의 신상품 추천
- **활용**: 어떤 신상품이 인기 상품과 유사한지 분석

#### 2. 고객 맞춤용 (고객 입력 → 신상품)
- **목적**: 개인화된 신상품 추천
- **입력**: 고객이 업로드하거나 클릭한 상품 이미지
- **출력**: 고객 취향에 맞는 신상품 추천
- **활용**: 고객에게 개인화된 신상품 노출

## 내부 데이터 변환 흐름

### 1. 이미지 입력 → 임베딩 변환 과정 (신상품 추천용)

#### Step 1: 입력 이미지 (상위 10% 상품 또는 고객 입력)
```python
# 상위 10% 상품 이미지 (내부용)
top_product_image = PIL.Image.open("top_seller_001.jpg")

# 또는 고객 입력 이미지 (고객용)
customer_image = PIL.Image.open("customer_upload_001.jpg")
```

#### Step 2: 이미지 전처리
```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

image_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]
```

#### Step 3: FashionCLIP 인코딩
```python
with torch.no_grad():  # FashionCLIP은 frozen 상태
    image_features = fashionclip_model.encode_image(image_tensor)
    query_embedding = F.normalize(image_features, p=2, dim=-1)  # [1, 512]
```

#### Step 4: 신상품 데이터베이스와 유사도 계산
```python
# 신상품 임베딩 로드
new_products_embeddings = load_new_products_embeddings()  # [N, 512]

# 코사인 유사도 계산
similarity_scores = torch.mm(query_embedding, new_products_embeddings.T)  # [1, N]

# Top-K 신상품 선택
top_scores, top_indices = torch.topk(similarity_scores, k=top_k)
```

### 2. JSON 입력 → 임베딩 변환 과정 (레거시)
#### Step 1: 원본 JSON 입력 (레거시 호환성)

```json
{
  "category": "상의",
  "style": ["레트로", "캐주얼"],
  "silhouette": "오버사이즈",
  "material": ["니트", "폴리에스터"],
  "detail": ["라운드넥", "긴소매"]
}
```
#### Step 2: Vocabulary 매핑

```json
{
  "category_id": 1,
  "style_ids": [5, 12],
  "silhouette_id": 8,
  "material_ids": [3, 7],
  "detail_ids": [15, 22]
}
```
```python
{
  "category": torch.tensor([1]),                    # [batch_size]
  "style": torch.tensor([[5, 12, 0, 0]]),         # [batch_size, max_len] (패딩)
  "silhouette": torch.tensor([8]),                 # [batch_size]
  "material": torch.tensor([[3, 7, 0]]),          # [batch_size, max_len] (패딩)
  "detail": torch.tensor([[15, 22, 0, 0, 0]]),    # [batch_size, max_len] (패딩)
  
  # 패딩 마스크
  "style_mask": torch.tensor([[1, 1, 0, 0]]),     # 유효한 토큰 = 1
  "material_mask": torch.tensor([[1, 1, 0]]),
  "detail_mask": torch.tensor([[1, 1, 0, 0, 0]])
}
```

#### Step 4: 임베딩 처리
```python
# 단일 범주형 필드
category_emb = embedding_layers['category'](category_ids)  # [batch_size, 128]
silhouette_emb = embedding_layers['silhouette'](silhouette_ids)  # [batch_size, 128]

# 다중 범주형 필드 (Mean Pooling)
style_emb_raw = embedding_layers['style'](style_ids)  # [batch_size, max_len, 128]
style_emb = masked_mean_pooling(style_emb_raw, style_mask)  # [batch_size, 128]

material_emb_raw = embedding_layers['material'](material_ids)
material_emb = masked_mean_pooling(material_emb_raw, material_mask)

detail_emb_raw = embedding_layers['detail'](detail_ids)
detail_emb = masked_mean_pooling(detail_emb_raw, detail_mask)
```

#### Step 5: MLP 처리 및 정규화
```python
# 연결
concat_emb = torch.cat([
    category_emb, style_emb, silhouette_emb, 
    material_emb, detail_emb
], dim=-1)  # [batch_size, 640]

# MLP 통과
hidden = F.relu(linear1(concat_emb))  # [batch_size, 256]
hidden = F.dropout(hidden, p=0.1, training=self.training)
output = linear2(hidden)  # [batch_size, 512]

# L2 정규화
final_embedding = F.normalize(output, p=2, dim=-1)  # [batch_size, 512]
```

### 3. 신상품 추천 생성 과정

#### Step 1: 신상품 데이터베이스 쿼리
```python
# 비즈니스 타입에 따른 데이터베이스 선택
if business_type == "internal_trend_analysis":
    # 내부용: 모든 신상품 대상
    target_db = "all_new_products"
elif business_type == "personalized_recommendation":
    # 고객용: 고객 선호도 필터링 적용 가능
    target_db = "filtered_new_products"

new_products = query_new_products_database(target_db)
```

#### Step 2: 실시간 유사도 계산
```python
# 쿼리 임베딩 (상위 10% 상품 또는 고객 입력)
query_embedding = torch.tensor([[0.1, 0.2, ..., 0.8]])  # [1, 512]

# 신상품 임베딩 (사전 계산됨)
new_products_embeddings = torch.tensor([
    [0.2, 0.1, ..., 0.7],  # new_item_001
    [0.3, 0.4, ..., 0.6],  # new_item_002
    # ... 더 많은 신상품 임베딩
])  # [N, 512]

# 코사인 유사도 계산
similarity_scores = torch.mm(query_embedding, new_products_embeddings.T)  # [1, N]
```

#### Step 3: 비즈니스 로직 적용
```python
# 내부용: 트렌드 분석을 위한 다양성 확보
if business_type == "internal_trend_analysis":
    # 카테고리별 균등 분배
    recommendations = diversify_by_category(top_recommendations)
    
# 고객용: 개인화 점수 적용
elif business_type == "personalized_recommendation":
    # 고객 프로필 기반 가중치 적용
    personalized_scores = apply_customer_preferences(similarity_scores, customer_profile)
    recommendations = select_top_k(personalized_scores)
```

## KPI 대시보드 데이터 구조

### 실시간 KPI 데이터

```json
{
  "timestamp": "2026-02-05T10:40:00Z",
  "kpi_cards": {
    "training_data": {
      "total_items": 2172,
      "categories": {
        "레트로": 196,
        "로맨틱": 994,
        "리조트": 998
      },
      "train_split": 1737,
      "validation_split": 435
    },
    "training_progress": {
      "current_epoch": 15,
      "total_epochs": 100,
      "progress_percentage": 15.0,
      "estimated_time_remaining": "2h 45m",
      "current_stage": "contrastive_learning"
    },
    "performance_metrics": {
      "top_1_accuracy": 0.0234,
      "top_5_accuracy": 0.1045,
      "mrr": 0.0543,
      "positive_similarity_avg": 0.0340,
      "negative_similarity_avg": 0.0146,
      "embedding_l2_norm": 1.0000
    }
  },
  "loss_curves": {
    "train_loss": [
      {"epoch": 1, "value": 4.2341},
      {"epoch": 2, "value": 3.8765},
      {"epoch": 3, "value": 3.5432},
      // ... 더 많은 데이터 포인트
    ],
    "validation_loss": [
      {"epoch": 1, "value": 4.1876},
      {"epoch": 2, "value": 3.9234},
      {"epoch": 3, "value": 3.6789},
      // ... 더 많은 데이터 포인트
    ]
  },
  "similarity_search_results": {
    "query_examples": [
      {
        "query_type": "json",
        "query_data": {
          "category": "상의",
          "style": ["레트로"]
        },
        "top_5_results": [
          {
            "image_id": "retro_001",
            "similarity_score": 0.8234,
            "image_url": "/results/similarity_search/query_1_레트로_스타일_검색.png"
          }
          // ... 4개 더
        ]
      }
    ]
  },
  "hyperparameters": {
    "current_config": {
      "batch_size": 64,
      "learning_rate": 0.0001,
      "temperature": 0.07,
      "embedding_dim": 128,
      "hidden_dim": 256,
      "dropout_rate": 0.1
    },
    "tuning_history": [
      {
        "experiment_id": "exp_001",
        "config": {"batch_size": 32, "learning_rate": 0.001},
        "best_top5_accuracy": 0.0876,
        "status": "completed"
      }
      // ... 더 많은 실험 기록
    ]
  }
}
```

### 데이터 증강 샘플 데이터

```json
{
  "augmentation_samples": {
    "image_augmentation": {
      "original_image": "/samples/original_001.jpg",
      "augmented_samples": [
        {
          "type": "rotation",
          "angle": 15,
          "image_url": "/samples/rotated_001.jpg"
        },
        {
          "type": "color_jitter",
          "brightness": 0.2,
          "contrast": 0.1,
          "image_url": "/samples/color_jitter_001.jpg"
        },
        {
          "type": "horizontal_flip",
          "image_url": "/samples/flipped_001.jpg"
        }
      ]
    },
    "json_augmentation": {
      "original_json": {
        "category": "상의",
        "style": ["레트로", "캐주얼"],
        "material": ["니트", "폴리에스터"]
      },
      "augmented_samples": [
        {
          "type": "field_dropout",
          "dropped_fields": ["material"],
          "result": {
            "category": "상의",
            "style": ["레트로", "캐주얼"]
          }
        },
        {
          "type": "synonym_replacement",
          "replacements": {"레트로": "빈티지"},
          "result": {
            "category": "상의",
            "style": ["빈티지", "캐주얼"],
            "material": ["니트", "폴리에스터"]
          }
        }
      ]
    }
  }
}
```

## 오류 응답 형식

### 일반적인 오류 응답

```json
{
  "status": "error",
  "error_code": "INVALID_INPUT",
  "error_message": "입력 데이터 형식이 올바르지 않습니다.",
  "details": {
    "field": "style",
    "expected_type": "array",
    "received_type": "string",
    "received_value": "레트로"
  },
  "request_id": "req_20260205_003",
  "timestamp": "2026-02-05T10:45:00Z"
}
```

### 모델 관련 오류 응답

```json
{
  "status": "error",
  "error_code": "MODEL_INFERENCE_FAILED",
  "error_message": "모델 추론 중 오류가 발생했습니다.",
  "details": {
    "model_name": "json_encoder",
    "error_type": "dimension_mismatch",
    "expected_shape": "[1, 512]",
    "actual_shape": "[1, 256]"
  },
  "request_id": "req_20260205_004",
  "timestamp": "2026-02-05T10:50:00Z",
  "retry_after": 30
}
```

## 성능 모니터링 데이터

### 실시간 성능 메트릭

```json
{
  "performance_monitoring": {
    "api_metrics": {
      "requests_per_second": 12.5,
      "average_response_time_ms": 156.7,
      "error_rate_percentage": 0.8,
      "cache_hit_rate_percentage": 78.3
    },
    "model_metrics": {
      "inference_time_ms": {
        "json_encoder": 8.3,
        "fashionclip_encoder": 45.2,
        "similarity_calculation": 12.8
      },
      "gpu_utilization_percentage": 67.4,
      "memory_usage_gb": {
        "gpu_memory": 4.2,
        "system_memory": 12.8
      }
    },
    "database_metrics": {
      "query_time_ms": {
        "embedding_cache": 5.2,
        "item_metadata": 8.7
      },
      "connection_pool_usage": 15,
      "cache_size_mb": 256.7
    }
  }
}
```

이 JSON 데이터 흐름 문서는 Fashion JSON Encoder 시스템의 모든 데이터 교환 형식과 변환 과정을 상세히 정의합니다. 프론트엔드 개발자와 백엔드 개발자가 API 통합 시 참조할 수 있는 완전한 명세서 역할을 합니다.