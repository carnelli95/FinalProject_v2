# 📚 Fashion JSON Encoder 파일 가이드

## 🚀 실행 가능한 스크립트들

### 1. 하이퍼파라미터 튜닝 (다음 단계)
```bash
python scripts/tuning/hyperparameter_tuning.py
```
**목적**: Optuna로 최적 하이퍼파라미터 찾기
**소요시간**: 약 30-60분 (12 trials)
**결과**: `tuning_results/optuna_similarity_gap_tuning_*.json`

### 2. JSON 필드 복구 테스트 (완료됨)
```bash
python scripts/testing/test_improved_training.py
```
**목적**: JSON 필드 복구 효과 검증
**결과**: `results/json_field_recovery_success.json`

### 3. 기본 학습 실행
```bash
python scripts/training/train.py
```
**목적**: 기본 설정으로 전체 학습 파이프라인 실행

### 4. 학습 계속하기
```bash
python scripts/training/continue_training.py
```
**목적**: 기존 체크포인트에서 학습 재개

### 5. 확장 학습
```bash
python scripts/training/run_extended_training.py
```
**목적**: 긴 에포크로 본격 학습

## 📁 새로운 폴더 구조

### scripts/
```
scripts/
├── training/           # 🎯 학습 관련 스크립트
│   ├── train.py                    # 기본 학습 실행
│   ├── continue_training.py        # 체크포인트에서 재개
│   └── run_extended_training.py    # 확장 학습
├── tuning/            # ⚙️ 하이퍼파라미터 튜닝
│   └── hyperparameter_tuning.py   # Optuna 튜닝
├── testing/           # 🧪 테스트 스크립트
│   ├── test_improved_training.py   # JSON 필드 복구 테스트
│   ├── test_similarity_search.py  # 유사도 검색 테스트
│   └── run_fast_tests.py          # 빠른 테스트
└── analysis/          # 📊 분석 및 시각화
    ├── analyze_progress.py        # 진행도 분석
    ├── generate_report.py         # 보고서 생성
    ├── visualize_embeddings.py    # 임베딩 시각화
    └── visualize_results.py       # 결과 시각화
```

## 🏗️ 핵심 모듈 상세 설명

### models/json_encoder.py
```python
class JSONEncoder(nn.Module):
    """패션 메타데이터를 512차원 임베딩으로 변환"""
```
**핵심 기능**:
- 5개 필드 처리: category, style, silhouette, material, detail
- 단일/다중 범주형 필드 구분 처리
- Mean pooling으로 가변 길이 처리
- L2 정규화된 512차원 출력

**사용법**:
```python
vocab_sizes = {'category': 4, 'style': 4, 'silhouette': 8, 'material': 22, 'detail': 38}
encoder = JSONEncoder(vocab_sizes)
embeddings = encoder(json_batch)  # [batch_size, 512]
```

### models/contrastive_learner.py
```python
class ContrastiveLearner(nn.Module):
    """JSON과 이미지 임베딩을 정렬하는 대조 학습 시스템"""
```
**핵심 기능**:
- CLIP 이미지 인코더 (고정)
- JSON 인코더 (학습 가능)
- InfoNCE 손실 함수
- 배치 내 네거티브 샘플링

**사용법**:
```python
learner = ContrastiveLearner(json_encoder, clip_encoder, temperature=0.07)
loss = learner(images, json_data)
```

### data/dataset_loader.py
```python
class KFashionDatasetLoader:
    """K-Fashion 데이터셋 로더 (JSON 필드 복구 완료)"""
```
**핵심 개선사항**:
- ✅ JSON 추출 경로 수정: `데이터셋 정보 → 데이터셋 상세설명 → 라벨링`
- ✅ 어휘 구축 로직 개선: 빈 문자열 처리 문제 해결
- ✅ 인덱스 오류 수정: 어휘 크기와 인덱스 범위 일치

**사용법**:
```python
loader = KFashionDatasetLoader("C:/sample/라벨링데이터")
items = loader.load_dataset_by_category()
vocabularies = loader.build_vocabularies()
```

### training/trainer.py
```python
class FashionTrainer:
    """2단계 학습 파이프라인"""
```
**Stage 1**: JSON Encoder 단독 학습 (sanity check)
**Stage 2**: 대조 학습 (JSON + CLIP)

**사용법**:
```python
trainer = FashionTrainer(config, vocab_sizes, device)
results = trainer.train_contrastive_learning(train_loader, val_loader, num_epochs)
```

### utils/training_monitor.py
```python
class TrainingMonitor:
    """tqdm + matplotlib 기반 간단한 모니터링"""
```
**기능**:
- tqdm 진행 바
- 실시간 메트릭 표시
- matplotlib 차트 자동 생성
- 학습 요약 JSON 저장

## 🔧 설정 파일들

### utils/config.py
```python
@dataclass
class TrainingConfig:
    """학습 하이퍼파라미터 설정"""
```
**주요 파라미터**:
- `batch_size`: 배치 크기 (기본: 64)
- `learning_rate`: 학습률 (기본: 1e-4)
- `temperature`: InfoNCE 온도 (기본: 0.07)
- `embedding_dim`: 필드 임베딩 차원 (기본: 128)
- `hidden_dim`: MLP 은닉층 차원 (기본: 256)
- `output_dim`: 최종 출력 차원 (고정: 512)

### requirements.txt
**핵심 의존성**:
```
torch>=1.9.0
transformers>=4.20.0
optuna>=3.0.0
matplotlib>=3.5.0
tqdm>=4.64.0
Pillow>=8.3.0
```

## 📊 결과 파일들

### results/json_field_analysis.json
JSON 필드 분석 결과:
```json
{
  "vocab_sizes": {"category": 4, "style": 4, "silhouette": 8, "material": 22, "detail": 38},
  "style_stats": {"리조트": 991, "로맨틱": 988, "레트로": 193},
  "material_stats": {"우븐": 1083, "린넨": 338, "시폰": 325, ...}
}
```

### results/training_summary.json
학습 진행 상황 및 메트릭:
```json
{
  "training_state": {"stage": "Stage 2", "current_epoch": 3},
  "metrics_history": {"train_loss": [...], "val_loss": [...], "top5_accuracy": [...]}
}
```

### tuning_results/optuna_similarity_gap_tuning_*.json
Optuna 튜닝 결과:
```json
{
  "best_params": {"learning_rate": 3e-4, "temperature": 0.07, "batch_size": 96},
  "best_value": 0.1234,
  "objective_function": "similarity_gap + category_precision@5 + mrr"
}
```

## 🎯 다음 실행 순서

1. **Optuna 튜닝 실행**:
   ```bash
   python hyperparameter_tuning.py
   ```

2. **결과 확인**:
   ```bash
   # 튜닝 결과 파일 확인
   ls tuning_results/
   ```

3. **최적 설정으로 본격 학습**:
   - 튜닝 결과의 `best_params`를 `TrainingConfig`에 적용
   - 50-100 에포크로 긴 학습 실행

4. **성능 모니터링**:
   - `results/training_charts.png`: 학습 곡선
   - `results/training_summary.json`: 상세 메트릭
   - Category-aware Precision@5 ≥ 0.9 달성 여부 확인

## ⚠️ 주의사항

1. **데이터 경로**: `C:/sample/라벨링데이터` 경로 확인 필요
2. **메모리**: 배치 크기 128 이상 시 GPU 메모리 부족 가능
3. **시간**: Optuna 튜닝은 30-60분 소요
4. **백업**: 중요한 체크포인트는 별도 백업 권장