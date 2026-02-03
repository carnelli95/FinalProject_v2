# 패션 JSON 인코더

패션 이미지와 JSON 메타데이터 간의 정렬된 임베딩을 대조 학습을 통해 학습하는 PyTorch 기반 시스템입니다.

## 개요

이 시스템은 패션 아이템 메타데이터(카테고리, 스타일, 실루엣, 소재, 디테일)를 512차원 임베딩으로 변환하여 대조 학습을 통해 CLIP 이미지 임베딩과 정렬하는 JSON 인코더를 구현합니다.

### 주요 기능

- **JSON 인코더**: 패션 메타데이터를 512차원 임베딩으로 변환
- **대조용 학습**: InfoNCE 손실을 사하여 JSON과 이미지 임베딩 정렬
- **CLIP 통합**: 고정된 CLIP 비전 인코더를 이미지 임베딩에 사용
- **포괄적 테스트**: 속성 기반 테스트 및 단위 테스트
- **사용하기 쉬운 CLI**: 학습 및 평가를 위한 간단한 명령줄 인터페이스

## 빠른 시작

### 1. 설치

```bash
# 저장소 복제
git clone <repository-url>
cd fashion-json-encoder

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터셋 준비

K-Fashion 데이터셋이 정리되고 접근 가능한지 확인하세요. 시스템은 다음을 기대합니다:
- 바운딩 박스 어노테이션이 있는 패션 이미지
- 다음 필드가 있는 JSON 메타데이터: category, style, silhouette, material, detail

### 3. 학습 실행

```bash
# 기본 설정으로 간단한 학습
python train.py --dataset_path /path/to/kfashion

# 사용자 정의 매개변수로 학습
python train.py --dataset_path /path/to/kfashion --epochs 50 --batch_size 32

# 설정 파일로 학습
python train.py --dataset_path /path/to/kfashion --config config.json
```

### 4. 정상성 검사 실행

```bash
# 합성 데이터로 빠른 정상성 검사
python train.py --sanity_check

# 실제 데이터로 정상성 검사
python train.py --sanity_check --dataset_path /path/to/kfashion
```

## 사용법

### 명령줄 인터페이스

시스템은 두 가지 주요 진입점을 제공합니다:

#### 1. `train.py` - 간소화된 학습 인터페이스

```bash
# 기본 사용법
python train.py --dataset_path /path/to/kfashion

# 고급 옵션
python train.py --dataset_path /path/to/kfashion \
                --epochs 50 \
                --batch_size 32 \
                --lr 0.001 \
                --output_dir my_training

# 독립 학습 건너뛰기
python train.py --dataset_path /path/to/kfashion --no_standalone

# 설정 파일 생성
python train.py --create_config my_config.json
```

#### 2. `main.py` - 전체 시스템 인터페이스

```bash
# 학습
python main.py train --dataset_path /path/to/kfashion

# 평가
python main.py evaluate --checkpoint_path checkpoints/best_model.pt

# 정상성 검사
python main.py sanity_check

# 설정 생성
python main.py create_config --output config.json
```

### 설정

학습 매개변수를 사용자 정의하기 위한 설정 파일 생성:

```bash
python train.py --create_config config.json
```

설정 예시:

```json
{
  "batch_size": 64,
  "learning_rate": 0.0001,
  "temperature": 0.07,
  "embedding_dim": 128,
  "hidden_dim": 256,
  "output_dim": 512,
  "dropout_rate": 0.1,
  "weight_decay": 1e-05,
  "max_epochs": 100,
  "target_categories": ["상의", "하의", "아우터"],
  "image_size": 224,
  "crop_padding": 0.1
}
```

## 학습 과정

시스템은 2단계 학습 접근법을 사용합니다:

### 1단계: 독립 JSON 인코더 학습
- JSON 인코더를 독립적으로 학습하여 기본 기능 검증
- 합성 타겟을 사용하여 모델이 학습할 수 있는지 확인
- 출력 차원 및 정규화 검증
- 기본값: 5 에포크

### 2단계: 대조 학습
- 고정된 CLIP 이미지 인코더와 함께 JSON 인코더 학습
- 배치 내 네거티브 샘플링과 함께 InfoNCE 손실 사용
- 공유된 512차원 공간에서 JSON과 이미지 임베딩 정렬
- 기본값: 전체 에포크에서 남은 에포크

## 학습 모니터링

### TensorBoard

TensorBoard로 학습 진행 상황 확인:

```bash
tensorboard --logdir logs
```

### 출력 파일

학습은 여러 출력 파일을 생성합니다:

```
training_output/
├── checkpoints/
│   ├── best_model.pt          # 최고 모델 체크포인트
│   └── checkpoint_epoch_*.pt  # 정기 체크포인트
├── logs/                      # TensorBoard 로그
└── results/
    └── training_results.json  # 학습 메트릭 및 결과
```

## 모델 아키텍처

### JSON 인코더 구조

```
JSON 입력 → 필드 임베딩 → 연결 → MLP → L2 정규화 → 512차원 출력
```

- **필드 임베딩**: 각 메타데이터 필드에 대한 별도 임베딩 레이어
- **다중 범주형 처리**: 여러 값을 가진 필드에 대한 평균 풀링
- **MLP**: ReLU 활성화 및 드롭아웃이 있는 2층 MLP
- **출력**: L2 정규화된 512차원 벡터

### 대조 학습

- **이미지 인코더**: 고정된 CLIP ViT-B/32
- **손실 함수**: 온도 τ=0.07인 InfoNCE
- **네거티브 샘플링**: 배치 내 네거티브
- **최적화**: 코사인 어닐링이 있는 Adam 옵티마이저

## 평가 메트릭

시스템은 여러 메트릭을 추적합니다:

- **InfoNCE 손실**: 주요 학습 목표
- **Top-1 정확도**: 올바른 이미지-JSON 매치 비율
- **Top-5 정확도**: 상위 5개에서 올바른 매치 비율
- **평균 역순위**: 올바른 매치의 평균 역순위
- **코사인 유사도**: 포지티브 쌍 간의 평균 유사도

## 테스트

### 속성 기반 테스트

시스템은 다음을 검증하는 속성 기반 테스트를 포함합니다:

1. **고정 출력 차원**: 모든 출력이 정확히 512차원
2. **L2 정규화**: 모든 출력 벡터가 단위 노름을 가짐
3. **CLIP 고정 상태**: 학습 중 CLIP 매개변수가 변경되지 않음
4. **다중 범주형 처리**: 리스트 값 필드의 적절한 처리

### 단위 테스트

테스트 스위트 실행:

```bash
python -m pytest tests/ -v
```

### 정상성 검사

내장된 정상성 검사는 다음을 검증합니다:

- 모델이 JSON 메타데이터를 올바르게 처리할 수 있음
- 출력 차원 및 정규화가 올바름
- 학습 중 그래디언트가 적절히 흐름
- 모든 메타데이터 유형에 대해 필드 처리가 작동함

## 문제 해결

### 일반적인 문제

1. **CUDA 메모리 부족**
   ```bash
   # 배치 크기 줄이기
   python train.py --dataset_path /path/to/data --batch_size 16
   ```

2. **데이터셋을 찾을 수 없음**
   ```bash
   # 데이터셋 경로가 존재하는지 확인
   ls /path/to/kfashion
   
   # 합성 데이터로 정상성 검사 사용
   python train.py --sanity_check
   ```

3. **느린 학습**
   ```bash
   # I/O 바운드인 경우 워커 수 줄이기
   # nvidia-smi로 GPU 사용률 확인
   # 설정에서 더 작은 이미지 크기 고려
   ```

### 디버깅

디버그 로깅 활성화:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

시스템 상태 확인:

```bash
# 정상성 검사 실행
python main.py sanity_check

# 작은 데이터셋으로 모델 확인
python train.py --dataset_path /path/to/data --epochs 1 --batch_size 4
```

## API 참조

### FashionEncoderSystem

모든 구성 요소를 통합하는 메인 시스템 클래스:

```python
from main import FashionEncoderSystem

# 시스템 초기화
system = FashionEncoderSystem(config_path='config.json')

# 데이터 설정
system.setup_data('/path/to/dataset')

# 트레이너 설정
system.setup_trainer()

# 모델 학습
results = system.train(standalone_epochs=5, contrastive_epochs=20)

# 모델 평가
eval_results = system.evaluate('checkpoints/best_model.pt')
```

### JSONEncoder

핵심 모델 클래스:

```python
from models.json_encoder import JSONEncoder

# 인코더 초기화
encoder = JSONEncoder(
    vocab_sizes={'category': 10, 'style': 20, ...},
    embedding_dim=128,
    hidden_dim=256
)

# 순전파
embeddings = encoder(json_batch)  # [batch_size, 512] 반환
```

### ContrastiveLearner

대조 학습 시스템:

```python
from models.contrastive_learner import ContrastiveLearner

# 학습자 초기화
learner = ContrastiveLearner(json_encoder, clip_encoder, temperature=0.07)

# 학습 스텝
loss = learner(images, json_data)

# 임베딩 얻기
embeddings = learner.get_embeddings(images, json_data)
```

## 기여

1. 저장소 포크
2. 기능 브랜치 생성
3. 새로운 기능에 대한 테스트 추가
4. 모든 테스트가 통과하는지 확인
5. 풀 리퀘스트 제출

## 라이선스

[여기에 라이선스 정보 추가]

## 인용

연구에서 이 코드를 사용하는 경우 다음과 같이 인용해 주세요:

```bibtex
@misc{fashion-json-encoder,
  title={Fashion JSON Encoder: Learning Aligned Embeddings for Fashion Image-Text Retrieval},
  author={[Your Name]},
  year={2024},
  url={[Repository URL]}
}
```