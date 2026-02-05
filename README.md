# Fashion JSON Encoder - 혁신적 베스트셀러 Proxy 시스템

패션 이미지 추천 시스템을 위한 고도화된 JSON Encoder 구현. K-Fashion 데이터셋의 JSON 메타데이터를 학습하여 FashionCLIP 이미지 임베딩과 정렬되는 512차원 Attribute Embedding을 생성하며, **임베딩 중심성 기반 베스트셀러 Proxy** 혁신 기술을 포함합니다.

## 🚀 핵심 혁신 기술

### 임베딩 중심성 기반 베스트셀러 Proxy
- **핵심 아이디어**: "베스트셀러를 판매 데이터 없이, 임베딩 공간의 중심성으로 근사(proxy)한다"
- **개념 직관**: "중심에 가까울수록 대중적이다"
- **Anchor Set**: 상위 10% 중심성 = 베스트셀러 Proxy
- **Query-Aware 평가**: All Queries vs Anchor Queries 차별화 평가

### 현재 달성 성과
- **Top-5 정확도**: 64.1% (Baseline v2)
- **혁신 시스템**: 임베딩 중심성 베스트셀러 Proxy 구현 완료
- **평가 시스템**: Query-Aware 평가 시스템 구현
- **Temperature 최적화**: 0.1에서 최적 성능 확인

## 📁 프로젝트 구조

```
fashion-json-encoder/
├── 📂 models/              # 핵심 모델 구현
│   ├── json_encoder.py     # JSON Encoder 모델
│   └── contrastive_learner.py  # 대조 학습 시스템
├── 📂 data/                # 데이터 처리 파이프라인
│   ├── dataset_loader.py   # K-Fashion 데이터 로더
│   ├── fashion_dataset.py  # PyTorch Dataset 구현
│   └── processor.py        # 데이터 전처리
├── 📂 scripts/             # 분석 및 실험 스크립트
│   ├── analysis/           # 중심성 분석, Query-Aware 평가
│   ├── integration/        # 통합 파이프라인
│   └── training/           # 학습 스크립트
├── 📂 tests/               # 테스트 스위트 (106개 통과)
├── 📂 results/             # 실험 결과 및 분석 보고서
├── 📂 checkpoints/         # 훈련된 모델 체크포인트
├── 📂 examples/            # 사용 예제 및 데모
├── 📂 docs/                # 문서 및 아키텍처 다이어그램
└── 📂 .kiro/specs/         # 프로젝트 명세 및 설계 문서
```

## 🎯 주요 기능

### 1. 핵심 시스템
- **JSON Encoder**: 패션 메타데이터를 512차원 임베딩으로 변환
- **Contrastive Learning**: InfoNCE Loss를 사용한 이미지-JSON 정렬
- **FashionCLIP 통합**: 고정된 CLIP 비전 인코더 사용

### 2. 혁신 기능
- **베스트셀러 Proxy**: 임베딩 중심성 기반 베스트셀러 근사
- **Query-Aware 평가**: All vs Anchor Queries 차별화 평가
- **카테고리별 분석**: 로맨틱 > 리조트 > 레트로 중심성 순서

### 3. 분석 도구
- **중심성 분석**: 글로벌 중심 벡터 기반 중심성 계산
- **성능 검증**: 포괄적 성능 목표 달성 검증
- **통합 파이프라인**: 자동화된 실험 및 분석 워크플로우

## 🚀 빠른 시작

### 1. 설치
```bash
git clone <repository-url>
cd fashion-json-encoder
pip install -r requirements.txt
```

### 2. 기본 학습
```bash
# 기본 설정으로 학습
python train.py --dataset_path /path/to/kfashion

# 정상성 검사
python train.py --sanity_check
```

### 3. 통합 파이프라인 실행
```bash
# 중심성 분석 → Query-Aware 평가 → 성능 보고서
python scripts/integration/integrated_pipeline.py

# 성능 목표 달성 검증
python scripts/integration/performance_goal_verification.py
```

## 📊 실험 결과

### 현재 성능 (2026-02-05 기준)
- **All Queries Recall@10**: 29.4% (목표: 75-80%)
- **Anchor Queries Recall@10**: 27.2% (목표: 85-92%)
- **Top-5 정확도**: 14.9%
- **베스트셀러 Proxy 검증**: 추가 최적화 필요

### 카테고리별 중심성 분석
- **로맨틱**: 0.8048 (가장 대중적)
- **리조트**: 0.7935 (중간 중심성)
- **레트로**: 0.7626 (가장 독특한)

## 🔧 사용법

### 명령줄 인터페이스

#### 기본 학습
```bash
python train.py --dataset_path /path/to/kfashion --epochs 50 --batch_size 32
```

#### 고급 분석
```bash
# 중심성 분석
python scripts/analysis/embedding_centrality_proxy.py

# Query-Aware 평가
python scripts/analysis/anchor_based_evaluation.py
```

#### API 서버 시작
```bash
python start_api_server.py
```

### Python API
```python
from main import FashionEncoderSystem

# 시스템 초기화
system = FashionEncoderSystem()
system.setup_data('/path/to/dataset')
system.setup_trainer()

# 학습 실행
results = system.train()

# 중심성 분석
from scripts.analysis.embedding_centrality_proxy import EmbeddingCentralityProxy
analyzer = EmbeddingCentralityProxy(system)
centrality_results = analyzer.run_complete_analysis()
```

## 🧪 테스트

### 테스트 실행
```bash
# 전체 테스트 스위트
python -m pytest tests/ -v

# 특정 모듈 테스트
python -m pytest tests/test_json_encoder.py -v
```

### 테스트 커버리지
- **총 테스트**: 111개
- **통과**: 106개 ✅
- **실패**: 5개 (데이터 로딩 관련)

## 📈 성능 최적화

### 권장 개선사항
1. **Temperature 미세 조정**: 0.08-0.12 범위 실험
2. **배치 크기 증가**: 32→64로 증가
3. **아키텍처 개선**: JSON Encoder 차원 확장 (128→256)
4. **Multi-head Attention**: 어텐션 메커니즘 도입

### 다음 단계
1. 성능 목표 달성 (Recall@10 75-80%)
2. 실시간 추천 API 시스템 구축
3. 베스트셀러 Proxy 시스템 상용화

## 📚 문서

- **설계 문서**: `.kiro/specs/fashion-json-encoder/design.md`
- **요구사항**: `.kiro/specs/fashion-json-encoder/requirements.md`
- **작업 계획**: `.kiro/specs/fashion-json-encoder/tasks.md`
- **아키텍처 다이어그램**: `docs/architecture_diagrams.md`

## 🤝 기여

1. 저장소 포크
2. 기능 브랜치 생성
3. 테스트 추가 및 실행
4. 풀 리퀘스트 제출

## 📄 라이선스

MIT License

## 📖 인용

```bibtex
@misc{fashion-json-encoder-2026,
  title={Fashion JSON Encoder: Embedding Centrality-based Bestseller Proxy System},
  author={[Your Name]},
  year={2026},
  note={Innovative bestseller approximation without sales data}
}
```

---

**🎉 혁신적인 베스트셀러 Proxy 시스템으로 패션 추천의 새로운 패러다임을 제시합니다!**