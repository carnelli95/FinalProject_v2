# 구현 계획: Fashion JSON Encoder

## 개요

K-Fashion 데이터셋의 JSON 메타데이터를 512차원 벡터로 변환하는 JSON Encoder를 구현합니다. CLIP 이미지 임베딩과 정렬되는 공통 임베딩 공간을 구축하여 패션 이미지 추천 시스템의 기반을 마련합니다.

## 작업 목록

- [x] 1. 프로젝트 구조 및 핵심 인터페이스 설정
  - 디렉토리 구조 생성 (models/, data/, utils/, tests/)
  - 핵심 데이터 클래스 및 설정 클래스 구현
  - PyTorch 및 필요한 라이브러리 의존성 설정
  - _Requirements: 1.3, 4.2_

- [ ] 2. 데이터 전처리 파이프라인 구현
  - [x] 2.1 K-Fashion 데이터셋 로더 구현
    - JSON 메타데이터 파싱 및 필터링 (상의/하의/아우터만)
    - Polygon to BBox 변환 함수 구현
    - BBox 기준 이미지 크롭 기능 구현
    - _Requirements: 전처리 요구사항_
  
  - [x] 2.2 데이터 전처리 단위 테스트 작성

    - Polygon to BBox 변환 정확성 테스트
    - 이미지 크롭 기능 테스트
    - 필터링 로직 테스트
    - _Requirements: 전처리 요구사항_
  
  - [x] 2.3 Vocabulary 구축 시스템 구현
    OOV(out-of-vocabulary) 토큰을 위한 <UNK> index 정의
    - 각 필드별 vocabulary 생성 (category, style, silhouette, material, detail)
    - JSON 필드를 vocabulary index로 변환하는 함수 구현
    - _Requirements: JSON 입력 스키마 가정_

- [-] 3. JSON Encoder 모델 구현
  - [x] 3.1 JSONEncoder 클래스 구현
    - 필드별 Embedding 레이어 구현
    - 단일/다중 범주형 필드 처리 로직 구현
    - MLP 레이어 및 L2 정규화 구현
    - _Requirements: 1.1, 1.2, 1.4, 2.1-2.5_
  
  - [ ]* 3.2 JSONEncoder 속성 테스트 작성
    - **Property 1: 고정 출력 차원**
    - **Validates: Requirements 1.1**
  
  - [ ]* 3.3 출력 정규화 속성 테스트 작성
    - **Property 2: 정규화된 출력 벡터**
    - **Validates: Requirements 1.2**
  
  - [ ]* 3.4 다중 범주형 필드 처리 속성 테스트 작성
    - **Property 4: 다중 범주형 필드 처리**
    - **Validates: Requirements 2.2, 2.4, 2.5**

- [x] 4. Contrastive Learning 시스템 구현
  - [x] 4.1 ContrastiveLearner 클래스 구현
    - CLIP Image Encoder 통합 (frozen 상태 유지)
    - InfoNCE Loss 구현 (temperature τ=0.07)
    - In-batch negative sampling 구현
    - _Requirements: 1.5, 3.1-3.3_
  
  - [ ]* 4.2 CLIP 고정 상태 속성 테스트 작성
    - **Property 3: CLIP 모델 고정 상태 유지**
    - **Validates: Requirements 1.5**
  
  - [ ]* 4.3 Positive/Negative Pair 생성 속성 테스트 작성
    - **Property 5: Positive Pair 생성**
    - **Property 6: Negative Pair 생성**
    - **Validates: Requirements 3.1, 3.2**
  
  - [ ]* 4.4 InfoNCE Loss 계산 속성 테스트 작성
    - **Property 7: InfoNCE Loss 계산**
    - **Validates: Requirements 3.3**

- [x] 5. 체크포인트 - 핵심 모델 검증
  - 모든 테스트가 통과하는지 확인하고, 질문이 있으면 사용자에게 문의

- [x] 6. 학습 파이프라인 구현
  - [x] 6.1 DataLoader 및 배치 처리 구현
    - PyTorch Dataset 클래스 구현
    - 배치 생성 및 패딩 처리 구현
    - 데이터 증강 및 전처리 파이프라인 통합
    - _Requirements: 학습 목표_
  
  - [x] 6.2 학습 루프 구현
    - [x] JSON Encoder 단독 학습/출력 분포 sanity check
    - Optimizer 설정 (Adam, learning_rate=1e-4)
    - 학습 스케줄러 및 체크포인트 저장 구현
    - 검증 루프 및 메트릭 계산 구현
    - _Requirements: 학습 목표_
  
  - [ ]* 6.3 학습 파이프라인 통합 테스트 작성
    - End-to-end 학습 과정 테스트
    - 메모리 사용량 및 성능 테스트
    - _Requirements: 학습 목표_

- [x] 7. 오류 처리 및 검증 시스템 구현
  - [x] 7.1 입력 데이터 검증 구현
    - InputValidator 클래스 구현
    - JSON 배치 및 이미지 배치 유효성 검사
    - _Requirements: 오류 처리_
  
  - [x] 7.2 모델 상태 검증 구현
    - ModelValidator 클래스 구현
    - 출력 차원 및 정규화 검증
    - _Requirements: 오류 처리_
  
  - [ ]* 7.3 오류 처리 단위 테스트 작성
    - 잘못된 입력에 대한 예외 처리 테스트
    - 엣지 케이스 처리 테스트
    - _Requirements: 오류 처리_

- [-] 8. 최종 통합 및 검증
  - [x] 8.1 전체 시스템 통합
    - 모든 컴포넌트를 연결하는 메인 스크립트 구현
    - 설정 파일 및 명령행 인터페이스 구현
    - _Requirements: 4.1, 4.4_
  
  - [ ]* 8.2 성능 벤치마크 테스트 작성
    - 임베딩 cosine similarity 기반 Top-K retrieval sanity check
    - 배치 크기별 처리 속도 측정
    - GPU 메모리 사용량 프로파일링
    - _Requirements: 성능 요구사항_

- [x] 9. 최종 체크포인트 - 전체 시스템 검증
  - 모든 테스트가 통과하는지 확인하고, 질문이 있으면 사용자에게 문의

## 참고사항

- `*` 표시된 작업은 선택사항으로 빠른 MVP를 위해 건너뛸 수 있습니다
- 각 작업은 특정 요구사항에 대한 추적 가능성을 위해 요구사항을 참조합니다
- 체크포인트는 점진적 검증을 보장합니다
- 속성 테스트는 범용 정확성 속성을 검증합니다
- 단위 테스트는 특정 예제 및 엣지 케이스를 검증합니다