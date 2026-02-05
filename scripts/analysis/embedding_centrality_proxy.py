#!/usr/bin/env python3
"""
임베딩 중심성 기반 베스트셀러 Proxy 시스템

🎯 핵심 아이디어:
"베스트셀러를 판매 데이터 없이, 임베딩 공간의 중심성으로 근사(proxy)한다."

🧠 개념 직관:
임베딩 공간에서 많은 상품과 비슷한 디자인 → 트렌드성 디자인 → 잘 팔릴 가능성 ↑
즉, "중심에 가까울수록 대중적이다"

🧱 설계 구조:
STEP 2-1: 전체 embedding 추출 (이미지만)
STEP 2-2: 글로벌 중심 벡터 계산
STEP 2-3: 중심성 점수 계산 (Cosine Similarity)
STEP 2-4: 상위 10% 선택 → Anchor Set (베스트셀러 Proxy)
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import torch
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from main import FashionEncoderSystem
from utils.config import TrainingConfig


class EmbeddingCentralityProxy:
    """임베딩 중심성 기반 베스트셀러 Proxy 시스템"""
    
    def __init__(self, system: FashionEncoderSystem):
        self.system = system
        self.data_module = system.data_module
        self.trainer = system.trainer
        
        # 임베딩 데이터 저장
        self.image_embeddings = None
        self.json_embeddings = None
        self.fashion_items = None
        
        # 중심성 분석 결과
        self.global_center = None
        self.centrality_scores = None
        self.anchor_indices = None
        self.tail_indices = None
        
    def extract_all_embeddings(self) -> Dict[str, Any]:
        """STEP 2-1: 전체 embedding 추출"""
        print("🔍 STEP 2-1: 전체 임베딩 추출 중...")
        print("   대상: Train + Validation 전체")
        print("   방법: 이미지 embedding만 사용 (JSON X)")
        
        # 전체 데이터셋 준비 (Train + Validation)
        all_items = []
        all_items.extend(self.data_module.train_dataset.fashion_items)
        all_items.extend(self.data_module.val_dataset.fashion_items)
        
        print(f"   총 아이템 수: {len(all_items)}")
        
        # 전체 데이터로더 생성
        from torch.utils.data import Dataset, DataLoader
        from data.fashion_dataset import collate_fashion_batch
        
        class FullDataset(Dataset):
            def __init__(self, fashion_items, base_dataset):
                self.fashion_items = fashion_items
                self.base_dataset = base_dataset
                
            def __len__(self):
                return len(self.fashion_items)
            
            def __getitem__(self, idx):
                item = self.fashion_items[idx]
                # 이미지 로드 및 전처리
                image = self.base_dataset.dataset_loader.get_cropped_image(item)
                image_tensor = self.base_dataset.image_transforms(image)
                
                # JSON 처리
                processed_json = self.base_dataset.dataset_loader.get_processed_json(item)
                
                return {
                    'image': image_tensor,
                    'category': processed_json['category'],
                    'style': processed_json['style'],
                    'silhouette': processed_json['silhouette'],
                    'material': processed_json['material'],
                    'detail': processed_json['detail']
                }
        
        full_dataset = FullDataset(all_items, self.data_module.train_dataset)
        full_loader = DataLoader(
            full_dataset,
            batch_size=32,  # 배치 크기 증가
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fashion_batch
        )
        
        # 임베딩 추출
        self.trainer.contrastive_learner.eval()
        
        all_image_embeddings = []
        all_json_embeddings = []
        
        print(f"   배치 수: {len(full_loader)}")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(full_loader):
                if batch_idx % 10 == 0:
                    print(f"   진행률: {batch_idx}/{len(full_loader)} ({batch_idx/len(full_loader)*100:.1f}%)")
                
                batch = self.trainer._move_batch_to_device(batch)
                json_batch = self.trainer._convert_batch_to_dict(batch)
                
                # 임베딩 추출
                embeddings = self.trainer.contrastive_learner.get_embeddings(batch.images, json_batch)
                
                all_image_embeddings.append(embeddings['image_embeddings'].cpu())
                all_json_embeddings.append(embeddings['json_embeddings'].cpu())
        
        # 결합
        self.image_embeddings = torch.cat(all_image_embeddings, dim=0).numpy()
        self.json_embeddings = torch.cat(all_json_embeddings, dim=0).numpy()
        self.fashion_items = all_items
        
        print(f"✅ 임베딩 추출 완료:")
        print(f"   이미지 임베딩: {self.image_embeddings.shape}")
        print(f"   JSON 임베딩: {self.json_embeddings.shape}")
        print(f"   아이템 수: {len(self.fashion_items)}")
        
        return {
            'image_embeddings': self.image_embeddings,
            'json_embeddings': self.json_embeddings,
            'num_items': len(self.fashion_items)
        }
    
    def compute_global_center(self) -> np.ndarray:
        """STEP 2-2: 글로벌 중심 벡터 계산"""
        print("\n🎯 STEP 2-2: 글로벌 중심 벡터 계산 중...")
        
        if self.image_embeddings is None:
            raise ValueError("먼저 extract_all_embeddings()를 실행하세요.")
        
        # 글로벌 중심 계산 (이미지 임베딩만 사용)
        global_center = np.mean(self.image_embeddings, axis=0)
        
        # 정규화
        global_center = global_center / np.linalg.norm(global_center)
        
        self.global_center = global_center
        
        print(f"✅ 글로벌 중심 벡터 계산 완료:")
        print(f"   차원: {global_center.shape}")
        print(f"   노름: {np.linalg.norm(global_center):.6f}")
        print(f"   의미: '전체 패션 데이터의 평균 스타일'")
        
        return global_center
    
    def compute_centrality_scores(self) -> np.ndarray:
        """STEP 2-3: 중심성 점수 계산 (Cosine Similarity)"""
        print("\n📐 STEP 2-3: 중심성 점수 계산 중...")
        
        if self.global_center is None:
            raise ValueError("먼저 compute_global_center()를 실행하세요.")
        
        # 각 상품에 대해 중심성 점수 계산
        centrality_scores = []
        
        for i, embedding in enumerate(self.image_embeddings):
            # 코사인 유사도 계산
            score = np.dot(embedding, self.global_center) / (
                np.linalg.norm(embedding) * np.linalg.norm(self.global_center)
            )
            centrality_scores.append(score)
        
        self.centrality_scores = np.array(centrality_scores)
        
        # 통계 정보
        print(f"✅ 중심성 점수 계산 완료:")
        print(f"   평균: {self.centrality_scores.mean():.4f}")
        print(f"   표준편차: {self.centrality_scores.std():.4f}")
        print(f"   최소값: {self.centrality_scores.min():.4f}")
        print(f"   최대값: {self.centrality_scores.max():.4f}")
        
        # 분포 정보
        percentiles = [10, 25, 50, 75, 90, 95]
        print(f"   분위수:")
        for p in percentiles:
            value = np.percentile(self.centrality_scores, p)
            print(f"     {p}%: {value:.4f}")
        
        return self.centrality_scores
    
    def create_anchor_and_tail_sets(self, anchor_percentile: int = 90, tail_percentile: int = 50) -> Dict[str, Any]:
        """STEP 2-4: Anchor Set (상위 10%) 및 Tail Set (하위 50%) 생성"""
        print(f"\n⚓ STEP 2-4: Anchor Set (상위 {100-anchor_percentile}%) 및 Tail Set (하위 {tail_percentile}%) 생성 중...")
        
        if self.centrality_scores is None:
            raise ValueError("먼저 compute_centrality_scores()를 실행하세요.")
        
        # 임계값 계산
        anchor_threshold = np.percentile(self.centrality_scores, anchor_percentile)
        tail_threshold = np.percentile(self.centrality_scores, tail_percentile)
        
        # 인덱스 선택
        self.anchor_indices = np.where(self.centrality_scores >= anchor_threshold)[0]
        self.tail_indices = np.where(self.centrality_scores <= tail_threshold)[0]
        
        # 카테고리별 분포 분석
        anchor_categories = {}
        tail_categories = {}
        all_categories = {}
        
        for idx in self.anchor_indices:
            category = self.fashion_items[idx].category
            anchor_categories[category] = anchor_categories.get(category, 0) + 1
        
        for idx in self.tail_indices:
            category = self.fashion_items[idx].category
            tail_categories[category] = tail_categories.get(category, 0) + 1
        
        for item in self.fashion_items:
            category = item.category
            all_categories[category] = all_categories.get(category, 0) + 1
        
        print(f"✅ Anchor & Tail Set 생성 완료:")
        print(f"   Anchor 임계값: {anchor_threshold:.4f}")
        print(f"   Tail 임계값: {tail_threshold:.4f}")
        print(f"   Anchor Set 크기: {len(self.anchor_indices)} ({len(self.anchor_indices)/len(self.fashion_items)*100:.1f}%)")
        print(f"   Tail Set 크기: {len(self.tail_indices)} ({len(self.tail_indices)/len(self.fashion_items)*100:.1f}%)")
        
        print(f"\n📊 카테고리별 분포:")
        print(f"   전체:")
        for cat, count in all_categories.items():
            print(f"     {cat}: {count}개 ({count/len(self.fashion_items)*100:.1f}%)")
        
        print(f"   Anchor Set (베스트셀러 Proxy):")
        for cat, count in anchor_categories.items():
            total_cat = all_categories[cat]
            print(f"     {cat}: {count}개 ({count/total_cat*100:.1f}% of {cat})")
        
        print(f"   Tail Set:")
        for cat, count in tail_categories.items():
            total_cat = all_categories[cat]
            print(f"     {cat}: {count}개 ({count/total_cat*100:.1f}% of {cat})")
        
        return {
            'anchor_indices': self.anchor_indices,
            'tail_indices': self.tail_indices,
            'anchor_threshold': anchor_threshold,
            'tail_threshold': tail_threshold,
            'anchor_categories': anchor_categories,
            'tail_categories': tail_categories,
            'all_categories': all_categories
        }
    
    def analyze_centrality_distribution(self) -> Dict[str, Any]:
        """중심성 분포 상세 분석"""
        print("\n📈 중심성 분포 상세 분석 중...")
        
        if self.centrality_scores is None:
            raise ValueError("먼저 compute_centrality_scores()를 실행하세요.")
        
        # 카테고리별 중심성 분석
        category_centrality = {}
        for i, item in enumerate(self.fashion_items):
            category = item.category
            if category not in category_centrality:
                category_centrality[category] = []
            category_centrality[category].append(self.centrality_scores[i])
        
        # 통계 계산
        category_stats = {}
        for category, scores in category_centrality.items():
            scores = np.array(scores)
            category_stats[category] = {
                'mean': float(scores.mean()),
                'std': float(scores.std()),
                'min': float(scores.min()),
                'max': float(scores.max()),
                'count': len(scores),
                'median': float(np.median(scores))
            }
        
        print(f"✅ 카테고리별 중심성 분석:")
        for category, stats in category_stats.items():
            print(f"   {category}:")
            print(f"     평균: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"     범위: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"     중앙값: {stats['median']:.4f}")
            print(f"     샘플 수: {stats['count']}")
        
        # 전체 분포 분석
        overall_stats = {
            'mean': float(self.centrality_scores.mean()),
            'std': float(self.centrality_scores.std()),
            'min': float(self.centrality_scores.min()),
            'max': float(self.centrality_scores.max()),
            'median': float(np.median(self.centrality_scores)),
            'skewness': float(self._compute_skewness(self.centrality_scores)),
            'kurtosis': float(self._compute_kurtosis(self.centrality_scores))
        }
        
        print(f"\n📊 전체 분포 특성:")
        print(f"   평균: {overall_stats['mean']:.4f}")
        print(f"   표준편차: {overall_stats['std']:.4f}")
        print(f"   왜도(Skewness): {overall_stats['skewness']:.4f}")
        print(f"   첨도(Kurtosis): {overall_stats['kurtosis']:.4f}")
        
        return {
            'category_stats': category_stats,
            'overall_stats': overall_stats,
            'category_centrality': {k: [float(x) for x in v] for k, v in category_centrality.items()}
        }
    
    def _compute_skewness(self, data):
        """왜도 계산"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3)
    
    def _compute_kurtosis(self, data):
        """첨도 계산"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def visualize_centrality_distribution(self, save_path: str = "results/centrality_analysis.png"):
        """중심성 분포 시각화 (간단 버전)"""
        print(f"\n📊 중심성 분포 시각화 생략 (matplotlib 의존성 문제)")
        
        if self.centrality_scores is None:
            raise ValueError("먼저 compute_centrality_scores()를 실행하세요.")
        
        # 텍스트 기반 간단 분석만 수행
        print(f"✅ 분포 분석 (텍스트):")
        print(f"   평균: {self.centrality_scores.mean():.4f}")
        print(f"   표준편차: {self.centrality_scores.std():.4f}")
        print(f"   범위: [{self.centrality_scores.min():.4f}, {self.centrality_scores.max():.4f}]")
        
        # 히스토그램 텍스트 버전
        hist, bin_edges = np.histogram(self.centrality_scores, bins=10)
        print(f"   히스토그램 (10 bins):")
        for i in range(len(hist)):
            bar = "█" * int(hist[i] / max(hist) * 20)  # 최대 20자 막대
            print(f"     [{bin_edges[i]:.3f}-{bin_edges[i+1]:.3f}]: {hist[i]:4d} {bar}")
        
        return True
    
    def run_complete_analysis(self, anchor_percentile: int = 90, tail_percentile: int = 50) -> Dict[str, Any]:
        """전체 분석 파이프라인 실행"""
        print("🚀 임베딩 중심성 기반 베스트셀러 Proxy 분석 시작")
        print("=" * 80)
        print("🎯 핵심 아이디어: '베스트셀러를 판매 데이터 없이, 임베딩 공간의 중심성으로 근사'")
        print("🧠 개념 직관: '중심에 가까울수록 대중적이다'")
        print("=" * 80)
        
        # STEP 2-1: 임베딩 추출
        embedding_info = self.extract_all_embeddings()
        
        # STEP 2-2: 글로벌 중심 계산
        global_center = self.compute_global_center()
        
        # STEP 2-3: 중심성 점수 계산
        centrality_scores = self.compute_centrality_scores()
        
        # STEP 2-4: Anchor & Tail Set 생성
        sets_info = self.create_anchor_and_tail_sets(anchor_percentile, tail_percentile)
        
        # 상세 분석
        distribution_analysis = self.analyze_centrality_distribution()
        
        # 시각화
        self.visualize_centrality_distribution()
        
        # 결과 종합 (JSON 직렬화 가능하도록 수정)
        complete_results = {
            'timestamp': datetime.now().isoformat(),
            'method': 'Embedding Centrality Proxy',
            'core_idea': '베스트셀러를 판매 데이터 없이, 임베딩 공간의 중심성으로 근사',
            'intuition': '중심에 가까울수록 대중적이다',
            
            'embedding_info': {
                'num_items': embedding_info['num_items'],
                'image_embeddings_shape': list(embedding_info['image_embeddings'].shape),
                'json_embeddings_shape': list(embedding_info['json_embeddings'].shape)
            },
            'global_center': {
                'norm': float(np.linalg.norm(global_center)),
                'dimension': int(global_center.shape[0])
            },
            'centrality_analysis': {
                'statistics': {
                    'mean': float(centrality_scores.mean()),
                    'std': float(centrality_scores.std()),
                    'min': float(centrality_scores.min()),
                    'max': float(centrality_scores.max())
                }
            },
            'sets_info': {
                'anchor_indices': self.anchor_indices.tolist(),
                'tail_indices': self.tail_indices.tolist(),
                'anchor_threshold': float(sets_info['anchor_threshold']),
                'tail_threshold': float(sets_info['tail_threshold']),
                'anchor_categories': sets_info['anchor_categories'],
                'tail_categories': sets_info['tail_categories'],
                'all_categories': sets_info['all_categories']
            },
            'distribution_analysis': distribution_analysis,
            
            'parameters': {
                'anchor_percentile': anchor_percentile,
                'tail_percentile': tail_percentile,
                'embedding_type': 'image_only',
                'similarity_metric': 'cosine'
            }
        }
        
        print(f"\n🎉 전체 분석 완료!")
        print(f"📊 주요 결과:")
        print(f"   총 아이템: {len(self.fashion_items)}")
        print(f"   Anchor Set (베스트셀러 Proxy): {len(self.anchor_indices)}개 ({len(self.anchor_indices)/len(self.fashion_items)*100:.1f}%)")
        print(f"   Tail Set: {len(self.tail_indices)}개 ({len(self.tail_indices)/len(self.fashion_items)*100:.1f}%)")
        print(f"   중심성 점수 범위: [{centrality_scores.min():.4f}, {centrality_scores.max():.4f}]")
        
        return complete_results


def run_embedding_centrality_analysis():
    """임베딩 중심성 분석 실행"""
    print("🎯 임베딩 중심성 기반 베스트셀러 Proxy 시스템")
    print("=" * 80)
    print("📌 논문/졸업작품의 핵심 아이디어 구현")
    print("🎯 목표: '베스트셀러를 판매 데이터 없이, 임베딩 공간의 중심성으로 근사'")
    print("=" * 80)
    
    # 데이터셋 경로
    dataset_path = "C:/sample/라벨링데이터"
    
    # Baseline v1 설정 (Temperature 0.1)
    config = TrainingConfig()
    config.temperature = 0.1
    config.batch_size = 16
    config.max_epochs = 8
    
    try:
        # 시스템 초기화
        system = FashionEncoderSystem()
        system.config = config
        
        # 데이터 설정
        print("📁 데이터 설정 중...")
        system.setup_data(dataset_path)
        
        # 트레이너 설정
        print("🏋️ 트레이너 설정 중...")
        system.setup_trainer()
        
        # Baseline v1 체크포인트 로드
        checkpoint_path = "checkpoints/baseline_v1_best_model.pt"
        if Path(checkpoint_path).exists():
            print(f"📦 Baseline v1 체크포인트 로드: {checkpoint_path}")
            system.trainer.load_checkpoint(checkpoint_path)
        else:
            # 일반 체크포인트 시도
            checkpoint_path = "checkpoints/best_model.pt"
            if Path(checkpoint_path).exists():
                print(f"📦 체크포인트 로드: {checkpoint_path}")
                system.trainer.load_checkpoint(checkpoint_path)
            else:
                print("⚠️ 체크포인트가 없습니다. 현재 모델 상태로 분석합니다.")
        
        # 중심성 분석 실행
        analyzer = EmbeddingCentralityProxy(system)
        results = analyzer.run_complete_analysis(
            anchor_percentile=90,  # 상위 10%
            tail_percentile=50     # 하위 50%
        )
        
        # 결과 저장
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        results_file = results_dir / "embedding_centrality_analysis.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 결과 저장: {results_file}")
        
        # 다음 단계 안내
        print(f"\n🔮 다음 단계:")
        print(f"   1. Query-Aware Evaluation에 Anchor Set 적용")
        print(f"   2. Anchor Queries Recall@10 ≥ 90% 목표 달성 확인")
        print(f"   3. Sensitivity Analysis (5%, 10%, 15% 비교)")
        print(f"   4. 논문/발표 자료용 결과 정리")
        
        # 정리
        system.cleanup()
        
        print(f"\n✨ 임베딩 중심성 분석 완료!")
        
        return results
        
    except Exception as e:
        print(f"\n❌ 분석 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    run_embedding_centrality_analysis()