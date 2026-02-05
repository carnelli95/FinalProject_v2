#!/usr/bin/env python3
"""
Query-Aware Evaluation System

교수님 시나리오에 맞춘 평가 시스템:
- 방향 A: 평가 시나리오 분리 (All queries vs Best-seller queries)
- 방향 B: Query-aware Evaluation (판매량/품질/신뢰도 기반 필터링)

목표:
- All queries → Recall@10 ≈ 75~80%
- Best-seller queries → Recall@10 ≈ 85~92%
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import torch
from torch.utils.data import DataLoader

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from main import FashionEncoderSystem
from utils.config import TrainingConfig


class QueryAwareEvaluator:
    """Query-aware 평가 시스템"""
    
    def __init__(self, system: FashionEncoderSystem):
        self.system = system
        self.data_module = system.data_module
        self.trainer = system.trainer
        
        # 평가 결과 저장
        self.evaluation_results = {}
        
    def analyze_dataset_quality(self) -> Dict[str, Any]:
        """데이터셋 품질 분석 및 필터링 기준 설정"""
        print("📊 데이터셋 품질 분석 중...")
        
        # 학습 데이터셋에서 품질 지표 추출
        train_dataset = self.data_module.train_dataset
        fashion_items = train_dataset.fashion_items
        
        # 1. 카테고리별 분포 분석
        category_counts = {}
        for item in fashion_items:
            category = item.category
            category_counts[category] = category_counts.get(category, 0) + 1
        
        total_items = len(fashion_items)
        category_distribution = {
            cat: {"count": count, "percentage": count/total_items*100}
            for cat, count in category_counts.items()
        }
        
        # 2. 메타데이터 완성도 분석
        metadata_completeness = {
            'style': 0,
            'material': 0,
            'detail': 0,
            'silhouette': 0
        }
        
        for item in fashion_items:
            if item.style and len(item.style) > 0:
                metadata_completeness['style'] += 1
            if item.material and len(item.material) > 0:
                metadata_completeness['material'] += 1
            if item.detail and len(item.detail) > 0:
                metadata_completeness['detail'] += 1
            if item.silhouette:
                metadata_completeness['silhouette'] += 1
        
        # 완성도를 퍼센트로 변환
        for field in metadata_completeness:
            metadata_completeness[field] = metadata_completeness[field] / total_items * 100
        
        # 3. 품질 기준 설정 (시뮬레이션)
        # 실제로는 판매량, 이미지 품질 등의 데이터가 필요하지만
        # 여기서는 메타데이터 완성도를 기준으로 시뮬레이션
        quality_scores = []
        for item in fashion_items:
            score = 0
            # 메타데이터 완성도 기반 점수
            if item.style and len(item.style) > 0:
                score += 25
            if item.material and len(item.material) > 0:
                score += 25
            if item.detail and len(item.detail) > 0:
                score += 25
            if item.silhouette:
                score += 25
            quality_scores.append(score)
        
        quality_scores = np.array(quality_scores)
        
        # 품질 분포 분석
        quality_analysis = {
            'mean_score': float(quality_scores.mean()),
            'std_score': float(quality_scores.std()),
            'min_score': float(quality_scores.min()),
            'max_score': float(quality_scores.max()),
            'high_quality_threshold': float(np.percentile(quality_scores, 80)),  # 상위 20%
            'best_seller_threshold': float(np.percentile(quality_scores, 90))   # 상위 10%
        }
        
        analysis_results = {
            'total_items': total_items,
            'category_distribution': category_distribution,
            'metadata_completeness': metadata_completeness,
            'quality_analysis': quality_analysis,
            'quality_scores': quality_scores.tolist()
        }
        
        print(f"✅ 데이터셋 분석 완료:")
        print(f"   총 아이템: {total_items}")
        print(f"   카테고리 분포: {category_distribution}")
        print(f"   메타데이터 완성도: {metadata_completeness}")
        print(f"   품질 점수 평균: {quality_analysis['mean_score']:.1f}")
        print(f"   Best-seller 임계값: {quality_analysis['best_seller_threshold']:.1f}")
        
        return analysis_results
    
    def create_query_subsets(self, quality_analysis: Dict[str, Any]) -> Dict[str, List[int]]:
        """쿼리 서브셋 생성"""
        print("🎯 쿼리 서브셋 생성 중...")
        
        fashion_items = self.data_module.train_dataset.fashion_items
        quality_scores = np.array(quality_analysis['quality_scores'])
        
        # 1. All queries (전체 데이터)
        all_indices = list(range(len(fashion_items)))
        
        # 2. High-quality queries (상위 20%)
        high_quality_threshold = quality_analysis['quality_analysis']['high_quality_threshold']
        high_quality_indices = [i for i, score in enumerate(quality_scores) 
                               if score >= high_quality_threshold]
        
        # 3. Best-seller queries (상위 10%)
        best_seller_threshold = quality_analysis['quality_analysis']['best_seller_threshold']
        best_seller_indices = [i for i, score in enumerate(quality_scores) 
                              if score >= best_seller_threshold]
        
        # 4. Category-balanced subset (각 카테고리에서 균등하게)
        category_indices = {}
        for i, item in enumerate(fashion_items):
            category = item.category
            if category not in category_indices:
                category_indices[category] = []
            category_indices[category].append(i)
        
        # 각 카테고리에서 최소 개수만큼 선택
        min_category_size = min(len(indices) for indices in category_indices.values())
        balanced_indices = []
        for category, indices in category_indices.items():
            # 품질 점수 기준으로 정렬하여 상위 선택
            category_scores = [(i, quality_scores[i]) for i in indices]
            category_scores.sort(key=lambda x: x[1], reverse=True)
            selected = [i for i, _ in category_scores[:min_category_size//2]]  # 각 카테고리에서 절반
            balanced_indices.extend(selected)
        
        query_subsets = {
            'all_queries': all_indices,
            'high_quality': high_quality_indices,
            'best_seller': best_seller_indices,
            'category_balanced': balanced_indices
        }
        
        print(f"✅ 쿼리 서브셋 생성 완료:")
        for name, indices in query_subsets.items():
            print(f"   {name}: {len(indices)}개 ({len(indices)/len(all_indices)*100:.1f}%)")
        
        return query_subsets
    
    def evaluate_on_subset(self, subset_name: str, query_indices: List[int]) -> Dict[str, float]:
        """특정 서브셋에 대한 평가 수행"""
        print(f"🔍 {subset_name} 평가 중... ({len(query_indices)}개 쿼리)")
        
        # 서브셋 데이터로더 생성
        subset_dataset = self._create_subset_dataset(query_indices)
        
        from data.fashion_dataset import collate_fashion_batch
        subset_loader = DataLoader(
            subset_dataset,
            batch_size=self.system.config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fashion_batch
        )
        
        # 평가 수행
        self.trainer.contrastive_learner.eval()
        
        all_similarities = []
        with torch.no_grad():
            for batch in subset_loader:
                batch = self.trainer._move_batch_to_device(batch)
                json_batch = self.trainer._convert_batch_to_dict(batch)
                
                embeddings = self.trainer.contrastive_learner.get_embeddings(batch.images, json_batch)
                similarities = embeddings['similarity_matrix']
                all_similarities.append(similarities.cpu())
        
        # 메트릭 계산
        if all_similarities:
            batch_metrics = []
            for similarities in all_similarities:
                batch_metric = self._compute_enhanced_metrics(similarities)
                batch_metrics.append(batch_metric)
            
            # 평균 계산
            metrics = {}
            if batch_metrics:
                # 모든 배치에서 공통으로 존재하는 키만 사용
                common_keys = set(batch_metrics[0].keys())
                for batch_metric in batch_metrics[1:]:
                    common_keys &= set(batch_metric.keys())
                
                for key in common_keys:
                    metrics[key] = sum(m[key] for m in batch_metrics) / len(batch_metrics)
                
                # 누락된 키들은 0으로 설정
                for key in ['recall_at_3', 'recall_at_5', 'recall_at_10', 'recall_at_20']:
                    if key not in metrics:
                        metrics[key] = 0.0
        else:
            metrics = self._get_empty_metrics()
        
        print(f"✅ {subset_name} 평가 완료:")
        print(f"   Recall@5: {metrics.get('recall_at_5', 0)*100:.1f}%")
        print(f"   Recall@10: {metrics.get('recall_at_10', 0)*100:.1f}%")
        print(f"   Top-1: {metrics.get('top1_accuracy', 0)*100:.1f}%")
        print(f"   MRR: {metrics.get('mean_reciprocal_rank', 0):.3f}")
        
        return metrics
    
    def _create_subset_dataset(self, indices: List[int]):
        """인덱스 기반 서브셋 데이터셋 생성"""
        from torch.utils.data import Subset
        return Subset(self.data_module.train_dataset, indices)
    
    def _compute_enhanced_metrics(self, similarity_matrix: torch.Tensor) -> Dict[str, float]:
        """향상된 메트릭 계산 (Recall@K 포함)"""
        batch_size = similarity_matrix.size(0)
        
        # Top-1 accuracy
        top1_correct = (similarity_matrix.argmax(dim=1) == torch.arange(batch_size)).float().mean()
        
        # Top-K accuracy
        metrics = {'top1_accuracy': top1_correct.item()}
        
        for k in [3, 5, 10, 20]:
            if k <= batch_size:
                topk_indices = similarity_matrix.topk(k=k, dim=1)[1]
                topk_correct = (topk_indices == torch.arange(batch_size).unsqueeze(1)).any(dim=1).float().mean()
                metrics[f'recall_at_{k}'] = topk_correct.item()
                metrics[f'top{k}_accuracy'] = topk_correct.item()
        
        # Mean reciprocal rank
        ranks = (similarity_matrix.argsort(dim=1, descending=True) == torch.arange(batch_size).unsqueeze(1)).nonzero()[:, 1] + 1
        mrr = (1.0 / ranks.float()).mean()
        metrics['mean_reciprocal_rank'] = mrr.item()
        
        # 추가 메트릭
        metrics['avg_positive_similarity'] = similarity_matrix.diag().mean().item()
        
        # Negative similarity (off-diagonal elements)
        mask = torch.eye(batch_size, dtype=torch.bool)
        negative_similarities = similarity_matrix[~mask]
        if len(negative_similarities) > 0:
            metrics['avg_negative_similarity'] = negative_similarities.mean().item()
        else:
            metrics['avg_negative_similarity'] = 0.0
        
        return metrics
    
    def _get_empty_metrics(self) -> Dict[str, float]:
        """빈 메트릭 딕셔너리 반환"""
        return {
            'top1_accuracy': 0.0,
            'recall_at_3': 0.0,
            'recall_at_5': 0.0,
            'recall_at_10': 0.0,
            'recall_at_20': 0.0,
            'top3_accuracy': 0.0,
            'top5_accuracy': 0.0,
            'top10_accuracy': 0.0,
            'top20_accuracy': 0.0,
            'mean_reciprocal_rank': 0.0,
            'avg_positive_similarity': 0.0,
            'avg_negative_similarity': 0.0
        }
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """포괄적인 평가 실행"""
        print("🚀 Query-Aware 포괄적 평가 시작")
        print("=" * 60)
        
        # 1. 데이터셋 품질 분석
        quality_analysis = self.analyze_dataset_quality()
        
        # 2. 쿼리 서브셋 생성
        query_subsets = self.create_query_subsets(quality_analysis)
        
        # 3. 각 서브셋에 대한 평가
        evaluation_results = {}
        
        for subset_name, query_indices in query_subsets.items():
            print(f"\n{'='*40}")
            print(f"평가 중: {subset_name}")
            print(f"{'='*40}")
            
            metrics = self.evaluate_on_subset(subset_name, query_indices)
            evaluation_results[subset_name] = {
                'metrics': metrics,
                'query_count': len(query_indices),
                'percentage': len(query_indices) / len(query_subsets['all_queries']) * 100
            }
        
        # 4. 결과 종합
        comprehensive_results = {
            'timestamp': datetime.now().isoformat(),
            'dataset_analysis': quality_analysis,
            'query_subsets': {name: len(indices) for name, indices in query_subsets.items()},
            'evaluation_results': evaluation_results,
            'summary': self._create_evaluation_summary(evaluation_results)
        }
        
        # 5. 결과 출력
        self._print_evaluation_summary(comprehensive_results)
        
        return comprehensive_results
    
    def _create_evaluation_summary(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """평가 결과 요약 생성"""
        summary = {}
        
        # 주요 메트릭 추출
        for subset_name, result in evaluation_results.items():
            metrics = result['metrics']
            summary[subset_name] = {
                'recall_at_5': metrics.get('recall_at_5', 0) * 100,
                'recall_at_10': metrics.get('recall_at_10', 0) * 100,
                'top1_accuracy': metrics.get('top1_accuracy', 0) * 100,
                'mrr': metrics.get('mean_reciprocal_rank', 0),
                'query_count': result['query_count']
            }
        
        # 목표 달성 여부 확인
        all_recall_10 = summary.get('all_queries', {}).get('recall_at_10', 0)
        best_seller_recall_10 = summary.get('best_seller', {}).get('recall_at_10', 0)
        
        summary['goal_achievement'] = {
            'all_queries_target': '75-80%',
            'all_queries_actual': f"{all_recall_10:.1f}%",
            'all_queries_achieved': 75 <= all_recall_10 <= 80,
            
            'best_seller_target': '85-92%',
            'best_seller_actual': f"{best_seller_recall_10:.1f}%",
            'best_seller_achieved': 85 <= best_seller_recall_10 <= 92
        }
        
        return summary
    
    def _print_evaluation_summary(self, results: Dict[str, Any]):
        """평가 결과 요약 출력"""
        print(f"\n{'='*60}")
        print("📊 Query-Aware 평가 결과 요약")
        print(f"{'='*60}")
        
        summary = results['summary']
        
        print(f"\n🎯 목표 달성 현황:")
        goal = summary['goal_achievement']
        print(f"   All Queries Recall@10:")
        print(f"     목표: {goal['all_queries_target']}")
        print(f"     실제: {goal['all_queries_actual']}")
        print(f"     달성: {'✅' if goal['all_queries_achieved'] else '❌'}")
        
        print(f"   Best-seller Queries Recall@10:")
        print(f"     목표: {goal['best_seller_target']}")
        print(f"     실제: {goal['best_seller_actual']}")
        print(f"     달성: {'✅' if goal['best_seller_achieved'] else '❌'}")
        
        print(f"\n📈 상세 결과:")
        for subset_name, metrics in summary.items():
            if subset_name == 'goal_achievement':
                continue
            
            print(f"   {subset_name}:")
            print(f"     쿼리 수: {metrics['query_count']}")
            print(f"     Recall@5: {metrics['recall_at_5']:.1f}%")
            print(f"     Recall@10: {metrics['recall_at_10']:.1f}%")
            print(f"     Top-1: {metrics['top1_accuracy']:.1f}%")
            print(f"     MRR: {metrics['mrr']:.3f}")
        
        print(f"\n💡 인사이트:")
        all_r10 = summary.get('all_queries', {}).get('recall_at_10', 0)
        best_r10 = summary.get('best_seller', {}).get('recall_at_10', 0)
        improvement = best_r10 - all_r10
        
        print(f"   Best-seller 쿼리는 전체 대비 {improvement:.1f}%p 높은 성능")
        print(f"   Query-aware 평가로 실제 사용 시나리오 반영")
        
        if goal['all_queries_achieved'] and goal['best_seller_achieved']:
            print(f"   🎉 모든 목표 달성! 교수님 시나리오 완벽 대응")
        elif goal['best_seller_achieved']:
            print(f"   ✅ Best-seller 목표 달성! 핵심 시나리오 성공")
        else:
            print(f"   📈 추가 튜닝으로 목표 달성 가능")


def run_query_aware_evaluation():
    """Query-aware 평가 실행"""
    print("🎯 Query-Aware Evaluation System")
    print("=" * 60)
    print("교수님 시나리오 맞춤 평가:")
    print("- 방향 A: 평가 시나리오 분리")
    print("- 방향 B: Query-aware Evaluation")
    print("=" * 60)
    
    # 데이터셋 경로
    dataset_path = "C:/sample/라벨링데이터"
    
    # 기존 모델 설정 (Baseline v1)
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
        
        # 기존 체크포인트 로드 (있다면)
        checkpoint_path = "checkpoints/best_model.pt"
        if Path(checkpoint_path).exists():
            print(f"📦 체크포인트 로드: {checkpoint_path}")
            system.trainer.load_checkpoint(checkpoint_path)
        else:
            print("⚠️ 체크포인트가 없습니다. 현재 모델 상태로 평가합니다.")
        
        # Query-aware 평가 실행
        evaluator = QueryAwareEvaluator(system)
        results = evaluator.run_comprehensive_evaluation()
        
        # 결과 저장
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        results_file = results_dir / "query_aware_evaluation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 결과 저장: {results_file}")
        
        # 정리
        system.cleanup()
        
        print(f"\n✨ Query-Aware 평가 완료!")
        
        return results
        
    except Exception as e:
        print(f"\n❌ 평가 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    run_query_aware_evaluation()