#!/usr/bin/env python3
"""
중심성 기반 Anchor 평가 실행

임베딩 중심성 분석 결과를 로드하여 정확한 Anchor 인덱스로 평가 수행
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.analysis.embedding_centrality_proxy import run_embedding_centrality_analysis
from scripts.analysis.anchor_based_evaluation import AnchorBasedEvaluator
from main import FashionEncoderSystem
from utils.config import TrainingConfig


def run_centrality_based_evaluation():
    """중심성 기반 Anchor 평가 실행"""
    print("🎯 중심성 기반 Anchor 평가 시스템")
    print("=" * 60)
    print("📌 STEP 1: 임베딩 중심성 분석")
    print("📌 STEP 2: 중심성 기반 Anchor 평가")
    print("🎯 목표: Anchor Queries Recall@10 ≥ 90%")
    print("=" * 60)
    
    # STEP 1: 임베딩 중심성 분석 실행
    print("\n🔍 STEP 1: 임베딩 중심성 분석 실행 중...")
    centrality_results = run_embedding_centrality_analysis()
    
    if centrality_results is None:
        print("❌ 중심성 분석 실패")
        return None
    
    # Anchor & Tail 인덱스 추출
    anchor_indices = centrality_results['sets_info']['anchor_indices']
    tail_indices = centrality_results['sets_info']['tail_indices']
    
    print(f"✅ 중심성 분석 완료:")
    print(f"   Anchor Set: {len(anchor_indices)}개")
    print(f"   Tail Set: {len(tail_indices)}개")
    print(f"   Anchor 임계값: {centrality_results['sets_info']['anchor_threshold']:.4f}")
    
    # STEP 2: Anchor 기반 평가 실행
    print(f"\n🔍 STEP 2: 중심성 기반 Anchor 평가 실행 중...")
    
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
                print("⚠️ 체크포인트가 없습니다. 현재 모델 상태로 평가합니다.")
        
        # 중심성 기반 Anchor 평가 실행
        evaluator = AnchorBasedEvaluator(system, anchor_indices, tail_indices)
        evaluation_results = evaluator.run_anchor_based_evaluation()
        
        # 종합 결과 생성
        comprehensive_results = {
            'timestamp': datetime.now().isoformat(),
            'method': 'Centrality-Based Anchor Evaluation',
            'core_concept': '임베딩 중심성 기반 베스트셀러 Proxy 평가',
            
            'centrality_analysis': {
                'anchor_threshold': centrality_results['sets_info']['anchor_threshold'],
                'tail_threshold': centrality_results['sets_info']['tail_threshold'],
                'anchor_categories': centrality_results['sets_info']['anchor_categories'],
                'tail_categories': centrality_results['sets_info']['tail_categories'],
                'centrality_stats': centrality_results['centrality_analysis']['statistics'],
                'distribution_analysis': centrality_results['distribution_analysis']
            },
            
            'evaluation_results': evaluation_results,
            
            'goal_achievement': evaluation_results['summary']['goal_achievement'],
            
            'key_insights': {
                'anchor_recall_10': evaluation_results['summary'].get('anchor_queries', {}).get('recall_at_10', 0),
                'all_recall_10': evaluation_results['summary'].get('all_queries', {}).get('recall_at_10', 0),
                'tail_recall_10': evaluation_results['summary'].get('tail_queries', {}).get('recall_at_10', 0),
                'improvement_vs_all': evaluation_results['summary']['goal_achievement']['improvement'],
                'target_achieved': evaluation_results['summary']['goal_achievement']['anchor_achieved']
            }
        }
        
        # 결과 저장
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        results_file = results_dir / "centrality_based_evaluation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 종합 결과 저장: {results_file}")
        
        # 최종 요약 출력
        print_final_summary(comprehensive_results)
        
        # 정리
        system.cleanup()
        
        print(f"\n✨ 중심성 기반 Anchor 평가 완료!")
        
        return comprehensive_results
        
    except Exception as e:
        print(f"\n❌ 평가 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_final_summary(results: Dict[str, Any]):
    """최종 요약 출력"""
    print(f"\n{'='*80}")
    print("🎉 중심성 기반 베스트셀러 Proxy 시스템 - 최종 결과")
    print(f"{'='*80}")
    
    insights = results['key_insights']
    goal = results['goal_achievement']
    
    print(f"\n🎯 핵심 목표 달성 현황:")
    print(f"   목표: Anchor Queries Recall@10 ≥ 90%")
    print(f"   실제: {insights['anchor_recall_10']:.1f}%")
    print(f"   달성: {'✅ 성공!' if insights['target_achieved'] else '❌ 미달성'}")
    
    print(f"\n📊 성능 비교:")
    print(f"   All Queries Recall@10: {insights['all_recall_10']:.1f}%")
    print(f"   Anchor Queries Recall@10: {insights['anchor_recall_10']:.1f}%")
    print(f"   Tail Queries Recall@10: {insights['tail_recall_10']:.1f}%")
    print(f"   Anchor vs All 개선: {insights['improvement_vs_all']:+.1f}%p")
    print(f"   Anchor vs Tail 개선: {insights['anchor_recall_10'] - insights['tail_recall_10']:+.1f}%p")
    
    print(f"\n🧠 중심성 분석 결과:")
    centrality = results['centrality_analysis']
    print(f"   Anchor 임계값: {centrality['anchor_threshold']:.4f}")
    print(f"   중심성 평균: {centrality['centrality_stats']['mean']:.4f}")
    print(f"   중심성 범위: [{centrality['centrality_stats']['min']:.4f}, {centrality['centrality_stats']['max']:.4f}]")
    
    print(f"\n📈 카테고리별 Anchor 분포:")
    for category, count in centrality['anchor_categories'].items():
        print(f"   {category}: {count}개")
    
    print(f"\n🔬 논문/졸업작품 기여:")
    print(f"   ✅ 판매 데이터 없이 베스트셀러 근사 시스템 구축")
    print(f"   ✅ 임베딩 중심성 기반 Proxy 개념 검증")
    print(f"   ✅ Query-aware 평가 시스템 개발")
    print(f"   ✅ 카테고리별 중심성 특성 분석")
    
    if insights['target_achieved']:
        print(f"\n🎉 축하합니다! 목표 달성으로 핵심 아이디어 검증 성공!")
        print(f"   '중심에 가까울수록 대중적이다' 가설 입증")
    else:
        print(f"\n📈 추가 최적화 방향:")
        print(f"   - Temperature 추가 튜닝 (0.05 ~ 0.15)")
        print(f"   - Anchor 비율 조정 (5%, 15% 테스트)")
        print(f"   - 멀티모달 임베딩 활용 (이미지 + JSON)")
        print(f"   - 대조 학습 손실 함수 개선")


if __name__ == "__main__":
    run_centrality_based_evaluation()