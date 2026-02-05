#!/usr/bin/env python3
"""
통합 파이프라인 구현

중심성 분석 → Query-Aware 평가 → 성능 보고서 생성
자동화된 실험 및 분석 워크플로우

Requirements: 전체 시스템 통합
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import torch

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from main import FashionEncoderSystem
from utils.config import TrainingConfig
from scripts.analysis.embedding_centrality_proxy import EmbeddingCentralityProxy
from scripts.analysis.anchor_based_evaluation import AnchorBasedEvaluator


class IntegratedPipeline:
    """통합 파이프라인: 중심성 분석 → Query-Aware 평가 → 성능 보고서"""
    
    def __init__(self, dataset_path: str, config: Optional[TrainingConfig] = None):
        self.dataset_path = dataset_path
        self.config = config or self._create_optimized_config()
        self.system = None
        
        # 결과 저장
        self.centrality_results = None
        self.evaluation_results = None
        self.performance_report = None
        
    def _create_optimized_config(self) -> TrainingConfig:
        """최적화된 설정 생성"""
        config = TrainingConfig()
        config.temperature = 0.1  # Baseline v1 최적 설정
        config.batch_size = 32    # Recall@10 계산을 위해 증가
        config.max_epochs = 8
        config.learning_rate = 1e-4
        return config
    
    def initialize_system(self) -> None:
        """시스템 초기화 및 설정"""
        print("🚀 통합 파이프라인 시스템 초기화")
        print("=" * 60)
        
        # 시스템 초기화
        self.system = FashionEncoderSystem()
        self.system.config = self.config
        
        # 데이터 설정
        print("📁 데이터 설정 중...")
        self.system.setup_data(self.dataset_path)
        
        # 트레이너 설정
        print("🏋️ 트레이너 설정 중...")
        self.system.setup_trainer()
        
        # 최적 체크포인트 로드
        self._load_best_checkpoint()
        
        print("✅ 시스템 초기화 완료")
        
    def _load_best_checkpoint(self) -> None:
        """최적 체크포인트 로드"""
        checkpoint_candidates = [
            "checkpoints/baseline_v1_best_model.pt",
            "checkpoints/baseline_v2_best_model.pt", 
            "checkpoints/best_model.pt"
        ]
        
        for checkpoint_path in checkpoint_candidates:
            if Path(checkpoint_path).exists():
                print(f"📦 체크포인트 로드: {checkpoint_path}")
                self.system.trainer.load_checkpoint(checkpoint_path)
                return
        
        print("⚠️ 체크포인트가 없습니다. 현재 모델 상태로 진행합니다.")
    
    def run_centrality_analysis(self) -> Dict[str, Any]:
        """STEP 1: 임베딩 중심성 분석 실행"""
        print("\n" + "=" * 60)
        print("STEP 1: 임베딩 중심성 기반 베스트셀러 Proxy 분석")
        print("=" * 60)
        print("🎯 핵심 아이디어: '베스트셀러를 판매 데이터 없이, 임베딩 공간의 중심성으로 근사'")
        
        # 중심성 분석 실행
        analyzer = EmbeddingCentralityProxy(self.system)
        self.centrality_results = analyzer.run_complete_analysis(
            anchor_percentile=90,  # 상위 10%
            tail_percentile=50     # 하위 50%
        )
        
        # 결과 저장
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        centrality_file = results_dir / "integrated_centrality_analysis.json"
        with open(centrality_file, 'w', encoding='utf-8') as f:
            json.dump(self.centrality_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 중심성 분석 결과 저장: {centrality_file}")
        
        return self.centrality_results
    
    def run_query_aware_evaluation(self) -> Dict[str, Any]:
        """STEP 2: Query-Aware 평가 실행"""
        print("\n" + "=" * 60)
        print("STEP 2: Query-Aware 평가 시스템")
        print("=" * 60)
        print("🎯 목표: Anchor Queries Recall@10 ≥ 85% 달성")
        
        if self.centrality_results is None:
            raise ValueError("먼저 중심성 분석을 실행하세요.")
        
        # Anchor & Tail 인덱스 추출
        anchor_indices = self.centrality_results['sets_info']['anchor_indices']
        tail_indices = self.centrality_results['sets_info']['tail_indices']
        
        print(f"📊 Anchor Set: {len(anchor_indices)}개 (베스트셀러 Proxy)")
        print(f"📊 Tail Set: {len(tail_indices)}개")
        
        # Query-Aware 평가 실행
        evaluator = AnchorBasedEvaluator(self.system, anchor_indices, tail_indices)
        self.evaluation_results = evaluator.run_anchor_based_evaluation()
        
        # 결과 저장
        results_dir = Path("results")
        evaluation_file = results_dir / "integrated_query_aware_evaluation.json"
        with open(evaluation_file, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Query-Aware 평가 결과 저장: {evaluation_file}")
        
        return self.evaluation_results
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """STEP 3: 포괄적 성능 보고서 생성"""
        print("\n" + "=" * 60)
        print("STEP 3: 포괄적 성능 보고서 생성")
        print("=" * 60)
        
        if self.centrality_results is None or self.evaluation_results is None:
            raise ValueError("먼저 중심성 분석과 Query-Aware 평가를 실행하세요.")
        
        # 성능 보고서 생성
        self.performance_report = self._create_comprehensive_report()
        
        # 결과 저장
        results_dir = Path("results")
        report_file = results_dir / "integrated_performance_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.performance_report, f, indent=2, ensure_ascii=False)
        
        # 마크다운 보고서 생성
        markdown_file = results_dir / "integrated_performance_report.md"
        self._generate_markdown_report(markdown_file)
        
        print(f"💾 성능 보고서 저장: {report_file}")
        print(f"📄 마크다운 보고서 저장: {markdown_file}")
        
        return self.performance_report
    
    def _create_comprehensive_report(self) -> Dict[str, Any]:
        """포괄적 성능 보고서 생성"""
        # 평가 결과에서 주요 메트릭 추출
        eval_summary = self.evaluation_results.get('summary', {})
        
        # 목표 달성 분석
        goal_achievement = eval_summary.get('goal_achievement', {})
        
        # 중심성 분석 통계
        centrality_stats = self.centrality_results.get('centrality_analysis', {}).get('statistics', {})
        
        # 카테고리별 분석
        category_analysis = self._analyze_category_performance()
        
        # 개선 권장사항
        recommendations = self._generate_recommendations()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'pipeline_version': '1.0',
            'system_configuration': {
                'temperature': self.config.temperature,
                'batch_size': self.config.batch_size,
                'dataset_items': self.centrality_results.get('embedding_info', {}).get('num_items', 0),
                'anchor_set_size': len(self.centrality_results.get('sets_info', {}).get('anchor_indices', [])),
                'tail_set_size': len(self.centrality_results.get('sets_info', {}).get('tail_indices', []))
            },
            'performance_summary': {
                'current_performance': {
                    'all_queries_recall_10': eval_summary.get('all_queries', {}).get('recall_at_10', 0),
                    'anchor_queries_recall_10': eval_summary.get('anchor_queries', {}).get('recall_at_10', 0),
                    'tail_queries_recall_10': eval_summary.get('tail_queries', {}).get('recall_at_10', 0),
                    'top5_accuracy': eval_summary.get('all_queries', {}).get('recall_at_5', 0),
                    'top1_accuracy': eval_summary.get('all_queries', {}).get('top1_accuracy', 0)
                },
                'target_performance': {
                    'all_queries_recall_10_target': '75-80%',
                    'anchor_queries_recall_10_target': '85-92%'
                },
                'goal_achievement': {
                    'all_queries_achieved': goal_achievement.get('all_queries_achieved', False),
                    'anchor_queries_achieved': goal_achievement.get('anchor_achieved', False),
                    'improvement_needed': goal_achievement.get('improvement', 0)
                }
            },
            'centrality_analysis_summary': {
                'mean_centrality': centrality_stats.get('mean', 0),
                'centrality_range': [centrality_stats.get('min', 0), centrality_stats.get('max', 0)],
                'anchor_threshold': self.centrality_results.get('sets_info', {}).get('anchor_threshold', 0),
                'proxy_validation': eval_summary.get('anchor_queries', {}).get('recall_at_10', 0) > eval_summary.get('all_queries', {}).get('recall_at_10', 0)
            },
            'category_analysis': category_analysis,
            'key_insights': self._extract_key_insights(),
            'recommendations': recommendations,
            'next_steps': self._generate_next_steps()
        }
        
        return report
    
    def _analyze_category_performance(self) -> Dict[str, Any]:
        """카테고리별 성능 분석"""
        category_stats = self.centrality_results.get('distribution_analysis', {}).get('category_stats', {})
        anchor_categories = self.centrality_results.get('sets_info', {}).get('anchor_categories', {})
        all_categories = self.centrality_results.get('sets_info', {}).get('all_categories', {})
        
        analysis = {}
        for category in category_stats.keys():
            anchor_count = anchor_categories.get(category, 0)
            total_count = all_categories.get(category, 1)
            anchor_ratio = anchor_count / total_count * 100
            
            analysis[category] = {
                'centrality_mean': category_stats[category]['mean'],
                'centrality_std': category_stats[category]['std'],
                'total_items': total_count,
                'anchor_items': anchor_count,
                'anchor_ratio': anchor_ratio,
                'popularity_rank': 0  # 나중에 계산
            }
        
        # 인기도 순위 계산 (중심성 평균 기준)
        sorted_categories = sorted(analysis.items(), key=lambda x: x[1]['centrality_mean'], reverse=True)
        for rank, (category, data) in enumerate(sorted_categories, 1):
            analysis[category]['popularity_rank'] = rank
        
        return analysis
    
    def _extract_key_insights(self) -> List[str]:
        """핵심 인사이트 추출"""
        insights = []
        
        # 베스트셀러 Proxy 검증
        eval_summary = self.evaluation_results.get('summary', {})
        anchor_recall = eval_summary.get('anchor_queries', {}).get('recall_at_10', 0)
        all_recall = eval_summary.get('all_queries', {}).get('recall_at_10', 0)
        
        if anchor_recall > all_recall:
            improvement = anchor_recall - all_recall
            insights.append(f"✅ 베스트셀러 Proxy 가설 검증: Anchor Queries가 {improvement:.1f}%p 더 높은 성능")
        else:
            insights.append("❌ 베스트셀러 Proxy 가설 미검증: 추가 최적화 필요")
        
        # 카테고리별 인사이트
        category_analysis = self._analyze_category_performance()
        most_popular = max(category_analysis.items(), key=lambda x: x[1]['centrality_mean'])
        least_popular = min(category_analysis.items(), key=lambda x: x[1]['centrality_mean'])
        
        insights.append(f"📊 가장 대중적 카테고리: {most_popular[0]} (중심성: {most_popular[1]['centrality_mean']:.4f})")
        insights.append(f"📊 가장 독특한 카테고리: {least_popular[0]} (중심성: {least_popular[1]['centrality_mean']:.4f})")
        
        # 성능 목표 달성 현황
        goal_achievement = eval_summary.get('goal_achievement', {})
        if goal_achievement.get('anchor_achieved', False):
            insights.append("🎯 Anchor Queries 목표 달성: 85-92% 범위 내")
        else:
            current = goal_achievement.get('anchor_actual', '0%')
            insights.append(f"🎯 Anchor Queries 목표 미달성: 현재 {current}, 목표 85-92%")
        
        return insights
    
    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """개선 권장사항 생성"""
        recommendations = []
        
        eval_summary = self.evaluation_results.get('summary', {})
        anchor_recall = eval_summary.get('anchor_queries', {}).get('recall_at_10', 0)
        all_recall = eval_summary.get('all_queries', {}).get('recall_at_10', 0)
        
        # 성능 개선 권장사항
        if anchor_recall < 85:
            recommendations.append({
                'category': '모델 최적화',
                'priority': 'High',
                'action': 'Temperature 미세 조정 (0.08, 0.09, 0.11, 0.12 실험)',
                'expected_impact': 'Anchor Queries Recall@10 5-10% 향상'
            })
            
            recommendations.append({
                'category': '아키텍처 개선',
                'priority': 'Medium',
                'action': 'JSON Encoder 차원 확장 (128→256)',
                'expected_impact': '전체적인 임베딩 품질 향상'
            })
        
        if all_recall < 75:
            recommendations.append({
                'category': '데이터 최적화',
                'priority': 'High',
                'action': '배치 크기 증가 (32→64) 및 전체 데이터 활용',
                'expected_impact': 'All Queries Recall@10 10-15% 향상'
            })
        
        # 중심성 분석 개선
        centrality_stats = self.centrality_results.get('centrality_analysis', {}).get('statistics', {})
        if centrality_stats.get('std', 0) > 0.06:
            recommendations.append({
                'category': '중심성 분석',
                'priority': 'Medium',
                'action': 'Anchor Set 비율 조정 (5%, 15% 실험)',
                'expected_impact': '베스트셀러 Proxy 정확도 향상'
            })
        
        return recommendations
    
    def _generate_next_steps(self) -> List[str]:
        """다음 단계 생성"""
        next_steps = []
        
        eval_summary = self.evaluation_results.get('summary', {})
        goal_achievement = eval_summary.get('goal_achievement', {})
        
        if not goal_achievement.get('anchor_achieved', False):
            next_steps.append("1. Temperature 최적화 실험 (0.08-0.12 범위)")
            next_steps.append("2. 배치 크기 증가 및 전체 데이터 활용")
            next_steps.append("3. JSON Encoder 아키텍처 개선")
        
        if not goal_achievement.get('all_queries_achieved', False):
            next_steps.append("4. 데이터 증강 기법 적용")
            next_steps.append("5. Multi-head Attention 메커니즘 도입")
        
        next_steps.extend([
            "6. 실시간 추천 API 시스템 구축",
            "7. 논문/졸업작품 결과 정리",
            "8. 베스트셀러 Proxy 시스템 상용화 검토"
        ])
        
        return next_steps
    
    def _generate_markdown_report(self, output_path: Path) -> None:
        """마크다운 형태의 보고서 생성"""
        report = self.performance_report
        
        markdown_content = f"""# 통합 파이프라인 성능 보고서

생성일시: {report['timestamp']}
파이프라인 버전: {report['pipeline_version']}

## 🎯 핵심 성과 요약

### 현재 성능
- **All Queries Recall@10**: {report['performance_summary']['current_performance']['all_queries_recall_10']:.1f}%
- **Anchor Queries Recall@10**: {report['performance_summary']['current_performance']['anchor_queries_recall_10']:.1f}%
- **Top-5 정확도**: {report['performance_summary']['current_performance']['top5_accuracy']:.1f}%
- **Top-1 정확도**: {report['performance_summary']['current_performance']['top1_accuracy']:.1f}%

### 목표 달성 현황
- **All Queries 목표**: {report['performance_summary']['target_performance']['all_queries_recall_10_target']} 
  → {'✅ 달성' if report['performance_summary']['goal_achievement']['all_queries_achieved'] else '❌ 미달성'}
- **Anchor Queries 목표**: {report['performance_summary']['target_performance']['anchor_queries_recall_10_target']} 
  → {'✅ 달성' if report['performance_summary']['goal_achievement']['anchor_queries_achieved'] else '❌ 미달성'}

## 📊 중심성 분석 결과

### 베스트셀러 Proxy 시스템
- **평균 중심성**: {report['centrality_analysis_summary']['mean_centrality']:.4f}
- **중심성 범위**: [{report['centrality_analysis_summary']['centrality_range'][0]:.4f}, {report['centrality_analysis_summary']['centrality_range'][1]:.4f}]
- **Anchor 임계값**: {report['centrality_analysis_summary']['anchor_threshold']:.4f}
- **Proxy 검증**: {'✅ 성공' if report['centrality_analysis_summary']['proxy_validation'] else '❌ 실패'}

### 카테고리별 분석
"""
        
        for category, data in report['category_analysis'].items():
            markdown_content += f"""
#### {category}
- 중심성: {data['centrality_mean']:.4f} ± {data['centrality_std']:.4f}
- 전체 아이템: {data['total_items']}개
- Anchor 아이템: {data['anchor_items']}개 ({data['anchor_ratio']:.1f}%)
- 인기도 순위: {data['popularity_rank']}위
"""
        
        markdown_content += f"""
## 💡 핵심 인사이트

"""
        for insight in report['key_insights']:
            markdown_content += f"- {insight}\n"
        
        markdown_content += f"""
## 🔧 개선 권장사항

"""
        for rec in report['recommendations']:
            markdown_content += f"""
### {rec['category']} (우선순위: {rec['priority']})
- **액션**: {rec['action']}
- **예상 효과**: {rec['expected_impact']}
"""
        
        markdown_content += f"""
## 🚀 다음 단계

"""
        for step in report['next_steps']:
            markdown_content += f"{step}\n"
        
        markdown_content += f"""
## 📈 시스템 설정

- **Temperature**: {report['system_configuration']['temperature']}
- **Batch Size**: {report['system_configuration']['batch_size']}
- **Dataset Items**: {report['system_configuration']['dataset_items']:,}개
- **Anchor Set**: {report['system_configuration']['anchor_set_size']}개
- **Tail Set**: {report['system_configuration']['tail_set_size']}개

---

*이 보고서는 통합 파이프라인에 의해 자동 생성되었습니다.*
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """전체 통합 파이프라인 실행"""
        print("🚀 통합 파이프라인 시작")
        print("=" * 80)
        print("🎯 목표: 중심성 분석 → Query-Aware 평가 → 성능 보고서 생성")
        print("🔄 자동화된 실험 및 분석 워크플로우")
        print("=" * 80)
        
        try:
            # 시스템 초기화
            self.initialize_system()
            
            # STEP 1: 중심성 분석
            centrality_results = self.run_centrality_analysis()
            
            # STEP 2: Query-Aware 평가
            evaluation_results = self.run_query_aware_evaluation()
            
            # STEP 3: 성능 보고서 생성
            performance_report = self.generate_performance_report()
            
            # 최종 결과 출력
            self._print_final_summary()
            
            # 정리
            self.system.cleanup()
            
            print("\n✨ 통합 파이프라인 완료!")
            
            return {
                'centrality_results': centrality_results,
                'evaluation_results': evaluation_results,
                'performance_report': performance_report
            }
            
        except Exception as e:
            print(f"\n❌ 통합 파이프라인 실패: {e}")
            if self.system:
                self.system.cleanup()
            raise
    
    def _print_final_summary(self) -> None:
        """최종 요약 출력"""
        print("\n" + "=" * 80)
        print("📊 통합 파이프라인 최종 요약")
        print("=" * 80)
        
        if self.performance_report:
            current_perf = self.performance_report['performance_summary']['current_performance']
            goal_achievement = self.performance_report['performance_summary']['goal_achievement']
            
            print(f"\n🎯 핵심 성능 지표:")
            print(f"   All Queries Recall@10: {current_perf['all_queries_recall_10']:.1f}% (목표: 75-80%)")
            print(f"   Anchor Queries Recall@10: {current_perf['anchor_queries_recall_10']:.1f}% (목표: 85-92%)")
            print(f"   Top-5 정확도: {current_perf['top5_accuracy']:.1f}%")
            
            print(f"\n✅ 목표 달성 현황:")
            print(f"   All Queries: {'달성' if goal_achievement['all_queries_achieved'] else '미달성'}")
            print(f"   Anchor Queries: {'달성' if goal_achievement['anchor_queries_achieved'] else '미달성'}")
            
            print(f"\n💡 핵심 인사이트:")
            for insight in self.performance_report['key_insights'][:3]:  # 상위 3개만 출력
                print(f"   {insight}")
            
            print(f"\n🔧 우선 권장사항:")
            high_priority_recs = [r for r in self.performance_report['recommendations'] if r['priority'] == 'High']
            for rec in high_priority_recs[:2]:  # 상위 2개만 출력
                print(f"   {rec['action']}")


def run_integrated_pipeline():
    """통합 파이프라인 실행 함수"""
    print("🎯 Fashion JSON Encoder - 통합 파이프라인")
    print("=" * 80)
    print("📌 중심성 분석 → Query-Aware 평가 → 성능 보고서 생성")
    print("🔄 자동화된 실험 및 분석 워크플로우")
    print("=" * 80)
    
    # 데이터셋 경로
    dataset_path = "C:/sample/라벨링데이터"
    
    # 최적화된 설정
    config = TrainingConfig()
    config.temperature = 0.1
    config.batch_size = 32
    config.max_epochs = 8
    
    try:
        # 통합 파이프라인 실행
        pipeline = IntegratedPipeline(dataset_path, config)
        results = pipeline.run_complete_pipeline()
        
        print(f"\n🎉 통합 파이프라인 성공적으로 완료!")
        print(f"📁 결과 파일:")
        print(f"   - results/integrated_centrality_analysis.json")
        print(f"   - results/integrated_query_aware_evaluation.json")
        print(f"   - results/integrated_performance_report.json")
        print(f"   - results/integrated_performance_report.md")
        
        return results
        
    except Exception as e:
        print(f"\n❌ 통합 파이프라인 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    run_integrated_pipeline()