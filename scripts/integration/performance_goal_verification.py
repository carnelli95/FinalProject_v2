#!/usr/bin/env python3
"""
성능 목표 달성 검증 시스템

- All Queries Recall@10: 75-80% 목표 달성
- Anchor Queries Recall@10: 85-92% 목표 달성  
- 베스트셀러 Proxy 시스템 완전 검증

Requirements: 성능 목표
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
from scripts.integration.integrated_pipeline import IntegratedPipeline


class PerformanceGoalVerifier:
    """성능 목표 달성 검증 시스템"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.performance_targets = {
            'all_queries_recall_10': {'min': 75.0, 'max': 80.0, 'unit': '%'},
            'anchor_queries_recall_10': {'min': 85.0, 'max': 92.0, 'unit': '%'},
            'top5_accuracy': {'min': 70.0, 'max': None, 'unit': '%'},  # 보조 목표
            'centrality_proxy_validation': {'min': 1.0, 'max': None, 'unit': '%p'}  # Anchor > All
        }
        
        self.verification_results = None
        
    def run_comprehensive_verification(self) -> Dict[str, Any]:
        """포괄적 성능 목표 검증 실행"""
        print("🎯 성능 목표 달성 검증 시스템")
        print("=" * 80)
        print("📊 목표:")
        print("   - All Queries Recall@10: 75-80%")
        print("   - Anchor Queries Recall@10: 85-92%")
        print("   - 베스트셀러 Proxy 시스템 완전 검증")
        print("=" * 80)
        
        # 다양한 설정으로 검증 실행
        verification_configs = self._generate_verification_configs()
        
        all_results = {}
        best_config = None
        best_score = 0
        
        for config_name, config in verification_configs.items():
            print(f"\n{'='*60}")
            print(f"검증 설정: {config_name}")
            print(f"{'='*60}")
            
            try:
                # 통합 파이프라인 실행
                pipeline = IntegratedPipeline(self.dataset_path, config)
                results = pipeline.run_complete_pipeline()
                
                # 성능 검증
                verification = self._verify_performance_goals(results, config_name)
                all_results[config_name] = verification
                
                # 최고 성능 설정 추적
                if verification['overall_score'] > best_score:
                    best_score = verification['overall_score']
                    best_config = config_name
                
                print(f"✅ {config_name} 검증 완료 (점수: {verification['overall_score']:.1f})")
                
            except Exception as e:
                print(f"❌ {config_name} 검증 실패: {e}")
                all_results[config_name] = {
                    'status': 'failed',
                    'error': str(e),
                    'overall_score': 0
                }
        
        # 종합 검증 결과 생성
        self.verification_results = self._create_comprehensive_verification_report(
            all_results, best_config
        )
        
        # 결과 저장
        self._save_verification_results()
        
        # 최종 검증 결과 출력
        self._print_verification_summary()
        
        return self.verification_results
    
    def _generate_verification_configs(self) -> Dict[str, TrainingConfig]:
        """검증용 다양한 설정 생성"""
        configs = {}
        
        # 1. Baseline v1 설정 (현재 최적)
        baseline_config = TrainingConfig()
        baseline_config.temperature = 0.1
        baseline_config.batch_size = 32
        baseline_config.max_epochs = 8
        baseline_config.learning_rate = 1e-4
        configs['baseline_v1'] = baseline_config
        
        # 2. 최적화된 설정 1 (배치 크기 증가)
        optimized_config1 = TrainingConfig()
        optimized_config1.temperature = 0.1
        optimized_config1.batch_size = 64  # 증가
        optimized_config1.max_epochs = 8
        optimized_config1.learning_rate = 1e-4
        configs['optimized_batch64'] = optimized_config1
        
        # 3. Temperature 미세 조정 1
        temp_config1 = TrainingConfig()
        temp_config1.temperature = 0.08  # 더 낮은 temperature
        temp_config1.batch_size = 32
        temp_config1.max_epochs = 8
        temp_config1.learning_rate = 1e-4
        configs['temperature_008'] = temp_config1
        
        # 4. Temperature 미세 조정 2
        temp_config2 = TrainingConfig()
        temp_config2.temperature = 0.12  # 더 높은 temperature
        temp_config2.batch_size = 32
        temp_config2.max_epochs = 8
        temp_config2.learning_rate = 1e-4
        configs['temperature_012'] = temp_config2
        
        return configs
    
    def _verify_performance_goals(self, pipeline_results: Dict[str, Any], 
                                config_name: str) -> Dict[str, Any]:
        """성능 목표 검증"""
        if 'performance_report' not in pipeline_results:
            return {
                'status': 'failed',
                'error': 'Performance report not found',
                'overall_score': 0
            }
        
        performance_report = pipeline_results['performance_report']
        current_perf = performance_report['performance_summary']['current_performance']
        
        # 각 목표별 검증
        verification = {
            'config_name': config_name,
            'timestamp': datetime.now().isoformat(),
            'goals': {},
            'overall_score': 0,
            'status': 'completed'
        }
        
        total_score = 0
        max_score = 0
        
        for goal_name, target in self.performance_targets.items():
            if goal_name == 'centrality_proxy_validation':
                # 베스트셀러 Proxy 검증 (Anchor > All)
                anchor_recall = current_perf.get('anchor_queries_recall_10', 0)
                all_recall = current_perf.get('all_queries_recall_10', 0)
                actual_value = anchor_recall - all_recall
                achieved = actual_value >= target['min']
                score = 100 if achieved else max(0, actual_value / target['min'] * 100)
            else:
                # 일반 메트릭 검증
                actual_value = current_perf.get(goal_name, 0)
                if goal_name.endswith('_recall_10'):
                    actual_value *= 100  # 백분율 변환
                
                if target['max'] is None:
                    # 최소값만 있는 경우
                    achieved = actual_value >= target['min']
                    score = min(100, actual_value / target['min'] * 100)
                else:
                    # 범위가 있는 경우
                    achieved = target['min'] <= actual_value <= target['max']
                    if achieved:
                        score = 100
                    elif actual_value < target['min']:
                        score = actual_value / target['min'] * 100
                    else:
                        score = max(0, 100 - (actual_value - target['max']) / target['max'] * 50)
            
            verification['goals'][goal_name] = {
                'target': target,
                'actual': actual_value,
                'achieved': achieved,
                'score': score
            }
            
            total_score += score
            max_score += 100
        
        verification['overall_score'] = total_score / max_score * 100 if max_score > 0 else 0
        
        return verification
    
    def _create_comprehensive_verification_report(self, all_results: Dict[str, Any], 
                                                best_config: str) -> Dict[str, Any]:
        """종합 검증 보고서 생성"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'verification_summary': {
                'total_configs_tested': len(all_results),
                'successful_configs': len([r for r in all_results.values() if r.get('status') == 'completed']),
                'best_config': best_config,
                'best_score': all_results.get(best_config, {}).get('overall_score', 0) if best_config else 0
            },
            'performance_targets': self.performance_targets,
            'detailed_results': all_results,
            'goal_achievement_analysis': self._analyze_goal_achievement(all_results),
            'recommendations': self._generate_optimization_recommendations(all_results, best_config),
            'next_steps': self._generate_verification_next_steps(all_results, best_config)
        }
        
        return report
    
    def _analyze_goal_achievement(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """목표 달성 분석"""
        analysis = {
            'goals_achieved_by_config': {},
            'best_performance_by_goal': {},
            'achievement_summary': {}
        }
        
        # 각 설정별 목표 달성 현황
        for config_name, result in all_results.items():
            if result.get('status') != 'completed':
                continue
                
            goals = result.get('goals', {})
            achieved_goals = [goal for goal, data in goals.items() if data.get('achieved', False)]
            analysis['goals_achieved_by_config'][config_name] = {
                'achieved_count': len(achieved_goals),
                'total_count': len(goals),
                'achieved_goals': achieved_goals,
                'overall_score': result.get('overall_score', 0)
            }
        
        # 각 목표별 최고 성능
        for goal_name in self.performance_targets.keys():
            best_performance = None
            best_config = None
            
            for config_name, result in all_results.items():
                if result.get('status') != 'completed':
                    continue
                    
                goal_data = result.get('goals', {}).get(goal_name, {})
                actual_value = goal_data.get('actual', 0)
                
                if best_performance is None or actual_value > best_performance:
                    best_performance = actual_value
                    best_config = config_name
            
            analysis['best_performance_by_goal'][goal_name] = {
                'best_value': best_performance,
                'best_config': best_config,
                'target_achieved': best_performance >= self.performance_targets[goal_name]['min'] if best_performance else False
            }
        
        # 전체 달성 요약
        total_goals = len(self.performance_targets)
        achieved_goals = len([g for g in analysis['best_performance_by_goal'].values() if g['target_achieved']])
        
        analysis['achievement_summary'] = {
            'total_goals': total_goals,
            'achieved_goals': achieved_goals,
            'achievement_rate': achieved_goals / total_goals * 100 if total_goals > 0 else 0,
            'critical_goals_status': {
                'all_queries_recall_10': analysis['best_performance_by_goal'].get('all_queries_recall_10', {}).get('target_achieved', False),
                'anchor_queries_recall_10': analysis['best_performance_by_goal'].get('anchor_queries_recall_10', {}).get('target_achieved', False)
            }
        }
        
        return analysis
    
    def _generate_optimization_recommendations(self, all_results: Dict[str, Any], 
                                             best_config: str) -> List[Dict[str, str]]:
        """최적화 권장사항 생성"""
        recommendations = []
        
        if not best_config or not all_results.get(best_config):
            recommendations.append({
                'priority': 'Critical',
                'category': '시스템 안정성',
                'action': '모든 설정에서 실패 - 기본 시스템 점검 필요',
                'expected_impact': '시스템 정상화'
            })
            return recommendations
        
        best_result = all_results[best_config]
        best_goals = best_result.get('goals', {})
        
        # All Queries Recall@10 개선
        all_queries_goal = best_goals.get('all_queries_recall_10', {})
        if not all_queries_goal.get('achieved', False):
            current_value = all_queries_goal.get('actual', 0)
            target_value = self.performance_targets['all_queries_recall_10']['min']
            gap = target_value - current_value
            
            if gap > 40:  # 40% 이상 차이
                recommendations.append({
                    'priority': 'Critical',
                    'category': '모델 아키텍처',
                    'action': 'JSON Encoder 차원 확장 (128→256→512) 및 Multi-head Attention 도입',
                    'expected_impact': f'All Queries Recall@10 {gap/2:.1f}% 향상 예상'
                })
            elif gap > 20:  # 20% 이상 차이
                recommendations.append({
                    'priority': 'High',
                    'category': '하이퍼파라미터',
                    'action': '배치 크기 증가 (64→128) 및 학습률 조정',
                    'expected_impact': f'All Queries Recall@10 {gap/3:.1f}% 향상 예상'
                })
            else:
                recommendations.append({
                    'priority': 'Medium',
                    'category': 'Temperature 최적화',
                    'action': 'Temperature 미세 조정 (0.05-0.15 범위 세밀 탐색)',
                    'expected_impact': f'All Queries Recall@10 {gap/2:.1f}% 향상 예상'
                })
        
        # Anchor Queries Recall@10 개선
        anchor_queries_goal = best_goals.get('anchor_queries_recall_10', {})
        if not anchor_queries_goal.get('achieved', False):
            current_value = anchor_queries_goal.get('actual', 0)
            target_value = self.performance_targets['anchor_queries_recall_10']['min']
            gap = target_value - current_value
            
            recommendations.append({
                'priority': 'High',
                'category': '베스트셀러 Proxy 최적화',
                'action': f'Anchor Set 비율 조정 (5%, 15%, 20% 실험) 및 중심성 계산 방법 개선',
                'expected_impact': f'Anchor Queries Recall@10 {gap/2:.1f}% 향상 예상'
            })
        
        # 베스트셀러 Proxy 검증
        proxy_goal = best_goals.get('centrality_proxy_validation', {})
        if not proxy_goal.get('achieved', False):
            recommendations.append({
                'priority': 'Medium',
                'category': '중심성 분석',
                'action': '글로벌 중심 벡터 계산 방법 개선 (가중 평균, 클러스터링 기반)',
                'expected_impact': '베스트셀러 Proxy 가설 검증 성공'
            })
        
        return recommendations
    
    def _generate_verification_next_steps(self, all_results: Dict[str, Any], 
                                        best_config: str) -> List[str]:
        """검증 기반 다음 단계 생성"""
        next_steps = []
        
        if not best_config:
            next_steps.extend([
                "1. 시스템 기본 설정 점검 및 디버깅",
                "2. 데이터 로딩 및 모델 초기화 문제 해결",
                "3. 기본 학습 파이프라인 안정화"
            ])
            return next_steps
        
        best_result = all_results[best_config]
        best_score = best_result.get('overall_score', 0)
        
        if best_score < 30:  # 매우 낮은 성능
            next_steps.extend([
                "1. 기본 모델 아키텍처 재검토",
                "2. 데이터 전처리 파이프라인 개선",
                "3. 학습 안정성 확보"
            ])
        elif best_score < 60:  # 중간 성능
            next_steps.extend([
                "1. 하이퍼파라미터 대규모 튜닝",
                "2. 모델 아키텍처 개선 (Multi-head Attention)",
                "3. 데이터 증강 기법 적용"
            ])
        else:  # 높은 성능
            next_steps.extend([
                "1. 미세 조정을 통한 성능 최적화",
                "2. 앙상블 기법 적용",
                "3. 실시간 API 시스템 구축 준비"
            ])
        
        # 공통 다음 단계
        next_steps.extend([
            "4. 논문/졸업작품 결과 정리",
            "5. 베스트셀러 Proxy 시스템 상용화 검토",
            "6. 추가 데이터셋 확장 실험"
        ])
        
        return next_steps
    
    def _save_verification_results(self) -> None:
        """검증 결과 저장"""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # JSON 결과 저장
        json_file = results_dir / "performance_goal_verification_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.verification_results, f, indent=2, ensure_ascii=False)
        
        # 마크다운 보고서 생성
        markdown_file = results_dir / "performance_goal_verification_report.md"
        self._generate_verification_markdown_report(markdown_file)
        
        print(f"💾 검증 결과 저장: {json_file}")
        print(f"📄 검증 보고서 저장: {markdown_file}")
    
    def _generate_verification_markdown_report(self, output_path: Path) -> None:
        """검증 마크다운 보고서 생성"""
        report = self.verification_results
        
        markdown_content = f"""# 성능 목표 달성 검증 보고서

생성일시: {report['timestamp']}

## 🎯 검증 개요

### 성능 목표
- **All Queries Recall@10**: 75-80%
- **Anchor Queries Recall@10**: 85-92%
- **베스트셀러 Proxy 검증**: Anchor > All Queries

### 검증 결과 요약
- **테스트된 설정**: {report['verification_summary']['total_configs_tested']}개
- **성공한 설정**: {report['verification_summary']['successful_configs']}개
- **최고 성능 설정**: {report['verification_summary']['best_config']}
- **최고 점수**: {report['verification_summary']['best_score']:.1f}점

## 📊 목표 달성 분석

### 전체 달성 현황
- **달성된 목표**: {report['goal_achievement_analysis']['achievement_summary']['achieved_goals']}/{report['goal_achievement_analysis']['achievement_summary']['total_goals']}개
- **달성률**: {report['goal_achievement_analysis']['achievement_summary']['achievement_rate']:.1f}%

### 핵심 목표 상태
"""
        
        critical_goals = report['goal_achievement_analysis']['achievement_summary']['critical_goals_status']
        for goal, achieved in critical_goals.items():
            status = "✅ 달성" if achieved else "❌ 미달성"
            markdown_content += f"- **{goal}**: {status}\n"
        
        markdown_content += f"""
## 🏆 최고 성능 분석

### 설정별 성과
"""
        
        for config_name, config_data in report['goal_achievement_analysis']['goals_achieved_by_config'].items():
            markdown_content += f"""
#### {config_name}
- 달성 목표: {config_data['achieved_count']}/{config_data['total_count']}개
- 전체 점수: {config_data['overall_score']:.1f}점
- 달성한 목표: {', '.join(config_data['achieved_goals']) if config_data['achieved_goals'] else '없음'}
"""
        
        markdown_content += f"""
## 🔧 최적화 권장사항

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
        for i, step in enumerate(report['next_steps'], 1):
            markdown_content += f"{step}\n"
        
        markdown_content += f"""
---

*이 보고서는 성능 목표 검증 시스템에 의해 자동 생성되었습니다.*
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
    
    def _print_verification_summary(self) -> None:
        """검증 요약 출력"""
        print("\n" + "=" * 80)
        print("🎯 성능 목표 달성 검증 최종 요약")
        print("=" * 80)
        
        if not self.verification_results:
            print("❌ 검증 결과가 없습니다.")
            return
        
        summary = self.verification_results['verification_summary']
        achievement = self.verification_results['goal_achievement_analysis']['achievement_summary']
        
        print(f"\n📊 검증 통계:")
        print(f"   테스트된 설정: {summary['total_configs_tested']}개")
        print(f"   성공한 설정: {summary['successful_configs']}개")
        print(f"   최고 성능 설정: {summary['best_config']}")
        print(f"   최고 점수: {summary['best_score']:.1f}점")
        
        print(f"\n🎯 목표 달성 현황:")
        print(f"   달성된 목표: {achievement['achieved_goals']}/{achievement['total_goals']}개")
        print(f"   달성률: {achievement['achievement_rate']:.1f}%")
        
        critical_goals = achievement['critical_goals_status']
        print(f"\n✅ 핵심 목표:")
        for goal, achieved in critical_goals.items():
            status = "달성" if achieved else "미달성"
            print(f"   {goal}: {status}")
        
        print(f"\n🔧 우선 권장사항:")
        high_priority_recs = [r for r in self.verification_results['recommendations'] if r['priority'] in ['Critical', 'High']]
        for rec in high_priority_recs[:3]:  # 상위 3개만 출력
            print(f"   {rec['action']}")


def run_performance_goal_verification():
    """성능 목표 달성 검증 실행 함수"""
    print("🎯 Fashion JSON Encoder - 성능 목표 달성 검증")
    print("=" * 80)
    print("📌 목표: All Queries Recall@10 75-80%, Anchor Queries Recall@10 85-92%")
    print("🔍 다양한 설정으로 포괄적 검증 수행")
    print("=" * 80)
    
    # 데이터셋 경로
    dataset_path = "C:/sample/라벨링데이터"
    
    try:
        # 성능 목표 검증 실행
        verifier = PerformanceGoalVerifier(dataset_path)
        results = verifier.run_comprehensive_verification()
        
        print(f"\n🎉 성능 목표 검증 완료!")
        print(f"📁 결과 파일:")
        print(f"   - results/performance_goal_verification_results.json")
        print(f"   - results/performance_goal_verification_report.md")
        
        return results
        
    except Exception as e:
        print(f"\n❌ 성능 목표 검증 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    run_performance_goal_verification()