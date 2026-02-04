"""
학습 진행도 간단한 모니터링 및 시각화

이 모듈은 Fashion JSON Encoder 학습 과정을 tqdm과 matplotlib을 사용하여
간단하게 모니터링하고 시각화하는 기능을 제공합니다.
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# 한글 폰트 설정
plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TrainingMonitor:
    """학습 진행도 간단한 모니터링 클래스 (tqdm + matplotlib 기반)"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # 학습 상태 추적
        self.training_state = {
            'stage': 'Stage 1',  # Stage 1: JSON Encoder, Stage 2: Contrastive Learning
            'current_epoch': 0,
            'total_epochs': 0,
            'start_time': None,
            'is_training': False
        }
        
        # 메트릭 저장
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'top1_accuracy': [],
            'top5_accuracy': [],
            'mrr': [],
            'positive_similarity': [],
            'negative_similarity': [],
            'l2_norm': [],
            'epochs': []
        }
        
        # tqdm 진행 바
        self.epoch_pbar = None
        
    def start_training(self, stage: str, total_epochs: int):
        """학습 시작"""
        self.training_state.update({
            'stage': stage,
            'current_epoch': 0,
            'total_epochs': total_epochs,
            'start_time': datetime.now(),
            'is_training': True
        })
        
        print(f"\n🚀 {stage} 학습 시작!")
        print(f"📊 총 에포크: {total_epochs}")
        print(f"⏰ 시작 시간: {self.training_state['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        # tqdm 진행 바 초기화
        self.epoch_pbar = tqdm(
            total=total_epochs,
            desc=f"{stage}",
            unit="epoch",
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
    def update_epoch(self, epoch: int, metrics: Dict[str, float]):
        """에포크 업데이트"""
        self.training_state['current_epoch'] = epoch
        
        # 메트릭 히스토리 업데이트
        self.metrics_history['epochs'].append(epoch)
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        # tqdm 진행 바 업데이트
        if self.epoch_pbar:
            # 주요 메트릭을 postfix로 표시
            postfix_dict = {}
            if 'val_loss' in metrics:
                postfix_dict['val_loss'] = f"{metrics['val_loss']:.4f}"
            if 'top5_accuracy' in metrics:
                postfix_dict['top5_acc'] = f"{metrics['top5_accuracy']:.2f}%"
            if 'mrr' in metrics:
                postfix_dict['mrr'] = f"{metrics['mrr']:.4f}"
                
            self.epoch_pbar.set_postfix(postfix_dict)
            self.epoch_pbar.update(1)
            
    def save_checkpoint(self, checkpoint_path: str):
        """체크포인트 저장 알림"""
        if self.epoch_pbar:
            self.epoch_pbar.write(f"� 체크포인트 저장: {checkpoint_path}")
        
    def finish_training(self):
        """학습 완료"""
        self.training_state['is_training'] = False
        end_time = datetime.now()
        duration = end_time - self.training_state['start_time']
        
        # tqdm 진행 바 종료
        if self.epoch_pbar:
            self.epoch_pbar.close()
        
        print(f"\n✅ {self.training_state['stage']} 학습 완료!")
        print(f"⏱️ 총 소요 시간: {duration}")
        
        # 최종 결과 저장 및 시각화
        self.save_training_summary()
        self.create_simple_charts()
        
    def create_simple_charts(self):
        """간단한 matplotlib 차트 생성"""
        if not self.metrics_history['epochs']:
            return
            
        # 2x2 서브플롯 생성
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Fashion JSON Encoder 학습 결과 - {self.training_state["stage"]}', fontsize=14)
        
        epochs = self.metrics_history['epochs']
        
        # 1. 손실 곡선
        if self.metrics_history['train_loss'] or self.metrics_history['val_loss']:
            if self.metrics_history['train_loss']:
                axes[0, 0].plot(epochs, self.metrics_history['train_loss'], 'b-', label='학습 손실', linewidth=2)
            if self.metrics_history['val_loss']:
                axes[0, 0].plot(epochs, self.metrics_history['val_loss'], 'r-', label='검증 손실', linewidth=2)
            axes[0, 0].set_title('학습/검증 손실')
            axes[0, 0].set_xlabel('에포크')
            axes[0, 0].set_ylabel('손실')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
        # 2. 정확도
        if self.metrics_history['top1_accuracy'] or self.metrics_history['top5_accuracy']:
            if self.metrics_history['top1_accuracy']:
                axes[0, 1].plot(epochs, self.metrics_history['top1_accuracy'], 'g-', label='Top-1', linewidth=2)
            if self.metrics_history['top5_accuracy']:
                axes[0, 1].plot(epochs, self.metrics_history['top5_accuracy'], 'orange', label='Top-5', linewidth=2)
            axes[0, 1].set_title('Top-K 정확도')
            axes[0, 1].set_xlabel('에포크')
            axes[0, 1].set_ylabel('정확도 (%)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
        # 3. MRR
        if self.metrics_history['mrr']:
            axes[1, 0].plot(epochs, self.metrics_history['mrr'], 'purple', label='MRR', linewidth=2)
            axes[1, 0].set_title('Mean Reciprocal Rank')
            axes[1, 0].set_xlabel('에포크')
            axes[1, 0].set_ylabel('MRR')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
        # 4. 유사도
        if self.metrics_history['positive_similarity'] or self.metrics_history['negative_similarity']:
            if self.metrics_history['positive_similarity']:
                axes[1, 1].plot(epochs, self.metrics_history['positive_similarity'], 'cyan', label='Positive', linewidth=2)
            if self.metrics_history['negative_similarity']:
                axes[1, 1].plot(epochs, self.metrics_history['negative_similarity'], 'red', label='Negative', linewidth=2)
            axes[1, 1].set_title('유사도')
            axes[1, 1].set_xlabel('에포크')
            axes[1, 1].set_ylabel('유사도')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
        plt.tight_layout()
        chart_path = self.results_dir / "training_charts.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 학습 차트 저장: {chart_path}")
        
    def save_training_summary(self):
        """학습 요약 저장"""
        summary = {
            'training_state': self.training_state.copy(),
            'metrics_history': self.metrics_history.copy(),
            'timestamp': datetime.now().isoformat()
        }
        
        # datetime 객체를 문자열로 변환
        if summary['training_state']['start_time']:
            summary['training_state']['start_time'] = summary['training_state']['start_time'].isoformat()
            
        summary_path = self.results_dir / "training_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        print(f"📄 학습 요약 저장: {summary_path}")


class DashboardDataProvider:
    """대시보드용 데이터 제공 클래스 (간단한 버전)"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        
    def get_dashboard_data(self) -> Dict[str, Any]:
        """대시보드용 데이터 반환"""
        # 학습 요약 로드
        summary_path = self.results_dir / "training_summary.json"
        if summary_path.exists():
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
        else:
            summary = {}
            
        # KPI 카드 데이터
        kpi_data = self._extract_kpi_data(summary)
        
        # 차트 데이터
        chart_data = self._extract_chart_data(summary)
        
        # 검색 결과 데이터
        search_data = self._extract_search_data()
        
        return {
            'kpi': kpi_data,
            'charts': chart_data,
            'search_results': search_data,
            'timestamp': datetime.now().isoformat()
        }
        
    def _extract_kpi_data(self, summary: Dict) -> Dict[str, Any]:
        """KPI 카드 데이터 추출"""
        metrics = summary.get('metrics_history', {})
        state = summary.get('training_state', {})
        
        return {
            'total_data': 2172,  # K-Fashion 데이터셋 크기
            'categories': {'레트로': 196, '로맨틱': 994, '리조트': 998},
            'current_epoch': state.get('current_epoch', 0),
            'total_epochs': state.get('total_epochs', 0),
            'stage': state.get('stage', 'N/A'),
            'top1_accuracy': metrics.get('top1_accuracy', [])[-1] if metrics.get('top1_accuracy') else 0,
            'top5_accuracy': metrics.get('top5_accuracy', [])[-1] if metrics.get('top5_accuracy') else 0,
            'mrr': metrics.get('mrr', [])[-1] if metrics.get('mrr') else 0,
            'positive_similarity': metrics.get('positive_similarity', [])[-1] if metrics.get('positive_similarity') else 0,
            'negative_similarity': metrics.get('negative_similarity', [])[-1] if metrics.get('negative_similarity') else 0,
            'l2_norm': metrics.get('l2_norm', [])[-1] if metrics.get('l2_norm') else 1.0
        }
        
    def _extract_chart_data(self, summary: Dict) -> Dict[str, Any]:
        """차트 데이터 추출"""
        metrics = summary.get('metrics_history', {})
        
        return {
            'loss_curve': {
                'epochs': metrics.get('epochs', []),
                'train_loss': metrics.get('train_loss', []),
                'val_loss': metrics.get('val_loss', [])
            },
            'accuracy_curve': {
                'epochs': metrics.get('epochs', []),
                'top1_accuracy': metrics.get('top1_accuracy', []),
                'top5_accuracy': metrics.get('top5_accuracy', [])
            },
            'similarity_curve': {
                'epochs': metrics.get('epochs', []),
                'positive_similarity': metrics.get('positive_similarity', []),
                'negative_similarity': metrics.get('negative_similarity', [])
            }
        }
        
    def _extract_search_data(self) -> Dict[str, Any]:
        """검색 결과 데이터 추출"""
        search_dir = self.results_dir / "similarity_search"
        if not search_dir.exists():
            return {}
            
        # 검색 결과 요약 로드
        summary_file = search_dir / "similarity_search_summary.md"
        if summary_file.exists():
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary_text = f.read()
        else:
            summary_text = ""
            
        # 검색 결과 이미지 목록
        image_files = list(search_dir.glob("*.png"))
        
        return {
            'summary': summary_text,
            'result_images': [str(img.name) for img in image_files],
            'total_queries': len(image_files)
        }