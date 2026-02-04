#!/usr/bin/env python3
"""
Optuna 기반 하이퍼파라미터 튜닝 (개선된 버전)

목표:
1순위: maximize (positive_similarity - negative_similarity)
2순위: MRR, Category-aware Precision@5
"""

import optuna
import torch
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from collections import defaultdict

from data.fashion_dataset import FashionDataModule
from training.trainer import create_trainer_from_data_module
from utils.config import TrainingConfig


class HyperparameterTuner:
    """Optuna 기반 하이퍼파라미터 튜너 (개선된 목적 함수)"""
    
    def __init__(self, 
                 dataset_path: str = "C:/sample/라벨링데이터",
                 n_trials: int = 20,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.dataset_path = dataset_path
        self.n_trials = n_trials
        self.device = device
        
        # 데이터 모듈 미리 준비
        print("📂 데이터 모듈 준비 중...")
        self.data_module = FashionDataModule(
            dataset_path=dataset_path,
            target_categories=['레트로', '로맨틱', '리조트'],
            batch_size=64  # 기본값, 나중에 튜닝됨
        )
        self.data_module.setup()
        print(f"✅ 데이터 준비 완료: {len(self.data_module.train_dataset)} 학습 샘플")
        
    def compute_category_aware_metrics(self, trainer, val_loader) -> Dict[str, float]:
        """카테고리별 정밀도 및 고급 메트릭 계산"""
        trainer.contrastive_learner.eval()
        
        all_image_embeddings = []
        all_json_embeddings = []
        all_categories = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = trainer._move_batch_to_device(batch)
                json_batch = trainer._convert_batch_to_dict(batch)
                
                # 임베딩 계산
                embeddings = trainer.contrastive_learner.get_embeddings(batch.images, json_batch)
                
                all_image_embeddings.append(embeddings['image_embeddings'].cpu())
                all_json_embeddings.append(embeddings['json_embeddings'].cpu())
                all_categories.append(batch.category_ids.cpu())
        
        # 전체 임베딩 연결
        image_embeddings = torch.cat(all_image_embeddings, dim=0)
        json_embeddings = torch.cat(all_json_embeddings, dim=0)
        categories = torch.cat(all_categories, dim=0)
        
        # 유사도 행렬 계산 (정사각형 행렬)
        similarity_matrix = torch.matmul(image_embeddings, json_embeddings.T)
        
        # 1순위 목표: positive/negative similarity gap
        batch_size = similarity_matrix.size(0)
        positive_similarities = similarity_matrix.diag()
        
        # negative similarities (대각선 제외)
        mask = torch.eye(batch_size, dtype=torch.bool)
        negative_similarities = similarity_matrix[~mask]
        
        pos_sim_mean = positive_similarities.mean().item()
        neg_sim_mean = negative_similarities.mean().item()
        similarity_gap = pos_sim_mean - neg_sim_mean
        
        # 2순위 목표: Category-aware Precision@5
        category_precision_5 = self._compute_category_precision_at_k(
            similarity_matrix, categories, k=5
        )
        
        # MRR 계산
        ranks = (similarity_matrix.argsort(dim=1, descending=True) == 
                torch.arange(similarity_matrix.size(0)).unsqueeze(1)).nonzero()[:, 1] + 1
        mrr = (1.0 / ranks.float()).mean().item()
        
        # Top-5 정확도
        top5_indices = similarity_matrix.topk(k=min(5, similarity_matrix.size(0)), dim=1)[1]
        top5_correct = (top5_indices == torch.arange(similarity_matrix.size(0)).unsqueeze(1)).any(dim=1).float().mean().item()
        
        return {
            'similarity_gap': similarity_gap,
            'positive_similarity': pos_sim_mean,
            'negative_similarity': neg_sim_mean,
            'category_precision_5': category_precision_5,
            'mrr': mrr,
            'top5_accuracy': top5_correct
        }
    
    def _compute_category_precision_at_k(self, similarity_matrix: torch.Tensor, 
                                       categories: torch.Tensor, k: int = 5) -> float:
        """카테고리별 Precision@K 계산"""
        batch_size = similarity_matrix.size(0)
        category_precisions = []
        
        # 각 카테고리별로 계산
        unique_categories = categories.unique()
        
        for category in unique_categories:
            # 해당 카테고리의 인덱스들
            category_mask = (categories == category)
            category_indices = category_mask.nonzero().squeeze(-1)
            
            if len(category_indices) < 2:  # 카테고리에 샘플이 1개 이하면 건너뛰기
                continue
            
            category_similarities = similarity_matrix[category_indices][:, category_indices]
            
            # Top-K 검색 결과에서 같은 카테고리 비율 계산
            topk_indices = category_similarities.topk(k=min(k+1, category_similarities.size(1)), dim=1)[1]
            
            # 자기 자신 제외하고 계산
            precision_scores = []
            for i, topk in enumerate(topk_indices):
                # 자기 자신(첫 번째) 제외
                relevant_topk = topk[1:k+1] if len(topk) > 1 else topk[1:]
                if len(relevant_topk) > 0:
                    precision = len(relevant_topk) / min(k, len(relevant_topk))
                    precision_scores.append(precision)
            
            if precision_scores:
                category_precisions.append(np.mean(precision_scores))
        
        return np.mean(category_precisions) if category_precisions else 0.0
        
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna 목적 함수 - 1순위: similarity gap 최대화"""
        
        # 하이퍼파라미터 제안 (사용자 추천 범위)
        config = TrainingConfig(
            # 핵심 하이퍼파라미터
            learning_rate=trial.suggest_categorical('learning_rate', [1e-4, 3e-4, 5e-4]),
            temperature=trial.suggest_categorical('temperature', [0.03, 0.05, 0.07, 0.1]),
            batch_size=trial.suggest_categorical('batch_size', [64, 96, 128]),
            
            # 모델 구조
            embedding_dim=trial.suggest_categorical('embedding_dim', [128, 256]),
            hidden_dim=trial.suggest_categorical('hidden_dim', [256, 512, 768]),
            dropout_rate=trial.suggest_float('dropout_rate', 0.1, 0.3),
            
            # 정규화
            weight_decay=trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
            
            # 고정값
            output_dim=512,  # 고정
            max_epochs=10,   # 튜닝용 짧은 에포크
        )
        
        print(f"\n🔍 Trial {trial.number + 1}/{self.n_trials}")
        print(f"   학습률: {config.learning_rate:.6f}")
        print(f"   온도: {config.temperature:.3f}")
        print(f"   배치 사이즈: {config.batch_size}")
        print(f"   임베딩 차원: {config.embedding_dim}")
        print(f"   은닉층 차원: {config.hidden_dim}")
        
        try:
            # 데이터 로더 업데이트
            self.data_module.batch_size = config.batch_size
            self.data_module._train_dataloader = None
            self.data_module._val_dataloader = None
            
            train_loader = self.data_module.train_dataloader()
            val_loader = self.data_module.val_dataloader()
            
            # 트레이너 생성
            trainer = create_trainer_from_data_module(
                data_module=self.data_module,
                config=config,
                device=self.device,
                checkpoint_dir=f'tuning_checkpoints/trial_{trial.number}',
                log_dir=f'tuning_logs/trial_{trial.number}'
            )
            
            # 학습 실행
            print(f"   🚀 학습 시작...")
            start_time = time.time()
            
            results = trainer.train_contrastive_learning(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=config.max_epochs
            )
            
            elapsed = time.time() - start_time
            
            # 고급 메트릭 계산
            advanced_metrics = self.compute_category_aware_metrics(trainer, val_loader)
            
            # 1순위 목적 함수: positive/negative similarity gap
            similarity_gap = advanced_metrics['similarity_gap']
            
            # 2순위 메트릭들
            category_precision_5 = advanced_metrics['category_precision_5']
            mrr = advanced_metrics['mrr']
            top5_accuracy = advanced_metrics['top5_accuracy']
            
            print(f"   ⏱️ 학습 완료: {elapsed:.1f}초")
            print(f"   📊 결과:")
            print(f"      🎯 Similarity Gap: {similarity_gap:.4f}")
            print(f"      📈 Category P@5: {category_precision_5:.4f}")
            print(f"      🔍 MRR: {mrr:.4f}")
            print(f"      ✅ Top-5 정확도: {top5_accuracy:.4f}")
            print(f"      ➕ Positive Sim: {advanced_metrics['positive_similarity']:.4f}")
            print(f"      ➖ Negative Sim: {advanced_metrics['negative_similarity']:.4f}")
            
            # 복합 목적 함수 (가중 평균)
            # 1순위: similarity gap (가중치 0.7)
            # 2순위: category precision@5 + MRR (가중치 0.3)
            objective_value = (
                0.7 * similarity_gap + 
                0.2 * category_precision_5 + 
                0.1 * mrr
            )
            
            print(f"      🏆 목적함수 값: {objective_value:.4f}")
            
            # 중간 결과 보고 (pruning용)
            trial.report(objective_value, config.max_epochs)
            
            # 리소스 정리
            trainer.close()
            
            return objective_value
            
        except Exception as e:
            print(f"   ❌ Trial 실패: {e}")
            import traceback
            traceback.print_exc()
            return -1.0  # 실패한 경우 최소값 반환
    
    def run_tuning(self) -> Dict[str, Any]:
        """하이퍼파라미터 튜닝 실행"""
        print(f"\n🎯 Optuna 하이퍼파라미터 튜닝 시작")
        print(f"   시행 횟수: {self.n_trials}")
        print(f"   디바이스: {self.device}")
        print(f"   🥇 1순위 목표: maximize (positive_similarity - negative_similarity)")
        print(f"   🥈 2순위 목표: Category-aware Precision@5, MRR")
        
        # Optuna 스터디 생성
        study = optuna.create_study(
            direction='maximize',  # 복합 목적 함수 최대화
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=3,
                n_warmup_steps=3
            )
        )
        
        # 튜닝 실행
        start_time = time.time()
        study.optimize(self.objective, n_trials=self.n_trials)
        total_time = time.time() - start_time
        
        # 결과 분석
        best_trial = study.best_trial
        best_params = best_trial.params
        best_value = best_trial.value
        
        print(f"\n🏆 튜닝 완료!")
        print(f"   총 소요 시간: {total_time/60:.1f}분")
        print(f"   최고 목적함수 값: {best_value:.4f}")
        print(f"\n📋 최적 하이퍼파라미터:")
        for key, value in best_params.items():
            print(f"   {key}: {value}")
        
        # 상위 3개 trial 분석
        print(f"\n🥇 상위 3개 Trial 결과:")
        sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value else -999, reverse=True)
        for i, trial in enumerate(sorted_trials[:3]):
            if trial.value is not None:
                print(f"   {i+1}위: Trial {trial.number}, 점수: {trial.value:.4f}")
                print(f"        lr: {trial.params.get('learning_rate', 'N/A')}, "
                      f"temp: {trial.params.get('temperature', 'N/A')}, "
                      f"batch: {trial.params.get('batch_size', 'N/A')}")
        
        # 결과 저장
        results = {
            'best_params': best_params,
            'best_value': best_value,
            'n_trials': self.n_trials,
            'total_time': total_time,
            'objective_function': 'similarity_gap + category_precision@5 + mrr',
            'top_trials': [
                {
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': trial.state.name
                }
                for trial in sorted_trials[:5] if trial.value is not None
            ],
            'all_trials': [
                {
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': trial.state.name
                }
                for trial in study.trials
            ]
        }
        
        # 결과 파일 저장
        results_dir = Path("tuning_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        results_file = results_dir / f"optuna_similarity_gap_tuning_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"📄 결과 저장: {results_file}")
        
        return results


def main():
    """메인 함수"""
    print("Fashion JSON Encoder 하이퍼파라미터 튜닝 (개선된 버전)")
    print("=" * 70)
    print("🎯 목표:")
    print("   1순위: maximize (positive_similarity - negative_similarity)")
    print("   2순위: Category-aware Precision@5, MRR")
    print("   교수님 요구사항: 동일 카테고리 내 정렬 정확도 ≥ 0.9")
    
    # 튜너 생성 및 실행
    tuner = HyperparameterTuner(
        dataset_path="C:/sample/라벨링데이터",
        n_trials=12,  # 시작은 12회로 (빠른 테스트)
        device='cpu'  # CPU에서 안정적으로
    )
    
    try:
        results = tuner.run_tuning()
        
        print(f"\n🎉 튜닝 성공!")
        print(f"   최적 설정:")
        best_params = results['best_params']
        print(f"     학습률: {best_params.get('learning_rate', 'N/A')}")
        print(f"     온도: {best_params.get('temperature', 'N/A')}")
        print(f"     배치 크기: {best_params.get('batch_size', 'N/A')}")
        print(f"     임베딩 차원: {best_params.get('embedding_dim', 'N/A')}")
        print(f"     은닉층 차원: {best_params.get('hidden_dim', 'N/A')}")
        
        print(f"\n📈 다음 단계:")
        print(f"   1. 최적 설정으로 본격 학습 (50-100 에포크)")
        print(f"   2. Category-aware 메트릭 모니터링")
        print(f"   3. Positive/Negative similarity gap 확인")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 튜닝 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()