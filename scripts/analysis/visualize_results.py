#!/usr/bin/env python3
"""
Fashion JSON Encoder 학습 결과 시각화 스크립트

이 스크립트는 학습 결과를 다양한 차트로 시각화합니다.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_training_results(results_path: str = "results/training_results.json"):
    """학습 결과 로드"""
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_training_losses(results):
    """학습 손실 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 독립 학습 손실
    standalone = results['standalone']
    epochs_standalone = range(1, len(standalone['train_losses']) + 1)
    
    axes[0].plot(epochs_standalone, standalone['train_losses'], 'b-o', label='Train Loss', linewidth=2, markersize=6)
    axes[0].plot(epochs_standalone, standalone['val_losses'], 'r-s', label='Validation Loss', linewidth=2, markersize=6)
    axes[0].set_title('독립 JSON 인코더 학습', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0.99, 1.01)
    
    # 대조 학습 손실
    contrastive = results['contrastive']
    epochs_contrastive = range(1, len(contrastive['train_losses']) + 1)
    
    axes[1].plot(epochs_contrastive, contrastive['train_losses'], 'b-o', label='Train Loss', linewidth=2, markersize=6)
    axes[1].plot(epochs_contrastive, contrastive['val_losses'], 'r-s', label='Validation Loss', linewidth=2, markersize=6)
    axes[1].set_title('대조 학습 (Contrastive Learning)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/training_losses.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_output_statistics(results):
    """출력 통계 시각화"""
    standalone = results['standalone']
    output_stats = standalone['output_stats']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = range(1, len(output_stats) + 1)
    
    # Mean 변화
    means = [stat['mean'] for stat in output_stats]
    axes[0, 0].plot(epochs, means, 'g-o', linewidth=2, markersize=6)
    axes[0, 0].set_title('임베딩 평균값 변화', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Mean')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Std 변화
    stds = [stat['std'] for stat in output_stats]
    axes[0, 1].plot(epochs, stds, 'm-s', linewidth=2, markersize=6)
    axes[0, 1].set_title('임베딩 표준편차 변화', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Standard Deviation')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Norm 변화
    norms = [stat['norm'] for stat in output_stats]
    axes[1, 0].plot(epochs, norms, 'c-^', linewidth=2, markersize=6)
    axes[1, 0].set_title('L2 정규화 상태', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('L2 Norm')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0.999, 1.001)
    
    # Learning Rate 변화 (대조 학습)
    contrastive = results['contrastive']
    lr_epochs = range(1, len(contrastive['learning_rates']) + 1)
    axes[1, 1].plot(lr_epochs, contrastive['learning_rates'], 'orange', linewidth=2)
    axes[1, 1].set_title('학습률 스케줄링', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch (Contrastive)')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig('results/output_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_performance_metrics(results):
    """성능 메트릭 시각화"""
    contrastive = results['contrastive']
    final_metrics = contrastive['final_metrics']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 정확도 메트릭
    accuracies = ['Top-1', 'Top-5']
    acc_values = [final_metrics['top1_accuracy'] * 100, final_metrics['top5_accuracy'] * 100]
    
    bars1 = axes[0, 0].bar(accuracies, acc_values, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
    axes[0, 0].set_title('검색 정확도', fontweight='bold')
    axes[0, 0].set_ylabel('정확도 (%)')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 바 위에 값 표시
    for bar, value in zip(bars1, acc_values):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # 유사도 분포
    pos_sim = final_metrics['positive_similarity_mean']
    neg_sim = final_metrics['negative_similarity_mean']
    
    similarities = ['Positive\nSimilarity', 'Negative\nSimilarity']
    sim_values = [pos_sim, neg_sim]
    
    bars2 = axes[0, 1].bar(similarities, sim_values, color=['#95E1D3', '#F38BA8'], alpha=0.8)
    axes[0, 1].set_title('유사도 비교', fontweight='bold')
    axes[0, 1].set_ylabel('Cosine Similarity')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 바 위에 값 표시
    for bar, value in zip(bars2, sim_values):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # MRR 시각화
    mrr = final_metrics['mean_reciprocal_rank']
    axes[1, 0].bar(['Mean Reciprocal\nRank'], [mrr], color='#A8E6CF', alpha=0.8)
    axes[1, 0].set_title('평균 역순위 (MRR)', fontweight='bold')
    axes[1, 0].set_ylabel('MRR')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].text(0, mrr + 0.002, f'{mrr:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 임베딩 정규화 상태
    norms = ['Image\nEmbedding', 'JSON\nEmbedding']
    norm_values = [final_metrics['image_embedding_norm'], final_metrics['json_embedding_norm']]
    
    bars4 = axes[1, 1].bar(norms, norm_values, color=['#FFB6C1', '#87CEEB'], alpha=0.8)
    axes[1, 1].set_title('임베딩 정규화 상태', fontweight='bold')
    axes[1, 1].set_ylabel('L2 Norm')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_ylim(0.99, 1.01)
    
    # 바 위에 값 표시
    for bar, value in zip(bars4, norm_values):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_dataset_distribution():
    """데이터셋 분포 시각화"""
    categories = ['레트로', '로맨틱', '리조트']
    counts = [196, 994, 998]
    total = sum(counts)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 파이 차트
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    wedges, texts, autotexts = axes[0].pie(counts, labels=categories, colors=colors, autopct='%1.1f%%',
                                          startangle=90, textprops={'fontsize': 12})
    axes[0].set_title('카테고리별 데이터 분포', fontsize=14, fontweight='bold')
    
    # 막대 차트
    bars = axes[1].bar(categories, counts, color=colors, alpha=0.8)
    axes[1].set_title('카테고리별 아이템 수', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('아이템 수')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # 막대 위에 값 표시
    for bar, count in zip(bars, counts):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/dataset_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"총 데이터셋 크기: {total:,}개 아이템")
    print(f"학습/검증 분할: {int(total * 0.8):,}개 / {int(total * 0.2):,}개")

def create_summary_report(results):
    """요약 보고서 생성"""
    standalone = results['standalone']
    contrastive = results['contrastive']
    
    print("=" * 60)
    print("🎯 Fashion JSON Encoder 학습 결과 요약")
    print("=" * 60)
    
    print("\n📊 데이터셋 정보:")
    print(f"  • 총 아이템 수: 2,172개")
    print(f"  • 카테고리: 레트로(196), 로맨틱(994), 리조트(998)")
    print(f"  • 학습/검증 분할: 1,737개 / 435개")
    
    print("\n🏋️ 독립 학습 (5 에포크):")
    print(f"  • 최종 Train Loss: {standalone['train_losses'][-1]:.4f}")
    print(f"  • 최종 Val Loss: {standalone['val_losses'][-1]:.4f}")
    print(f"  • 임베딩 정규화: ✅ (norm={standalone['final_analysis']['norm_mean']:.6f})")
    
    print("\n🔄 대조 학습 (10 에포크):")
    print(f"  • 최고 Val Loss: {contrastive['best_val_loss']:.4f}")
    print(f"  • 최종 Train Loss: {contrastive['train_losses'][-1]:.4f}")
    print(f"  • 최종 Val Loss: {contrastive['val_losses'][-1]:.4f}")
    
    print("\n📈 최종 성능 메트릭:")
    final_metrics = contrastive['final_metrics']
    print(f"  • Top-1 정확도: {final_metrics['top1_accuracy']*100:.2f}%")
    print(f"  • Top-5 정확도: {final_metrics['top5_accuracy']*100:.2f}%")
    print(f"  • Mean Reciprocal Rank: {final_metrics['mean_reciprocal_rank']:.4f}")
    print(f"  • 평균 Positive Similarity: {final_metrics['avg_positive_similarity']:.4f}")
    print(f"  • 평균 Negative Similarity: {final_metrics['negative_similarity_mean']:.4f}")
    
    print("\n✅ 모델 상태:")
    print(f"  • 이미지 임베딩 정규화: {final_metrics['image_embedding_norm']:.3f}")
    print(f"  • JSON 임베딩 정규화: {final_metrics['json_embedding_norm']:.3f}")
    print(f"  • 임베딩 차원: {final_metrics['embedding_dim']}차원")
    
    print("\n💾 저장된 파일:")
    print(f"  • 모델 체크포인트: checkpoints/best_model.pt")
    print(f"  • 학습 결과: results/training_results.json")
    print(f"  • 시각화 결과: results/*.png")
    
    print("=" * 60)

def main():
    """메인 함수"""
    # 결과 디렉토리 생성
    Path("results").mkdir(exist_ok=True)
    
    # 학습 결과 로드
    try:
        results = load_training_results()
    except FileNotFoundError:
        print("❌ 학습 결과 파일을 찾을 수 없습니다: results/training_results.json")
        return
    
    print("🎨 Fashion JSON Encoder 학습 결과 시각화를 시작합니다...")
    
    # 요약 보고서 출력
    create_summary_report(results)
    
    # 시각화 생성
    print("\n📊 시각화 생성 중...")
    
    try:
        # 1. 학습 손실 그래프
        print("  • 학습 손실 그래프 생성...")
        plot_training_losses(results)
        
        # 2. 출력 통계 그래프
        print("  • 출력 통계 그래프 생성...")
        plot_output_statistics(results)
        
        # 3. 성능 메트릭 그래프
        print("  • 성능 메트릭 그래프 생성...")
        plot_performance_metrics(results)
        
        # 4. 데이터셋 분포 그래프
        print("  • 데이터셋 분포 그래프 생성...")
        plot_dataset_distribution()
        
        print("\n✅ 모든 시각화가 완료되었습니다!")
        print("📁 결과 파일들이 results/ 디렉토리에 저장되었습니다.")
        
    except Exception as e:
        print(f"❌ 시각화 중 오류 발생: {e}")
        print("matplotlib 설치 확인: pip install matplotlib seaborn")

if __name__ == "__main__":
    main()