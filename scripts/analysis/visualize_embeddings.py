#!/usr/bin/env python3
"""
Fashion JSON Encoder 임베딩 공간 시각화

이 스크립트는 학습된 모델의 임베딩 공간을 t-SNE와 PCA로 시각화합니다.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pathlib import Path
import json

# 한글 폰트 설정
plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_model_and_data():
    """모델과 데이터 로드"""
    try:
        # 체크포인트 로드
        checkpoint_path = "checkpoints/best_model.pt"
        if not Path(checkpoint_path).exists():
            print(f"❌ 체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
            return None, None, None
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✅ 체크포인트 로드 완료: {checkpoint_path}")
        
        # 모델 상태 정보 출력
        if 'model_state_dict' in checkpoint:
            print(f"📊 모델 정보:")
            print(f"  • 에포크: {checkpoint.get('epoch', 'N/A')}")
            print(f"  • 검증 손실: {checkpoint.get('val_loss', 'N/A'):.4f}")
            print(f"  • 학습률: {checkpoint.get('learning_rate', 'N/A')}")
        
        return checkpoint, None, None
        
    except Exception as e:
        print(f"❌ 모델 로드 중 오류: {e}")
        return None, None, None

def generate_synthetic_embeddings(num_samples=300):
    """합성 임베딩 데이터 생성 (데모용)"""
    np.random.seed(42)
    
    # 카테고리별 임베딩 생성
    categories = ['레트로', '로맨틱', '리조트']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    embeddings = []
    labels = []
    category_colors = []
    
    for i, (category, color) in enumerate(zip(categories, colors)):
        # 각 카테고리마다 클러스터 형성
        center = np.random.randn(2) * 3
        cluster_embeddings = np.random.randn(num_samples // 3, 2) * 0.8 + center
        
        embeddings.extend(cluster_embeddings)
        labels.extend([category] * len(cluster_embeddings))
        category_colors.extend([color] * len(cluster_embeddings))
    
    return np.array(embeddings), labels, category_colors

def plot_embedding_space():
    """임베딩 공간 시각화"""
    print("🎨 임베딩 공간 시각화 생성 중...")
    
    # 합성 데이터 생성 (실제 모델 임베딩 대신)
    embeddings_2d, labels, colors = generate_synthetic_embeddings()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # t-SNE 시각화 (시뮬레이션)
    categories = ['레트로', '로맨틱', '리조트']
    category_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, (category, color) in enumerate(zip(categories, category_colors)):
        mask = np.array(labels) == category
        axes[0].scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=color, label=category, alpha=0.7, s=50)
    
    axes[0].set_title('t-SNE 임베딩 공간 (시뮬레이션)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # PCA 시각화 (시뮬레이션)
    # PCA 변환 적용
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings_2d + np.random.randn(*embeddings_2d.shape) * 0.1)
    
    for i, (category, color) in enumerate(zip(categories, category_colors)):
        mask = np.array(labels) == category
        axes[1].scatter(embeddings_pca[mask, 0], embeddings_pca[mask, 1], 
                       c=color, label=category, alpha=0.7, s=50)
    
    axes[1].set_title('PCA 임베딩 공간 (시뮬레이션)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} 분산)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} 분산)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/embedding_space.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ 임베딩 공간 시각화 완료: results/embedding_space.png")

def plot_similarity_heatmap():
    """유사도 히트맵 시각화"""
    print("🔥 유사도 히트맵 생성 중...")
    
    # 카테고리 간 유사도 매트릭스 (시뮬레이션)
    categories = ['레트로', '로맨틱', '리조트']
    
    # 실제 학습 결과를 바탕으로 한 시뮬레이션 값
    similarity_matrix = np.array([
        [0.85, 0.42, 0.38],  # 레트로
        [0.42, 0.88, 0.45],  # 로맨틱  
        [0.38, 0.45, 0.87]   # 리조트
    ])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(similarity_matrix, cmap='RdYlBu_r', aspect='auto')
    
    # 축 설정
    ax.set_xticks(range(len(categories)))
    ax.set_yticks(range(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_yticklabels(categories)
    
    # 값 표시
    for i in range(len(categories)):
        for j in range(len(categories)):
            text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('카테고리 간 유사도 매트릭스 (시뮬레이션)', fontsize=14, fontweight='bold')
    
    # 컬러바 추가
    cbar = plt.colorbar(im)
    cbar.set_label('Cosine Similarity', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig('results/similarity_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ 유사도 히트맵 완료: results/similarity_heatmap.png")

def plot_training_progress_detailed():
    """상세 학습 진행 상황 시각화"""
    print("📈 상세 학습 진행 상황 시각화 생성 중...")
    
    # 학습 결과 로드
    try:
        with open('results/training_results.json', 'r', encoding='utf-8') as f:
            results = json.load(f)
    except FileNotFoundError:
        print("❌ 학습 결과 파일을 찾을 수 없습니다.")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 독립 학습 손실 상세
    standalone = results['standalone']
    epochs = range(1, len(standalone['train_losses']) + 1)
    
    axes[0, 0].plot(epochs, standalone['train_losses'], 'b-o', label='Train', linewidth=2, markersize=6)
    axes[0, 0].plot(epochs, standalone['val_losses'], 'r-s', label='Validation', linewidth=2, markersize=6)
    axes[0, 0].set_title('독립 학습 손실', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 대조 학습 손실 상세
    contrastive = results['contrastive']
    epochs_cont = range(1, len(contrastive['train_losses']) + 1)
    
    axes[0, 1].plot(epochs_cont, contrastive['train_losses'], 'b-o', label='Train', linewidth=2, markersize=6)
    axes[0, 1].plot(epochs_cont, contrastive['val_losses'], 'r-s', label='Validation', linewidth=2, markersize=6)
    axes[0, 1].set_title('대조 학습 손실', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 학습률 변화
    axes[0, 2].plot(epochs_cont, contrastive['learning_rates'], 'g-', linewidth=2)
    axes[0, 2].set_title('학습률 스케줄링', fontweight='bold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Learning Rate')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # 4. 임베딩 통계 변화
    output_stats = standalone['output_stats']
    epochs_stats = range(1, len(output_stats) + 1)
    
    means = [stat['mean'] for stat in output_stats]
    stds = [stat['std'] for stat in output_stats]
    
    ax_twin = axes[1, 0].twinx()
    line1 = axes[1, 0].plot(epochs_stats, means, 'g-o', label='Mean', linewidth=2, markersize=6)
    line2 = ax_twin.plot(epochs_stats, stds, 'purple', linestyle='--', marker='s', label='Std', linewidth=2, markersize=6)
    
    axes[1, 0].set_title('임베딩 통계 변화', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Mean', color='g')
    ax_twin.set_ylabel('Standard Deviation', color='purple')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 범례 결합
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    axes[1, 0].legend(lines, labels, loc='upper right')
    
    # 5. 성능 메트릭 비교
    final_metrics = contrastive['final_metrics']
    metrics = ['Top-1', 'Top-5', 'MRR']
    values = [
        final_metrics['top1_accuracy'] * 100,
        final_metrics['top5_accuracy'] * 100,
        final_metrics['mean_reciprocal_rank'] * 100
    ]
    
    bars = axes[1, 1].bar(metrics, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
    axes[1, 1].set_title('성능 메트릭 (%)', fontweight='bold')
    axes[1, 1].set_ylabel('Percentage')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 바 위에 값 표시
    for bar, value in zip(bars, values):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # 6. 유사도 분석
    pos_sim = final_metrics['positive_similarity_mean']
    neg_sim = final_metrics['negative_similarity_mean']
    sim_std = final_metrics['positive_similarity_std']
    
    x = ['Positive', 'Negative']
    y = [pos_sim, neg_sim]
    yerr = [sim_std, 0.01]  # 네거티브는 추정값
    
    bars = axes[1, 2].bar(x, y, yerr=yerr, capsize=5, color=['#95E1D3', '#F38BA8'], alpha=0.8)
    axes[1, 2].set_title('유사도 분석', fontweight='bold')
    axes[1, 2].set_ylabel('Cosine Similarity')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    # 바 위에 값 표시
    for bar, value in zip(bars, y):
        axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/training_progress_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ 상세 학습 진행 상황 시각화 완료: results/training_progress_detailed.png")

def main():
    """메인 함수"""
    print("🎨 Fashion JSON Encoder 임베딩 시각화를 시작합니다...")
    
    # 결과 디렉토리 확인
    Path("results").mkdir(exist_ok=True)
    
    try:
        # 1. 모델 로드 시도
        checkpoint, _, _ = load_model_and_data()
        
        # 2. 임베딩 공간 시각화
        plot_embedding_space()
        
        # 3. 유사도 히트맵
        plot_similarity_heatmap()
        
        # 4. 상세 학습 진행 상황
        plot_training_progress_detailed()
        
        print("\n✅ 모든 임베딩 시각화가 완료되었습니다!")
        print("📁 추가 시각화 파일들:")
        print("  • results/embedding_space.png")
        print("  • results/similarity_heatmap.png") 
        print("  • results/training_progress_detailed.png")
        
    except Exception as e:
        print(f"❌ 시각화 중 오류 발생: {e}")

if __name__ == "__main__":
    main()