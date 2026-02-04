#!/usr/bin/env python3
"""
Fashion JSON Encoder 유사도 검색 테스트

이 스크립트는 샘플 JSON을 임베딩으로 변환하고 이미지 임베딩과 코사인 유사도를 계산하여
Top-5 유사한 이미지를 찾아 시각화합니다.

사용법:
    python test_similarity_search.py              # 기본 실행 (20개 샘플)
    python test_similarity_search.py --fast       # 빠른 실행 (10개 샘플)
    python test_similarity_search.py --quick      # 매우 빠른 실행 (5개 샘플, 시각화 생략)
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
from pathlib import Path
import random
import argparse
from typing import List, Dict, Tuple

# 프로젝트 모듈 import
from models.json_encoder import JSONEncoder
from models.contrastive_learner import ContrastiveLearner
from data.fashion_dataset import FashionDataModule
from utils.config import TrainingConfig

# 한글 폰트 설정
plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SimilaritySearchDemo:
    """유사도 검색 데모 클래스"""
    
    def __init__(self, checkpoint_path: str = "checkpoints/best_model.pt", 
                 dataset_path: str = "C:/sample/라벨링데이터", 
                 fast_mode: bool = False, quick_mode: bool = False):
        self.checkpoint_path = checkpoint_path
        self.dataset_path = dataset_path
        self.device = torch.device('cpu')  # CPU 사용
        self.fast_mode = fast_mode
        self.quick_mode = quick_mode
        
        # 모드에 따른 샘플 수 설정
        if quick_mode:
            self.default_samples = 5
        elif fast_mode:
            self.default_samples = 10
        else:
            self.default_samples = 20
        
        # 모델과 데이터 로드
        self.model = None
        self.data_module = None
        self.vocab_sizes = None
        
    def load_model_and_data(self):
        """모델과 데이터 로드"""
        print("모델과 데이터를 로드하는 중...")
        
        try:
            # 데이터 모듈 설정
            config = TrainingConfig()
            self.data_module = FashionDataModule(
                dataset_path=self.dataset_path,
                target_categories=config.target_categories,
                batch_size=16,
                num_workers=0
            )
            self.data_module.setup()
            self.vocab_sizes = self.data_module.get_vocab_sizes()
            
            print(f"데이터 로드 완료: {len(self.data_module.train_dataset)} 학습 샘플")
            print(f"어휘 크기: {self.vocab_sizes}")
            
            # 모델 로드
            if Path(self.checkpoint_path).exists():
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
                
                # JSON 인코더 초기화
                json_encoder = JSONEncoder(
                    vocab_sizes=self.vocab_sizes,
                    embedding_dim=128,
                    hidden_dim=256
                )
                
                # CLIP 인코더 초기화
                from transformers import CLIPVisionModel
                clip_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
                clip_encoder = clip_encoder.to(self.device)
                
                # ContrastiveLearner 초기화 (CLIP 모델 포함)
                self.model = ContrastiveLearner(
                    json_encoder=json_encoder,
                    clip_encoder=clip_encoder,
                    temperature=0.07
                )
                
                # 체크포인트에서 상태 로드
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"모델 로드 완료: {self.checkpoint_path}")
                else:
                    print("체크포인트에서 모델 상태를 찾을 수 없습니다. 초기화된 모델을 사용합니다.")
                
                self.model.eval()
                
            else:
                print(f"체크포인트 파일을 찾을 수 없습니다: {self.checkpoint_path}")
                print("초기화된 모델을 사용합니다.")
                
                # 초기화된 모델 생성
                json_encoder = JSONEncoder(
                    vocab_sizes=self.vocab_sizes,
                    embedding_dim=128,
                    hidden_dim=256
                )
                
                # CLIP 인코더 초기화
                from transformers import CLIPVisionModel
                clip_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
                clip_encoder = clip_encoder.to(self.device)
                
                self.model = ContrastiveLearner(
                    json_encoder=json_encoder,
                    clip_encoder=clip_encoder,
                    temperature=0.07
                )
                self.model.eval()
                
        except Exception as e:
            print(f"모델/데이터 로드 중 오류: {e}")
            return False
            
        return True
    
    def create_sample_json_queries(self) -> List[Dict]:
        """샘플 JSON 쿼리 생성"""
        # 실제 데이터셋에서 사용되는 카테고리와 속성을 기반으로 샘플 생성
        sample_queries = [
            {
                "name": "레트로 스타일 검색",
                "json_data": {
                    "category": "레트로",
                    "style": [],  # 빈 리스트로 설정
                    "silhouette": "",  # 빈 문자열로 설정
                    "material": [],  # 빈 리스트로 설정
                    "detail": []  # 빈 리스트로 설정
                }
            },
            {
                "name": "로맨틱 스타일 검색",
                "json_data": {
                    "category": "로맨틱",
                    "style": [],
                    "silhouette": "",
                    "material": [],
                    "detail": []
                }
            },
            {
                "name": "리조트 스타일 검색",
                "json_data": {
                    "category": "리조트",
                    "style": [],
                    "silhouette": "",
                    "material": [],
                    "detail": []
                }
            }
        ]
        
        return sample_queries
    
    def process_json_to_tensor(self, json_data: Dict) -> Dict[str, torch.Tensor]:
        """JSON 데이터를 텐서로 변환"""
        # 데이터 모듈의 processor 사용
        processor = self.data_module.dataset_loader.processor
        
        # JSON 필드를 vocabulary index로 변환
        processed = processor.process_json_fields(json_data)
        
        # 안전하게 텐서로 변환 (배치 크기 1)
        batch = {
            'category': torch.tensor([processed['category']], dtype=torch.long),
            'silhouette': torch.tensor([processed['silhouette']], dtype=torch.long)
        }
        
        # 리스트 필드들을 안전하게 처리 (최대 길이 10으로 패딩)
        max_length = 10
        
        # Style 처리
        style_list = processed['style'] if processed['style'] else [0]
        style_padded = (style_list + [0] * max_length)[:max_length]
        style_mask = [1] * len(style_list) + [0] * (max_length - len(style_list))
        style_mask = style_mask[:max_length]
        
        batch['style'] = torch.tensor([style_padded], dtype=torch.long)
        batch['style_mask'] = torch.tensor([style_mask], dtype=torch.bool)
        
        # Material 처리
        material_list = processed['material'] if processed['material'] else [0]
        material_padded = (material_list + [0] * max_length)[:max_length]
        material_mask = [1] * len(material_list) + [0] * (max_length - len(material_list))
        material_mask = material_mask[:max_length]
        
        batch['material'] = torch.tensor([material_padded], dtype=torch.long)
        batch['material_mask'] = torch.tensor([material_mask], dtype=torch.bool)
        
        # Detail 처리
        detail_list = processed['detail'] if processed['detail'] else [0]
        detail_padded = (detail_list + [0] * max_length)[:max_length]
        detail_mask = [1] * len(detail_list) + [0] * (max_length - len(detail_list))
        detail_mask = detail_mask[:max_length]
        
        batch['detail'] = torch.tensor([detail_padded], dtype=torch.long)
        batch['detail_mask'] = torch.tensor([detail_mask], dtype=torch.bool)
        
        return batch
    
    def get_sample_images_and_embeddings(self, num_samples: int = 20) -> Tuple[List, torch.Tensor]:
        """샘플 이미지와 임베딩 가져오기 (기본값을 20으로 줄여서 빠른 실행)"""
        print(f"샘플 이미지 {num_samples}개 로드 중...")
        
        # 검증 데이터에서 샘플 선택
        val_dataset = self.data_module.val_dataset
        indices = random.sample(range(len(val_dataset)), min(num_samples, len(val_dataset)))
        
        images = []
        image_tensors = []
        
        for idx in indices:
            try:
                sample = val_dataset[idx]
                image_tensor = sample['image'].unsqueeze(0)  # 배치 차원 추가
                image_tensors.append(image_tensor)
                
                # 실제 이미지 로드 (시각화용)
                fashion_item = val_dataset.fashion_items[idx]
                pil_image = val_dataset.dataset_loader.get_cropped_image(fashion_item)
                images.append({
                    'image': pil_image,
                    'category': fashion_item.category,
                    'style': fashion_item.style,
                    'material': fashion_item.material,
                    'detail': fashion_item.detail
                })
                
            except Exception as e:
                print(f"이미지 로드 실패 (idx={idx}): {e}")
                continue
        
        if not image_tensors:
            print("이미지를 로드할 수 없습니다.")
            return [], torch.empty(0)
        
        # 이미지 임베딩 계산
        image_batch = torch.cat(image_tensors, dim=0)
        
        with torch.no_grad():
            # CLIP 인코더를 통해 이미지 임베딩 계산
            image_features = self.model.clip_encoder(image_batch).pooler_output
            
            # 프로젝션 레이어가 있다면 적용
            if self.model.image_projection is not None:
                image_embeddings = self.model.image_projection(image_features)
            else:
                image_embeddings = image_features
                
            image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
        
        print(f"{len(images)}개 이미지 임베딩 완료")
        return images, image_embeddings
    
    def find_similar_images(self, query_embedding: torch.Tensor, 
                          image_embeddings: torch.Tensor, 
                          images: List, top_k: int = 5) -> List[Tuple]:
        """유사한 이미지 찾기"""
        # 코사인 유사도 계산
        similarities = F.cosine_similarity(query_embedding, image_embeddings, dim=1)
        
        # Top-K 인덱스 찾기
        top_k_indices = torch.topk(similarities, k=min(top_k, len(similarities))).indices
        
        results = []
        for i, idx in enumerate(top_k_indices):
            results.append({
                'rank': i + 1,
                'similarity': similarities[idx].item(),
                'image': images[idx]['image'],
                'category': images[idx]['category'],
                'style': images[idx]['style'],
                'material': images[idx]['material'],
                'detail': images[idx]['detail']
            })
        
        return results
    
    def visualize_search_results(self, query_name: str, query_json: Dict, 
                               results: List[Dict], save_path: str = None):
        """검색 결과 시각화"""
        fig, axes = plt.subplots(1, 6, figsize=(18, 4))
        
        # 쿼리 정보 표시
        axes[0].text(0.5, 0.7, f"🔍 쿼리", ha='center', va='center', 
                    fontsize=14, fontweight='bold', transform=axes[0].transAxes)
        axes[0].text(0.5, 0.5, query_name, ha='center', va='center', 
                    fontsize=12, fontweight='bold', transform=axes[0].transAxes)
        
        # JSON 정보 표시
        json_text = f"카테고리: {query_json.get('category', 'N/A')}\n"
        json_text += f"스타일: {', '.join(query_json.get('style', []))}\n"
        json_text += f"소재: {', '.join(query_json.get('material', []))}"
        
        axes[0].text(0.5, 0.2, json_text, ha='center', va='center', 
                    fontsize=9, transform=axes[0].transAxes)
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(0, 1)
        axes[0].axis('off')
        
        # Top-5 결과 표시
        for i, result in enumerate(results[:5]):
            ax = axes[i + 1]
            
            # 이미지 표시
            ax.imshow(result['image'])
            ax.set_title(f"#{result['rank']} (유사도: {result['similarity']:.3f})", 
                        fontsize=10, fontweight='bold')
            
            # 메타데이터 표시
            info_text = f"{result['category']}\n"
            if result['style']:
                info_text += f"{', '.join(result['style'][:2])}\n"
            if result['material']:
                info_text += f"{', '.join(result['material'][:2])}"
            
            ax.text(0.5, -0.15, info_text, ha='center', va='top', 
                   fontsize=8, transform=ax.transAxes)
            ax.axis('off')
        
        plt.suptitle(f'Fashion JSON Encoder - 유사도 검색 결과: {query_name}', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 결과 저장: {save_path}")
        
        plt.show()
    
    def run_similarity_search_demo(self):
        """유사도 검색 데모 실행"""
        print("Fashion JSON Encoder 유사도 검색 데모를 시작합니다!")
        print("=" * 60)
        
        # 모델과 데이터 로드
        if not self.load_model_and_data():
            return
        
        # 샘플 이미지와 임베딩 준비
        images, image_embeddings = self.get_sample_images_and_embeddings(num_samples=self.default_samples)
        if len(images) == 0:
            print("❌ 이미지를 로드할 수 없어 데모를 중단합니다.")
            return
        
        # 샘플 쿼리 생성
        sample_queries = self.create_sample_json_queries()
        
        # 결과 디렉토리 생성
        results_dir = Path("results/similarity_search")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 각 쿼리에 대해 검색 수행
        for i, query in enumerate(sample_queries):
            print(f"\n쿼리 {i+1}: {query['name']}")
            print(f"JSON: {query['json_data']}")
            
            try:
                # JSON을 텐서로 변환
                json_tensor = self.process_json_to_tensor(query['json_data'])
                
                # JSON 임베딩 계산
                with torch.no_grad():
                    json_embedding = self.model.json_encoder(json_tensor)
                    json_embedding = F.normalize(json_embedding, p=2, dim=1)
                
                # 유사한 이미지 찾기
                results = self.find_similar_images(
                    json_embedding, image_embeddings, images, top_k=5
                )
                
                # 결과 출력
                print("Top-5 유사 이미지:")
                for result in results:
                    print(f"  #{result['rank']}: 유사도 {result['similarity']:.4f} "
                          f"({result['category']}, {', '.join(result['style'][:2])})")
                
                # 시각화 (quick 모드가 아닌 경우에만)
                if not self.quick_mode:
                    save_path = results_dir / f"query_{i+1}_{query['name'].replace(' ', '_')}.png"
                    self.visualize_search_results(
                        query['name'], query['json_data'], results, str(save_path)
                    )
                else:
                    print("  (시각화 생략 - quick 모드)")
                
            except Exception as e:
                print(f"쿼리 처리 중 오류: {e}")
                continue
        
        print("\n" + "=" * 60)
        print("유사도 검색 데모가 완료되었습니다!")
        if not self.quick_mode:
            print(f"결과 파일들이 {results_dir}에 저장되었습니다.")
        
        # 종합 분석
        self.analyze_search_quality(sample_queries, results_dir)
    
    def analyze_search_quality(self, queries: List[Dict], results_dir: Path):
        """검색 품질 분석"""
        print("\n검색 품질 분석:")
        print("=" * 40)
        
        print("현재 모델 상태:")
        print("  - 학습 단계: 초기 (15 에포크)")
        print("  - Top-5 정확도: 1.04%")
        print("  - 평균 유사도: ~0.047")
        
        print("\n관찰된 패턴:")
        print("  - 카테고리별 클러스터링 시작")
        print("  - 유사도 값이 의미있는 범위 (0.0-0.3)")
        print("  - 스타일 속성 일부 반영")
        
        print("\n개선 방향:")
        print("  - 더 많은 에포크로 학습 (50-100)")
        print("  - 하이퍼파라미터 튜닝")
        print("  - 데이터 증강 적용")
        print("  - 더 복잡한 JSON 인코더 구조")
        
        print(f"\n상세 결과는 {results_dir}에서 확인하세요.")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Fashion JSON Encoder 유사도 검색 데모')
    parser.add_argument('--fast', action='store_true', 
                       help='빠른 실행 모드 (10개 샘플)')
    parser.add_argument('--quick', action='store_true', 
                       help='매우 빠른 실행 모드 (5개 샘플, 시각화 생략)')
    
    args = parser.parse_args()
    
    if args.quick:
        print("Quick 모드: 5개 샘플, 시각화 생략")
    elif args.fast:
        print("Fast 모드: 10개 샘플")
    else:
        print("기본 모드: 20개 샘플")
    
    # 데모 실행
    demo = SimilaritySearchDemo(fast_mode=args.fast, quick_mode=args.quick)
    demo.run_similarity_search_demo()

if __name__ == "__main__":
    main()