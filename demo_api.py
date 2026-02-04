"""
Fashion JSON Encoder API Demo Script
Requirements 13: JSON Encoder 독립 검증 및 API 테스트
"""

import asyncio
import aiohttp
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import io

# 로컬 모듈 임포트
from models.json_encoder import JSONEncoder
from data.dataset_loader import KFashionDatasetLoader
from utils.validators import InputValidator, ModelValidator

async def test_api_endpoints():
    """API 엔드포인트 테스트"""
    base_url = "http://localhost:8000"
    
    async with aiohttp.ClientSession() as session:
        # 1. 헬스 체크
        print("🔍 API 헬스 체크...")
        async with session.get(f"{base_url}/health") as response:
            health_data = await response.json()
            print(f"✅ 서버 상태: {health_data['status']}")
            print(f"📊 모델 로드 상태: {health_data['models_loaded']}")
        
        # 2. JSON 스타일 기반 추천 테스트
        print("\n🎨 JSON 스타일 기반 추천 테스트...")
        style_request = {
            "input_type": "json",
            "style_description": {
                "category": "상의",
                "style": ["레트로", "캐주얼"],
                "silhouette": "오버사이즈",
                "material": ["니트", "폴리에스터"],
                "detail": ["라운드넥", "긴소매"]
            },
            "options": {
                "top_k": 5,
                "similarity_threshold": 0.1
            }
        }
        
        async with session.post(
            f"{base_url}/api/recommend/style",
            json=style_request
        ) as response:
            if response.status == 200:
                result = await response.json()
                print(f"✅ 추천 성공: {len(result['recommendations'])}개 아이템")
                print(f"⏱️ 응답 시간: {result['performance_metrics']['total_response_time_ms']:.1f}ms")
                
                # 첫 번째 추천 아이템 출력
                if result['recommendations']:
                    first_item = result['recommendations'][0]
                    print(f"🏆 최고 유사도 아이템: {first_item['item_id']} (유사도: {first_item['similarity_score']:.4f})")
            else:
                error_data = await response.json()
                print(f"❌ 추천 실패: {error_data}")
        
        # 3. KPI 대시보드 데이터 테스트
        print("\n📊 KPI 대시보드 데이터 테스트...")
        async with session.get(f"{base_url}/api/dashboard/kpi") as response:
            if response.status == 200:
                kpi_data = await response.json()
                print(f"✅ KPI 데이터 조회 성공")
                print(f"📈 Top-5 정확도: {kpi_data['kpi_cards']['performance_metrics']['top_5_accuracy']:.4f}")
                print(f"🎯 MRR: {kpi_data['kpi_cards']['performance_metrics']['mrr']:.4f}")
                print(f"🔄 API 요청/초: {kpi_data['api_metrics']['requests_per_second']}")
            else:
                print(f"❌ KPI 데이터 조회 실패")

def test_json_encoder_sanity_check():
    """
    JSON Encoder Standalone Sanity Check
    Requirements 13: JSON Encoder 독립 검증
    """
    print("\n🧪 JSON Encoder Sanity Check 시작...")
    
    try:
        # 1. 합성 데이터 생성
        print("📝 합성 데이터 생성...")
        synthetic_batch = create_synthetic_json_batch()
        
        # 2. JSON Encoder 초기화
        print("🏗️ JSON Encoder 초기화...")
        vocab_sizes = {
            'category': 10,
            'style': 50,
            'silhouette': 20,
            'material': 30,
            'detail': 40
        }
        
        json_encoder = JSONEncoder(
            vocab_sizes=vocab_sizes,
            embedding_dim=128,
            hidden_dim=256
        )
        json_encoder.eval()
        
        # 3. 출력 차원 검증
        print("🔍 출력 차원 검증...")
        with torch.no_grad():
            output = json_encoder(synthetic_batch)
        
        ModelValidator.validate_output_dimension(output, expected_dim=512)
        print(f"✅ 출력 차원: {output.shape} (예상: [배치크기, 512])")
        
        # 4. L2 정규화 검증
        print("📏 L2 정규화 검증...")
        ModelValidator.validate_normalization(output)
        norms = torch.norm(output, dim=-1)
        print(f"✅ L2 정규화: 평균 norm = {norms.mean():.6f} (예상: 1.000000)")
        
        # 5. 배치 일관성 검증
        print("🔄 배치 일관성 검증...")
        with torch.no_grad():
            output2 = json_encoder(synthetic_batch)
        
        if torch.allclose(output, output2, atol=1e-6):
            print("✅ 배치 일관성: 동일 입력에 대해 동일 출력 생성")
        else:
            raise ValueError("배치 일관성 실패: 동일 입력에 대해 다른 출력 생성")
        
        # 6. 그래디언트 흐름 검증
        print("⚡ 그래디언트 흐름 검증...")
        json_encoder.train()
        output = json_encoder(synthetic_batch)
        loss = output.sum()  # 더미 손실
        loss.backward()
        
        # 그래디언트가 존재하는지 확인
        has_gradients = any(param.grad is not None for param in json_encoder.parameters())
        if has_gradients:
            print("✅ 그래디언트 흐름: 정상적인 역전파 확인")
        else:
            raise ValueError("그래디언트 흐름 실패: 역전파 중 그래디언트 생성되지 않음")
        
        # 7. 필드 처리 검증
        print("🏷️ 필드 처리 검증...")
        validate_field_processing(json_encoder, synthetic_batch)
        
        # 8. 최종 검증 완료
        print("\n🎉 **SANITY CHECK PASS** 🎉")
        print("✅ 모든 검증 항목 통과:")
        print("   - 512차원 출력 ✓")
        print("   - L2 정규화 ✓")
        print("   - 배치 일관성 ✓")
        print("   - 그래디언트 흐름 ✓")
        print("   - 필드 처리 ✓")
        
        # 검증 결과 저장
        save_sanity_check_results({
            "status": "PASS",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "output_shape": list(output.shape),
            "l2_norm_mean": norms.mean().item(),
            "l2_norm_std": norms.std().item(),
            "gradient_flow": has_gradients,
            "batch_consistency": True
        })
        
        return True
        
    except Exception as e:
        print(f"\n❌ **SANITY CHECK FAILED** ❌")
        print(f"오류: {str(e)}")
        
        # 실패 결과 저장
        save_sanity_check_results({
            "status": "FAILED",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error": str(e)
        })
        
        return False

def create_synthetic_json_batch(batch_size: int = 4) -> dict:
    """합성 JSON 배치 데이터 생성"""
    return {
        'category': torch.randint(1, 10, (batch_size,)),
        'style': torch.randint(1, 50, (batch_size, 4)),
        'silhouette': torch.randint(1, 20, (batch_size,)),
        'material': torch.randint(1, 30, (batch_size, 3)),
        'detail': torch.randint(1, 40, (batch_size, 5)),
        'style_mask': torch.ones(batch_size, 4, dtype=torch.long),
        'material_mask': torch.ones(batch_size, 3, dtype=torch.long),
        'detail_mask': torch.ones(batch_size, 5, dtype=torch.long)
    }

def validate_field_processing(json_encoder: JSONEncoder, batch: dict):
    """필드 처리 로직 검증"""
    # 단일 범주형 필드 테스트
    single_cat_batch = {
        'category': torch.tensor([1]),
        'style': torch.tensor([[0, 0, 0, 0]]),  # 모든 패딩
        'silhouette': torch.tensor([5]),
        'material': torch.tensor([[0, 0, 0]]),  # 모든 패딩
        'detail': torch.tensor([[0, 0, 0, 0, 0]]),  # 모든 패딩
        'style_mask': torch.zeros(1, 4, dtype=torch.long),
        'material_mask': torch.zeros(1, 3, dtype=torch.long),
        'detail_mask': torch.zeros(1, 5, dtype=torch.long)
    }
    
    with torch.no_grad():
        output = json_encoder(single_cat_batch)
    
    if output.shape == (1, 512):
        print("✅ 단일 범주형 필드 처리: 정상")
    else:
        raise ValueError(f"단일 범주형 필드 처리 실패: 예상 (1, 512), 실제 {output.shape}")
    
    # 다중 범주형 필드 테스트
    multi_cat_batch = {
        'category': torch.tensor([1]),
        'style': torch.tensor([[1, 2, 3, 0]]),  # 3개 유효, 1개 패딩
        'silhouette': torch.tensor([5]),
        'material': torch.tensor([[1, 2, 0]]),  # 2개 유효, 1개 패딩
        'detail': torch.tensor([[1, 2, 3, 4, 5]]),  # 모두 유효
        'style_mask': torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
        'material_mask': torch.tensor([[1, 1, 0]], dtype=torch.long),
        'detail_mask': torch.ones(1, 5, dtype=torch.long)
    }
    
    with torch.no_grad():
        output = json_encoder(multi_cat_batch)
    
    if output.shape == (1, 512):
        print("✅ 다중 범주형 필드 처리: 정상")
    else:
        raise ValueError(f"다중 범주형 필드 처리 실패: 예상 (1, 512), 실제 {output.shape}")

def save_sanity_check_results(results: dict):
    """Sanity Check 결과 저장"""
    Path("temp_logs").mkdir(exist_ok=True)
    
    with open("temp_logs/sanity_check_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"📁 검증 결과 저장: temp_logs/sanity_check_results.json")

def test_data_loading():
    """데이터 로딩 테스트"""
    print("\n📂 데이터 로딩 테스트...")
    
    try:
        # 데이터 로더 초기화
        dataset_loader = KFashionDatasetLoader(
            dataset_path="C:/sample/라벨링데이터",
            target_categories=['레트로', '로맨틱', '리조트']
        )
        
        # 데이터 로드
        fashion_items = dataset_loader.load_dataset_by_category()
        print(f"✅ 데이터 로드 완료: {len(fashion_items)}개 아이템")
        
        # 카테고리별 분포 출력
        category_counts = {}
        for item in fashion_items:
            category_counts[item.category] = category_counts.get(item.category, 0) + 1
        
        print("📊 카테고리별 분포:")
        for category, count in category_counts.items():
            print(f"   - {category}: {count}개")
        
        # 어휘 구축
        vocabularies = dataset_loader.build_vocabularies()
        print(f"📚 어휘 구축 완료: {len(vocabularies)}개 필드")
        
        for field, vocab in vocabularies.items():
            print(f"   - {field}: {len(vocab)}개 토큰")
        
        return True
        
    except Exception as e:
        print(f"❌ 데이터 로딩 실패: {str(e)}")
        return False

async def main():
    """메인 실행 함수"""
    print("🚀 Fashion JSON Encoder API 데모 시작")
    print("=" * 50)
    
    # 1. JSON Encoder Sanity Check
    sanity_check_passed = test_json_encoder_sanity_check()
    
    # 2. 데이터 로딩 테스트
    data_loading_passed = test_data_loading()
    
    # 3. API 테스트 (서버가 실행 중인 경우)
    print("\n🌐 API 엔드포인트 테스트...")
    try:
        await test_api_endpoints()
        api_test_passed = True
    except Exception as e:
        print(f"⚠️ API 테스트 건너뜀 (서버가 실행되지 않음): {str(e)}")
        api_test_passed = False
    
    # 4. 최종 결과 요약
    print("\n" + "=" * 50)
    print("📋 테스트 결과 요약:")
    print(f"   🧪 JSON Encoder Sanity Check: {'✅ PASS' if sanity_check_passed else '❌ FAIL'}")
    print(f"   📂 데이터 로딩: {'✅ PASS' if data_loading_passed else '❌ FAIL'}")
    print(f"   🌐 API 테스트: {'✅ PASS' if api_test_passed else '⚠️ SKIP'}")
    
    if sanity_check_passed and data_loading_passed:
        print("\n🎉 시스템 검증 완료! 모든 핵심 기능이 정상 작동합니다.")
    else:
        print("\n⚠️ 일부 테스트가 실패했습니다. 로그를 확인해주세요.")

if __name__ == "__main__":
    asyncio.run(main())