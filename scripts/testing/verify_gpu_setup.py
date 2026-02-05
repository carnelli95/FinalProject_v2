#!/usr/bin/env python3
"""
GPU 환경 및 대용량 학습 준비 상태 검증 스크립트
"""

import sys
import torch
import torchvision
import transformers
import numpy as np
from pathlib import Path

def check_python_version():
    """Python 버전 확인"""
    print("🐍 Python 버전 확인")
    version = sys.version_info
    print(f"   현재 버전: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and 9 <= version.minor <= 11:
        print("   ✅ 권장 버전 범위 (3.9-3.11)")
        return True
    else:
        print("   ⚠️  권장: Python 3.9-3.11")
        return False

def check_cuda_setup():
    """CUDA 설정 확인"""
    print("\n🔥 CUDA 환경 확인")
    
    # CUDA 사용 가능 여부
    cuda_available = torch.cuda.is_available()
    print(f"   CUDA 사용 가능: {'✅' if cuda_available else '❌'}")
    
    if not cuda_available:
        print("   ❌ CUDA가 설치되지 않았거나 PyTorch가 CPU 버전입니다")
        return False
    
    # CUDA 버전
    cuda_version = torch.version.cuda
    print(f"   CUDA 버전: {cuda_version}")
    
    # GPU 정보
    gpu_count = torch.cuda.device_count()
    print(f"   GPU 개수: {gpu_count}")
    
    total_memory = 0
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        total_memory += gpu_memory
        print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    print(f"   총 GPU 메모리: {total_memory:.1f}GB")
    
    # 65GB 학습을 위한 권장사항
    if total_memory >= 48:  # 24GB x 2 이상
        print("   ✅ 65GB 대용량 학습 가능")
        return True
    elif total_memory >= 24:  # 24GB 이상
        print("   ⚠️  단일 GPU 학습 가능, 분산 학습 권장")
        return True
    else:
        print("   ❌ GPU 메모리 부족 (최소 24GB 권장)")
        return False

def check_pytorch_version():
    """PyTorch 버전 확인"""
    print("\n🔥 PyTorch 버전 확인")
    
    torch_version = torch.__version__
    torchvision_version = torchvision.__version__
    
    print(f"   PyTorch: {torch_version}")
    print(f"   TorchVision: {torchvision_version}")
    
    # 버전 파싱
    torch_major, torch_minor = map(int, torch_version.split('.')[:2])
    
    if torch_major >= 2 and torch_minor >= 1:
        print("   ✅ 권장 버전 (2.1.0+)")
        return True
    elif torch_major >= 2:
        print("   ⚠️  최소 버전 충족, 2.1.0+ 권장")
        return True
    else:
        print("   ❌ PyTorch 2.0+ 필요")
        return False

def check_transformers():
    """Transformers 라이브러리 확인"""
    print("\n🤗 Transformers 확인")
    
    transformers_version = transformers.__version__
    print(f"   Transformers: {transformers_version}")
    
    # CLIP 모델 로드 테스트
    try:
        from transformers import CLIPVisionModel
        model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        print("   ✅ CLIP 모델 로드 성공")
        return True
    except Exception as e:
        print(f"   ❌ CLIP 모델 로드 실패: {e}")
        return False

def check_mixed_precision():
    """Mixed Precision 지원 확인"""
    print("\n⚡ Mixed Precision 확인")
    
    if not torch.cuda.is_available():
        print("   ❌ CUDA 필요")
        return False
    
    try:
        # AMP 테스트
        scaler = torch.cuda.amp.GradScaler()
        
        # 간단한 연산 테스트
        x = torch.randn(2, 3, device='cuda')
        with torch.cuda.amp.autocast():
            y = x * 2
        
        print("   ✅ Mixed Precision 지원")
        return True
    except Exception as e:
        print(f"   ❌ Mixed Precision 실패: {e}")
        return False

def check_distributed_training():
    """분산 학습 지원 확인"""
    print("\n🔗 분산 학습 확인")
    
    try:
        import torch.distributed as dist
        
        # NCCL 백엔드 확인
        if torch.distributed.is_nccl_available():
            print("   ✅ NCCL 백엔드 사용 가능")
        else:
            print("   ⚠️  NCCL 백엔드 사용 불가")
        
        # 멀티 GPU 확인
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            print(f"   ✅ 멀티 GPU 분산 학습 가능 ({gpu_count}개)")
            return True
        else:
            print("   ⚠️  단일 GPU (분산 학습 불가)")
            return True
            
    except Exception as e:
        print(f"   ❌ 분산 학습 설정 실패: {e}")
        return False

def check_memory_optimization():
    """메모리 최적화 기능 확인"""
    print("\n💾 메모리 최적화 확인")
    
    try:
        import psutil
        import gc
        
        # 시스템 메모리
        memory = psutil.virtual_memory()
        print(f"   시스템 RAM: {memory.total / 1024**3:.1f}GB")
        print(f"   사용 가능: {memory.available / 1024**3:.1f}GB")
        
        if memory.total >= 64 * 1024**3:  # 64GB
            print("   ✅ 충분한 시스템 메모리")
        else:
            print("   ⚠️  시스템 메모리 부족 (64GB 권장)")
        
        # GPU 메모리 정리 테스트
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print("   ✅ 메모리 정리 기능 정상")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 메모리 최적화 확인 실패: {e}")
        return False

def main():
    """전체 환경 검증"""
    print("🎯 65GB 대용량 학습 환경 검증")
    print("=" * 60)
    
    checks = [
        check_python_version(),
        check_cuda_setup(),
        check_pytorch_version(),
        check_transformers(),
        check_mixed_precision(),
        check_distributed_training(),
        check_memory_optimization()
    ]
    
    passed = sum(checks)
    total = len(checks)
    
    print("\n" + "=" * 60)
    print(f"🎯 검증 결과: {passed}/{total} 통과")
    
    if passed == total:
        print("✅ 모든 검증 통과! 65GB 대용량 학습 준비 완료")
        print("\n🚀 다음 단계:")
        print("   1. 65GB 데이터셋 경로 확인")
        print("   2. scripts/training/large_scale_v2_training.py 실행")
        print("   3. 학습 모니터링 및 체크포인트 관리")
    elif passed >= 5:
        print("⚠️  대부분 검증 통과, 일부 최적화 필요")
        print("   기본 학습은 가능하지만 성능 최적화 권장")
    else:
        print("❌ 환경 설정 필요")
        print("   Python, CUDA, PyTorch 설치 및 설정 확인 필요")

if __name__ == "__main__":
    main()