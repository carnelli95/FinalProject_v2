#!/usr/bin/env python3
"""
확장 학습 실행 스크립트

현재 체크포인트에서 50 에포크 추가 학습을 진행합니다.
"""

import subprocess
import sys
import time
from pathlib import Path


def check_checkpoint():
    """체크포인트 존재 확인"""
    checkpoint_path = Path("checkpoints/best_model.pt")
    if not checkpoint_path.exists():
        print("❌ 체크포인트 파일이 없습니다!")
        print("   먼저 기본 학습을 완료해주세요.")
        return False
    
    print(f"✅ 체크포인트 발견: {checkpoint_path}")
    return True


def run_extended_training():
    """확장 학습 실행"""
    print("🚀 확장 학습 시작!")
    print("=" * 60)
    
    # 체크포인트 확인
    if not check_checkpoint():
        return False
    
    # 학습 실행
    cmd = [
        sys.executable, "continue_training.py",
        "--epochs", "50",
        "--checkpoint", "checkpoints/best_model.pt"
    ]
    
    print(f"실행 명령어: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ 확장 학습 완료!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 학습 실행 실패: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⏹️ 사용자에 의해 중단되었습니다.")
        return False


def show_tensorboard_info():
    """TensorBoard 실행 안내"""
    print("\n📊 TensorBoard로 학습 진행 상황 확인:")
    print("   tensorboard --logdir=logs")
    print("   브라우저에서 http://localhost:6006 접속")


def main():
    """메인 함수"""
    print("Fashion JSON Encoder 확장 학습")
    print(f"시작 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 확장 학습 실행
    success = run_extended_training()
    
    if success:
        print("\n🎉 모든 작업이 완료되었습니다!")
        show_tensorboard_info()
        
        print("\n📈 다음 단계:")
        print("1. TensorBoard로 학습 곡선 확인")
        print("2. 성능 개선이 확인되면 유사도 검색 테스트")
        print("3. 만족스러우면 Optuna 튜닝 진행")
        
    else:
        print("\n❌ 작업이 실패했습니다.")
        print("로그를 확인하고 문제를 해결해주세요.")


if __name__ == "__main__":
    main()