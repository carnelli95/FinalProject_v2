#!/usr/bin/env python3
"""
빠른 테스트 실행 스크립트

이 스크립트는 모든 테스트를 빠르게 실행하여 개발 과정에서 빠른 피드백을 제공합니다.

사용법:
    python run_fast_tests.py              # 모든 테스트 실행
    python run_fast_tests.py --unit       # 단위 테스트만 실행
    python run_fast_tests.py --demo       # 데모만 실행
    python run_fast_tests.py --quick      # 매우 빠른 모드
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path


def run_command(cmd, description, timeout=60):
    """명령어 실행 및 결과 출력"""
    print(f"\n{'='*60}")
    print(f"실행 중: {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout,
            encoding='utf-8',
            errors='replace'
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"성공: {description} 완료 ({elapsed:.1f}초)")
            if result.stdout:
                print("출력:")
                print(result.stdout[-500:])  # 마지막 500자만 출력
        else:
            print(f"실패: {description} 실패 ({elapsed:.1f}초)")
            if result.stderr:
                print("오류:")
                print(result.stderr[-500:])  # 마지막 500자만 출력
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"시간 초과: {description} 시간 초과 ({timeout}초)")
        return False
    except Exception as e:
        print(f"오류: {description} 실행 중 오류: {e}")
        return False


def run_unit_tests():
    """단위 테스트 실행"""
    tests = [
        ("python -m pytest tests/test_json_encoder.py -v --tb=short", "JSON Encoder 테스트"),
        ("python -m pytest tests/test_validators.py -v --tb=short", "Validator 테스트"),
        ("python -m pytest tests/test_data_processing.py -v --tb=short", "데이터 처리 테스트"),
    ]
    
    results = []
    for cmd, desc in tests:
        success = run_command(cmd, desc, timeout=30)
        results.append((desc, success))
    
    return results


def run_integration_tests():
    """통합 테스트 실행"""
    tests = [
        ("python -m pytest tests/test_training.py::TestFashionTrainer::test_trainer_initialization -v --tb=short", "트레이너 초기화 테스트"),
        ("python -m pytest tests/test_fashion_dataset.py -v --tb=short", "데이터셋 테스트"),
    ]
    
    results = []
    for cmd, desc in tests:
        success = run_command(cmd, desc, timeout=45)
        results.append((desc, success))
    
    return results


def run_demo_tests(quick_mode=False):
    """데모 테스트 실행"""
    if quick_mode:
        cmd = "python test_similarity_search.py --quick"
        desc = "유사도 검색 데모 (Quick 모드)"
        timeout = 30
    else:
        cmd = "python test_similarity_search.py --fast"
        desc = "유사도 검색 데모 (Fast 모드)"
        timeout = 60
    
    success = run_command(cmd, desc, timeout=timeout)
    return [(desc, success)]


def print_summary(all_results):
    """테스트 결과 요약 출력"""
    print(f"\n{'='*60}")
    print("테스트 결과 요약")
    print(f"{'='*60}")
    
    total_tests = len(all_results)
    passed_tests = sum(1 for _, success in all_results if success)
    failed_tests = total_tests - passed_tests
    
    print(f"총 테스트: {total_tests}")
    print(f"성공: {passed_tests}")
    print(f"실패: {failed_tests}")
    
    if failed_tests > 0:
        print(f"\n실패한 테스트:")
        for desc, success in all_results:
            if not success:
                print(f"  • {desc}")
    
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    print(f"\n성공률: {success_rate:.1f}%")
    
    if success_rate == 100:
        print("모든 테스트가 성공했습니다!")
    elif success_rate >= 80:
        print("대부분의 테스트가 성공했습니다.")
    else:
        print("일부 테스트가 실패했습니다. 확인이 필요합니다.")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='빠른 테스트 실행 스크립트')
    parser.add_argument('--unit', action='store_true', help='단위 테스트만 실행')
    parser.add_argument('--demo', action='store_true', help='데모만 실행')
    parser.add_argument('--quick', action='store_true', help='매우 빠른 모드 (최소 테스트)')
    
    args = parser.parse_args()
    
    print("Fashion JSON Encoder 빠른 테스트 시작")
    print(f"시작 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = []
    
    if args.quick:
        print("Quick 모드: 최소 테스트만 실행")
        # 가장 중요한 테스트만 실행
        success = run_command(
            "python -m pytest tests/test_json_encoder.py::TestJSONEncoder::test_forward_output_shape -v", 
            "핵심 기능 테스트", 
            timeout=30
        )
        all_results.append(("핵심 기능 테스트", success))
        
        # 빠른 데모 실행
        demo_results = run_demo_tests(quick_mode=True)
        all_results.extend(demo_results)
        
    elif args.unit:
        print("단위 테스트만 실행")
        unit_results = run_unit_tests()
        all_results.extend(unit_results)
        
    elif args.demo:
        print("데모만 실행")
        demo_results = run_demo_tests(quick_mode=False)
        all_results.extend(demo_results)
        
    else:
        print("전체 테스트 실행")
        
        # 단위 테스트
        print("\n1단계: 단위 테스트")
        unit_results = run_unit_tests()
        all_results.extend(unit_results)
        
        # 통합 테스트
        print("\n2단계: 통합 테스트")
        integration_results = run_integration_tests()
        all_results.extend(integration_results)
        
        # 데모 테스트
        print("\n3단계: 데모 테스트")
        demo_results = run_demo_tests(quick_mode=False)
        all_results.extend(demo_results)
    
    # 결과 요약
    print_summary(all_results)
    
    # 종료 코드 설정
    failed_count = sum(1 for _, success in all_results if not success)
    sys.exit(failed_count)


if __name__ == "__main__":
    main()