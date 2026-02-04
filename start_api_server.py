"""
Fashion JSON Encoder API Server Startup Script
Requirements 14: API 통신 구조 구현
"""

import uvicorn
import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """FastAPI 서버 시작"""
    print("🚀 Fashion JSON Encoder API 서버 시작...")
    print("📍 서버 주소: http://localhost:8000")
    print("📖 API 문서: http://localhost:8000/docs")
    print("🔍 헬스 체크: http://localhost:8000/health")
    print("=" * 50)
    
    # 환경 변수 설정
    os.environ.setdefault("PYTHONPATH", str(project_root))
    
    # FastAPI 서버 실행
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(project_root)],
        log_level="info"
    )

if __name__ == "__main__":
    main()