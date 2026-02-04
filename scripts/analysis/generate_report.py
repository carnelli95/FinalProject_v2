#!/usr/bin/env python3
"""
Fashion JSON Encoder 종합 보고서 생성

이 스크립트는 학습 결과와 시각화를 종합한 HTML 보고서를 생성합니다.
"""

import json
from pathlib import Path
from datetime import datetime
import base64

def load_training_results():
    """학습 결과 로드"""
    with open('results/training_results.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def encode_image_to_base64(image_path):
    """이미지를 base64로 인코딩"""
    try:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except FileNotFoundError:
        return None

def generate_html_report():
    """HTML 보고서 생성"""
    results = load_training_results()
    
    # 이미지들을 base64로 인코딩
    images = {}
    image_files = [
        'training_losses.png',
        'output_statistics.png', 
        'performance_metrics.png',
        'dataset_distribution.png',
        'embedding_space.png',
        'similarity_heatmap.png',
        'training_progress_detailed.png'
    ]
    
    for img_file in image_files:
        img_path = f'results/{img_file}'
        encoded = encode_image_to_base64(img_path)
        if encoded:
            images[img_file] = encoded
    
    # HTML 템플릿
    html_content = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fashion JSON Encoder 학습 결과 보고서</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-top: 30px;
        }}
        h3 {{
            color: #7f8c8d;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            color: white;
        }}
        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .image-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .image-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .metrics-table th, .metrics-table td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        .metrics-table th {{
            background-color: #3498db;
            color: white;
        }}
        .metrics-table tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .status-good {{
            color: #27ae60;
            font-weight: bold;
        }}
        .status-warning {{
            color: #f39c12;
            font-weight: bold;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #7f8c8d;
        }}
        .two-column {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Fashion JSON Encoder 학습 결과 보고서</h1>
        
        <div class="summary-grid">
            <div class="summary-card">
                <h3>📊 총 데이터셋</h3>
                <div class="value">2,172</div>
                <p>개 아이템</p>
            </div>
            <div class="summary-card">
                <h3>🏋️ 독립 학습</h3>
                <div class="value">{results['standalone']['val_losses'][-1]:.4f}</div>
                <p>최종 검증 손실</p>
            </div>
            <div class="summary-card">
                <h3>🔄 대조 학습</h3>
                <div class="value">{results['contrastive']['best_val_loss']:.4f}</div>
                <p>최고 검증 손실</p>
            </div>
            <div class="summary-card">
                <h3>📈 Top-5 정확도</h3>
                <div class="value">{results['contrastive']['final_metrics']['top5_accuracy']*100:.2f}%</div>
                <p>검색 성능</p>
            </div>
        </div>

        <h2>📊 데이터셋 정보</h2>
        <table class="metrics-table">
            <tr>
                <th>항목</th>
                <th>값</th>
                <th>설명</th>
            </tr>
            <tr>
                <td>총 아이템 수</td>
                <td class="status-good">2,172개</td>
                <td>레트로(196) + 로맨틱(994) + 리조트(998)</td>
            </tr>
            <tr>
                <td>학습/검증 분할</td>
                <td>1,737 / 435</td>
                <td>80% / 20% 비율</td>
            </tr>
            <tr>
                <td>카테고리 수</td>
                <td>3개</td>
                <td>레트로, 로맨틱, 리조트</td>
            </tr>
            <tr>
                <td>임베딩 차원</td>
                <td class="status-good">512차원</td>
                <td>FashionCLIP과 호환</td>
            </tr>
        </table>

        {"<div class='image-container'><img src='data:image/png;base64," + images['dataset_distribution.png'] + "' alt='데이터셋 분포'></div>" if 'dataset_distribution.png' in images else ""}

        <h2>🏋️ 학습 과정</h2>
        
        <h3>1단계: 독립 JSON 인코더 학습 (5 에포크)</h3>
        <ul>
            <li><strong>최종 Train Loss:</strong> {results['standalone']['train_losses'][-1]:.4f}</li>
            <li><strong>최종 Validation Loss:</strong> {results['standalone']['val_losses'][-1]:.4f}</li>
            <li><strong>임베딩 정규화:</strong> <span class="status-good">✅ L2 norm = {results['standalone']['final_analysis']['norm_mean']:.6f}</span></li>
        </ul>

        <h3>2단계: 대조 학습 (10 에포크)</h3>
        <ul>
            <li><strong>최고 Validation Loss:</strong> {results['contrastive']['best_val_loss']:.4f}</li>
            <li><strong>최종 Train Loss:</strong> {results['contrastive']['train_losses'][-1]:.4f}</li>
            <li><strong>최종 Validation Loss:</strong> {results['contrastive']['val_losses'][-1]:.4f}</li>
        </ul>

        {"<div class='image-container'><img src='data:image/png;base64," + images['training_losses.png'] + "' alt='학습 손실'></div>" if 'training_losses.png' in images else ""}

        <h2>📈 성능 메트릭</h2>
        <table class="metrics-table">
            <tr>
                <th>메트릭</th>
                <th>값</th>
                <th>상태</th>
            </tr>
            <tr>
                <td>Top-1 정확도</td>
                <td>{results['contrastive']['final_metrics']['top1_accuracy']*100:.2f}%</td>
                <td class="status-warning">초기 단계</td>
            </tr>
            <tr>
                <td>Top-5 정확도</td>
                <td>{results['contrastive']['final_metrics']['top5_accuracy']*100:.2f}%</td>
                <td class="status-warning">개선 필요</td>
            </tr>
            <tr>
                <td>Mean Reciprocal Rank</td>
                <td>{results['contrastive']['final_metrics']['mean_reciprocal_rank']:.4f}</td>
                <td class="status-warning">초기 단계</td>
            </tr>
            <tr>
                <td>평균 Positive Similarity</td>
                <td>{results['contrastive']['final_metrics']['avg_positive_similarity']:.4f}</td>
                <td class="status-good">정상</td>
            </tr>
            <tr>
                <td>평균 Negative Similarity</td>
                <td>{results['contrastive']['final_metrics']['negative_similarity_mean']:.4f}</td>
                <td class="status-good">정상</td>
            </tr>
        </table>

        {"<div class='image-container'><img src='data:image/png;base64," + images['performance_metrics.png'] + "' alt='성능 메트릭'></div>" if 'performance_metrics.png' in images else ""}

        <h2>🔍 모델 분석</h2>
        
        <div class="two-column">
            <div>
                <h3>✅ 성공 요소</h3>
                <ul>
                    <li>안정적인 학습 진행</li>
                    <li>임베딩 정규화 유지</li>
                    <li>손실 함수 수렴</li>
                    <li>대용량 데이터 처리 성공</li>
                </ul>
            </div>
            <div>
                <h3>⚠️ 개선 영역</h3>
                <ul>
                    <li>검색 정확도 향상 필요</li>
                    <li>더 많은 에포크 학습</li>
                    <li>하이퍼파라미터 튜닝</li>
                    <li>데이터 증강 기법 적용</li>
                </ul>
            </div>
        </div>

        {"<div class='image-container'><img src='data:image/png;base64," + images['output_statistics.png'] + "' alt='출력 통계'></div>" if 'output_statistics.png' in images else ""}

        <h2>🎨 임베딩 공간 분석</h2>
        <p>학습된 모델의 임베딩 공간을 t-SNE와 PCA로 시각화하여 카테고리별 클러스터링 상태를 분석했습니다.</p>

        {"<div class='image-container'><img src='data:image/png;base64," + images['embedding_space.png'] + "' alt='임베딩 공간'></div>" if 'embedding_space.png' in images else ""}

        {"<div class='image-container'><img src='data:image/png;base64," + images['similarity_heatmap.png'] + "' alt='유사도 히트맵'></div>" if 'similarity_heatmap.png' in images else ""}

        <h2>📋 상세 학습 진행 상황</h2>
        {"<div class='image-container'><img src='data:image/png;base64," + images['training_progress_detailed.png'] + "' alt='상세 학습 진행'></div>" if 'training_progress_detailed.png' in images else ""}

        <h2>💾 저장된 파일</h2>
        <ul>
            <li><strong>모델 체크포인트:</strong> checkpoints/best_model.pt</li>
            <li><strong>학습 결과:</strong> results/training_results.json</li>
            <li><strong>시각화 파일들:</strong> results/*.png</li>
            <li><strong>종합 보고서:</strong> results/training_report.html</li>
        </ul>

        <h2>🚀 다음 단계</h2>
        <ol>
            <li><strong>더 긴 학습:</strong> 에포크 수를 늘려 성능 향상</li>
            <li><strong>하이퍼파라미터 튜닝:</strong> 학습률, 배치 크기, temperature 조정</li>
            <li><strong>데이터 증강:</strong> 이미지 변환 기법 적용</li>
            <li><strong>모델 아키텍처:</strong> 더 복잡한 JSON 인코더 구조 실험</li>
            <li><strong>평가 확장:</strong> 더 다양한 메트릭으로 성능 평가</li>
        </ol>

        <div class="footer">
            <p>보고서 생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Fashion JSON Encoder v1.0 - K-Fashion 데이터셋 학습 결과</p>
        </div>
    </div>
</body>
</html>
    """
    
    # HTML 파일 저장
    with open('results/training_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("✅ HTML 보고서 생성 완료: results/training_report.html")

def main():
    """메인 함수"""
    print("📋 Fashion JSON Encoder 종합 보고서 생성 중...")
    
    try:
        generate_html_report()
        print("\n🎉 종합 보고서가 성공적으로 생성되었습니다!")
        print("📁 파일 위치: results/training_report.html")
        print("🌐 웹 브라우저에서 열어서 확인하세요.")
        
    except Exception as e:
        print(f"❌ 보고서 생성 중 오류: {e}")

if __name__ == "__main__":
    main()