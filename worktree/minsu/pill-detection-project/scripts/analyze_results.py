# analyze_results.py
"""학습 결과 종합 분석"""

import torch
import argparse
from pathlib import Path
from src.utils.visualization import Visualizer
from src.data.dataset import get_dataloader
from src.models import create_model

def comprehensive_analysis(checkpoint_path, output_dir='outputs/analysis'):
    """종합적인 결과 분석"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    visualizer = Visualizer()
    
    # 1. 학습 곡선 시각화
    if 'history' in checkpoint:
        print("학습 곡선 생성 중...")
        visualizer.plot_training_curves(
            checkpoint['history'], 
            save_path=output_dir / 'training_curves.png'
        )
    
    # 2. 혼동행렬 시각화 (모델 로드 필요)
    print("혼동행렬 생성 중...")
    model = create_model('faster_rcnn', num_classes=5)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 검증 데이터로 예측
    val_dataloader = get_dataloader('val')
    predictions, targets = evaluate_model(model, val_dataloader)
    
    visualizer.plot_confusion_matrix(
        predictions, targets,
        save_path=output_dir / 'confusion_matrix.png'
    )
    
    print(f"분석 완료! 결과: {output_dir}")

# 실행 예시
if __name__ == '__main__':
    comprehensive_analysis('outputs/checkpoints/best_model.pth')
