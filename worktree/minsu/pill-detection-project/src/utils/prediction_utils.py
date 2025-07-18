# src/utils/prediction_utils.py
"""예측 관련 공통 유틸리티"""

from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm


def find_images(image_dir, extensions=['*.png', '*.jpg', '*.jpeg']):
    """디렉토리에서 이미지 파일들 찾기 (대소문자 무관)"""
    image_dir = Path(image_dir)
    image_paths = []
    
    for ext in extensions:
        image_paths.extend(sorted(image_dir.rglob(ext)))
        image_paths.extend(sorted(image_dir.rglob(ext.upper())))
    
    return image_paths


def load_image_as_array(img_path):
    """이미지를 numpy 배열로 로드"""
    image = Image.open(img_path).convert('RGB')
    return np.array(image)


def save_predictions_summary(predictions, output_file):
    """예측 결과 요약을 텍스트 파일로 저장"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== 예측 결과 요약 ===\n\n")
        
        total_detections = sum(len(pred['boxes']) for pred in predictions)
        f.write(f"처리된 이미지: {len(predictions)}개\n")
        f.write(f"총 검출 객체: {total_detections}개\n")
        f.write(f"평균 검출 수: {total_detections/len(predictions):.1f}개/이미지\n\n")
        
        f.write("=== 이미지별 검출 결과 ===\n")
        for i, pred in enumerate(predictions):
            f.write(f"\n{i+1}. {Path(pred['image_path']).name}\n")
            f.write(f"   검출 객체: {len(pred['boxes'])}개\n")
            
            if len(pred['boxes']) > 0:
                for j, (label, score) in enumerate(zip(pred['labels'], pred['scores'])):
                    f.write(f"   - 객체 {j+1}: 클래스 {label.item()}, 신뢰도 {score.item():.3f}\n")


def visualize_predictions_batch(predictions, visualizer, output_dir, max_images=10):
    """예측 결과들을 일괄 시각화"""
    vis_dir = Path(output_dir) / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    saved_count = 0
    for i, pred in enumerate(predictions[:max_images]):
        try:
            img = load_image_as_array(pred['image_path'])
            save_path = vis_dir / f"pred_{i}.png"
            visualizer.plot_image_with_boxes(img, pred, save_path)
            saved_count += 1
        except Exception as e:
            print(f"시각화 실패 {pred['image_path']}: {e}")
    
    return saved_count, vis_dir