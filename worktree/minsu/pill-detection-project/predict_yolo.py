# predict_yolo.py
"""YOLOv11 전용 예측 실행 (기존 predict.py 구조 재사용)"""

import torch
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# 기존 코드 재사용
from src.utils import Visualizer


def predict_images_yolo(model, image_paths, conf_threshold=0.25):
    """YOLOv11로 이미지들에 대해 예측 수행 (기존 predict.py 구조 참고)"""
    
    predictions = []
    
    for img_path in tqdm(image_paths, desc="예측 중"):
        # 이미지 로드 (기존 방식과 동일)
        image = Image.open(img_path).convert('RGB')
        
        # YOLOv11 예측 (YOLO 전용 부분)
        results = model.predict(image, conf=conf_threshold, verbose=False)
        result = results[0]
        
        # 기존 predict.py 형식으로 변환 (호환성 유지)
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu()  # x1, y1, x2, y2
            scores = result.boxes.conf.cpu()
            labels = result.boxes.cls.int().cpu() + 1  # 0-based → 1-based (기존과 동일)
        else:
            boxes = torch.zeros((0, 4))
            scores = torch.zeros((0,))
            labels = torch.zeros((0,), dtype=torch.int64)
        
        # 기존 predict.py와 동일한 형식으로 저장
        predictions.append({
            'image_id': int(img_path.stem) if img_path.stem.isdigit() else hash(img_path.stem),
            'image_path': str(img_path),
            'boxes': boxes,
            'labels': labels,
            'scores': scores
        })
    
    return predictions


def main():
    # 기존 predict.py와 동일한 argument parser
    parser = argparse.ArgumentParser(description='YOLOv11 이미지 예측')
    parser.add_argument('--model', type=str, required=True, help='YOLOv11 모델 파일 경로 (.pt)')
    parser.add_argument('--images', type=str, required=True, help='이미지 디렉토리')
    parser.add_argument('--output', type=str, default='outputs/predictions_yolo')
    parser.add_argument('--visualize', action='store_true', help='결과 시각화')
    parser.add_argument('--conf-threshold', type=float, default=0.25, help='신뢰도 임계값')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("YOLOv11 알약 검출 예측")  # 제목만 수정
    print("=" * 50)
    
    # YOLOv11 모델 로드 (YOLO 전용 부분)
    print("\nYOLOv11 모델 로딩...")
    model = YOLO(args.model)
    print(f"모델 로드 완료: {args.model}")
    
    # 이미지 찾기 (기존 방식 재사용 + 확장)
    image_dir = Path(args.images)
    image_paths = []
    
    # png, jpg 모두 지원 (기존보다 확장)
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        image_paths.extend(sorted(image_dir.rglob(ext)))
    
    print(f"\n발견된 이미지: {len(image_paths)}개")
    
    if len(image_paths) == 0:
        print("❌ 예측할 이미지가 없습니다!")
        return
    
    # YOLOv11 예측 수행
    predictions = predict_images_yolo(model, image_paths, args.conf_threshold)
    
    # 결과 저장 (기존 방식 재사용)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 기존 Visualizer 재사용
    visualizer = Visualizer()
    
    # 시각화 (기존 predict.py 로직 재사용)
    if args.visualize:
        print("\n시각화 중...")
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
        
        # 기존과 동일하게 처음 10개만
        for i, pred in enumerate(predictions[:10]):
            img = Image.open(pred['image_path'])
            save_path = vis_dir / f"pred_{i}.png"
            visualizer.plot_image_with_boxes(np.array(img), pred, save_path)
            print(f"저장됨: {save_path}")
    
    # 결과 요약 (추가 기능)
    total_detections = sum(len(pred['boxes']) for pred in predictions)
    print(f"\n예측 완료!")
    print(f"처리된 이미지: {len(predictions)}개")
    print(f"총 검출 객체: {total_detections}개")
    print(f"평균 검출 수: {total_detections/len(predictions):.1f}개/이미지")
    print(f"결과 저장: {output_dir}")
    
    if args.visualize:
        print(f"시각화 이미지: {len(predictions[:10])}개 (최대 10개)")
        print(f"시각화 위치: {vis_dir}")


if __name__ == '__main__':
    main()