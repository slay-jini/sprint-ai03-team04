# predict.py
"""예측 실행"""

import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

from src.models import create_model
from src.data.transforms import get_transform
from src.utils import load_config, Visualizer


def predict_images(model, image_paths, transform, device, config):
    """이미지들에 대해 예측 수행"""
    model.eval()
    model.to(device)
    
    predictions = []
    
    for img_path in tqdm(image_paths, desc="예측 중"):
        # 이미지 로드
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # 변환
        if transform:
            #transformed = transform(image=image, bboxes=[], labels=[])
            transformed = transform(image=image, target={})
            #image_tensor = transformed['image']
            image_tensor = transformed[0]
        
        # 배치 차원 추가
        image_tensor = image_tensor.to(device)
        
        # 예측
        with torch.no_grad():
            pred = model([image_tensor])[0]
        
        # 후처리
        pred = model.postprocess([pred], 
                               score_threshold=config['inference']['score_threshold'],
                               nms_threshold=config['inference']['nms_threshold'])[0]
        
        predictions.append({
            'image_id': int(img_path.stem),
            'image_path': str(img_path),
            'boxes': pred['boxes'].cpu(),
            'labels': pred['labels'].cpu(),
            'scores': pred['scores'].cpu()
        })
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description='이미지 예측')
    parser.add_argument('--model', type=str, required=True, help='모델 파일 경로')
    parser.add_argument('--images', type=str, required=True, help='이미지 디렉토리')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml')
    parser.add_argument('--output', type=str, default='outputs/predictions')
    parser.add_argument('--visualize', action='store_true', help='결과 시각화')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("알약 검출 예측")
    print("=" * 50)
    
    # 설정 로드
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 체크포인트 로드
    print("\n모델 로딩...")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    saved_config = checkpoint.get('config', config)
    
    # 모델 생성
    model = create_model(
        saved_config['model']['name'],        
        **saved_config['model']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 이미지 찾기
    image_dir = Path(args.images)
    image_paths = sorted(image_dir.rglob("*.png"))
    print(f"\n이미지 수: {len(image_paths)}")
    
    # 변환
    transform = get_transform(train=False)
    
    # 예측
    predictions = predict_images(model, image_paths, transform, device, config)
    
    # 결과 저장
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer = Visualizer()
    visualizer.save_detection_results(predictions, output_dir)
    
    # 시각화
    if args.visualize:
        print("\n시각화 중...")
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
        
        for i, pred in enumerate(predictions[:10]):  # 처음 10개만
            img = Image.open(pred['image_path'])
            save_path = vis_dir / f"pred_{i}.png"
            visualizer.plot_image_with_boxes(np.array(img), pred, save_path)
    
    print(f"\n예측 완료! 결과: {output_dir}")


if __name__ == '__main__':
    main()