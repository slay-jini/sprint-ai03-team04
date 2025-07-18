# evaluate.py
"""모델 평가 실행"""

import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

from src.data import COCODataset, get_transform
from src.models import create_model
from src.training import Evaluator
from src.utils import load_config, Visualizer


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def main():
    parser = argparse.ArgumentParser(description='모델 평가')
    parser.add_argument('--model', type=str, required=True, help='모델 파일 경로')
    parser.add_argument('--config', type=str, default=None, help='설정 파일 경로(옵션)')
    parser.add_argument('--visualize', action='store_true', help='결과 시각화')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("모델 평가")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 체크포인트 로드 및 config 추출
    print(f"\n체크포인트 로딩: {args.model}")
    if not Path(args.model).exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {args.model}")
    
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    
    # Config 우선순위: 1) 명령줄 지정 2) 체크포인트 저장된 config 3) 기본값

    if args.config:
        print(f"명령줄 지정 Config 사용: {args.config}")
        config = load_config(args.config)
        saved_config = config  # 명령줄 우선
    elif 'config' in checkpoint:
        print("체크포인트에서 저장된 Config 자동 사용")
        saved_config = checkpoint['config']
        config = saved_config
        print(f"모델 타입: {config['model']['name']}")
    else:
        print("체크포인트에 Config가 없음. 기본 Config 사용")
        config = load_config('configs/base_config.yaml')
        saved_config = config
    
    print(f"체크포인트 에폭: {checkpoint.get('epoch', 'N/A')}")
    print(f"체크포인트 손실: {checkpoint.get('loss', 'N/A')}")
    
    # 데이터셋
    print("\n데이터셋 로딩...")
    test_dataset = COCODataset(
        img_dir=config['data']['train_path'],  # 또는 별도의 테스트 경로
        ann_dir=config['data']['train_path'].parent / "train_annotations",
        transform=get_transform(train=False)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # 모델 생성 및 로드
    print("\n모델 로딩...")
    
    if saved_config['model']['name'] == 'yolo':
        # YOLO 모델은 다른 방식으로 로드
        from src.models.yolo import load_yolo_model
        model = load_yolo_model(args.model)
    else:
        # Faster R-CNN 등은 기존 방식
        model = create_model(
            saved_config['model']['name'],        
            **saved_config['model']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)  # 모델을 GPU로 이동
        model.eval()      # 평가 모드 설정

    print(f"모델 타입: {saved_config['model']['name']}")
    print(f"클래스 수: {saved_config['model']['num_classes']}")
    print(f"디바이스: {device}")

    
    # 평가
    evaluator = Evaluator(config, device=device)
    results = evaluator.evaluate(model, test_loader)
    
    # 결과 출력
    print("\n=== 평가 결과 ===")
    print(f"mAP: {results['mAP']:.4f}")
    print(f"mAP@0.5: {results['mAP_50']:.4f}")
    print(f"mAP@0.75: {results['mAP_75']:.4f}")
    
    print("\n클래스별 AP:")
    for class_id, ap in results['ap_per_class'].items():
        print(f"  클래스 {class_id}: {ap:.4f}")
    
    # 시각화
    if args.visualize:
        print("\n결과 시각화...")
        visualizer = Visualizer()
        output_dir = Path('outputs/evaluation')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 샘플 이미지에 대한 예측 시각화
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        with torch.no_grad():
            for i, (images, targets) in enumerate(test_loader):
                if i >= 5:  # 5개만 시각화
                    break
                
                images = [img.to(device) for img in images]
                predictions = model(images)
                
                for j, (img, pred) in enumerate(zip(images, predictions)):
                    save_path = output_dir / f'sample_{i}_{j}.png'
                    visualizer.plot_image_with_boxes(img, pred, save_path)


if __name__ == '__main__':
    main()

