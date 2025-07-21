# evaluate_yolo.py
"""YOLOv11 전용 평가 스크립트"""

import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

from src.data import COCODataset, get_transform
from src.training.evaluator import calculate_map
from src.utils import Visualizer


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def evaluate_yolo_model(model, data_loader, conf_threshold=0.25):
    """YOLOv11 모델 평가"""
    all_predictions = []
    all_targets = []
    
    print(f"YOLOv11 평가 중... (신뢰도 임계값: {conf_threshold})")
    
    for images, targets in tqdm(data_loader, desc="평가 중"):
        # YOLO 예측
        batch_preds = []
        for img in images:
            # PIL 이미지로 변환해서 YOLO에 전달
            img_np = img.numpy().transpose(1, 2, 0)  # CHW → HWC
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype('uint8')
            
            # YOLO 예측 수행
            results = model.predict(img_np, conf=conf_threshold, verbose=False)
            result = results[0]
            
            # Faster R-CNN 형식으로 변환
            if result.boxes is not None and len(result.boxes) > 0:
                # YOLO 클래스 매핑: 0,1,2,3 → 우리 데이터셋의 1,2,3,4
                yolo_classes = result.boxes.cls.int().cpu()
                
                # YOLO 클래스를 우리 클래스로 매핑
                class_mapping = {0: 1, 1: 2, 2: 3, 3: 4}  # YOLO → 우리 데이터셋
                mapped_labels = torch.tensor([class_mapping[cls.item()] for cls in yolo_classes])
                pred = {
                    'boxes': result.boxes.xyxy.cpu(),
                    'labels': mapped_labels,
                    'scores': result.boxes.conf.cpu()
                }
            else:
                pred = {
                    'boxes': torch.zeros((0, 4)),
                    'labels': torch.zeros((0,), dtype=torch.int64),
                    'scores': torch.zeros((0,))
                }
            
            batch_preds.append(pred)
        
        all_predictions.extend(batch_preds)
        
        # 타겟은 그대로 CPU로 이동
        for target in targets:
            all_targets.append({
                'boxes': target['boxes'].cpu(),
                'labels': target['labels'].cpu()
            })
    
    # mAP 계산 (기존 calculate_map 재사용)
    results = calculate_map(
        all_predictions,
        all_targets,
        iou_thresholds=[0.5, 0.75],
        num_classes=4,  # 4개 알약 클래스
        score_threshold=conf_threshold
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(description='YOLOv11 모델 평가')
    parser.add_argument('--model', type=str, required=True, help='YOLOv11 모델 파일 경로 (.pt)')
    parser.add_argument('--images', type=str, default='./ai03-level1-project/train_images', help='이미지 디렉토리')
    parser.add_argument('--visualize', action='store_true', help='결과 시각화')
    parser.add_argument('--conf-threshold', type=float, default=0.5, help='신뢰도 임계값')
    parser.add_argument('--subset-size', type=int, default=None, help='테스트용 서브셋 크기 (선택)')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("YOLOv11 모델 평가")
    print("=" * 50)
    print(f"모델: {args.model}")
    print(f"신뢰도 임계값: {args.conf_threshold}")
    
    # YOLOv11 모델 로드
    print("\nYOLOv11 모델 로딩...")
    model = YOLO(args.model)
    print(f"모델 로드 완료!")
    
    # 데이터셋 로딩
    print("\n데이터셋 로딩...")
    dataset = COCODataset(
        img_dir=args.images,
        ann_dir=Path(args.images).parent / "train_annotations",
        transform=get_transform(train=False)
    )
    
    # 서브셋 생성 (디버깅용)
    if args.subset_size and args.subset_size < len(dataset):
        print(f"서브셋 사용: {args.subset_size}개 이미지")
        subset_indices = list(range(args.subset_size))
        dataset = torch.utils.data.Subset(dataset, subset_indices)
    
    test_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    print(f"평가할 이미지: {len(dataset)}개")
    print(f"배치 수: {len(test_loader)}")
    
    # 평가 수행
    results = evaluate_yolo_model(model, test_loader, args.conf_threshold)
    
    # 결과 출력 (Faster R-CNN과 동일한 형식)
    print("\n=== YOLOv11 평가 결과 ===")
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
        output_dir = Path('outputs/evaluation_yolo')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 샘플 이미지 시각화
        count = 0
        for images, targets in test_loader:
            if count >= 5:  # 5개 배치만
                break
                
            for i, img in enumerate(images):
                # YOLOv11 예측
                img_np = img.numpy().transpose(1, 2, 0)
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype('uint8')
                
                results = model.predict(img_np, conf=0.3, verbose=False)  # 시각화용은 낮은 임계값
                result = results[0]
                
                # 예측 결과 변환
                if result.boxes is not None and len(result.boxes) > 0:
                    # YOLO 클래스 매핑
                    yolo_classes = result.boxes.cls.int().cpu()
                    class_mapping = {0: 1, 1: 2, 2: 3, 3: 4}  # YOLO → 우리 데이터셋
                    mapped_labels = torch.tensor([class_mapping[cls.item()] for cls in yolo_classes])
                    pred = {
                        'boxes': result.boxes.xyxy.cpu(),
                        'labels': mapped_labels,
                        'scores': result.boxes.conf.cpu()
                    }
                else:
                    pred = {
                        'boxes': torch.zeros((0, 4)),
                        'labels': torch.zeros((0,), dtype=torch.int64),
                        'scores': torch.zeros((0,))
                    }
                
                save_path = output_dir / f'yolo_sample_{count}_{i}.png'
                visualizer.plot_image_with_boxes(img, pred, save_path)
                print(f"시각화 저장: {save_path}")
            
            count += 1
        
        print(f"시각화 완료: {output_dir}")


if __name__ == '__main__':
    main()