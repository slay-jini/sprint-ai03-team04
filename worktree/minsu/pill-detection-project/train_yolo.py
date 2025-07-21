# train_yolo.py
"""YOLOv11 전용 학습 스크립트"""

from ultralytics import YOLO
import argparse
from pathlib import Path
from src.utils import load_config

def convert_coco_to_yolo(config):
    """COCO 형식을 YOLO 형식으로 변환"""
    from src.data import COCODataset
    import json
    from pathlib import Path
    
    print("COCO → YOLO 형식 변환 중...")
    
    # 출력 디렉토리 생성
    yolo_dir = Path('yolo_dataset')
    yolo_dir.mkdir(exist_ok=True)
    (yolo_dir / 'images').mkdir(exist_ok=True)
    (yolo_dir / 'labels').mkdir(exist_ok=True)
    
    # 데이터셋 로드
    dataset = COCODataset(
        img_dir=config['data']['train_path'],
        ann_dir=Path(config['data']['train_path']).parent / "train_annotations",
        transform=None
    )
    
    converted_count = 0
    
    for i in range(len(dataset)):
        try:
            image, target = dataset[i]
            img_info = dataset.get_img_info(i)
            
            # 이미지 파일 복사 (심볼릭 링크로 효율적으로)
            src_img = Path(config['data']['train_path']) / img_info['file_name']
            dst_img = yolo_dir / 'images' / img_info['file_name']
            
            if not dst_img.exists():
                import shutil
                shutil.copy2(src_img, dst_img)
            
            # 어노테이션 변환
            if len(target['boxes']) > 0:
                img_width = img_info['width'] 
                img_height = img_info['height']
                
                yolo_annotations = []
                for box, label in zip(target['boxes'], target['labels']):
                    # COCO (x1, y1, x2, y2) → YOLO (x_center, y_center, width, height) 정규화
                    x1, y1, x2, y2 = box.numpy()
                    
                    x_center = (x1 + x2) / 2 / img_width
                    y_center = (y1 + y2) / 2 / img_height  
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    
                    # YOLO 클래스 ID (1-based → 0-based)
                    yolo_class = label.item() - 1
                    
                    yolo_annotations.append(f"{yolo_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                
                # TXT 파일 저장
                txt_file = yolo_dir / 'labels' / (Path(img_info['file_name']).stem + '.txt')
                with open(txt_file, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
                
                converted_count += 1
        
        except Exception as e:
            print(f"변환 실패 {i}: {e}")
    
    print(f"변환 완료: {converted_count}개 이미지")
    return str(yolo_dir)
def create_yolo_data_yaml(yolo_dir_path):
    """YOLO 형식 데이터 설정 파일 생성"""
    data_yaml = {
        'path': yolo_dir_path,
        'train': 'images',
        'val': 'images',  # 같은 데이터로 검증 (간단히)
        'nc': 4,  # 4개 알약 클래스
        'names': {
            0: '기넥신에프정',
            1: '뮤테란캡슐', 
            2: '가바토파정',
            3: '보령부스파정'
        }
    }
    
    # YAML 파일로 저장
    yaml_path = Path('yolo_data.yaml')
    import yaml
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)
    
    return str(yaml_path)

def main():
    parser = argparse.ArgumentParser(description='YOLOv11 학습')
    parser.add_argument('--config', type=str, default='configs/yolo.yaml')
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    print("=" * 50)
    print("YOLOv11 전용 학습")
    print("=" * 50)
    
    # YOLOv11 모델 로드
    version = config['model'].get('version', '11n')
    model = YOLO(f'yolo{version}.pt')
    
    # 데이터 변환
    yolo_dir = convert_coco_to_yolo(config)
    
    # 데이터 설정 파일 생성
    data_yaml = create_yolo_data_yaml(yolo_dir)
    import torch
    
    print(f"데이터 설정 파일: {data_yaml}")
    
    # 학습 실행
    results = model.train(
        data=data_yaml,
        epochs=config['training']['epochs'],
        imgsz=640,
        batch=config['training']['batch_size'],
        lr0=config['training']['learning_rate'],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        project='outputs',
        name='yolo_experiment',
        save_period=5,  # 5 에폭마다 저장
        patience=50,    # 50 에폭 동안 성능 개선 없으면 중단
    )
    
    print("YOLOv11 학습 완료!")
    print(f"결과 저장 위치: outputs/yolo_experiment")

if __name__ == '__main__':
    main()