# train.py
"""모델 학습 실행"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path

from src.data import COCODataset, get_transform
from src.models import create_model
from src.training import Trainer, Evaluator
from src.utils import load_config, merge_configs, create_parser


def collate_fn(batch):
    """배치 정리 함수"""
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    return images, targets


def main():
    # 인자 파싱
    parser = create_parser()
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)

    # output 키가 없으면 기본값 설정
    if 'output' not in config:
        config['output'] = {
            'checkpoint_dir': 'outputs/checkpoints',
            'log_dir': 'outputs/logs',
            'prediction_dir': 'outputs/predictions'
        }

    #if args.model_config:
    # if args.config:
    #     model_config = load_config(args.config)
    #     config = merge_configs(config, model_config)
    
    # # 명령줄 인자로 오버라이드
    # if config['training']['batch_size']:
    #     config['training']['batch_size'] = args.batch_size
    # if args.epochs:
    #     config['training']['epochs'] = args.epochs
    # if args.device:
    #     config['device'] = args.device

    # 명령줄 인자로 환경 설정만 오버라이드
    if args.output:
        config['output']['checkpoint_dir'] = args.output + '/checkpoints'
        config['output']['log_dir'] = args.output + '/logs'
    
    if args.gpu is not None:
        config['device'] = f'cuda:{args.gpu}'
    
    # 체크포인트 재개
    if args.resume:
        config['resume'] = args.resume
    
    print("=" * 50)
    print("알약 검출 모델 학습")
    print("=" * 50)
    print(f"모델: {config['model']['name']}")
    print(f"에폭: {config['training']['epochs']}")
    print(f"배치 크기: {config['training']['batch_size']}")
    
    # 데이터셋 준비
    print("\n데이터셋 로딩...")
    train_dataset = COCODataset(
        img_dir=config['data']['train_path'],
        ann_dir=config['data']['train_path'].parent / "train_annotations",
        transform=get_transform(train=True, config=config.get('augmentation', {}))
    )
    
    # 검증 데이터셋 (간단히 학습 데이터의 일부 사용)
    val_dataset = COCODataset(
        img_dir=config['data']['train_path'],
        ann_dir=config['data']['train_path'].parent / "train_annotations",
        transform=get_transform(train=False)
    )
    
    # 데이터 로더
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=10,
        pin_memory=True,   # GPU 메모리 효율성을 위해 추가
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=10,
        pin_memory=True,   # GPU 메모리 효율성을 위해 추가
        collate_fn=collate_fn
    )
    
    # 모델 생성
    print("\n모델 생성...")
    # num_classes 먼저 계산
    if config['model']['num_classes'] == 'auto':
        num_classes = train_dataset.get_num_classes() + 1  # 배경 포함
        config['model']['num_classes'] = num_classes
    else:
        num_classes = config['model']['num_classes']
    model = create_model(
        config['model']['name'],
        num_classes=num_classes,
        backbone=config['model'].get('backbone', 'resnet50'),     
        pretrained=config['model'].get('pretrained', True)        
    )
    
    print(f"모델 생성 완료: {config['model']['name']}")
    print(f"클래스 수: {num_classes}")  # config가 아닌 계산된 값 출력

    # 학습기 생성
    trainer = Trainer(model, config, device=config.get('device', 'cuda'))
    evaluator = Evaluator(config, device=config.get('device', 'cuda'))
    
    # 학습 실행
    print("\n학습 시작!")
    trainer.train(train_loader, val_loader, evaluator)
    
    print("\n학습 완료!")
    print(f"체크포인트: {config['output']['checkpoint_dir']}")



if __name__ == '__main__':
    main()
