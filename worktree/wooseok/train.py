import torch
from pathlib import Path
from dataset import PillDetectionDataset
from models.yolo_model import create_model  # YOLOv3 모델 사용
from trainer import Trainer
from config import cfg
from torch.utils.data import DataLoader, random_split

def collate_fn(batch):
    return tuple(zip(*batch))

def main(dataset=None):
    # 데이터셋 로드 (외부에서 전달받거나 기본값 사용)
    if dataset is None:
        dataset = PillDetectionDataset(
            json_file=cfg.TRAIN_JSON,
            img_dir=cfg.TRAIN_IMG_DIR
        )
    
    print(f"Using dataset with {len(dataset)} samples")
    
    # 학습/검증 데이터 분할
    val_size = int(len(dataset) * cfg.VAL_RATIO)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn
    )
    
    # 모델 생성
    model = create_model(cfg)
    model = model.to(cfg.DEVICE)
    
    # 트레이너 생성 및 학습
    trainer = Trainer(model, train_loader, val_loader, cfg)
    trainer.train()

if __name__ == "__main__":
    # CUDA 사용 가능 여부 확인
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
        cfg.DEVICE = 'cpu'
    
    main()

# Colab에서 데이터셋 객체와 함께 학습을 실행하는 함수
def train_with_dataset(dataset):
    """
    Colab에서 생성된 데이터셋 객체를 사용하여 학습을 실행합니다.
    
    Args:
        dataset: PillDetectionDataset 객체
    """
    # CUDA 사용 가능 여부 확인
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
        cfg.DEVICE = 'cpu'
    
    # 데이터셋 객체와 함께 학습 시작
    main(dataset)
