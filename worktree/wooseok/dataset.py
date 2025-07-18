import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from pathlib import Path

class PillDetectionDataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None):
        """
        Args:
            json_file (str): COCO format annotation file path
            img_dir (str): Images directory path
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.img_dir = Path(img_dir)
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        # Create image_id to annotations mapping
        self.img_to_anns = {}
        for ann in self.data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
            
        # Create category_id to index mapping
        self.cat_ids = {cat['id']: idx for idx, cat in enumerate(self.data['categories'])}
        self.num_classes = len(self.cat_ids)
        
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data['images'])

    def __getitem__(self, idx):
        # Get image info
        img_info = self.data['images'][idx]
        img_path = self.img_dir / img_info['file_name']
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        
        # Get annotations
        img_id = img_info['id']
        boxes = []
        labels = []
        
        if img_id in self.img_to_anns:
            for ann in self.img_to_anns[img_id]:
                # Get bbox coordinates
                x, y, w, h = ann['bbox']
                # Convert to [x_min, y_min, x_max, y_max] format
                boxes.append([x, y, x + w, y + h])
                # Convert category_id to index
                labels.append(self.cat_ids[ann['category_id']])
        
        # Convert to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Prepare target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'orig_size': torch.as_tensor([orig_h, orig_w])
        }
        
        # Apply transformations
        image = self.transform(image)
        
        return image, target

if __name__ == "__main__":
    # 데이터셋 생성
    train_dataset = PillDetectionDataset(
        json_file='dataset/train.json',
        img_dir='dataset/images/train'
    )
    
    # DataLoader 생성
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x))  # 가변 크기 박스 처리를 위한 collate 함수
    )
    
    # 데이터셋 테스트
    for images, targets in train_loader:
        print("배치 크기:", len(images))
        print("이미지 크기:", images[0].shape)
        print("박스 수:", len(targets[0]['boxes']))
        print("레이블:", targets[0]['labels'])
        break

# Colab에서 사용할 수 있는 간단한 데이터셋 생성 함수
def create_colab_dataset(json_path, img_dir):
    """
    Colab에서 쉽게 데이터셋을 생성할 수 있는 함수
    
    Args:
        json_path (str): COCO format annotation file path
        img_dir (str): Images directory path
    
    Returns:
        PillDetectionDataset: 생성된 데이터셋 객체
    """
    try:
        dataset = PillDetectionDataset(json_file=json_path, img_dir=img_dir)
        print(f"✅ 데이터셋 생성 성공: {len(dataset)} 개의 이미지")
        return dataset
    except Exception as e:
        print(f"❌ 데이터셋 생성 실패: {e}")
        return None
