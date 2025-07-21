# src/data/transforms.py
"""최소한의 데이터 변환 - 단계별 테스트용"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch


class PillTransform:
    """알약 검출을 위한 변환 클래스 - 최소한 버전"""
    
    def __init__(self, train=True, config=None):
        self.train = train
        self.config = config or {}
        self.transform = self._build_transform()
    
    def _build_transform(self):
        """가장 기본적인 변환만"""
        transforms = [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        # ToTensorV2도 제거하고 수동으로 처리
        return A.Compose(transforms)
    
    def __call__(self, image, target):
        """변환 적용 - 최소한 처리"""
        # 빈 target 처리
        if not target or 'boxes' not in target:
            boxes = []
            labels = []
        else:
            boxes = target['boxes']
            labels = target['labels']
        
        # 기본 변환 적용
        transformed = self.transform(image=image)
        
        # 수동으로 텐서 변환
        image = torch.from_numpy(transformed['image'].transpose(2, 0, 1)).float()
        
        # boxes가 이미 tensor라면 그대로, 아니면 변환
        if not torch.is_tensor(target.get('boxes', [])):
            if len(boxes) > 0:
                target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
                target['labels'] = torch.tensor(labels, dtype=torch.int64)
            else:
                target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['labels'] = torch.zeros((0,), dtype=torch.int64)
        
        return image, target


def get_transform(train=True, config=None):
    """변환 객체 생성 헬퍼 함수"""
    return PillTransform(train=train, config=config)