# src/data/transforms.py
"""데이터 증강 및 변환"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch


class PillTransform:
    """알약 검출을 위한 변환 클래스"""
    
    def __init__(self, train=True, config=None):
        self.train = train
        self.config = config or {}
        self.transform = self._build_transform()
    
    def _build_transform(self):
        """변환 파이프라인 구성"""
        transforms = []
        has_bbox_transform = False
        
        if self.train:
            # 학습용 증강
            #if self.config.get('horizontal_flip', True):
            #    transforms.append(A.HorizontalFlip(p=0.5))
            # 기본적으로 HorizontalFlip은 항상 포함 (경고 방지)
            transforms.append(A.HorizontalFlip(p=0.5 if self.config.get('horizontal_flip', True) else 0.0))
            has_bbox_transform = True
            
            if self.config.get('vertical_flip', False):
                transforms.append(A.VerticalFlip(p=0.5))
                has_bbox_transform = True
            
            if self.config.get('rotation_range', 0) > 0:
                transforms.append(
                    A.Rotate(limit=self.config['rotation_range'], p=0.5)
                )
                has_bbox_transform = True
            
            # 밝기/대비 조정
            brightness = self.config.get('brightness_range', [1.0, 1.0])
            if brightness[0] != 1.0 or brightness[1] != 1.0:
                transforms.append(
                    A.RandomBrightnessContrast(
                        brightness_limit=(brightness[0]-1, brightness[1]-1),
                        p=0.5
                    )
                )
        
        # 정규화
        transforms.append(
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
        transforms.append(ToTensorV2())
         # bbox 변환이 있을 때만 bbox_params 설정
        if has_bbox_transform:
            return A.Compose(
                transforms,
                bbox_params=A.BboxParams(
                    format='pascal_voc',
                    label_fields=['labels'],
                    min_visibility=0.1 # 0.3 -> 0.1로 완화
                )
            )
        else:
            return A.Compose(transforms)
    
    def __call__(self, image, target):
        """변환 적용"""
        # 예측 시 target이 빈 딕셔너리일 경우 처리
        if not target or 'boxes' not in target:
            # 예측 시: target이 비어있거나 boxes가 없는 경우
            boxes = []
            labels = []
        else:
            # 학습 시: 정상적인 boxes와 labels 처리
            boxes = target['boxes'].numpy() if torch.is_tensor(target['boxes']) else target['boxes']
            labels = target['labels'].numpy() if torch.is_tensor(target['labels']) else target['labels']
        
        # 변환 적용
        transformed = self.transform(
            image=image,
            bboxes=boxes,
            labels=labels
        )
        
        # 다시 PyTorch 형식으로
        image = transformed['image']
        
        if transformed['bboxes'] is not None and len(transformed['bboxes']) > 0:
            target['boxes'] = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            target['labels'] = torch.tensor(transformed['labels'], dtype=torch.int64)
        else:
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.int64)
        
        return image, target


def get_transform(train=True, config=None):
    """변환 객체 생성 헬퍼 함수"""
    return PillTransform(train=train, config=config)
