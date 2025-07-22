# models/base_model.py
"""기본 검출 모델 클래스"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from pathlib import Path


class BaseDetectionModel(nn.Module, ABC):
    """모든 검출 모델의 기본 클래스"""
    
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
    
    @abstractmethod
    def forward(self, images, targets=None):
        """순전파"""
        pass
    
    @abstractmethod
    def predict(self, images):
        """예측 수행"""
        pass
    
    def save(self, path):
        """모델 저장"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'num_classes': self.num_classes,
            'model_type': self.__class__.__name__
        }, path)
        print(f"모델 저장: {path}")
    
    def load(self, path, device='cpu'):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"모델 로드: {path}")
        return checkpoint
    
    def get_num_parameters(self):
        """파라미터 수 계산"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}
