# src/models/__init__.py
from .base_model import BaseDetectionModel
from .faster_rcnn import FasterRCNNModel


MODEL_REGISTRY = {
    'faster_rcnn': FasterRCNNModel,
    'yolo': None,
    'dino': None,
}


def create_model(model_name, num_classes, **kwargs):
    """모델 생성 팩토리 함수"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_class = MODEL_REGISTRY[model_name]
    return model_class(num_classes=num_classes, **kwargs)
