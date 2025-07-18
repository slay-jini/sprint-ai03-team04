# src/models/yolo.py
"""YOLOv11 모델 구현 (평가/예측 전용)"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from .base_model import BaseDetectionModel
from pathlib import Path


class YOLOModel(BaseDetectionModel):
    """YOLOv11 모델 (평가/예측 전용)"""
    
    def __init__(self, num_classes, version='11n', pretrained=True, **kwargs):
        super().__init__(num_classes)
        self.version = version
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # YOLOv11 모델 로드
        if isinstance(version, str) and version.startswith('11'):
            model_name = f'yolo{version}.pt'
        else:
            model_name = f'yolo11{version}.pt'
            
        self.model = YOLO(model_name)
        
        # 평가/예측용이므로 클래스 수 설정은 로드된 모델 그대로
        print(f"YOLOv11{version} 로드 완료 - 평가/예측 모드")
    
    @classmethod
    def load_pretrained(cls, model_path, **kwargs):
        """학습된 YOLOv11 모델 로드"""
        instance = cls.__new__(cls)
        instance.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 학습된 모델 로드
        instance.model = YOLO(model_path)
        
        # num_classes는 모델에서 자동 추출
        try:
            if hasattr(instance.model.model, 'model') and len(instance.model.model.model) > 0:
                instance.num_classes = instance.model.model.model[-1].nc + 1  # 배경 포함
            else:
                instance.num_classes = 5  # 기본값
        except:
            instance.num_classes = 5  # 기본값
            
        print(f"학습된 YOLOv11 모델 로드 완료: {model_path}")
        return instance
    
    def forward(self, images, targets=None):
        """순전파 (추론 전용)"""
        # 학습 모드는 지원하지 않음 (train_yolo.py 사용)
        if self.training and targets is not None:
            raise NotImplementedError("YOLOv11 학습은 train_yolo.py를 사용하세요")
        
        # 추론 모드
        self.model.eval()
        return self._inference_forward(images)
    
    def _inference_forward(self, images):
        """추론시 순전파"""
        if isinstance(images, list):
            # 리스트 처리
            results = self.model(images, verbose=False)
        else:
            # Tensor인 경우 numpy로 변환
            if images.dim() == 4:  # 배치
                # GPU 텐서를 CPU로 이동 후 numpy 변환
                images_np = images.cpu().numpy()
                # (B, C, H, W) → (B, H, W, C)
                images_np = images_np.transpose(0, 2, 3, 1)
                # 0-1 범위를 0-255로 변환
                if images_np.max() <= 1.0:
                    images_np = (images_np * 255).astype('uint8')
                results = self.model(images_np, verbose=False)
            else:
                # 단일 이미지
                img_np = images.cpu().numpy().transpose(1, 2, 0)
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype('uint8')
                results = self.model([img_np], verbose=False)
        
        return self._format_outputs(results)
    
    def _format_outputs(self, results):
        """YOLOv11 출력을 Faster R-CNN 형식으로 변환"""
        predictions = []
        
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy  # x1, y1, x2, y2 형식
                scores = result.boxes.conf
                labels = result.boxes.cls.int() + 1  # 0-based를 1-based로 변환 (배경=0)
                
                predictions.append({
                    'boxes': boxes.cpu(),
                    'labels': labels.cpu(),
                    'scores': scores.cpu()
                })
            else:
                # 검출된 객체가 없는 경우
                predictions.append({
                    'boxes': torch.zeros((0, 4)),
                    'labels': torch.zeros((0,), dtype=torch.int64),
                    'scores': torch.zeros((0,))
                })
        
        return predictions
    
    def predict(self, images, conf_threshold=0.25, nms_threshold=0.45):
        """예측 수행 (신뢰도 임계값 지원)"""
        self.eval()
        
        # YOLO 모델에 임계값 설정
        self.model.conf = conf_threshold
        self.model.iou = nms_threshold
        
        with torch.no_grad():
            return self.forward(images)
    
    def to(self, device):
        """디바이스 이동"""
        self.device = device
        try:
            self.model.to(device)
        except:
            # YOLO 모델은 자동으로 디바이스 관리
            pass
        return self
    
    def train(self, mode=True):
        """학습 모드 설정 (평가 전용 모델이므로 경고)"""
        if mode:
            print("⚠️ 이 YOLOv11 모델은 평가/예측 전용입니다. 학습은 train_yolo.py를 사용하세요.")
        self.training = False  # 항상 False
        return self
    
    def eval(self):
        """평가 모드 설정"""
        self.training = False
        if hasattr(self.model, 'eval'):
            self.model.eval()
        return self
    
    def state_dict(self):
        """모델 상태 딕셔너리 (YOLO는 자체 저장 방식 사용)"""
        print("⚠️ YOLOv11는 자체 저장 방식을 사용합니다. model.save() 또는 model.export() 사용을 권장합니다.")
        return {}
    
    def load_state_dict(self, state_dict):
        """모델 상태 로드 (YOLO는 자체 로딩 방식 사용)"""
        print("⚠️ YOLOv11는 자체 로딩 방식을 사용합니다. YOLO.load() 사용을 권장합니다.")
        return
    
    def parameters(self):
        """모델 파라미터 (평가 전용이므로 빈 iterator 반환)"""
        # 평가/예측 전용이므로 옵티마이저가 필요하지 않음
        return iter([])


def create_yolo_model(num_classes, version='11n', pretrained=True, **kwargs):
    """YOLOv11 모델 생성 함수"""
    return YOLOModel(num_classes, version, pretrained, **kwargs)


def load_yolo_model(model_path, **kwargs):
    """학습된 YOLOv11 모델 로드 함수"""
    return YOLOModel.load_pretrained(model_path, **kwargs)