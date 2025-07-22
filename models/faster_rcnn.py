# models/faster_rcnn.py

def create_faster_rcnn_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# models/faster_rcnn.py
"""Faster R-CNN 모델 구현"""

import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import configs.base_config as configs

from .base_model import BaseDetectionModel


class FasterRCNNModel(BaseDetectionModel):
    """Faster R-CNN 모델"""
    
    def __init__(self, num_classes, backbone='resnet50', pretrained=True):
        super().__init__(num_classes)

        # 백본에 따라 모델 생성
        if backbone == 'resnet50':
            # self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            #     pretrained=pretrained
            if pretrained:
                weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            else:
                weights = None

            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights=weights
            )
        else:
            # 커스텀 백본
            backbone_model = self._create_backbone(backbone, pretrained)
            anchor_generator = self._create_anchor_generator(configs)
            self.model = FasterRCNN(
                backbone_model,
                num_classes=num_classes,
                rpn_anchor_generator=anchor_generator
            )

        # 분류기 헤드 교체
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes
        )

        print(f"Faster R-CNN 생성 완료 - 백본: {backbone}, 클래스: {num_classes}")
    
    def _create_backbone(self, backbone_name, pretrained):
        """백본 네트워크 생성"""
        # 간단한 예시 - 실제로는 더 복잡함
        from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
        return resnet_fpn_backbone(backbone_name, pretrained)
    
    def _create_anchor_generator(self, config):
        """앵커 생성기 구성"""
        anchor_sizes = config.get('anchor_sizes', ((32,), (64,), (128,), (256,), (512,)))
        aspect_ratios = config.get('aspect_ratios', ((0.5, 1.0, 2.0),)) * len(anchor_sizes)

        return AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios
        )
    
    def forward(self, images, targets=None):
        """순전파"""
        return self.model(images, targets)
    
    def predict(self, images):
        """예측 모드"""
        self.eval()
        with torch.no_grad():
            predictions = self.model(images)
        return predictions
    
    def postprocess(self, predictions, score_threshold=0.5, nms_threshold=0.5):
        """후처리 - NMS 및 필터링"""
        processed = []
        
        for pred in predictions:
            # 점수 필터링
            keep = pred['scores'] > score_threshold
            
            boxes = pred['boxes'][keep]
            labels = pred['labels'][keep]
            scores = pred['scores'][keep]
            
            # NMS 적용
            keep_nms = torchvision.ops.nms(boxes, scores, nms_threshold)
            
            processed.append({
                'boxes': boxes[keep_nms],
                'labels': labels[keep_nms],
                'scores': scores[keep_nms]
            })
        
        return processed