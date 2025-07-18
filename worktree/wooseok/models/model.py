import torch
import torch.nn as nn
import torchvision

class PillDetector(nn.Module):
    """알약 검출을 위한 기본 모델 클래스"""
    def __init__(self, num_classes, backbone='resnet50'):
        super().__init__()
        self.num_classes = num_classes
        
        # 백본 모델 로드
        if backbone == 'resnet50':
            self.backbone = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                pretrained=True,
                min_size=800,
                max_size=1333,
                box_detections_per_img=100,
                box_nms_thresh=0.5,
                box_score_thresh=0.05
            )
            
            # 클래스 수 변경
            in_features = self.backbone.roi_heads.box_predictor.cls_score.in_features
            self.backbone.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                in_features,
                num_classes
            )
    
    def forward(self, images, targets=None):
        """
        Args:
            images (List[Tensor]): 입력 이미지들
            targets (List[Dict[Tensor]]): 학습시 정답 데이터
                - boxes (FloatTensor[N, 4]): 바운딩 박스 좌표
                - labels (Int64Tensor[N]): 레이블
        """
        if self.training and targets is None:
            raise ValueError("학습 모드에서는 targets가 필요합니다.")
            
        if self.training:
            return self.backbone(images, targets)
        
        return self.backbone(images)
    
def create_model(cfg):
    """설정에 따라 모델 생성"""
    model = PillDetector(
        num_classes=cfg.NUM_CLASSES,
        backbone=cfg.BACKBONE
    )
    return model
