import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels//2, kernel_size=1),
            ConvBlock(channels//2, channels, kernel_size=3)
        )

    def forward(self, x):
        return x + self.block(x)

class YOLOv3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        # Darknet-53 백본
        self.backbone = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64, stride=2),
            self._make_layer(64, 1),
            ConvBlock(64, 128, stride=2),
            self._make_layer(128, 2),
            ConvBlock(128, 256, stride=2),
            self._make_layer(256, 8),
            ConvBlock(256, 512, stride=2),
            self._make_layer(512, 8),
            ConvBlock(512, 1024, stride=2),
            self._make_layer(1024, 4),
        )

        # YOLO 디텍션 헤드
        self.head = nn.Sequential(
            ConvBlock(1024, 512, kernel_size=1),
            ConvBlock(512, 1024, kernel_size=3),
            ConvBlock(1024, 512, kernel_size=1),
            ConvBlock(512, 1024, kernel_size=3),
            ConvBlock(1024, 512, kernel_size=1),
            nn.Conv2d(512, (num_classes + 5) * 3, kernel_size=1)  # 3개의 앵커박스
        )

    def _make_layer(self, channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResBlock(channels))
        return nn.Sequential(*layers)

    def forward(self, images, targets=None):
        if isinstance(images, list):
            images = torch.stack(images)
            
        features = self.backbone(images)
        detections = self.head(features)
        
        if self.training and targets is not None:
            # 학습 모드에서는 loss 반환
            return {"loss": torch.tensor(1.0, requires_grad=True, device=images.device)}
        else:
            # 추론 모드에서는 예측값 반환
            return self.transform_predictions(detections, images.shape)

    def compute_loss(self, predictions, targets):
        """YOLOv3 loss 계산"""
        # 임시 구현 - 실제로는 YOLO loss (좌표, 신뢰도, 클래스 loss) 계산
        device = predictions.device
        dummy_loss = torch.tensor(1.0, requires_grad=True, device=device)
        return {"loss": dummy_loss}

    def transform_predictions(self, detections, input_shape):
        """예측값을 바운딩 박스로 변환"""
        batch_size = detections.shape[0]
        grid_size = detections.shape[-1]  # 그리드 크기
        
        results = []
        
        for b in range(batch_size):
            # 더미 예측값 생성 (실제 구현에서는 NMS와 threshold 적용)
            boxes = torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]], 
                               dtype=torch.float32, device=detections.device)
            labels = torch.tensor([0, 1], dtype=torch.long, device=detections.device)
            scores = torch.tensor([0.9, 0.8], dtype=torch.float32, device=detections.device)
            
            results.append({
                'boxes': boxes,
                'labels': labels,
                'scores': scores
            })
        
        return results

def create_model(cfg):
    """설정에 따라 YOLOv3 모델 생성"""
    model = YOLOv3(num_classes=cfg.NUM_CLASSES)
    return model
