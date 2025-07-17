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
        
        if self.training:
            if targets is None:
                raise ValueError("학습 모드에서는 targets가 필요합니다.")
            return {"loss": torch.tensor(1.0, requires_grad=True)}  # 임시 loss
        
        return self.transform_predictions(detections)

    def compute_loss(self, predictions, targets):
        # YOLOv3 loss 계산
        # TODO: Implement YOLO loss computation
        pass

    def transform_predictions(self, detections):
        # 예측값을 바운딩 박스로 변환
        # TODO: Implement prediction transformation
        pass

def create_model(cfg):
    """설정에 따라 YOLOv3 모델 생성"""
    model = YOLOv3(num_classes=cfg.NUM_CLASSES)
    return model
