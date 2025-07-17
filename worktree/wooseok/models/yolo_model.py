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
            # 학습 모드에서는 실제 loss 계산
            return self.compute_loss(detections, targets)
        else:
            # 추론 모드에서는 예측값 반환
            return self.transform_predictions(detections, images.shape)

    def compute_loss(self, predictions, targets):
        """YOLO loss 계산 - 간단한 회귀 손실"""
        device = predictions.device
        batch_size = predictions.shape[0]
        
        # 예측값을 적절한 형태로 변환
        # predictions: [batch_size, (num_classes + 5) * 3, grid_height, grid_width]
        pred_reshaped = predictions.view(batch_size, -1)
        
        # 타겟에서 박스와 레이블 정보 추출
        total_loss = 0.0
        valid_samples = 0
        
        for i, target in enumerate(targets):
            if len(target['boxes']) > 0:
                # 박스 좌표 정규화 (이미지 크기로 나누기)
                boxes = target['boxes']
                labels = target['labels']
                
                # 간단한 회귀 손실 계산
                # 실제로는 grid cell assignment, objectness, class prediction 등 복잡한 계산 필요
                
                # 예측값의 일부를 사용하여 박스 좌표 예측
                pred_sample = pred_reshaped[i]
                
                # 박스 좌표를 평균내어 단일 값으로 만들기
                box_center_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
                box_center_y = (boxes[:, 1] + boxes[:, 3]) / 2.0
                
                # 타겟 값 (정규화)
                target_val = (box_center_x.mean() + box_center_y.mean()) / 400.0  # 대략적인 정규화
                
                # 예측값의 평균과 타겟 값 간의 MSE 손실
                pred_val = pred_sample.mean()
                loss = torch.nn.functional.mse_loss(pred_val, target_val.to(device))
                
                total_loss += loss
                valid_samples += 1
        
        if valid_samples > 0:
            avg_loss = total_loss / valid_samples
        else:
            avg_loss = torch.tensor(0.0, requires_grad=True, device=device)
        
        return {"loss": avg_loss}

    def transform_predictions(self, detections, input_shape):
        """예측값을 바운딩 박스로 변환"""
        batch_size = detections.shape[0]
        device = detections.device
        
        results = []
        
        for b in range(batch_size):
            # 예측값에서 박스 정보 추출 (간단한 구현)
            pred_flat = detections[b].view(-1)
            
            # 예측값에서 몇 개의 박스 생성 (실제로는 NMS 등 후처리 필요)
            num_predictions = min(5, pred_flat.shape[0] // 6)  # 최대 5개 박스
            
            if num_predictions > 0:
                # 예측값에서 박스 좌표 추출
                boxes_list = []
                labels_list = []
                scores_list = []
                
                for i in range(num_predictions):
                    start_idx = i * 6
                    if start_idx + 5 < pred_flat.shape[0]:
                        # 박스 좌표 (x1, y1, x2, y2)
                        x1 = torch.clamp(pred_flat[start_idx] * 200, 0, 400)
                        y1 = torch.clamp(pred_flat[start_idx + 1] * 200, 0, 400)
                        x2 = torch.clamp(pred_flat[start_idx + 2] * 200 + x1, x1 + 10, 400)
                        y2 = torch.clamp(pred_flat[start_idx + 3] * 200 + y1, y1 + 10, 400)
                        
                        # 신뢰도 점수
                        score = torch.sigmoid(pred_flat[start_idx + 4])
                        
                        # 클래스 (간단히 0으로 설정)
                        label = 0
                        
                        if score > 0.1:  # 최소 신뢰도 임계값
                            boxes_list.append([x1.item(), y1.item(), x2.item(), y2.item()])
                            labels_list.append(label)
                            scores_list.append(score.item())
                
                if boxes_list:
                    boxes = torch.tensor(boxes_list, dtype=torch.float32, device=device)
                    labels = torch.tensor(labels_list, dtype=torch.long, device=device)
                    scores = torch.tensor(scores_list, dtype=torch.float32, device=device)
                else:
                    # 빈 예측
                    boxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
                    labels = torch.zeros((0,), dtype=torch.long, device=device)
                    scores = torch.zeros((0,), dtype=torch.float32, device=device)
            else:
                # 빈 예측
                boxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
                labels = torch.zeros((0,), dtype=torch.long, device=device)
                scores = torch.zeros((0,), dtype=torch.float32, device=device)
            
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
