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
        """YOLO loss 계산 - 개선된 회귀 손실"""
        device = predictions.device
        batch_size = predictions.shape[0]
        
        # 예측값을 적절한 형태로 변환
        # predictions: [batch_size, (num_classes + 5) * 3, grid_height, grid_width]
        grid_h, grid_w = predictions.shape[2], predictions.shape[3]
        num_anchors = 3
        
        # 예측값을 (batch_size, grid_h, grid_w, num_anchors, 5 + num_classes) 형태로 변환
        predictions = predictions.view(batch_size, num_anchors, 5 + self.num_classes, grid_h, grid_w)
        predictions = predictions.permute(0, 3, 4, 1, 2).contiguous()  # [B, H, W, 3, 5+C]
        
        # 예측값 분리
        pred_xy = torch.sigmoid(predictions[..., 0:2])  # center x, y
        pred_wh = predictions[..., 2:4]  # width, height
        pred_conf = torch.sigmoid(predictions[..., 4:5])  # objectness
        pred_cls = predictions[..., 5:]  # class probabilities
        
        # 손실 계산
        total_loss = 0.0
        valid_samples = 0
        
        for i, target in enumerate(targets):
            if len(target['boxes']) > 0:
                # 타겟 박스를 grid cell에 할당
                boxes = target['boxes'] / 400.0  # 정규화 (이미지 크기가 400이라고 가정)
                labels = target['labels']
                
                # 각 타겟 박스에 대해 가장 가까운 grid cell 찾기
                for box, label in zip(boxes, labels):
                    # 박스 중심점을 grid 좌표로 변환
                    center_x = (box[0] + box[2]) / 2.0
                    center_y = (box[1] + box[3]) / 2.0
                    
                    grid_x = int(center_x * grid_w)
                    grid_y = int(center_y * grid_h)
                    
                    # grid 범위 확인
                    if 0 <= grid_x < grid_w and 0 <= grid_y < grid_h:
                        # 박스 크기
                        width = box[2] - box[0]
                        height = box[3] - box[1]
                        
                        # 가장 적합한 앵커 선택 (간단히 첫 번째 앵커 사용)
                        anchor_idx = 0
                        
                        # 타겟 값 생성
                        target_xy = torch.tensor([center_x * grid_w - grid_x, center_y * grid_h - grid_y], device=device)
                        target_wh = torch.tensor([width * grid_w, height * grid_h], device=device)
                        target_conf = torch.tensor([1.0], device=device)
                        
                        # 해당 grid cell의 예측값
                        pred_xy_cell = pred_xy[i, grid_y, grid_x, anchor_idx]
                        pred_wh_cell = pred_wh[i, grid_y, grid_x, anchor_idx]
                        pred_conf_cell = pred_conf[i, grid_y, grid_x, anchor_idx]
                        
                        # 좌표 손실 (MSE)
                        xy_loss = torch.nn.functional.mse_loss(pred_xy_cell, target_xy)
                        wh_loss = torch.nn.functional.mse_loss(pred_wh_cell, target_wh)
                        
                        # objectness 손실
                        conf_loss = torch.nn.functional.binary_cross_entropy(pred_conf_cell, target_conf)
                        
                        # 클래스 손실 (간단히 0번 클래스로 설정)
                        if label < self.num_classes:
                            target_cls = torch.zeros(self.num_classes, device=device)
                            target_cls[label] = 1.0
                            pred_cls_cell = torch.softmax(pred_cls[i, grid_y, grid_x, anchor_idx], dim=0)
                            cls_loss = torch.nn.functional.cross_entropy(
                                pred_cls[i, grid_y, grid_x, anchor_idx].unsqueeze(0), 
                                label.unsqueeze(0).to(device)
                            )
                        else:
                            cls_loss = torch.tensor(0.0, device=device)
                        
                        # 총 손실
                        sample_loss = 5.0 * xy_loss + 5.0 * wh_loss + 1.0 * conf_loss + 1.0 * cls_loss
                        total_loss += sample_loss
                        valid_samples += 1
        
        # no-object 손실 (background)
        no_obj_loss = torch.nn.functional.binary_cross_entropy(
            pred_conf.view(-1), 
            torch.zeros_like(pred_conf.view(-1))
        )
        total_loss += 0.5 * no_obj_loss
        
        if valid_samples > 0:
            avg_loss = total_loss / max(valid_samples, 1)
        else:
            avg_loss = total_loss
        
        return {"loss": avg_loss}

    def transform_predictions(self, detections, input_shape):
        """예측값을 바운딩 박스로 변환"""
        batch_size = detections.shape[0]
        device = detections.device
        grid_h, grid_w = detections.shape[2], detections.shape[3]
        num_anchors = 3
        
        # 예측값을 적절한 형태로 변환
        detections = detections.view(batch_size, num_anchors, 5 + self.num_classes, grid_h, grid_w)
        detections = detections.permute(0, 3, 4, 1, 2).contiguous()  # [B, H, W, 3, 5+C]
        
        # 예측값 분리
        pred_xy = torch.sigmoid(detections[..., 0:2])  # center x, y
        pred_wh = detections[..., 2:4]  # width, height
        pred_conf = torch.sigmoid(detections[..., 4:5])  # objectness
        pred_cls = torch.softmax(detections[..., 5:], dim=-1)  # class probabilities
        
        results = []
        
        for b in range(batch_size):
            boxes_list = []
            labels_list = []
            scores_list = []
            
            for y in range(grid_h):
                for x in range(grid_w):
                    for a in range(num_anchors):
                        conf = pred_conf[b, y, x, a, 0]
                        
                        # 신뢰도 임계값 확인
                        if conf > 0.3:  # 더 높은 임계값
                            # 박스 좌표 계산
                            center_x = (x + pred_xy[b, y, x, a, 0]) / grid_w
                            center_y = (y + pred_xy[b, y, x, a, 1]) / grid_h
                            width = torch.exp(pred_wh[b, y, x, a, 0]) / grid_w
                            height = torch.exp(pred_wh[b, y, x, a, 1]) / grid_h
                            
                            # 이미지 좌표로 변환 (400x400 이미지 가정)
                            img_w, img_h = 400, 400
                            
                            x1 = (center_x - width/2) * img_w
                            y1 = (center_y - height/2) * img_h
                            x2 = (center_x + width/2) * img_w
                            y2 = (center_y + height/2) * img_h
                            
                            # 박스 좌표 클램핑
                            x1 = torch.clamp(x1, 0, img_w)
                            y1 = torch.clamp(y1, 0, img_h)
                            x2 = torch.clamp(x2, 0, img_w)
                            y2 = torch.clamp(y2, 0, img_h)
                            
                            # 유효한 박스인지 확인
                            if x2 > x1 and y2 > y1:
                                # 가장 높은 클래스 확률의 클래스 선택
                                class_conf, class_pred = torch.max(pred_cls[b, y, x, a], dim=0)
                                final_score = conf * class_conf
                                
                                if final_score > 0.1:  # 최종 신뢰도 임계값
                                    boxes_list.append([x1.item(), y1.item(), x2.item(), y2.item()])
                                    labels_list.append(class_pred.item())
                                    scores_list.append(final_score.item())
            
            if boxes_list:
                boxes = torch.tensor(boxes_list, dtype=torch.float32, device=device)
                labels = torch.tensor(labels_list, dtype=torch.long, device=device)
                scores = torch.tensor(scores_list, dtype=torch.float32, device=device)
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
