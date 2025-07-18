import torch
import torch.nn as nn
import torchvision.models as models

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
    def __init__(self, num_classes, freeze_backbone_layers=3):
        super().__init__()
        self.num_classes = num_classes

        # ResNet-50 사전 훈련된 백본 사용
        resnet = models.resnet50(pretrained=True)
        
        # ResNet의 마지막 두 레이어 제거 (avgpool, fc)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # 백본의 초기 레이어들 고정 (선택적)
        if freeze_backbone_layers > 0:
            layers_to_freeze = list(self.backbone.children())[:freeze_backbone_layers]
            for layer in layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
            print(f"Frozen first {freeze_backbone_layers} layers of ResNet backbone")
        
        # ResNet-50의 출력 채널 수는 2048
        backbone_out_channels = 2048
        
        # Feature Pyramid Network (FPN) 스타일 neck 추가
        self.neck = nn.Sequential(
            ConvBlock(backbone_out_channels, 1024, kernel_size=1),
            ConvBlock(1024, 512, kernel_size=3),
            ConvBlock(512, 1024, kernel_size=1),
            ConvBlock(1024, 512, kernel_size=3)
        )

        # YOLO 디텍션 헤드
        self.head = nn.Sequential(
            ConvBlock(512, 256, kernel_size=1),
            ConvBlock(256, 512, kernel_size=3),
            ConvBlock(512, 256, kernel_size=1),
            ConvBlock(256, 512, kernel_size=3),
            ConvBlock(512, 256, kernel_size=1),
            nn.Conv2d(256, (num_classes + 5) * 3, kernel_size=1)  # 3개의 앵커박스
        )

    def _make_layer(self, channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResBlock(channels))
        return nn.Sequential(*layers)

    def forward(self, images, targets=None):
        if isinstance(images, list):
            images = torch.stack(images)
            
        # ResNet backbone을 통한 특징 추출
        backbone_features = self.backbone(images)
        
        # Neck을 통한 특징 정제
        neck_features = self.neck(backbone_features)
        
        # Detection head를 통한 최종 예측
        detections = self.head(neck_features)
        
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
        pred_wh = predictions[..., 2:4]  # width, height (log space)
        pred_conf = torch.sigmoid(predictions[..., 4:5])  # objectness
        pred_cls = predictions[..., 5:]  # class probabilities
        
        # 손실 초기화
        coord_loss = 0.0
        conf_loss = 0.0
        cls_loss = 0.0
        valid_targets = 0
        
        # 배치별 처리
        for b in range(batch_size):
            target = targets[b]
            
            if len(target['boxes']) == 0:
                continue
                
            # 타겟 박스 정규화 (0-1 범위로)
            boxes = target['boxes'].float() / 400.0  # 이미지 크기로 정규화
            labels = target['labels']
            
            # 각 타겟 박스에 대해 처리
            for box, label in zip(boxes, labels):
                # 박스 중심점과 크기 계산
                x_center = (box[0] + box[2]) / 2.0
                y_center = (box[1] + box[3]) / 2.0
                width = box[2] - box[0]
                height = box[3] - box[1]
                
                # Grid cell 계산
                grid_x = int(x_center * grid_w)
                grid_y = int(y_center * grid_h)
                
                # Grid 범위 체크
                grid_x = max(0, min(grid_x, grid_w - 1))
                grid_y = max(0, min(grid_y, grid_h - 1))
                
                # 가장 적합한 앵커 선택 (첫 번째 앵커 사용)
                anchor_idx = 0
                
                # 타겟 좌표 (grid cell 내 상대 좌표)
                target_x = x_center * grid_w - grid_x
                target_y = y_center * grid_h - grid_y
                target_w = torch.log(width * grid_w + 1e-8)  # log space, epsilon 추가
                target_h = torch.log(height * grid_h + 1e-8)
                
                # 해당 grid cell의 예측값
                pred_x = pred_xy[b, grid_y, grid_x, anchor_idx, 0]
                pred_y = pred_xy[b, grid_y, grid_x, anchor_idx, 1]
                pred_w = pred_wh[b, grid_y, grid_x, anchor_idx, 0]
                pred_h = pred_wh[b, grid_y, grid_x, anchor_idx, 1]
                pred_conf_val = pred_conf[b, grid_y, grid_x, anchor_idx, 0]
                
                # 좌표 손실 (MSE)
                coord_loss += torch.nn.functional.mse_loss(pred_x, torch.tensor(target_x, device=device))
                coord_loss += torch.nn.functional.mse_loss(pred_y, torch.tensor(target_y, device=device))
                coord_loss += torch.nn.functional.mse_loss(pred_w, target_w.to(device))
                coord_loss += torch.nn.functional.mse_loss(pred_h, target_h.to(device))
                
                # Objectness 손실 (해당 위치에 객체가 있음)
                conf_loss += torch.nn.functional.binary_cross_entropy(
                    pred_conf_val, torch.tensor(1.0, device=device)
                )
                
                # 클래스 손실
                if label < self.num_classes:
                    pred_cls_val = pred_cls[b, grid_y, grid_x, anchor_idx]
                    cls_loss += torch.nn.functional.cross_entropy(
                        pred_cls_val.unsqueeze(0), 
                        label.unsqueeze(0).to(device)
                    )
                
                valid_targets += 1
        
        # No-object 손실 (배경에 대한 confidence 억제)
        # 타겟이 할당되지 않은 모든 위치에서 confidence를 0으로 만들기
        no_obj_mask = torch.ones_like(pred_conf, device=device)
        
        # 타겟이 있는 위치는 마스크에서 제외
        for b in range(batch_size):
            target = targets[b]
            if len(target['boxes']) > 0:
                boxes = target['boxes'].float() / 400.0
                for box in boxes:
                    x_center = (box[0] + box[2]) / 2.0
                    y_center = (box[1] + box[3]) / 2.0
                    grid_x = int(x_center * grid_w)
                    grid_y = int(y_center * grid_h)
                    grid_x = max(0, min(grid_x, grid_w - 1))
                    grid_y = max(0, min(grid_y, grid_h - 1))
                    no_obj_mask[b, grid_y, grid_x, 0, 0] = 0  # 첫 번째 앵커만 사용
        
        no_obj_conf = pred_conf * no_obj_mask
        no_obj_loss = torch.nn.functional.binary_cross_entropy(
            no_obj_conf, torch.zeros_like(no_obj_conf)
        )
        
        # 전체 손실 계산
        if valid_targets > 0:
            coord_loss = coord_loss / valid_targets
            conf_loss = conf_loss / valid_targets
            cls_loss = cls_loss / valid_targets
        
        # 가중치 적용
        total_loss = 5.0 * coord_loss + 1.0 * conf_loss + 1.0 * cls_loss + 0.5 * no_obj_loss
        
        return {"loss": total_loss}

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
        pred_wh = detections[..., 2:4]  # width, height (log space)
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
                        
                        # 더 낮은 신뢰도 임계값으로 변경
                        if conf > 0.05:  # 0.3에서 0.05로 낮춤
                            # 박스 좌표 계산
                            center_x = (x + pred_xy[b, y, x, a, 0]) / grid_w
                            center_y = (y + pred_xy[b, y, x, a, 1]) / grid_h
                            
                            # width, height를 log space에서 변환
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
                            if x2 > x1 + 1 and y2 > y1 + 1:  # 최소 크기 확인
                                # 가장 높은 클래스 확률의 클래스 선택
                                class_conf, class_pred = torch.max(pred_cls[b, y, x, a], dim=0)
                                final_score = conf * class_conf
                                
                                # 최종 신뢰도 임계값도 낮춤
                                if final_score > 0.01:  # 0.1에서 0.01로 낮춤
                                    boxes_list.append([x1.item(), y1.item(), x2.item(), y2.item()])
                                    labels_list.append(class_pred.item())
                                    scores_list.append(final_score.item())
            
            if boxes_list:
                boxes = torch.tensor(boxes_list, dtype=torch.float32, device=device)
                labels = torch.tensor(labels_list, dtype=torch.long, device=device)
                scores = torch.tensor(scores_list, dtype=torch.float32, device=device)
                
                # NMS 적용하여 중복 박스 제거
                from torchvision.ops import nms
                keep_indices = nms(boxes, scores, iou_threshold=0.5)
                
                boxes = boxes[keep_indices]
                labels = labels[keep_indices]
                scores = scores[keep_indices]
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
    # 사전 훈련된 ResNet 백본을 사용한 YOLO 모델 생성
    freeze_layers = getattr(cfg, 'FREEZE_BACKBONE_LAYERS', 3)  # 기본값 3
    model = YOLOv3(num_classes=cfg.NUM_CLASSES, freeze_backbone_layers=freeze_layers)
    
    print(f"Created YOLO model with ResNet-50 backbone")
    print(f"Number of classes: {cfg.NUM_CLASSES}")
    print(f"Frozen backbone layers: {freeze_layers}")
    
    return model
