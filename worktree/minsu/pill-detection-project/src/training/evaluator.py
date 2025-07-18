# src/training/evaluator.py
"""모델 평가 및 mAP 계산"""

import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm


class Evaluator:
    """모델 평가 클래스"""
    
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
    def evaluate(self, model, data_loader):
        """모델 평가 수행"""
        model.eval()
        
        # config에서 threshold 가져오기
        score_threshold = self.config.get('inference', {}).get('score_threshold', 0.1)

        # 예측 및 정답 수집
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in tqdm(data_loader, desc="평가 중"):
                # GPU로 이동
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) if hasattr(v, 'to') else v 
                            for k, v in t.items()} for t in targets]
                
                # 예측
                predictions = model(images)
                
                # CPU로 이동 및 저장
                for pred, target in zip(predictions, targets):
                    all_predictions.append({
                        'boxes': pred['boxes'].cpu(),
                        'labels': pred['labels'].cpu(),
                        'scores': pred['scores'].cpu()
                    })
                    all_targets.append({
                        'boxes': target['boxes'].cpu(),
                        'labels': target['labels'].cpu()
                    })
        
        # mAP 계산
        results = calculate_map(
            all_predictions,
            all_targets,
            iou_thresholds=[0.5, 0.75],
            num_classes=self.config['model']['num_classes'] - 1,
            score_threshold=score_threshold
        )
        
        return results


def calculate_map(predictions, targets, iou_thresholds=[0.5], num_classes=4, score_threshold=0.1):
    """mAP (mean Average Precision) 계산"""
    # 클래스별, IoU 임계값별 AP 저장
    ap_per_class_per_iou = defaultdict(lambda: defaultdict(float))
    
    # 각 클래스별로 계산
    for class_id in range(1, num_classes + 1):
        for iou_threshold in iou_thresholds:
            ap = calculate_ap_for_class(
                predictions, targets, class_id, iou_threshold, score_threshold
            )
            ap_per_class_per_iou[class_id][iou_threshold] = ap
    
    # 전체 mAP 계산
    mAP = np.mean([
        ap_per_class_per_iou[c][0.5] 
        for c in range(1, num_classes + 1)
    ])
    
    mAP_50 = mAP
    mAP_75 = np.mean([
        ap_per_class_per_iou[c][0.75] 
        for c in range(1, num_classes + 1)
    ]) if 0.75 in iou_thresholds else 0.0
    
    # 클래스별 AP (IoU=0.5)
    ap_per_class = {
        c: ap_per_class_per_iou[c][0.5] 
        for c in range(1, num_classes + 1)
    }
    
    return {
        'mAP': mAP,
        'mAP_50': mAP_50,
        'mAP_75': mAP_75,
        'ap_per_class': ap_per_class,
        'ap_per_class_per_iou': dict(ap_per_class_per_iou)
    }


def calculate_ap_for_class(predictions, targets, class_id, iou_threshold, score_threshold):
    """특정 클래스에 대한 AP 계산"""
    class_preds = []
    class_targets = []
    
    for pred, target in zip(predictions, targets):
        # 예측에서 해당 클래스 + score threshold 적용
        mask = (pred['labels'] == class_id) & (pred['scores'] > score_threshold)
        if mask.sum() > 0:
            indices = torch.where(mask)[0]
            class_preds.extend([
                {
                    'box': pred['boxes'][i].numpy(),
                    'score': pred['scores'][i].item()
                }
                for i in indices
            ])
        
        # 정답에서 해당 클래스
        mask = target['labels'] == class_id
        if mask.sum() > 0:
            indices = torch.where(mask)[0]
            class_targets.extend([
                {'box': target['boxes'][i].numpy()}
                for i in indices
            ])
    
    if len(class_targets) == 0:
        return 0.0 if len(class_preds) > 0 else 1.0
    
    if len(class_preds) == 0:
        return 0.0
    
    # 점수 순으로 정렬
    class_preds.sort(key=lambda x: x['score'], reverse=True)
    
    # TP/FP 계산
    tp = np.zeros(len(class_preds))
    fp = np.zeros(len(class_preds))
    matched_targets = set()
    
    for i, pred in enumerate(class_preds):
        best_iou = 0
        best_target_idx = -1
        
        # 가장 높은 IoU를 가진 타겟 찾기
        for j, target in enumerate(class_targets):
            if j in matched_targets:
                continue
            
            iou = calculate_iou(pred['box'], target['box'])
            if iou > best_iou:
                best_iou = iou
                best_target_idx = j
        
        # TP/FP 판단
        if best_iou >= iou_threshold and best_target_idx != -1:
            tp[i] = 1
            matched_targets.add(best_target_idx)
        else:
            fp[i] = 1
    
    # Precision-Recall 계산
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / len(class_targets)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    
    # AP 계산 (11-point interpolation)
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11
    
    return ap


def calculate_iou(box1, box2):
    """두 박스 간 IoU 계산"""
    # numpy array로 변환
    if torch.is_tensor(box1):
        box1 = box1.numpy()
    if torch.is_tensor(box2):
        box2 = box2.numpy()

    # 교집합 영역
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # 합집합 영역
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-10)