import torch
import numpy as np
from collections import defaultdict

def calculate_ap(recalls, precisions):
    """Average Precision 계산"""
    # 11-point interpolation
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap = ap + p / 11.
    return ap

def calculate_mAP(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_threshold=0.5):
    """Mean Average Precision 계산
    
    Args:
        pred_boxes (list[tensor]): 예측된 박스들 [N, 4]
        pred_labels (list[tensor]): 예측된 레이블들 [N]
        pred_scores (list[tensor]): 예측 신뢰도 [N]
        gt_boxes (list[tensor]): 정답 박스들 [M, 4]
        gt_labels (list[tensor]): 정답 레이블들 [M]
    """
    # 입력 검증
    if not pred_boxes or not gt_boxes:
        print("Warning: Empty prediction or ground truth boxes")
        return 0.0
    
    try:
        # 빈 텐서 필터링 및 디버깅 정보
        valid_pred_count = sum(1 for label in pred_labels if len(label) > 0)
        valid_gt_count = sum(1 for label in gt_labels if len(label) > 0)
        
        print(f"Debug: Valid predictions: {valid_pred_count}, Valid ground truths: {valid_gt_count}")
        
        filtered_pred_labels = [label for label in pred_labels if len(label) > 0]
        filtered_gt_labels = [label for label in gt_labels if len(label) > 0]
        
        if not filtered_pred_labels or not filtered_gt_labels:
            print("Warning: No valid labels found")
            return 0.0
            
        # 클래스 수 계산
        all_pred_labels = torch.cat(filtered_pred_labels)
        all_gt_labels = torch.cat(filtered_gt_labels)
        
        print(f"Debug: Pred label range: {all_pred_labels.min()}-{all_pred_labels.max()}")
        print(f"Debug: GT label range: {all_gt_labels.min()}-{all_gt_labels.max()}")
        
        n_classes = max(all_pred_labels.max().item(), all_gt_labels.max().item()) + 1
        print(f"Debug: Number of classes: {n_classes}")
        
    except Exception as e:
        print(f"Error calculating number of classes: {e}")
        return 0.0
    
    # 클래스별 AP 계산
    aps = []
    
    for class_id in range(n_classes):
        # 현재 클래스의 예측과 정답만 선택
        class_preds = defaultdict(list)  # 이미지별 예측
        class_gts = defaultdict(list)    # 이미지별 정답
        
        for img_idx in range(len(pred_boxes)):
            pred_mask = pred_labels[img_idx] == class_id
            gt_mask = gt_labels[img_idx] == class_id
            
            class_preds[img_idx] = [
                pred_boxes[img_idx][pred_mask],
                pred_scores[img_idx][pred_mask]
            ]
            
            class_gts[img_idx] = gt_boxes[img_idx][gt_mask]
        
        # 모든 예측을 신뢰도 순으로 정렬
        all_preds = []
        all_gts = []
        for img_idx in class_preds.keys():
            boxes, scores = class_preds[img_idx]
            all_preds.extend([(img_idx, box, score) for box, score in zip(boxes, scores)])
            all_gts.extend([(img_idx, box) for box in class_gts[img_idx]])
        
        if not all_preds:  # 예측이 없으면 AP = 0
            continue
            
        all_preds.sort(key=lambda x: x[2], reverse=True)
        
        # TP, FP 계산
        tp = np.zeros(len(all_preds))
        fp = np.zeros(len(all_preds))
        
        for pred_idx, (img_idx, pred_box, _) in enumerate(all_preds):
            gt_boxes_img = class_gts[img_idx]
            
            if len(gt_boxes_img) == 0:
                fp[pred_idx] = 1
                continue
                
            # IoU 계산
            ious = box_iou(pred_box.unsqueeze(0), gt_boxes_img)
            max_iou, max_idx = torch.max(ious, dim=1)
            
            if max_iou >= iou_threshold:
                tp[pred_idx] = 1
            else:
                fp[pred_idx] = 1
        
        # compute precision/recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        recalls = tp / float(len(all_gts)) if len(all_gts) > 0 else tp
        precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        
        ap = calculate_ap(recalls, precisions)
        aps.append(ap)
    
    # aps가 비어있지 않은 경우에만 평균 계산
    if aps:
        return np.mean(aps)
    else:
        print("Warning: No valid AP values calculated")
        return 0.0

def box_iou(box1, box2):
    """두 박스 세트 간의 IoU 계산"""
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])
    
    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    
    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    union = area1[:, None] + area2 - inter
    
    return inter / union
