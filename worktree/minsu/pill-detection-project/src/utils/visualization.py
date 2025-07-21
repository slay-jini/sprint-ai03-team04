# src/utils/visualization.py
"""시각화 도구"""

from .font_config import setup_korean_font
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
import torch
import seaborn as sns


class Visualizer:
    """검출 결과 시각화 클래스"""
    
    def __init__(self, class_names=None, num_classes=None):
        # config에서 전달받은 num_classes 사용
        if num_classes is None:
            num_classes = 4  # 기본값
        
        self.num_classes = num_classes
        self.class_names = class_names or {i: f"Class {i}" for i in range(1, num_classes+1)}
        self.colors = self._generate_colors(num_classes)
    
    def _generate_colors(self, n):
        """클래스별 색상 생성 - Matplotlib 공식 방법"""
        import matplotlib as mpl
    
        # 공식 문서 권장: qualitative colormap 사용
        if n <= 10:
            # 적은 클래스: 'tab10' 사용
            cmap = mpl.colormaps['tab10']
        elif n <= 20:
            # 중간 클래스: 'tab20' 사용  
            cmap = mpl.colormaps['tab20']
        else:
            # 많은 클래스: 'viridis' 등 sequential colormap 사용
            cmap = mpl.colormaps['viridis']
        
        # 0부터 1까지 균등하게 분포된 값들로 색상 추출
        colors = cmap(np.linspace(0, 1, n))
        return colors
    
    def plot_image_with_boxes(self, image, predictions, save_path=None):
        """이미지와 박스 시각화"""
        fig, ax = plt.subplots(1, figsize=(12, 8))
        
        # 이미지 표시
        if torch.is_tensor(image):
            image = image.permute(1, 2, 0).cpu().numpy()
            # 정규화 해제
            image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            image = np.clip(image, 0, 1)
        
        ax.imshow(image)
        
        # 박스 그리기
        if 'boxes' in predictions:
            boxes = predictions['boxes'].cpu().numpy()
            labels = predictions['labels'].cpu().numpy()
            scores = predictions['scores'].cpu().numpy()
            
            for box, label, score in zip(boxes, labels, scores):
                x, y, x2, y2 = box
                w = x2 - x
                h = y2 - y
                
                # 박스
                rect = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=2,
                    edgecolor=self.colors[label-1],
                    facecolor='none'
                )
                ax.add_patch(rect)
                
                # 레이블
                class_name = self.class_names.get(label, f"Class {label}")
                label_text = f'{class_name}: {score:.2f}'
                ax.text(
                    x, y - 5, label_text,
                    bbox=dict(facecolor=self.colors[label-1], alpha=0.7),
                    fontsize=10, color='white'
                )
        
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=100)
            plt.close()
        else:
            plt.show()
    
    def plot_training_curves(self, history, save_path=None):
        """학습 곡선 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss
        if 'train_loss' in history:
            axes[0, 0].plot(history['train_loss'], label='Train')
            if 'val_loss' in history:
                axes[0, 0].plot(history['val_loss'], label='Val')
            axes[0, 0].set_title('Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].legend()
        
        # mAP
        if 'val_map' in history:
            axes[0, 1].plot(history['val_map'])
            axes[0, 1].set_title('Validation mAP')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylim([0, 1])
        
        # Learning Rate
        if 'lr' in history:
            axes[1, 0].plot(history['lr'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_yscale('log')
        
        # 클래스별 AP
        if 'ap_per_class' in history:
            latest_ap = history['ap_per_class'][-1]
            classes = list(latest_ap.keys())
            values = list(latest_ap.values())
            
            axes[1, 1].bar(classes, values)
            axes[1, 1].set_title('AP per Class')
            axes[1, 1].set_xlabel('Class')
            axes[1, 1].set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_confusion_matrix(self, predictions, targets, save_path=None):
        """혼동 행렬 시각화"""
        # 예측과 정답에서 클래스 추출
        pred_classes = []
        true_classes = []
        
        for pred, target in zip(predictions, targets):
            # 간단한 매칭 (실제로는 IoU 기반 매칭 필요)
            pred_classes.extend(pred['labels'].cpu().numpy())
            true_classes.extend(target['labels'].cpu().numpy())
        
        # 혼동 행렬 계산
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true_classes, pred_classes)
        
        # 시각화
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(self.class_names.values()),
            yticklabels=list(self.class_names.values())
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def save_detection_results(self, results, output_dir):
        """검출 결과를 CSV로 저장"""
        import csv
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = output_dir / 'predictions.csv'
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'annotation_id', 'image_id', 'category_id',
                'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'score'
            ])
            
            ann_id = 1
            for result in results:
                image_id = result['image_id']
                
                if 'boxes' in result:
                    boxes = result['boxes'].cpu().numpy()
                    labels = result['labels'].cpu().numpy()
                    scores = result['scores'].cpu().numpy()
                    
                    for box, label, score in zip(boxes, labels, scores):
                        x1, y1, x2, y2 = box
                        writer.writerow([
                            ann_id, image_id, int(label),
                            x1, y1, x2-x1, y2-y1, float(score)
                        ])
                        ann_id += 1
        
        print(f"예측 결과 저장: {csv_path}")