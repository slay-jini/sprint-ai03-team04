import torch
import torch.optim as optim
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from datetime import datetime

from utils.metrics import calculate_mAP

class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        
        # 옵티마이저 설정 (Adam으로 변경)
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY
        )
        
        # 학습 이력
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_map': []
        }
        
        # Early stopping 설정
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        
        # 출력 디렉토리 설정
        self.output_dir = Path(cfg.OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True)
        
        # 로깅 설정
        self.setup_logging()
    
    def setup_logging(self):
        """로깅 설정"""
        self.logger = logging.getLogger('trainer')
        self.logger.setLevel(logging.INFO)
        
        # 파일 핸들러
        fh = logging.FileHandler(self.output_dir / 'training.log')
        fh.setLevel(logging.INFO)
        
        # 콘솔 핸들러
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 포맷터
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def train_one_epoch(self, epoch):
        """한 에폭 학습"""
        self.model.train()
        epoch_loss = 0
        
        with tqdm(self.train_loader, desc=f'Epoch {epoch}') as pbar:
            for images, targets in pbar:
                # GPU로 데이터 이동
                images = [image.to(self.cfg.DEVICE) for image in images]
                targets = [{k: v.to(self.cfg.DEVICE) for k, v in t.items()} for t in targets]
                
                # 순전파
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                # 역전파
                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()
                
                # 손실 기록
                epoch_loss += losses.item()
                pbar.set_postfix({'loss': losses.item()})
        
        return epoch_loss / len(self.train_loader)
    
    def validate(self, epoch):
        """검증"""
        self.model.eval()
        val_loss = 0
        pred_boxes, pred_labels, pred_scores = [], [], []
        gt_boxes, gt_labels = [], []
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validation'):
                images = [image.to(self.cfg.DEVICE) for image in images]
                targets = [{k: v.to(self.cfg.DEVICE) for k, v in t.items()} for t in targets]
                
                # 검증 손실 계산을 위해 일시적으로 train 모드로 전환
                self.model.train()
                loss_dict = self.model(images, targets)
                if loss_dict is not None:
                    losses = sum(loss for loss in loss_dict.values())
                    val_loss += losses.item()
                else:
                    # loss_dict가 None인 경우 스킵
                    print(f"Warning: loss_dict is None during validation")
                
                # 예측을 위해 다시 eval 모드로 전환
                self.model.eval()
                outputs = self.model(images)
                
                # outputs가 None이 아닌지 확인
                if outputs is not None:
                    # mAP 계산을 위한 예측값과 정답 수집
                    for out, target in zip(outputs, targets):
                        # out이 예상한 형태인지 확인
                        if isinstance(out, dict) and 'boxes' in out:
                            pred_boxes.append(out['boxes'].cpu())
                            pred_labels.append(out['labels'].cpu())
                            pred_scores.append(out['scores'].cpu())
                            gt_boxes.append(target['boxes'].cpu())
                            gt_labels.append(target['labels'].cpu())
                        else:
                            print(f"Warning: Invalid output format: {type(out)}")
                else:
                    print(f"Warning: Model output is None during validation")
        
        # mAP 계산 (예측값이 있는 경우에만)
        if pred_boxes and gt_boxes:
            val_map = calculate_mAP(pred_boxes, pred_labels, pred_scores, 
                                  gt_boxes, gt_labels)
        else:
            val_map = 0.0
            print("Warning: No valid predictions for mAP calculation")
        
        val_loss = val_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0.0
        
        return val_loss, val_map
    
    def train(self):
        """전체 학습 과정"""
        self.logger.info("학습 시작")
        start_time = datetime.now()
        
        for epoch in range(self.cfg.NUM_EPOCHS):
            # 학습
            train_loss = self.train_one_epoch(epoch)
            
            # 검증
            val_loss, val_map = self.validate(epoch)
            
            # 히스토리 저장
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_map'].append(val_map)
            
            # 로깅
            self.logger.info(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, val_mAP={val_map:.4f}"
            )
            
            # 최고 성능 모델 저장
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.save_checkpoint(epoch, is_best=True)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Early stopping 체크
            if self.patience_counter >= self.cfg.PATIENCE:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # 주기적 체크포인트 저장
            if epoch % self.cfg.SAVE_INTERVAL == 0:
                self.save_checkpoint(epoch)
            
            # 학습 곡선 그리기
            self.plot_learning_curves()
        
        # 학습 완료
        end_time = datetime.now()
        training_time = end_time - start_time
        self.logger.info(f"학습 완료. 총 소요 시간: {training_time}")
        self.logger.info(f"최고 성능 에폭: {self.best_epoch}, 검증 손실: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, epoch, is_best=False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.cfg.__dict__
        }
        
        if is_best:
            path = self.output_dir / 'best_model.pth'
        else:
            path = self.output_dir / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, path)
        self.logger.info(f"체크포인트 저장됨: {path}")
    
    def plot_learning_curves(self):
        """학습 곡선 그리기"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 손실 그래프
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # mAP 그래프
        ax2.plot(self.history['val_map'], label='Val mAP')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mAP')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'learning_curves.png')
        plt.close()
