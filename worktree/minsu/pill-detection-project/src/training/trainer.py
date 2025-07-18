# src/training/trainer.py
"""모델 학습 관리"""

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import time


class Trainer:
    """모델 학습 클래스"""
    
    def __init__(self, model, config, device='cuda'):
        self.model = model
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 옵티마이저 설정
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # 로깅
        self.writer = SummaryWriter(config['output']['log_dir'])
        self.best_map = 0.0
        
    def _create_optimizer(self):
        """옵티마이저 생성"""
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        opt_type = self.config['optimizer'].get('type', 'sgd')
        lr = self.config['training']['learning_rate']
        
        if opt_type == 'sgd':
            return optim.SGD(
                params,
                lr=lr,
                momentum=self.config['optimizer'].get('momentum', 0.9),
                weight_decay=self.config['optimizer'].get('weight_decay', 0.0005)
            )
        elif opt_type == 'adam':
            return optim.Adam(params, lr=lr)
        elif opt_type == 'adamw':  # 추가!
            return optim.AdamW(
                params,
                lr=lr,
                weight_decay=self.config['optimizer'].get('weight_decay', 0.0005),
                betas=(0.9, 0.999)
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")
    
    def _create_scheduler(self):
        """학습률 스케줄러 생성"""
        scheduler_type = self.config['scheduler'].get('type', 'step')
        
        if scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config['scheduler'].get('step_size', 30),
                gamma=self.config['scheduler'].get('gamma', 0.1)
            )
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs']
            )
        else:
            return None
    
    def train_epoch(self, data_loader, epoch):
        """한 에폭 학습"""
        self.model.train()
        
        epoch_loss = 0.0
        num_batches = len(data_loader)
        
        pbar = tqdm(data_loader, desc=f'Epoch {epoch}')
        for batch_idx, (images, targets) in enumerate(pbar):
            # GPU로 이동
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) if hasattr(v, 'to') else v 
                        for k, v in t.items()} for t in targets]
            
            # 순전파
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # 역전파
            self.optimizer.zero_grad()
            losses.backward()
            
            # 그래디언트 클리핑
            if self.config['training'].get('gradient_clip'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
            
            self.optimizer.step()
            
            # 로깅
            loss_value = losses.item()
            epoch_loss += loss_value
            
            if batch_idx % 10 == 0:
                pbar.set_postfix({'loss': f'{loss_value:.4f}'})
                
                # Tensorboard 로깅
                step = epoch * num_batches + batch_idx
                self.writer.add_scalar('Loss/train', loss_value, step)
                
                for k, v in loss_dict.items():
                    self.writer.add_scalar(f'Loss/{k}', v.item(), step)
        
        return epoch_loss / num_batches
    
    def validate(self, data_loader, evaluator, epoch):
        """검증 수행"""
        print(f"\n검증 중...")
        
        # mAP 계산
        results = evaluator.evaluate(self.model, data_loader)
        
        # 로깅
        self.writer.add_scalar('mAP/val', results['mAP'], epoch)
        self.writer.add_scalar('mAP@.5/val', results['mAP_50'], epoch)
        self.writer.add_scalar('mAP@.75/val', results['mAP_75'], epoch)
        
        # 클래스별 AP 로깅
        for class_id, ap in results['ap_per_class'].items():
            self.writer.add_scalar(f'AP/class_{class_id}', ap, epoch)
        
        return results
    
    def train(self, train_loader, val_loader, evaluator):
        """전체 학습 프로세스"""
        epochs = self.config['training']['epochs']
        save_interval = self.config['checkpoint']['save_interval']
        
        print(f"학습 시작: {epochs} 에폭")
        print(f"디바이스: {self.device}")
        print(f"배치 크기: {self.config['training']['batch_size']}")
        print("=" * 50)
        
        for epoch in range(1, epochs + 1):
            # 학습
            start_time = time.time()
            train_loss = self.train_epoch(train_loader, epoch)
            epoch_time = time.time() - start_time
            
            # 학습률 조정
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('LR', current_lr, epoch)
            
            # 검증 및 최고 모델 저장
            val_results = None
            best_updated = False

            if val_loader and epoch % self.config['validation']['interval'] == 0:
                val_results = self.validate(val_loader, evaluator, epoch)
                
                print(f"\nEpoch {epoch}/{epochs}")
                print(f"Train Loss: {train_loss:.4f}")
                print(f"Val mAP: {val_results['mAP']:.4f}")
                print(f"Val mAP@.5: {val_results['mAP_50']:.4f}")
                print(f"Val mAP@.75: {val_results['mAP_75']:.4f}")
                print(f"Time: {epoch_time:.1f}s")
                
                # 최고 성능 모델 저장
                if val_results['mAP'] >= self.best_map:
                    self.best_map = val_results['mAP']
                    self.save_checkpoint(epoch, train_loss, val_results, best=True)
                    best_updated = True
                    print(f"최고 성능 갱신! mAP: {self.best_map:.4f}")
            else:
                print(f"\nEpoch {epoch}/{epochs} - Loss: {train_loss:.4f} - Time: {epoch_time:.1f}s")
            
            # 효율적인 정기 저장
            should_save = False
            save_reason = ""
            
            if epoch == 1:
                # 첫 번째 에폭: 무조건 저장
                should_save = True
                save_reason = "첫 번째 에폭"
            elif epoch % save_interval == 0:
                # 정기 저장 (모든 에폭)
                should_save = True
                save_reason = f"정기 저장 (interval: {save_interval})"
            
            if should_save:
                self.save_checkpoint(epoch, train_loss, val_results)
                print(f"📁 정기 체크포인트 저장: epoch_{epoch}.pth ({save_reason})")
            
            print("-" * 50)
            
        self.writer.close()
        print(f"\n학습 완료! 최고 mAP: {self.best_map:.4f}")
        print(f"체크포인트 저장 위치: {self.config['output']['checkpoint_dir']}")
    
    def save_checkpoint(self, epoch, loss, metrics=None, best=False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'loss': loss,
            'metrics': metrics,
            'config': self.config
        }
        
        # 저장 경로
        if best:
            path = Path(self.config['output']['checkpoint_dir']) / 'best_model.pth'
        else:
            path = Path(self.config['output']['checkpoint_dir']) / f'epoch_{epoch}.pth'
        
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f"체크포인트 저장: {path}")
