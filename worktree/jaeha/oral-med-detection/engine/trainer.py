# engine/trainer.py
import torch
from tqdm.auto import tqdm

def train_one_epoch(model, optimizer, data_loader, device, epoch, num_epochs):
    """한 에폭의 훈련을 수행합니다."""
    total_loss = 0.0
    
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs} [훈련]")
    for images, targets in progress_bar:
        if images is None: continue

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        progress_bar.set_postfix(loss=losses.item())
        
    return total_loss / len(data_loader)
