# engine/evaluator.py
import torch
from tqdm.auto import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# 이 함수는 model.eval() 상태에서만 호출되어야 함
@torch.no_grad() # 함수 전체는 그래디언트 계산이 필요 없음
def evaluate(model, data_loader, device, calculate_map_metric=True):
    """
    모델의 검증 손실과 mAP를 한 번의 순회로 모두 계산합니다.
    이 함수는 model.eval() 상태에서 호출되어야 합니다.
    """
    # mAP 계산기 초기화 (IoU 0.5 임계값 사용)
    metric = MeanAveragePrecision(iou_type="bbox")# 모든 IoU 임계값에 대해 계산
    total_loss = 0.0
    
    progress_bar = tqdm(data_loader, desc="[검증/평가]")
    for images, targets in progress_bar:
        if images is None: continue
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # --- 이 부분이 핵심 ---
        # --- 손실 계산 ---
        # with 블록 안에서만 임시로 모드를 바꿔 손실을 얻고, 벗어나면 자동으로 eval 복귀
        # 1. 손실 계산을 위해 잠시 train 모드로 전환 (컨텍스트 관리자 사용)
        # 이 블록을 벗어나면 자동으로 원래 모드(eval)로 돌아옵니다.
        with torch.enable_grad(): # with 블록 안에서만 임시로 그래디언트 계산 허용
            model.train()
            loss_dict = model(images, targets)
            model.eval() # 손실 계산 후 즉시 eval 모드로 복귀

        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

        # --- mAP 계산을 위한 추론 ---
        # --- mAP 계산은 플래그가 True일 때만 수행 ---

        if calculate_map_metric:
            # 이미 eval() 상태이므로 바로 추론
            outputs = model(images)

            # torchmetrics가 요구하는 형식으로 변환
            # preds: 리스트 안의 각 딕셔너리는 'boxes', 'scores', 'labels' 키를 가짐
            # target: 리스트 안의 각 딕셔너리는 'boxes', 'labels' 키를 가짐
            preds = [{k: v.cpu() for k, v in out.items()} for out in outputs]
            targets_formatted = [{k: v.cpu() for k, v in t.items()} for t in targets] # 원본 targets 사용
            
            # 배치 단위로 계산기에 업데이트
            metric.update(preds, targets_formatted)

    # 평균 손실
    avg_loss = total_loss / len(data_loader)

    map_results = None
    if calculate_map_metric:
        # mAP 계산 및 결과 출력
        map_results = metric.compute()

    return avg_loss, map_results

