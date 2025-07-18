import torch
from data.dataset import PillDataset
from data.transforms import get_basic_transform

# 테스트를 위한 가짜 데이터셋 경로 (작은 샘플 데이터만 넣어둠)
FAKE_ROOT_DIR = "path/to/your/fake_test_data" 

def test_dataset_output_shape_and_type():
    """
    PillDataset이 올바른 모양과 타입의 텐서를 출력하는지 테스트합니다.
    """
    # 1. 준비 (Arrange)
    dataset = PillDataset(root=FAKE_ROOT_DIR, transforms=get_basic_transform())
    
    # 데이터셋에서 샘플 하나를 가져옵니다.
    img, target = dataset[0]

    # 2. 실행 (Act) - 이미 준비 단계에서 실행됨

    # 3. 단언 (Assert)
    assert isinstance(img, torch.Tensor)
    assert img.shape[0] == 3 # 3-channel (RGB) 이미지인지 확인
    
    assert isinstance(target, dict)
    assert "boxes" in target
    assert "labels" in target
    assert isinstance(target["boxes"], torch.Tensor)
    assert target["boxes"].shape[1] == 4 # Bbox는 [x1, y1, x2, y2] 4개의 값을 가짐