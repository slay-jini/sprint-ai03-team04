from pathlib import Path

class Config:
    # 데이터 경로
    DATA_DIR = Path("dataset")
    TRAIN_JSON = DATA_DIR / "train.json"
    TRAIN_IMG_DIR = DATA_DIR / "images/train"
    OUTPUT_DIR = Path("outputs")
    
    # 모델 파라미터
    NUM_CLASSES = 100  # 알약 종류 수 (실제 클래스 수보다 크게 설정)
    FREEZE_BACKBONE_LAYERS = 3  # ResNet 백본의 처음 몇 개 레이어를 고정할지 (0: 고정하지 않음)
    
    # YOLO 파라미터
    ANCHORS = [[10, 13], [16, 30], [33, 23],  # 작은 객체용
               [30, 61], [62, 45], [59, 119],  # 중간 객체용
               [116, 90], [156, 198], [373, 326]]  # 큰 객체용
    CONF_THRESHOLD = 0.5  # 객체 감지 신뢰도 임계값
    NMS_THRESHOLD = 0.4   # Non-Maximum Suppression 임계값
    
    # 학습 파라미터
    BATCH_SIZE = 4  # ResNet 백본 사용으로 인해 배치 크기 감소 (8 → 4)
    NUM_EPOCHS = 10  # 에폭 수 증가 (1 → 10)
    LEARNING_RATE = 0.0001  # 더 낮은 학습률로 변경
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    
    # 데이터 전처리
    IMG_SIZE = (800, 800)  # 입력 이미지 크기
    
    # Early Stopping
    PATIENCE = 10  # 성능 향상이 없을 때 기다리는 에폭 수
    MIN_DELTA = 0.001  # 최소 성능 향상치
    
    # 학습 관련
    SAVE_INTERVAL = 5  # 모델 저장 주기 (에폭)
    VAL_RATIO = 0.2  # 검증 데이터 비율
    
    # GPU 설정
    DEVICE = 'cuda'  # 'cuda' or 'cpu'
    NUM_WORKERS = 0  # 멀티프로세싱 에러 방지를 위해 0으로 설정
    
    def update(self, **kwargs):
        """설정 업데이트 메소드"""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError(f"Config has no attribute '{k}'")

# 전역 설정 객체
cfg = Config()
