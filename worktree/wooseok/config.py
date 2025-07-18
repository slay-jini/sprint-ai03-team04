from pathlib import Path

class Config:
    # 데이터 경로
    DATA_DIR = Path("dataset")
    TRAIN_JSON = DATA_DIR / "train.json"
    TRAIN_IMG_DIR = DATA_DIR / "images/train"
    OUTPUT_DIR = Path("outputs")
    
    # 모델 파라미터
    NUM_CLASSES = 100  # 알약 종류 수 (실제 클래스 수보다 크게 설정)
    FREEZE_BACKBONE_LAYERS = 0  # ResNet 백본 고정 해제 (3 → 0)
    
    # YOLO 파라미터
    ANCHORS = [[10, 13], [16, 30], [33, 23],  # 작은 객체용
               [30, 61], [62, 45], [59, 119],  # 중간 객체용
               [116, 90], [156, 198], [373, 326]]  # 큰 객체용
    CONF_THRESHOLD = 0.5  # 객체 감지 신뢰도 임계값
    NMS_THRESHOLD = 0.4   # Non-Maximum Suppression 임계값
    
    # 학습 파라미터
    BATCH_SIZE = 8  # 배치 크기 증가로 안정적인 gradient
    NUM_EPOCHS = 10  # 에폭 수 증가 (1 → 10)
    LEARNING_RATE = 0.001  # 학습률 증가 (0.0001 → 0.001)
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    
    # 데이터 전처리
    IMG_SIZE = (416, 416)  # 입력 이미지 크기 (800 → 416)
    
    # Early Stopping
    PATIENCE = 10  # 성능 향상이 없을 때 기다리는 에폭 수
    MIN_DELTA = 0.001  # 최소 성능 향상치
    
    # 학습 관련
    SAVE_INTERVAL = 5  # 모델 저장 주기 (에폭)
    ENABLE_VALIDATION = True  # Validation 활성화/비활성화
    ENABLE_MAP_CALCULATION = False  # mAP 계산 비활성화 (시간 단축)
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
