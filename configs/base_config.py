# configs/base_config.py
import torch

# --- 설정 변수 ---
# --- 기본 설정 ---
# 데이터 경로 (필요 시 실제 경로로 수정)
ROOT_DIRECTORY = "path/to/your/dataset"
# 체크포인트 저장 경로
CHECKPOINT_DIR = "checkpoints"

# --- 훈련 설정 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

NUM_EPOCHS = 20 # or 10
BATCH_SIZE = 4 # or 2 or 8
BATCH_SIZE_VAL = 1
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
MAP_CALC_CYCLE = 5

# [추가] 그래디언트 클리핑에 사용할 최대 norm 값 (0 이하면 비활성화)
GRADIENT_CLIP_NORM = 1.0 

# --- 콜백 설정 ---
# EarlyStopping
ES_PATIENCE = 5
# CheckpointSaver
CS_TOP_K = 3

# --- 데이터셋 설정 ---
TRAIN_VALID_SPLIT = 0.8
NUM_WORKERS = 2 # os.cpu_count()

# [추가] 데이터셋에서 무시할 최소 박스 크기 (픽셀 단위)
MIN_BOX_SIZE = 10 