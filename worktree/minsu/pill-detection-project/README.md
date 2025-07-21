# 알약 검출 프로젝트 (균형잡힌 버전)

헬스잇(Health Eat)의 알약 검출 시스템입니다. 핵심 기능을 모두 포함하면서도 이해하기 쉬운 구조로 구성되었습니다.

## 📋 주요 기능

✅ **데이터 전처리 및 분석**

- COCO 형식 데이터 처리
- 데이터 통계 분석
- 이상치 제거

✅ **모델 교체 가능**

- Faster R-CNN (기본)
- YOLO (추가 가능)
- 쉬운 모델 추가 구조

✅ **데이터 증강**

- 회전, 플립, 밝기 조정
- 모델별 최적화된 증강

✅ **mAP 평가**

- IoU 기반 정확한 mAP 계산
- 클래스별 성능 분석
- 상세 평가 리포트

✅ **시각화**

- 학습 곡선
- 검출 결과
- 혼동 행렬

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

# PyTorch 문제 시 삭제 후 따로 설치

# pip uninstall torch torchvision torchaudio ultralytics -y

# pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1 --extra-index-url https://download.pytorch.org/whl/cu128

### 2. 데이터 전처리

```bash
# 데이터 분석
python preprocess.py --data-path ./ai03-level1-project --analyze

# 데이터 정리 (선택사항)
python preprocess.py --data-path ./ai03-level1-project --clean
```

### 3. 모델 학습

#### Faster R-CNN (기본)

```bash
python train.py --config configs/base_config.yaml
```

#### YOLO로 학습

pip install ultralytics # 먼저 설치
python train_yolo.py --config configs/yolo.yaml

🎯 모델 버전 선택
configs/yolo.yaml에서 version만 변경:

v11n: 가장 빠름, 낮은 정확도
v11s: 균형잡힌 선택 ⭐
v11m: 더 높은 정확도
v11l: 높은 정확도, 느림
v11x: 최고 정확도, 가장 느림

```bash
python train.py --config configs/base_config.yaml --model-config configs/yolo.yaml
```

#### 커스텀 설정

```bash
python train.py --config configs/base_config.yaml --batch-size 16 --epochs 100
```

### 4. 모델 평가

```bash
자동 평가(추천) : python evaluate.py --model outputs/checkpoints/best_model.pth --visualize
```

YOLOv11로 지정 평가 : python evaluate_yolo.py --model outputs/yolo_experiment/weights/best.pt --visualize

### 5. 예측 수행

```bash
python predict.py \
    --model outputs/checkpoints/best_model.pth \
    --images ./ai03-level1-project/test_images \
    --output outputs/predictions \
    --visualize
```

YOLOv11로 지정 예측 :
python predict_yolo.py \
 --model outputs/yolo_experiment/weights/best.pt \
 --images ./ai03-level1-project/test_images \
 --output outputs/predictions_yolo \
 --visualize \
 --conf-threshold 0.25

## 📁 프로젝트 구조

```
pill-detection-project/
├── configs/              # 설정 파일
│   ├── base_config.yaml    # 기본 설정
│   ├── faster_rcnn.yaml    # Faster R-CNN 설정
│   └── yolo.yaml           # YOLO 설정
│
├── src/                  # 소스 코드
│   ├── data/               # 데이터 처리
│   ├── models/             # 모델 구현
│   ├── training/           # 학습/평가
│   └── utils/              # 유틸리티
│
├── train.py             # 학습 실행
├── evaluate.py          # 평가 실행
├── predict.py           # 예측 실행
└── preprocess.py        # 전처리 실행
```

## 🔧 설정 가이드

### 기본 설정 (base_config.yaml)

```yaml
# 데이터 경로
data:
  train_path: "./ai03-level1-project/train_images"

# 학습 설정
training:
  batch_size: 8 # GPU 메모리에 따라 조정
  epochs: 50 # 학습 시간
  learning_rate: 0.001

# 데이터 증강
augmentation:
  horizontal_flip: true
  rotation_range: 15 # 회전 각도
  brightness_range: [0.8, 1.2]
```

### 모델별 설정

- **Faster R-CNN**: 높은 정확도, 느린 속도
- **YOLO**: 빠른 속도, 약간 낮은 정확도

## 📊 평가 지표

### mAP (mean Average Precision)

- **mAP@0.5**: IoU 0.5 기준 평균 정밀도
- **mAP@0.75**: IoU 0.75 기준 (더 엄격)
- **클래스별 AP**: 각 알약별 성능

### 예상 성능

- Faster R-CNN: mAP@0.5 ~0.80-0.85
- YOLO: mAP@0.5 ~0.75-0.80

## 💡 팁

### 메모리 부족 시

```yaml
training:
  batch_size: 4 # 8에서 4로 감소
```

### 학습 속도 개선

```yaml
training:
  num_workers: 8 # CPU 코어 수에 맞게
```

### 과적합 방지

```yaml
augmentation:
  rotation_range: 30 # 더 강한 증강
  scale_range: [0.7, 1.3]
```

## 🐛 문제 해결

### CUDA 오류

```bash
# CPU로 실행
python train.py --device cpu
```

### 데이터 로드 오류

```bash
# 데이터 경로 확인
python preprocess.py --data-path [정확한_경로] --analyze
```

## 📈 학습 모니터링

TensorBoard로 실시간 모니터링:

```bash
tensorboard --logdir outputs/logs
```

브라우저에서 http://localhost:6006 접속

## 🏆 결과 예시

### 학습 곡선

- Loss 감소 추이
- mAP 상승 추이
- 학습률 변화

### 검출 결과

- 바운딩 박스와 신뢰도
- 클래스별 정확도
- 시각화된 예측 결과

## 📝 추가 개발

### 새 모델 추가

1. `src/models/`에 모델 클래스 생성
2. `BaseDetectionModel` 상속
3. `MODEL_REGISTRY`에 등록

### 새 증강 추가

1. `src/data/transforms.py` 수정
2. `configs/`에 설정 추가

---

**개발팀**: 헬스잇(Health Eat) AI Team  
**라이선스**: Private
