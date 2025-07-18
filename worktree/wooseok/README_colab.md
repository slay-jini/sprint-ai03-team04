# Colab에서 데이터셋과 함께 학습하기

이 가이드는 Google Colab에서 데이터셋을 생성하고 `train.py`를 실행하는 방법을 설명합니다.

## 🚀 빠른 시작

### 1. Colab에서 기본 설정

```python
# 필요한 라이브러리 설치
!pip install torch torchvision pillow

# Google Drive 마운트 (데이터가 Drive에 있는 경우)
from google.colab import drive
drive.mount('/content/drive')

# 프로젝트 폴더로 이동
import os
os.chdir('/content/drive/MyDrive/your_project_folder')
```

### 2. 방법 1: 기존 데이터셋 클래스 사용

```python
from dataset import PillDetectionDataset, create_colab_dataset
from train import train_with_dataset

# 데이터셋 생성
dataset = create_colab_dataset(
    json_path='/content/drive/MyDrive/your_dataset/train.json',
    img_dir='/content/drive/MyDrive/your_dataset/images/train'
)

# 학습 실행
if dataset is not None:
    train_with_dataset(dataset)
```

### 3. 방법 2: 직접 데이터셋 생성

```python
from dataset import PillDetectionDataset
from train import train_with_dataset

# 데이터셋 객체 생성
dataset = PillDetectionDataset(
    json_file='/path/to/your/train.json',
    img_dir='/path/to/your/images/train'
)

print(f"Dataset loaded with {len(dataset)} samples")

# 학습 실행
train_with_dataset(dataset)
```

## 📁 파일 구조

```
your_colab_project/
├── train.py           # 수정된 학습 스크립트
├── dataset.py         # 데이터셋 클래스
├── config.py          # 설정 파일
├── trainer.py         # 학습 로직
├── models/            # 모델 정의
│   ├── model.py
│   └── yolo_model.py
├── utils/             # 유틸리티 함수
│   └── metrics.py
└── your_dataset/      # 데이터셋 폴더
    ├── train.json
    └── images/
        └── train/
```

## 🔧 주요 변경사항

### train.py 수정사항
- `main()` 함수에 `dataset` 매개변수 추가
- `train_with_dataset()` 함수 추가 (Colab 전용)
- 외부에서 생성된 데이터셋 객체를 받아서 학습 진행

### dataset.py 추가사항
- `create_colab_dataset()` 함수 추가
- 에러 처리 및 성공/실패 메시지 출력

## 💡 사용 팁

1. **데이터 경로 확인**: Colab에서 실제 데이터 경로를 정확히 지정하세요.
2. **GPU 사용**: 런타임 타입을 GPU로 설정하세요.
3. **메모리 관리**: 큰 데이터셋의 경우 배치 크기를 조정하세요.
4. **파일 권한**: 업로드된 파일의 권한을 확인하세요.

## 🐛 문제 해결

### 일반적인 오류와 해결방법

1. **파일 경로 오류**
   ```python
   import os
   print(os.listdir('/content/drive/MyDrive/'))  # 파일 목록 확인
   ```

2. **메모리 부족**
   ```python
   # config.py에서 배치 크기 줄이기
   cfg.BATCH_SIZE = 4  # 기본값에서 줄이기
   ```

3. **CUDA 오류**
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA version: {torch.version.cuda}")
   ```

## 📝 예제 코드

완전한 실행 예제:

```python
# 1. 라이브러리 설치 및 설정
!pip install torch torchvision pillow
from google.colab import drive
drive.mount('/content/drive')

# 2. 프로젝트 폴더로 이동
import os
os.chdir('/content/drive/MyDrive/pill_detection_project')

# 3. 데이터셋 생성 및 학습
from dataset import create_colab_dataset
from train import train_with_dataset

# 데이터셋 생성
dataset = create_colab_dataset(
    json_path='dataset/train.json',
    img_dir='dataset/images/train'
)

# 학습 실행
if dataset:
    train_with_dataset(dataset)
```

이제 Colab에서 로컬 파일 시스템에 의존하지 않고 데이터셋을 생성하여 학습할 수 있습니다! 🎉
