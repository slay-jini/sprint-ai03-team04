# data/dataset.py
import torch
import os
import glob
import json
from PIL import Image
from tqdm.auto import tqdm
from collections import defaultdict
import configs.base_config as cfg
import numpy as np

# =================================================================================
# 1. 데이터셋 클래스 정의 (PillDataset)
# =================================================================================
class PillDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None, min_box_size=10):  # min_box_size 인자 추가
        self.root = root
        self.transforms = transforms
        self.min_box_size = min_box_size # 추가

        # --- 최종 데이터를 담을 인스턴스 변수 초기화 ---
        self.images = []
        self.annotations = defaultdict(list)
        self.id_to_cat = {}
        self.map_cat_id_to_label = {}
        self.map_label_to_cat_id = {}

        # --- 모든 전처리를 수행하는 단일 메소드 호출 ---
        self._load_data()

        print(f"데이터셋 준비 완료: {len(self.images)}개 이미지, {len(self.map_cat_id_to_label)}개 클래스")


    def _load_data(self):
        """
        데이터를 로드하고 전처리하는 모든 과정을 통합하여 관리합니다.
        단 한 번의 파일 순회로 효율성을 극대화합니다.
        """
        print("데이터 로딩 및 전처리 시작...")
        annotation_paths = glob.glob(os.path.join(self.root, 'train_annotations', '**', '*.json'), recursive=True)

        # 1. 임시 변수: 모든 정보를 한 번의 루프로 수집
        raw_images = {}
        # defaultdict를 여기서 사용합니다!
        raw_annotations = defaultdict(list)
        present_category_ids = set()

        for ann_path in tqdm(annotation_paths, desc="어노테이션 파일 처리 중"):
            with open(ann_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 이미지 정보 수집
            for img in data.get('images', []):
                if img['id'] not in raw_images:
                    raw_images[img['id']] = img
            
            # 카테고리 이름 정보 수집
            for cat in data.get('categories', []):
                if cat['id'] not in self.id_to_cat:
                    self.id_to_cat[cat['id']] = cat['name']

            # 유효한 어노테이션 및 실제 사용된 카테고리 ID 수집
            for ann in data.get('annotations', []):
                bbox = ann.get('bbox', [])
                if len(bbox) == 4 and bbox[2] > self.min_box_size and bbox[3] > self.min_box_size:
                    raw_annotations[ann['image_id']].append(ann)
                    present_category_ids.add(ann['category_id'])
        
        # 2. 클래스 매핑 생성
        sorted_ids = sorted(list(present_category_ids))
        self.map_cat_id_to_label = {cat_id: i + 1 for i, cat_id in enumerate(sorted_ids)}
        self.map_label_to_cat_id = {v: k for k, v in self.map_cat_id_to_label.items()}

        # 3. 최종 어노테이션 생성 (클래스 ID 변환)
        for img_id, anns in raw_annotations.items():
            # 이 이미지에 유효한 어노테이션이 하나라도 있는지 확인
            if raw_images.get(img_id):
                for ann in anns:
                    # ann['category_id']는 원본 ID이므로, 매핑된 라벨로 교체
                    original_cat_id = ann['category_id']
                    if original_cat_id in self.map_cat_id_to_label:
                         ann['category_id'] = self.map_cat_id_to_label[original_cat_id]
                         self.annotations[img_id].append(ann)
        
        # 4. 최종 이미지 목록 생성 (유효한 어노테이션이 있는 이미지만)
        self.images = [img for img_id, img in raw_images.items() if img_id in self.annotations]
        self.images.sort(key=lambda x: x['file_name']) # 파일명 순으로 정렬하여 일관성 유지

    def __getitem__(self, idx):
        # image = self.images[idx]
        # filename = os.path.basename(img_path)
        # abspath?
        
        # filename = image.file_name
        # img_path = os.path.join(cfg.ROOT_DIRECTORY, cfg.TRAIN_IMAGE_DIRECTORY, filename)
        # try:
        #     img = Image.open(cfg.ROOT_DIRECTORY + filename).convert("RGB")
        # except FileNotFoundError:
        #     # Colab 환경에서는 파일을 못찾으면 다음으로 넘어가는게 중요
        #     return None

        # annotation = self.image_to_labels[filename]
        # annotation =image[filename]

        img_info = self.images[idx]
        img_path = os.path.join(self.root, 'train_images', img_info['file_name'])
        # image를 numpy 배열로 바로 읽습니다.
        image = np.array(Image.open(img_path).convert("RGB"))

        anns = self.annotations[img_info['id']]
        
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            x1, y1, x2, y2 = x, y, x + w, y + h
            boxes.append([x1, y1, x2, y2])
            labels.append(ann['category_id'])

        target = {
            'boxes': torch.as_tensor(boxes),
            'labels': torch.as_tensor(labels)
        }
        # print(f"target: {target['boxes'].shape}")

        # target = {}
        # target["boxes"] = annotation["boxes"]
        # target["labels"] = annotation["labels"]
        #target["image_id"] = annotation["image_id"]
        #target["area"] = annotation["area"]
        #target["iscrowd"] = annotation["iscrowd"]

        if self.transforms:
            # torchvision v2 transform은 이미지와 타겟을 함께 받습니다.
            # Albumentations는 numpy 배열을 입력으로 기대
            transformed = self.transforms(image=image, boxes=boxes, labels=labels)
            image = transformed['image']

            # 변환 후의 박스와 라벨을 다시 가져와 텐서로 만듭니다.
            target = {
                'boxes': torch.as_tensor(transformed['boxes'], dtype=torch.float32) if transformed['boxes'] else torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.as_tensor(transformed['labels'], dtype=torch.int64) if transformed['labels'] else torch.zeros((0,), dtype=torch.int64)
            }
        return image, target

    def __len__(self):
        return len(self.images)

    def get_num_classes(self):
        return len(self.map_cat_id_to_label) + 1  # 배경 클래스 포함
