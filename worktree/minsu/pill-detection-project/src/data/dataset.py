# src/data/dataset.py
"""COCO 형식 데이터셋 처리"""

import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
from collections import defaultdict


class COCODataset(Dataset):
    """COCO 형식의 알약 검출 데이터셋"""
    CLASS_MAPPING = {
    3482: 1,   # 첫 번째 알약
    2482: 2,   # 두 번째 알약  
    16547: 3,  # 세 번째 알약
    1899: 4,   # 네 번째 알약
    }
    def __init__(self, img_dir, ann_dir, transform=None, preprocessing=True):
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.transform = transform
        
        # 데이터 저장
        self.images = []
        self.annotations = defaultdict(list)
        self.categories = {}
        
        # 데이터 로드
        self._load_annotations()
        
        # 전처리 (선택사항)
        if preprocessing:
            self._preprocess_data()
            
        print(f"데이터셋 준비 완료: {len(self.images)}개 이미지, {len(self.categories)}개 클래스")
    
    def _load_annotations(self):
        """JSON 어노테이션 로드"""
        json_files = list(self.ann_dir.rglob("*.json"))
        
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 이미지 정보
            for img in data.get('images', []):
                img_id = img['id']
                self.images.append({
                    'id': img_id,
                    'file_name': img['file_name'],
                    'width': img['width'],
                    'height': img['height'],
                    'metadata': img  # 전체 메타데이터 보존
                })
            
            # 어노테이션
            for ann in data.get('annotations', []):
                if self._is_valid_annotation(ann):
                    original_category_id = ann['category_id']
                    mapped_category_id = self.map_category_id(original_category_id)
                
                    # 매핑된 클래스만 사용
                    if mapped_category_id is not None:
                        ann['category_id'] = mapped_category_id  # 매핑된 ID로 변경
                        self.annotations[ann['image_id']].append(ann)
                
            # 카테고리
            for cat in data.get('categories', []):
                original_id = cat['id']
                mapped_id = self.map_category_id(original_id)
                if mapped_id is not None:
                    self.categories[mapped_id] = cat['name']
        
        # 정렬
        self.images.sort(key=lambda x: x['id'])
    
    def _is_valid_annotation(self, ann):
        """유효한 어노테이션인지 확인"""
        bbox = ann.get('bbox', [])
        return len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0
    
    def _preprocess_data(self):
        """데이터 전처리 - 이상치 제거, 정규화 등"""
        # 너무 작은 박스 제거
        min_size = 10
        for img_id in self.annotations:
            valid_anns = []
            for ann in self.annotations[img_id]:
                bbox = ann['bbox']
                if bbox[2] >= min_size and bbox[3] >= min_size:
                    valid_anns.append(ann)
            self.annotations[img_id] = valid_anns
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 이미지 로드
        img_info = self.images[idx]
        img_path = self.img_dir / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        
        # 어노테이션 가져오기
        img_id = img_info['id']
        anns = self.annotations.get(img_id, [])
        
        # 형식 변환
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x+w, y+h])
            labels.append(ann['category_id'])
        
        # numpy 배열로 변환
        image = np.array(image)
        
        # 타겟 생성
        target = {}
        if boxes:
            target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
            target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
        else:
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.int64)
        
        target['image_id'] = img_id
        
        # 변환 적용
        if self.transform:
            image, target = self.transform(image, target)
        
        return image, target
    
    def get_img_info(self, idx):
        """이미지 메타데이터 반환"""
        return self.images[idx]
    
    def map_category_id(self, original_id):
        """원본 클래스 ID를 모델 클래스 ID로 매핑"""
        return self.CLASS_MAPPING.get(original_id, None)  # 매핑되지 않으면 배경(0)


