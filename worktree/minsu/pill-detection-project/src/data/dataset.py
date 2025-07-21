# src/data/dataset.py
"""COCO 형식 데이터셋 처리 - 개선된 버전"""

import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
from collections import defaultdict


class COCODataset(Dataset):
    """COCO 형식의 알약 검출 데이터셋"""
    
    def __init__(self, img_dir, ann_dir, transform=None, min_box_size=10):
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.transform = transform
        self.min_box_size = min_box_size
        
        # 데이터 저장
        self.images = []
        self.annotations = defaultdict(list)
        self.categories = {}
        self.class_mapping = {}  # 자동 생성될 클래스 매핑
        
        # 데이터 로드 및 전처리
        self._load_and_process_data()
            
        print(f"데이터셋 준비 완료: {len(self.images)}개 이미지, {len(self.class_mapping)}개 클래스")
        print(f"클래스 범위: 1 ~ {len(self.class_mapping)}")
    
    def _load_and_process_data(self):
        """데이터 로드 및 전처리를 한 번에 수행"""
        print("데이터 로드 및 전처리 중...")
        
        json_files = list(self.ann_dir.rglob("*.json"))
        all_category_ids = set()
        raw_annotations = defaultdict(list)
        
        # 1단계: 원본 데이터 로드
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 이미지 정보
            for img in data.get('images', []):
                self.images.append({
                    'id': img['id'],
                    'file_name': img['file_name'],
                    'width': img['width'],
                    'height': img['height']
                })
            
            # 유효한 어노테이션만 수집
            for ann in data.get('annotations', []):
                if self._is_valid_annotation(ann):
                    all_category_ids.add(ann['category_id'])
                    raw_annotations[ann['image_id']].append(ann)
                
            # 카테고리 정보
            for cat in data.get('categories', []):
                self.categories[cat['id']] = cat['name']
        
        print(f"발견된 총 {len(all_category_ids)}개 클래스: {sorted(all_category_ids)}")
        
        # 2단계: 클래스 매핑 생성 (연속적인 1, 2, 3, ... 형태)
        sorted_ids = sorted(all_category_ids)
        self.class_mapping = {original_id: new_id for new_id, original_id in enumerate(sorted_ids, 1)}
        
        # 3단계: 어노테이션 전처리 및 클래스 매핑 적용
        total_boxes = 0
        filtered_boxes = 0
        
        for img_id, anns in raw_annotations.items():
            valid_anns = []
            for ann in anns:
                total_boxes += 1
                
                # 박스 크기 필터링
                bbox = ann['bbox']
                if bbox[2] >= self.min_box_size and bbox[3] >= self.min_box_size:
                    # 클래스 매핑 적용
                    ann['category_id'] = self.class_mapping[ann['category_id']]
                    valid_anns.append(ann)
                    filtered_boxes += 1
            
            if valid_anns:  # 유효한 어노테이션이 있는 이미지만 유지
                self.annotations[img_id] = valid_anns
        
        print(f"전처리 완료: {total_boxes}개 → {filtered_boxes}개 박스 (너무 작은 박스 {total_boxes - filtered_boxes}개 제거)")
        
        # 이미지 정렬 및 유효한 이미지만 유지
        self.images = [img for img in self.images if img['id'] in self.annotations]
        self.images.sort(key=lambda x: x['id'])
        
        print(f"유효한 이미지: {len(self.images)}개")
    
    def _is_valid_annotation(self, ann):
        """유효한 어노테이션인지 확인"""
        bbox = ann.get('bbox', [])
        return (len(bbox) == 4 and 
                bbox[2] > 0 and bbox[3] > 0 and  # 너비, 높이 > 0
                bbox[0] >= 0 and bbox[1] >= 0)   # x, y >= 0
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 이미지 로드
        img_info = self.images[idx]
        img_path = self.img_dir / img_info['file_name']
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"이미지 로드 실패: {img_path} - {e}")
            # 더미 이미지 반환
            image = Image.new('RGB', (224, 224), color='gray')
        
        # 어노테이션 가져오기
        img_id = img_info['id']
        anns = self.annotations.get(img_id, [])
        
        # 바운딩 박스와 라벨 변환 (공식 방법)
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']  # COCO format: [x, y, width, height]
            
            # 유효성 검증 먼저
            if w <= 0 or h <= 0 or w < self.min_box_size or h < self.min_box_size:
                continue  # 잘못된 박스 건너뛰기
            
            # COCO → PyTorch 변환: [x, y, w, h] → [x1, y1, x2, y2]
            x1, y1, x2, y2 = x, y, x + w, y + h
            
            # 변환 후 재검증
            if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0:
                boxes.append([x1, y1, x2, y2])
                labels.append(ann['category_id'])  # 이미 매핑된 ID
        
        # numpy 배열로 변환
        image = np.array(image)
        
        # 타겟 생성
        target = {
            'image_id': img_id,
            'boxes': torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        }
        
        # 변환 적용
        if self.transform:
            image, target = self.transform(image, target)
        
        return image, target
    
    def get_img_info(self, idx):
        """이미지 메타데이터 반환"""
        return self.images[idx]
    
    def get_class_names(self):
        """클래스 이름 딕셔너리 반환 (매핑된 ID → 이름)"""
        class_names = {}
        for original_id, mapped_id in self.class_mapping.items():
            class_names[mapped_id] = self.categories.get(original_id, f"Class_{original_id}")
        return class_names
    
    def get_num_classes(self):
        """총 클래스 수 반환 (배경 클래스 제외)"""
        return len(self.class_mapping)
    
    def print_dataset_info(self):
        """데이터셋 상세 정보 출력"""
        print("=" * 60)
        print("데이터셋 정보")
        print("=" * 60)
        print(f"총 이미지 수: {len(self.images)}")
        print(f"총 클래스 수: {len(self.class_mapping)}")
        
        # 클래스별 통계
        class_counts = defaultdict(int)
        total_annotations = 0
        box_areas = []
        
        for img_id in self.annotations:
            for ann in self.annotations[img_id]:
                class_counts[ann['category_id']] += 1
                total_annotations += 1
                
                # 박스 면적 계산
                bbox = ann['bbox']
                area = bbox[2] * bbox[3]
                box_areas.append(area)
        
        print(f"총 어노테이션 수: {total_annotations}")
        print(f"이미지당 평균 객체 수: {total_annotations / len(self.images):.2f}")
        
        if box_areas:
            print(f"박스 면적 통계: 평균 {np.mean(box_areas):.0f}, 중간값 {np.median(box_areas):.0f}")
        
        print(f"\n클래스별 분포:")
        class_names = self.get_class_names()
        for class_id in sorted(class_counts.keys()):
            count = class_counts[class_id]
            percentage = count / total_annotations * 100
            name = class_names.get(class_id, f"Unknown_{class_id}")
            print(f"  클래스 {class_id:2d} ({name}): {count:4d}개 ({percentage:5.1f}%)")
        
        # 클래스 매핑 테이블
        print(f"\n클래스 매핑 (원본 ID → 새 ID):")
        for original_id, new_id in sorted(self.class_mapping.items()):
            name = self.categories.get(original_id, f"Unknown_{original_id}")
            print(f"  {original_id:5d} → {new_id:2d} ({name})")


# 별도 전처리 클래스 (필요한 경우에만 사용)
class DataPreprocessor:
    """사전 데이터 정리 및 분석 도구"""
    
    def __init__(self, data_root):
        self.data_root = Path(data_root)
    
    def analyze_raw_data(self):
        """원본 데이터 분석 (전처리 전)"""
        print("원본 데이터 분석 중...")
        
        ann_dir = self.data_root / "train_annotations"
        json_files = list(ann_dir.rglob("*.json"))
        
        all_categories = set()
        all_images = set()
        total_annotations = 0
        invalid_annotations = 0
        
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 이미지 수집
            for img in data.get('images', []):
                all_images.add(img['id'])
            
            # 어노테이션 분석
            for ann in data.get('annotations', []):
                total_annotations += 1
                all_categories.add(ann['category_id'])
                
                # 유효성 검사
                bbox = ann.get('bbox', [])
                if not (len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0):
                    invalid_annotations += 1
        
        print(f"총 이미지: {len(all_images)}개")
        print(f"총 카테고리: {len(all_categories)}개")
        print(f"총 어노테이션: {total_annotations}개")
        print(f"유효하지 않은 어노테이션: {invalid_annotations}개")
        print(f"발견된 카테고리 ID: {sorted(all_categories)}")
        
        return {
            'num_images': len(all_images),
            'num_categories': len(all_categories),
            'total_annotations': total_annotations,
            'invalid_annotations': invalid_annotations,
            'category_ids': sorted(all_categories)
        }
    
    def create_train_val_split(self, val_ratio=0.2, seed=42):
        """학습/검증 분할을 위한 이미지 ID 리스트 생성"""
        np.random.seed(seed)
        
        # 모든 이미지 ID 수집
        ann_dir = self.data_root / "train_annotations"
        json_files = list(ann_dir.rglob("*.json"))
        
        all_image_ids = set()
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for img in data.get('images', []):
                all_image_ids.add(img['id'])
        
        # 분할
        image_ids = sorted(all_image_ids)
        np.random.shuffle(image_ids)
        
        val_size = int(len(image_ids) * val_ratio)
        train_ids = image_ids[val_size:]
        val_ids = image_ids[:val_size]
        
        print(f"데이터 분할 완료:")
        print(f"  학습: {len(train_ids)}개 이미지")
        print(f"  검증: {len(val_ids)}개 이미지")
        
        return train_ids, val_ids


# 사용 예시
def create_dataset(img_dir, ann_dir, min_box_size=10):
    """간단한 데이터셋 생성"""
    dataset = COCODataset(
        img_dir=img_dir,
        ann_dir=ann_dir,
        min_box_size=min_box_size
    )
    dataset.print_dataset_info()
    return dataset


def analyze_before_training(data_root):
    """학습 전 데이터 분석"""
    preprocessor = DataPreprocessor(data_root)
    stats = preprocessor.analyze_raw_data()
    train_ids, val_ids = preprocessor.create_train_val_split()
    
    return stats, train_ids, val_ids