# src/data/preprocessor.py
"""데이터 전처리 도구"""

import json
import shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
from tqdm import tqdm


class DataPreprocessor:
    """데이터 전처리 클래스"""
    
    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self.stats = defaultdict(int)
    
    def analyze_dataset(self):
        """데이터셋 분석"""
        print("데이터셋 분석 중...")
        
        # 이미지 분석
        img_dir = self.data_root / "train_images"
        images = list(img_dir.glob("*.png"))
        print(f"전체 이미지 수: {len(images)}")
        
        # 어노테이션 분석
        ann_dir = self.data_root / "train_annotations"
        json_files = list(ann_dir.rglob("*.json"))
        
        total_boxes = 0
        class_counts = defaultdict(int)
        box_sizes = []
        
        for json_file in tqdm(json_files, desc="JSON 파일 분석"):
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            for ann in data.get('annotations', []):
                if 'bbox' in ann and len(ann['bbox']) == 4:
                    total_boxes += 1
                    class_counts[ann['category_id']] += 1
                    box_sizes.append((ann['bbox'][2], ann['bbox'][3]))
        
        # 통계 출력
        print(f"\n=== 데이터셋 통계 ===")
        print(f"전체 박스 수: {total_boxes}")
        print(f"이미지당 평균 박스: {total_boxes/len(images):.2f}")
        print(f"\n클래스별 분포:")
        for cls_id, count in sorted(class_counts.items()):
            print(f"  클래스 {cls_id}: {count}개 ({count/total_boxes*100:.1f}%)")
        
        if box_sizes:
            widths, heights = zip(*box_sizes)
            print(f"\n박스 크기 통계:")
            print(f"  평균: {np.mean(widths):.1f} x {np.mean(heights):.1f}")
            print(f"  최소: {min(widths):.0f} x {min(heights):.0f}")
            print(f"  최대: {max(widths):.0f} x {max(heights):.0f}")
        
        return {
            'num_images': len(images),
            'num_boxes': total_boxes,
            'class_distribution': dict(class_counts),
            'avg_box_size': (np.mean(widths), np.mean(heights)) if box_sizes else (0, 0)
        }
    
    def create_validation_split(self, train_dir, val_ratio=0.2):
        """학습/검증 데이터 분할"""
        print(f"\n검증 데이터 생성 중 (비율: {val_ratio*100}%)...")
        
        # 이미지 목록
        img_dir = train_dir / "train_images"
        images = list(img_dir.glob("*.png"))
        np.random.shuffle(images)
        
        # 분할
        val_size = int(len(images) * val_ratio)
        val_images = images[:val_size]
        
        # 검증 폴더 생성
        val_img_dir = train_dir / "val_images"
        val_ann_dir = train_dir / "val_annotations"
        val_img_dir.mkdir(exist_ok=True)
        val_ann_dir.mkdir(exist_ok=True)
        
        print(f"검증 이미지: {len(val_images)}개")
        
        # 파일 복사는 실제로는 하지 않고 인덱스만 반환
        val_indices = [int(img.stem) for img in val_images]
        
        return val_indices
    
    def clean_annotations(self, ann_dir):
        """어노테이션 정리 - 잘못된 데이터 제거"""
        print("\n어노테이션 정리 중...")
        
        cleaned = 0
        json_files = list(Path(ann_dir).rglob("*.json"))
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # 유효한 어노테이션만 필터링
            valid_anns = []
            for ann in data.get('annotations', []):
                bbox = ann.get('bbox', [])
                if len(bbox) == 4 and all(v > 0 for v in bbox[2:]):
                    valid_anns.append(ann)
                else:
                    cleaned += 1
            
            data['annotations'] = valid_anns
            
            # 저장
            with open(json_file, 'w') as f:
                json.dump(data, f)
        
        print(f"정리된 어노테이션: {cleaned}개")