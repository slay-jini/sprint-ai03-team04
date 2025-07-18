import os
import json
import shutil
from pathlib import Path
from collections import defaultdict

def prepare_dataset(base_dir):
    base_dir = Path(base_dir)
    
    # 출력 디렉토리 생성
    output_dir = base_dir / 'dataset'
    output_dir.mkdir(exist_ok=True)
    (output_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    
    # 데이터 수집
    annotations_dir = base_dir / 'data' / 'train_annotations'
    images_dir = base_dir / 'data' / 'train_images'
    test_images_dir = base_dir / 'data' / 'test_images'
    
    # COCO 형식의 데이터셋 초기화
    dataset = {
        'images': [],
        'annotations': [],
        'categories': []
    }
    
    # 카테고리 정보 수집
    categories = {}
    category_count = defaultdict(int)
    
    # JSON 파일들을 재귀적으로 찾기
    for json_path in annotations_dir.rglob('*.json'):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            if not data.get('annotations'):
                continue
                
            # 카테고리 정보 수집
            for cat in data['categories']:
                if cat['id'] not in categories:
                    categories[cat['id']] = cat
                category_count[cat['id']] += 1
            
            # 이미지와 어노테이션 정보 수집
            for img in data['images']:
                dataset['images'].append(img)
                
                # 이미지 파일 복사
                img_name = img['file_name'] if 'file_name' in img else f"{img['id']}.jpg"
                src_path = images_dir / img_name
                if src_path.exists():
                    shutil.copy2(src_path, output_dir / 'images' / 'train' / img_name)
            
            dataset['annotations'].extend(data['annotations'])
    
    # 카테고리 정보 정리
    dataset['categories'] = list(categories.values())
    
    # 데이터셋 통계
    print("\n=== 데이터셋 통계 ===")
    print(f"총 이미지 수: {len(dataset['images'])}")
    print(f"총 어노테이션 수: {len(dataset['annotations'])}")
    print(f"총 카테고리 수: {len(dataset['categories'])}")
    print("\n카테고리별 이미지 수:")
    for cat_id, count in category_count.items():
        cat_name = categories[cat_id]['name']
        print(f"- {cat_name}: {count}개")
    
    # COCO 형식으로 저장
    with open(output_dir / 'train.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print("\n데이터셋 준비 완료!")
    print(f"- 어노테이션 파일: {output_dir / 'train.json'}")
    print(f"- 학습 이미지: {output_dir / 'images/train'}")

if __name__ == "__main__":
    base_dir = Path(r"c:\Users\ADMIN\OneDrive\바탕 화면\baseline")
    prepare_dataset(base_dir)
