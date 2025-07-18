import json
import random
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
from PIL import Image, ImageFile
from pathlib import Path

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우의 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False     # 마이너스 기호 깨짐 방지

# PIL 관련 경고 무시
warnings.filterwarnings('ignore', category=UserWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Matplotlib 경고 무시
plt.rcParams['figure.max_open_warning'] = 0

def visualize_samples(dataset_dir, num_samples=5):
    dataset_dir = Path(dataset_dir)
    
    # Load annotations
    with open(dataset_dir / 'train.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create mapping from image_id to annotations
    image_to_anns = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_to_anns:
            image_to_anns[img_id] = []
        image_to_anns[img_id].append(ann)
    
    # Create category id to name mapping
    cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    
    # Randomly select images
    selected_images = random.sample(data['images'], min(num_samples, len(data['images'])))
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
    if num_samples == 1:
        axes = [axes]
    
    # Plot each image with its bounding boxes
    for ax, img_info in zip(axes, selected_images):
        # Load image
        img_path = dataset_dir / 'images' / 'train' / img_info['file_name']
        if not img_path.exists():
            print(f"이미지를 찾을 수 없습니다: {img_path}")
            continue
            
        img = Image.open(img_path)
        ax.imshow(img)
        
        # Draw bounding boxes
        if img_info['id'] in image_to_anns:
            for ann in image_to_anns[img_info['id']]:
                bbox = ann['bbox']  # [x, y, width, height]
                category_name = cat_id_to_name[ann['category_id']]
                
                # Create a Rectangle patch
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]), bbox[2], bbox[3],
                    linewidth=2, edgecolor='r', facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add label with category ID
                label_text = f"ID:{ann['category_id']} - {category_name}"
                ax.text(
                    bbox[0], bbox[1]-10, label_text,
                    color='red', fontsize=8, 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
                )
        
        ax.axis('off')
    
    plt.tight_layout()
    
    # 저장할 디렉토리 생성
    output_dir = dataset_dir.parent / 'visualization'
    output_dir.mkdir(exist_ok=True)
    
    # 결과 저장
    plt.savefig(output_dir / 'sample_visualizations.png', 
                bbox_inches='tight', 
                dpi=300)
    print(f"시각화 결과가 저장되었습니다: {output_dir / 'sample_visualizations.png'}")
    plt.show()

if __name__ == "__main__":
    dataset_dir = Path(r"c:\Users\ADMIN\OneDrive\바탕 화면\baseline\dataset")
    visualize_samples(dataset_dir, num_samples=5)
