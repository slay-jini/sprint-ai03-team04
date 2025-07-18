# preprocess.py
"""데이터 전처리 실행"""

import argparse
from pathlib import Path
from src.data.preprocessor import DataPreprocessor


def main():
    parser = argparse.ArgumentParser(description='데이터 전처리')
    parser.add_argument('--data-path', type=str, required=True, help='데이터 루트 경로')
    parser.add_argument('--analyze', action='store_true', help='데이터 분석 수행')
    parser.add_argument('--clean', action='store_true', help='어노테이션 정리')
    parser.add_argument('--split', action='store_true', help='train/val 분할')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("데이터 전처리 시작")
    print("=" * 50)
    
    preprocessor = DataPreprocessor(args.data_path)
    
    if args.analyze or not (args.clean or args.split):
        # 기본: 분석만 수행
        stats = preprocessor.analyze_dataset()
        print("\n분석 완료!")
    
    if args.clean:
        ann_dir = Path(args.data_path) / "train_annotations"
        preprocessor.clean_annotations(ann_dir)
        print("\n정리 완료!")
    
    if args.split:
        val_indices = preprocessor.create_validation_split(Path(args.data_path))
        print(f"\n검증 세트 생성 완료: {len(val_indices)}개 이미지")


if __name__ == '__main__':
    main()
