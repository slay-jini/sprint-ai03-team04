# src/utils/__init__.py
from .config import load_config, merge_configs
from .visualization import Visualizer
import argparse

def create_parser():
    """명령줄 인자 파서 생성"""
    parser = argparse.ArgumentParser(description='알약 검출 모델 학습')
    parser.add_argument('--config', type=str, required=True,
                       help='설정 파일 경로 (예: configs/faster_rcnn.yaml)')
    parser.add_argument('--resume', type=str, default=None,
                       help='체크포인트 파일 경로')
    parser.add_argument('--output', type=str, default='outputs/',
                       help='출력 디렉토리 경로')
    parser.add_argument('--gpu', type=int, default=0,
                       help='사용할 GPU 번호')

    return parser

__all__ = ['load_config', 'merge_configs', 'Visualizer', 'create_parser']

