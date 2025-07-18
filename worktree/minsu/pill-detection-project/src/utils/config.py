# src/utils/config.py
"""설정 파일 관리"""

import yaml
from pathlib import Path
import argparse


def load_config(config_path):
    """YAML 설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
 # 상속 처리
    if '_base_' in config:
        base_path = Path(config_path).parent / config['_base_']
        base_config = load_config(base_path)
        
        # 기본 설정을 복사하고 현재 설정으로 오버라이드
        merged_config = deep_merge(base_config, config)
        merged_config.pop('_base_', None)  # _base_ 키 제거
        
        return merged_config

    # Path 객체로 변환
    if 'data' in config:
        for key in ['train_path', 'val_path', 'test_path']:
            if key in config['data']:
                config['data'][key] = Path(config['data'][key])
    
    if 'output' in config:
        for key in config['output']:
            config['output'][key] = Path(config['output'][key])
    
    return config

def deep_merge(base_dict, override_dict):
    """딕셔너리 깊은 병합"""
    result = base_dict.copy()
    
    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result

def merge_configs(base_config, model_config):
    """베이스 설정과 모델 설정 병합"""
    import copy
    merged = copy.deepcopy(base_config)
    
    # 재귀적으로 병합
    def update_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = update_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    return update_dict(merged, model_config)


def create_parser():
    """명령줄 인자 파서 생성"""
    parser = argparse.ArgumentParser(description='알약 검출 시스템')
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/base_config.yaml',
        help='설정 파일 경로'
    )
    
    parser.add_argument(
        '--model-config',
        type=str,
        help='모델별 설정 파일 (선택사항)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        help='실행 디바이스'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='배치 크기 (설정 파일 오버라이드)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='학습 에폭 수 (설정 파일 오버라이드)'
    )
    
    return parser