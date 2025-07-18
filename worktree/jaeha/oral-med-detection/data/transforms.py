# data/transforms.py
import torch
import torchvision.transforms.v2 as T

def get_basic_transform():
    transforms = []
    transforms.append(T.ToImage())
    transforms.append(T.ToDtype(torch.float32, scale=True))
    return T.Compose(transforms)


def get_transform(train: bool):
    transforms = []
    transforms.append(T.ToImage())
    transforms.append(T.ToDtype(torch.float32, scale=True))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_augmented_transform():
    # 추후 이 곳에 회전, 밝기 조절, CopyPaste 등의 증강 로직을 추가합니다.
    pass

