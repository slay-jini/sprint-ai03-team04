import torch

def collate_fn(batch):
    """
    데이터로더에서 None 값을 안전하게 처리하는 collate 함수.

    In short, `lambda batch: tuple(zip(*filter(None, batch)))`
    """
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return (None, None)
    return tuple(zip(*batch))
