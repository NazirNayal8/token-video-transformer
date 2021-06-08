import numpy as np
import torch
from torch import Tensor
from pytorchvideo import transforms as video_transforms
from typing import Union


def to_tensor(frames: Union[np.ndarray, Tensor]) -> Tensor:
    """
    frames of shape (T, 3, H, W) if torch tensor, or (T, H, W, 3) if numpy array
    """
    assert len(frames.shape) == 4, "Shape of frames should be 4-dimensional"

    if frames.shape[3] == 3:
        frames = frames.transpose(0, 3, 1, 2).astype(np.float32) # -> (T, 3, H, W)
        frames = torch.from_numpy(frames)
    
    frames /= 255.0
    frames = frames.permute(1, 0, 2, 3)
    return frames


def uniform_crop(size: int):

    def uniform_crop_func(frames: Tensor) -> Tensor:
        result = video_transforms.functional.uniform_crop(frames, size)
        return result
    
    return uniform_crop_func

def permute(permutation):

    def permute_func(frames: Tensor) -> Tensor:
        frames = frames.permute(*permutation)
        return frames
    
    return permute_func