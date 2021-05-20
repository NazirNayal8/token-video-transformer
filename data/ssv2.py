import os
import cv2
import json
import numpy as np
import random
import math
import torch
from torch.utils.data import Dataset
from typing import Any

class SSV2Dataset(Dataset):
    """
    Pytorch Dataset for Something-Something-V2 Dataset
    """
    
    dataset_root = '/datasets/20bn_something_something/v2/'
    labels_root = os.path.join(dataset_root, 'labels')
    videos_path = os.path.join(dataset_root, 'videos')
    train_path = os.path.join(labels_root, 'something-something-v2-train.json')
    valid_path = os.path.join(labels_root, 'something-something-v2-validation.json')
    test_path = os.path.join(labels_root, 'something-something-v2-test.json')
    labels_path = os.path.join(labels_root, 'something-something-v2-labels.json')
    
    def __init__(self, mode: str, num_samples: int, transforms=None, filter_by_labels=None):
        super().__init__()
        
        self.mode = mode
        self.num_samples = num_samples
        self.transforms = transforms
        self.labels_dict = self._read_labels_dict()
        self.labels = self._read_labels(mode)
        
        if filter_by_labels is not None:
            self.labels = filter_by_labels(self.labels)
        
        self.data_len = len(self.labels)
    
    def _read_labels(self, mode: str):
        
        path = None
        
        if mode == 'train':
            path = self.train_path
        elif mode == 'valid':
            path = self.valid_path
        elif mode == 'test':
            path = self.test_path
        else:
            raise ValueError(f"Undefined SSV2 Dataset Mode: {mode}")
        
        with open(path, 'rb') as f:
            data = json.load(f)
            
        if mode == 'test':
            return data
            
        for i in range(len(data)):
            idx = data[i]['id']
            label = self._clean_label(data[i]['template'])
            data[i] = {
                'id': data[i]['id'],
                'label_str': label,
                'label': int(self.labels_dict[label])
            }

        return data
    
    def _clean_label(self, label: str):
        
        label = label.replace("[", "")
        label = label.replace("]", "")
        
        return label
    
    def _read_labels_dict(self):
        
        with open(self.labels_path, 'rb') as f:
            labels = json.load(f)
        
        return labels
    
    def _read_video_frames(self, path: str) -> (np.ndarray, float):
    
        vid = cv2.VideoCapture(path)
        fps = vid.get(cv2.CAP_PROP_FPS)

        frames = []
        while vid.isOpened():
            ret, frame = vid.read()

            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        return np.asarray(frames), fps
    
    def _sample_random(self, frames: np.ndarray) -> np.ndarray:
        
        num_frames = frames.shape[0]
        num_samples = self.num_samples
        # if number of required frames is more than available frames
        # then set number of required frames to available frames
        if num_frames < num_samples:
            
            frame = frames[-1][None,:,:,:]
            diff = num_samples - num_frames
            frames = np.concatenate([frames, np.concatenate([frame for _ in range(diff)])])
            num_frames = num_samples
       
        seg_len = math.ceil(num_frames / num_samples)
        
        segments = np.array_split(range(num_frames), num_samples)
        selected_frames = []
        for seg in segments:
            selected_frames.append(np.random.choice(seg))
    
        return frames[selected_frames]
            
        
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, index: int):
        
        if self.mode != 'test':
            label = self.labels[index]['label']
        
        vid_id = self.labels[index]['id']
        vid_path = os.path.join(self.videos_path, str(vid_id) + '.webm')
        frames, fps = self._read_video_frames(vid_path)
        
        frames = self._sample_random(frames)
        
        if self.transforms:
            frames_transformed = [0] * frames.shape[0]
            for i in range(frames.shape[0]):
                frames_transformed[i] = self.transforms(frames[i]).unsqueeze(0)
            frames = torch.cat(frames_transformed, axis=0)
        
        if self.mode == 'test':
            return frames
        
        return frames, label