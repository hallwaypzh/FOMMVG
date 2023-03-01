#from misc.utils import make_image
import os
from matplotlib import transforms 

import numpy as np
from PIL import Image 

import torch 
from torch.utils.data import Dataset 
from torchvision.transforms import ToTensor

from random import randint

class KpsDataset(Dataset):
    def __init__(self, kps_root, seq_len):
        super().__init__()
        self.kps_paths = [os.path.join(kps_root, name) for name in sorted(os.listdir(kps_root))]
        self.kps_paths = [name for name in self.kps_paths if np.load(name).shape[0] > seq_len]
        self.seq_len = seq_len
    
    def __getitem__(self, idx):
        kps_path = self.kps_paths[idx]
        kps = torch.from_numpy(np.load(kps_path)).squeeze(1)
        vid_len = kps.shape[0]
        idx = randint(0, vid_len - self.seq_len)
        idx0 = randint(0, vid_len - 1)
        idx1 = randint(0, vid_len - 1)
        data_dict = dict()
        data_dict['kps'] = kps[idx:idx+self.seq_len]
        data_dict['x0'] = kps[idx0]
        data_dict['x1'] = kps[idx1]
        return data_dict

    def __len__(self):
        return len(self.kps_paths)

class KpsImageDataset(Dataset):
    def __init__(self, vid_root, kps_root, seq_len, return_images=False):
        super().__init__()
        self.kps_paths = [os.path.join(kps_root, name) for name in sorted(os.listdir(kps_root))]
        self.vid_paths = [os.path.join(vid_root, name) for name in sorted(os.listdir(vid_root))]
        
        self.vid_paths = [name for name in self.vid_paths if len(os.listdir(name)) > seq_len]
        self.kps_paths = [name for name in self.kps_paths if np.load(name).shape[0] > seq_len]
        print(len(self.kps_paths), len(self.vid_paths))
        
        assert len(self.kps_paths) == len(self.vid_paths)
        self.seq_len = seq_len
        self.return_images = return_images
    
    def __getitem__(self, idx):
        kps_path = self.kps_paths[idx]
        vid_path = self.vid_paths[idx]
        img_paths = [os.path.join(vid_path, name) for name in sorted(os.listdir(vid_path))]
    
        kps = torch.from_numpy(np.load(kps_path)).squeeze(1)
        start_idx = randint(0, kps.shape[0] - self.seq_len)
        source_image = ToTensor()(Image.open(img_paths[start_idx])).unsqueeze(0)
        target_image = ToTensor()(Image.open(img_paths[start_idx+self.seq_len-1])).unsqueeze(0)
        data_dict = dict()
        if self.return_images:
            imgs = []
            for i in range(self.seq_len):
                imgs.append(ToTensor()(Image.open(img_paths[start_idx + i])))
            imgs = torch.stack(imgs)
            data_dict['imgs'] = imgs 
        data_dict['source'] = source_image 
        data_dict['kps'] = kps[start_idx:start_idx+self.seq_len]
        data_dict['vid_path'] = vid_path 
        data_dict['start_idx'] = start_idx
        data_dict['seq_len'] = self.seq_len 
        data_dict['target'] = target_image
        return data_dict
    
    def get_images(self, vid_path, start_idx, num_frames):
        img_paths = [os.path.join(vid_path, name) for name in sorted(os.listdir(vid_path))]
        imgs = []
        for i in range(num_frames):
            imgs.append(np.array(Image.open(img_paths[start_idx + i])) / 255.)
        return imgs
    
    def __len__(self):
        return len(self.kps_paths)


