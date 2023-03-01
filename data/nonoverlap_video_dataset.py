import os
from matplotlib import transforms 

import numpy as np
from PIL import Image 

import torch 
from torch.utils.data import Dataset 
from torchvision.transforms import ToTensor

from random import randint
from .data_utils import make_imageclip_dataset, find_classes
import random

class KpsNonOverlapDataset(Dataset):
    
    def __init__(self, vid_root, kps_root, nframes=16, split='full', dataset_type='RAV'):
        super().__init__()
        self.vid_root = vid_root 
        self.kps_root = kps_root 
        if 'RAV' in dataset_type.upper():
            self.dataset_mode = 'RAV'
        elif 'VOX' in dataset_type.upper():
            self.dataset_mode = 'VOX'
        else:
            assert False, "Unknown dataset, support [VOX | RAV]"
        self.nframes = nframes
        self.to_tensor = ToTensor()
        classes, class_to_idx = find_classes(vid_root)
        imgs = make_imageclip_dataset(vid_root, nframes, class_to_idx=class_to_idx,
                                      vid_diverse_sampling=False, split=split)
        print(f'Total {len(imgs)} sequences')
        self.data = [(item[0][0].split('/')[-2], 
                      int(item[0][0].split('/')[-1].split('_')[-1].split('.')[0]),
                      int(item[-1][0].split('/')[-1].split('_')[-1].split('.')[0])) for item in imgs]
        self._total_size = len(self.data)
        self.shuffle_indices = [i for i in range(self._total_size)]
        self.return_images = False
        random.shuffle(self.shuffle_indices)
    
    def __len__(self):
        return self._total_size
    
    def __getitem__(self, index):
        index = self.shuffle_indices[index]
        video_name, idx0, idx1 = self.data[index]        
        if self.dataset_mode == 'VOX':
            kps_data = torch.from_numpy(np.load(os.path.join(self.kps_root, video_name.replace('.mp4', '.npy')))).squeeze(1)
        else:
            kps_data = torch.from_numpy(np.load(os.path.join(self.kps_root, f'{video_name}.npy'))).squeeze(1)
        vid_len = kps_data.shape[0]
        idx2 = random.randint(0, vid_len-1)
        data_dict = dict()
        data_dict['kps'] = kps_data[idx0:idx1+1]
        data_dict['x0'] = kps_data[idx2]
        return data_dict
    
    def sample_aug_index(self):
        index = random.choice(self.shuffle_indices)
        video_name, _, _ = self.data[index]
        #kps_data = torch.from_numpy(np.load(os.path.join(self.kps_root, f'{video_name}.npy'))).squeeze(1)
        if self.dataset_mode == 'VOX':
            kps_data = torch.from_numpy(np.load(os.path.join(self.kps_root, video_name.replace('.mp4', '.npy')))).squeeze(1)
        else:
            kps_data = torch.from_numpy(np.load(os.path.join(self.kps_root, f'{video_name}.npy'))).squeeze(1)
        seq_len = kps_data.shape[0]
        indice = random.randint(0, seq_len-1)
        return kps_data[indice:indice+1]
        
class KpsImageNonOverlapAugDataset(Dataset):
    
    def __init__(self, vid_root, kps_root, nframes=16, split='full', return_images=False, dataset_type='RAVD'):
        super().__init__()
        self.vid_root = vid_root 
        self.kps_root = kps_root 
        self.nframes = nframes
        if 'RAV' in dataset_type.upper():
            self.dataset_mode = 'RAV'
        elif 'VOX' in dataset_type.upper():
            self.dataset_mode = 'VOX'
        else:
            assert False, "Unknown dataset, support [VOX | RAV]"
            
        self.to_tensor = ToTensor()
        classes, class_to_idx = find_classes(vid_root)
        
        imgs = make_imageclip_dataset(vid_root, nframes, class_to_idx=class_to_idx,
                                      vid_diverse_sampling=False, split=split)
        print(f'Total {len(imgs)} sequences')
        self.imgs = imgs
        self._total_size = len(self.imgs)
        self.shuffle_indices = [i for i in range(self._total_size)]
        self.return_images = return_images
        random.shuffle(self.shuffle_indices)
    
    def __len__(self):
        return self._total_size
    
    def __getitem__(self, index):
        index = self.shuffle_indices[index]
        clip = self.imgs[index]
        video_name = clip[0][0].split('/')[-2]
        idx0 = int(clip[0][0].split('/')[-1].split('_')[-1].split('.')[0])
        idx1 = int(clip[-1][0].split('/')[-1].split('_')[-1].split('.')[0])
        #kps_data = torch.from_numpy(np.load(os.path.join(self.kps_root, f'{video_name}.npy'))).squeeze(1)
        if self.dataset_mode == 'VOX':
            kps_data = torch.from_numpy(np.load(os.path.join(self.kps_root, video_name.replace('.mp4', '.npy')))).squeeze(1)
        else:
            kps_data = torch.from_numpy(np.load(os.path.join(self.kps_root, f'{video_name}.npy'))).squeeze(1)
        data_dict = dict()
        data_dict['kps'] = kps_data[idx0:idx1+1]
        source_image = ToTensor()(Image.open(clip[0][0])).unsqueeze(0)
        data_dict['source'] = source_image 
        return data_dict