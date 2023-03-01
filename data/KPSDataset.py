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
    def __init__(self, vid_root, kps_root, seq_len, return_images=False):
        super().__init__()
        self.kps_paths = [os.path.join(kps_root, name) for name in sorted(os.listdir(kps_root))]
        self.vid_paths = [os.path.join(vid_root, name) for name in sorted(os.listdir(vid_root))]
        
        self.vid_paths = [name for name in self.vid_paths if len(os.listdir(name)) > seq_len]
        self.kps_paths = [name for name in self.kps_paths if np.load(name).shape[0] > seq_len]
        #import pdb; pdb.set_trace()
        # self.vid_paths = [self.vid_paths[199]] * 200
        # self.kps_paths = [self.kps_paths[199]] * 200
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

from .data_utils import make_imageclip_dataset, find_classes
import random 

class KpsNonOverlapDataset(Dataset):
    
    def __init__(self, vid_root, kps_root, nframes=16, split='full'):
        super().__init__()
        self.vid_root = vid_root 
        self.kps_root = kps_root 
        self.nframes = nframes
        self.to_tensor = ToTensor()
        classes, class_to_idx = find_classes(vid_root)
        
        imgs = make_imageclip_dataset(vid_root, nframes, class_to_idx=class_to_idx,
                                      vid_diverse_sampling=False, split=split)
        print(f'Total {len(imgs)} sequences')
        self.imgs = imgs
        self._total_size = len(self.imgs)
        self.shuffle_indices = [i for i in range(self._total_size)]
        self.return_images = False
        random.shuffle(self.shuffle_indices)
    
    def set_return_image(self, status):
        self.return_images = status 
    
    def __len__(self):
        return self._total_size
    
    def __getitem__(self, index):
        index = self.shuffle_indices[index]
        clip = self.imgs[index]
        video_name = clip[0][0].split('/')[-2]
        idx0 = int(clip[0][0].split('/')[-1].split('_')[-1].split('.')[0])
        idx1 = int(clip[-1][0].split('/')[-1].split('_')[-1].split('.')[0])
        
        kps_data = torch.from_numpy(np.load(os.path.join(self.kps_root, f'{video_name}.npy'))).squeeze(1)
        #import pdb; pdb.set_trace()
        vid_len = kps_data.shape[0]
        idx2 = random.randint(0, vid_len-1)
        
        data_dict = dict()
        data_dict['kps'] = kps_data[idx0:idx1+1]
        data_dict['x0'] = kps_data[idx2]
        source_image = ToTensor()(Image.open(clip[0][0])).unsqueeze(0)
        data_dict['source'] = source_image 
        return data_dict
    
    def sample_aug_index(self):
        index = random.choice(self.shuffle_indices)
        index = random.choice(self.imgs[index])[0]
        video_name = index.split('/')[-2]
        kps_data = torch.from_numpy(np.load(os.path.join(self.kps_root, f'{video_name}.npy'))).squeeze(1)
        
        seq_len = kps_data.shape[0]
        indice = random.randint(0, seq_len-1)
        
        return kps_data[indice:indice+1]
        
class KpsNonOverlapAugDataset(Dataset):
    
    def __init__(self, vid_root, kps_root, nframes=16, split='full', return_images=False):
        super().__init__()
        self.vid_root = vid_root 
        self.kps_root = kps_root 
        self.nframes = nframes
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
        
        kps_data = torch.from_numpy(np.load(os.path.join(self.kps_root, f'{video_name}.npy'))).squeeze(1)
        
        data_dict = dict()
        data_dict['kps'] = kps_data[idx0:idx1+1]

        source_image = ToTensor()(Image.open(clip[0][0])).unsqueeze(0)
        data_dict['source'] = source_image 
        return data_dict
        

# if __name__ == '__main__':
#     # dataset = KpsDataset(vid_root='/data/pzh/datasets/vox/images/full_images', kps_root='/data/pzh/datasets/vox/tpsmm_kps', seq_len=11)
#     # from torch.utils.data.dataloader import DataLoader
#     # dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
#     # for i, data in enumerate(dataloader):
#     #     print(i, data['images'].shape, data['kps'].shape)
#     #     exit()
#     from data_utils import make_imageclip_dataset 
#     from data_utils import find_classes
    
#     classes, class2idx = find_classes('/data/pzh/datasets/vox/images/')
#     print(class2idx)
    
#     training_dataset = make_imageclip_dataset('/data/pzh/datasets/vox/images/',
#                                      16, class2idx, vid_diverse_sampling=False, split='train')
#     classes, class2idx = find_classes('/data/pzh/datasets/vox/images/test')
#     print(class2idx)
#     testing_dataset = make_imageclip_dataset('/data/pzh/datasets/vox/images/test',
#                                      16, class2idx, vid_diverse_sampling=False, split='test')
    
#     import pdb; pdb.set_trace()

