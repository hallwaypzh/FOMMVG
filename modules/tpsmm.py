from torch import nn 
import torch 
import yaml 
import numpy as np
from PIL import Image

from .inpainting_network import InpaintingNetwork 
from .keypoint_detector import KPDetector 
from .dense_motion import DenseMotionNetwork 
from .avd_network import AVDNetwork
from skimage.draw import circle
import matplotlib.pyplot as plt

def load_checkpoints(config_path, checkpoint_path, device):
    with open(config_path) as f:
        config = yaml.load(f)

    inpainting = InpaintingNetwork(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    kp_detector = KPDetector(**config['model_params']['common_params'])
    dense_motion_network = DenseMotionNetwork(**config['model_params']['common_params'],
                                              **config['model_params']['dense_motion_params'])
    avd_network = AVDNetwork(num_tps=config['model_params']['common_params']['num_tps'],
                             **config['model_params']['avd_network_params'])
    kp_detector.to(device)
    dense_motion_network.to(device)
    inpainting.to(device)
    avd_network.to(device)
       
    checkpoint = torch.load(checkpoint_path, map_location=device)
 
    inpainting.load_state_dict(checkpoint['inpainting_network'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    dense_motion_network.load_state_dict(checkpoint['dense_motion_network'])
    if 'avd_network' in checkpoint:
        avd_network.load_state_dict(checkpoint['avd_network'])
    
    inpainting.eval()
    kp_detector.eval()
    dense_motion_network.eval()
    avd_network.eval()
    
    return inpainting, kp_detector, dense_motion_network, avd_network

class TPSMM:
    def __init__(self, \
        config_path='./configs/vox-256.yaml', \
        checkpoint_path='./pretrained_models/vox.pth.tar', device='cuda', colormap='gist_rainbow'):
        inpainting, kp_detector, dense_motion_network, avd_network = load_checkpoints(config_path=config_path, checkpoint_path=checkpoint_path, device=device)
        self.inpainting = inpainting.eval()
        self.kp_detector = kp_detector.eval()
        self.dense_motion_network = dense_motion_network.eval()
        self.avd_network = avd_network.eval()
        self.colormap = plt.get_cmap(colormap)
        self.kp_size = 10
    
    @torch.no_grad()
    def animate(self, source_image, kp_sequence, vis_kp=True):
        # kp_sequence Lx(Nx2)
        # source_image 1x3xHxW
        imgs = []
        kp_source = { 'fg_kp' : kp_sequence[0].view(50, 2).unsqueeze(0) }
        for i in range(kp_sequence.shape[0]):
            kp_norm = { 'fg_kp' : kp_sequence[i].view(50, 2).unsqueeze(0) }
            dense_motion = self.dense_motion_network(source_image=source_image, kp_driving=kp_norm,
                                                    kp_source=kp_source, bg_param = None, 
                                                    dropout_flag = False)
            out = self.inpainting(source_image, dense_motion)
            imgs.append(out['prediction'][0].permute(1, 2, 0).data.cpu().numpy())
        
        if vis_kp:
            for i in range(kp_sequence.shape[0]):
                imgs[i] = self.draw_image_with_kp(imgs[i], kp_sequence[i].view(50, 2).view(-1, 5, 2).detach().cpu().numpy())
        return imgs

    
    def draw_image_with_kp(self, image, kp_array):
        image = np.copy(image)
        spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
        kp_array = spatial_size * (kp_array + 1) / 2
        # for kp_ind, kp in enumerate(kp_array):
        #     rr, cc = circle(kp[1], kp[0], 3, shape=image.shape[:2])
        #     image[rr, cc] = np.array(self.colormap(kp_ind // 5 / 10))[:3]
        #     print(kp_ind, kp_ind // 5 / 10, kp)
        num_cluster = kp_array.shape[0]
        for i in range(num_cluster):
            for j in range(kp_array.shape[1]):
                kp = kp_array[i][j]
                rr, cc = circle(kp[1], kp[0], 3, shape=image.shape[:2])
                image[rr, cc] = np.array(self.colormap(i / 10))[:3]
        return image
    
    
    def generate(self, source_image, kp_src, kp_tgt):
        kp_source = { 'fg_kp' : kp_src.view(-1, 50, 2) }
        kp_norm = { 'fg_kp' : kp_tgt.view(-1, 50, 2) }
        
        dense_motion = self.dense_motion_network(source_image=source_image, kp_driving=kp_norm,
                                                kp_source=kp_source, bg_param = None, 
                                                dropout_flag = False)
        return self.inpainting(source_image, dense_motion)['prediction']


    @torch.no_grad()
    def animate_seq(self, source_image, kp_sequence, vis_kp=True):
        # kp_sequence Lx(Nx2)
        # source_image 1x3xHxW
        imgs = []
        kp_source = { 'fg_kp' : kp_sequence[0].view(50, 2).unsqueeze(0) }
        for i in range(kp_sequence.shape[0]):
            kp_norm = { 'fg_kp' : kp_sequence[i].view(50, 2).unsqueeze(0) }
            dense_motion = self.dense_motion_network(source_image=source_image, kp_driving=kp_norm,
                                                    kp_source=kp_source, bg_param = None, 
                                                    dropout_flag = False)
            out = self.inpainting(source_image, dense_motion)
            source_image = out['prediction']
            kp_source = kp_norm 
            imgs.append(out['prediction'][0].permute(1, 2, 0).data.cpu().numpy())
        if vis_kp:
            for i in range(kp_sequence.shape[0]):
                imgs[i] = self.draw_image_with_kp(imgs[i], kp_sequence[i].view(50, 2).view(-1, 5, 2).detach().cpu().numpy())
        return imgs
    
    def get_kps(self, image):
        bs = image.shape[0]
        return self.kp_detector(image)['fg_kp'].view(bs, -1)

    
    @torch.no_grad()
    def augment_kp_seq(self, source_kp, kp_sequences):
        B, T, N = kp_sequences.shape 
        kp_sequences = kp_sequences.view(-1, N)
        source_kp = source_kp.repeat(kp_sequences.shape[0], 1)
        aug_kp_sequences = self.avd_network(source_kp, kp_sequences)
        return aug_kp_sequences.view(B, T, N)

    @torch.no_grad()
    def animate_batch(self, source_image, kp_sequence, vis_kp=False):
        # kp_sequence  LxBx(Nx2)
        # source_image Bx3xHxW
        batch_size = source_image.shape[0]
        imgs = []
        
        kp_source = {'fg_kp' : kp_sequence[0].view(batch_size, 50, 2)}
        
        for i in range(kp_sequence.shape[0]):
            kp_norm = {'fg_kp' : kp_sequence[i].view(batch_size, 50, 2)}
            dense_motion = self.dense_motion_network(source_image=source_image, kp_driving=kp_norm,
                                                    kp_source=kp_source, bg_param = None,
                                                    dropout_flag = False)
            out = self.inpainting(source_image, dense_motion)
            imgs.append(out['prediction'].permute(0, 2, 3, 1).data.cpu().numpy())
        return imgs