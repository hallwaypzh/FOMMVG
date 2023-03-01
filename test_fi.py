from argparse import ArgumentParser
import os

from PIL import Image
from torchvision import transforms
import torch 

from modules.tpsmm import TPSMM
from  modules.cpc_model import P2PModelBDG_MLP_LM as P2PModel
import utils
import imageio
import numpy as np

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--src_path', type=str, default='/data/pzh/datasets/face_forensics/croped/images/634/00030.jpg')
    parser.add_argument('--tgt_path', type=str, default='/data/pzh/datasets/face_forensics/croped/images/634/00030.jpg')
    parser.add_argument('--ckpt', type=str, default='./ckpts/fi_best.pth')
    parser.add_argument('--sav_path', type=str, default='./0.gif')
    
    opt = parser.parse_args()

    device = 'cuda'
    MODEL_PATH = opt.ckpt 
    model_opt = torch.load(MODEL_PATH)['opt']
    model = P2PModel(n_dim=100, opt=model_opt)
    model.load(MODEL_PATH)
    model = model.to(device)
    
    src_img = Image.open(opt.src_path).resize((256, 256))
    tgt_img = Image.open(opt.tgt_path).resize((256, 256))
    tpsmm = TPSMM()
    
    img_id = opt.src_path.split('/')[-2]
    src_frame_index = opt.src_path.split('/')[-1]
    tgt_frame_index = opt.tgt_path.split('/')[-1]
    
    src_img = transforms.ToTensor()(src_img).unsqueeze(0).to(device)
    tgt_img = transforms.ToTensor()(tgt_img).unsqueeze(0).to(device)
    
    src_kps = tpsmm.get_kps(src_img)
    tgt_kps = tpsmm.get_kps(tgt_img)
    
    gen_seq_cpc, _, _ = model.G_step(src_kps, tgt_kps, 30)
    gen_vid_cpc = tpsmm.animate(src_img, gen_seq_cpc, vis_kp=False)
    gen_vid_cpc = (np.stack(gen_vid_cpc) * 255.).astype('uint8')

    kps_vis = []
    for i in range(gen_seq_cpc.shape[0]):
        kps_vis.append(tpsmm.draw_image_with_kp(1. * np.ones_like(gen_vid_cpc[0]), gen_seq_cpc[i].view(50, 2).view(-1, 5, 2).detach().cpu().numpy()))
    kps_vis = np.uint8(np.stack(kps_vis) *255.)

    imageio.mimsave(opt.sav_path, [img2 for img2 in gen_vid_cpc], duration=0.25)
