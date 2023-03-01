import os 
import click
import sys
sys.path.extend(['.', 'src'])
from tqdm import tqdm
import random

import numpy as np
import torch
from modules.tpsmm import TPSMM
from torchvision.io import read_video, write_video, write_png
from torchvision.transforms import transforms
from PIL import Image 


@click.command()
@click.pass_context
@click.option('--network_pkl', help='Network pickle filename', default='./ckpts/vp_best.pth', required=True)
@click.option('--image_path', default='./figs/imgs/00002.jpg', required=True)
@click.option('--nframes', type=int, default=59, help='# of frames to generate.')
@click.option('--result_dir', type=str, default='./0.mp4')

def main(
    ctx: click.Context,
    network_pkl: str,
    image_path:str,
    nframes: int,
    result_dir: str,
    device='cuda'
    ):
    opt = torch.load(network_pkl)['opt']
    from modules.sivg_model import KPSeqGAN_MLP_PCAMD as KPSeqGAN
    from PIL import Image
    motion_generator = KPSeqGAN(opt=opt)
    motion_generator = motion_generator.to(device)
    motion_generator.load(network_pkl)
    tpsmm = TPSMM()
    

    src = Image.open(image_path)
    src = transforms.ToTensor()(src).unsqueeze(0).cuda()
    os.makedirs(result_dir, exist_ok=True)
    src = src.cuda()
    src_kps = tpsmm.get_kps(src)
        
    gen_seq, _, _ = motion_generator.G_step(src_kps, nframes)
    gen_vid = tpsmm.animate_batch(src, gen_seq, vis_kp=False)
    gen_vid = (np.stack(gen_vid) * 255.).astype('uint8')
    gen_vid = gen_vid.transpose(1, 0, 2, 3, 4)
    write_video(result_dir, gen_vid[0], fps=15.)

if __name__ == '__main__':
    main()