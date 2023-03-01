import os
import sys
import time
import random
import logging
import argparse

import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import progressbar
from progressbar import Percentage, Bar, ETA
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from misc import utils
from misc import visualize

import math 

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int, help='gpu to use')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--log_dir', default='logs/VP', help='base directory to save logs')
parser.add_argument('--ckpt', type=str, default='', help='load ckpt for continued training') # load ckpt

parser.add_argument('--dataset', default='vox', help='dataset to train with (mnist | weizmann | h36m | bair)')
parser.add_argument('--nepochs', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--epoch_size', type=int, default=600, help='how many batches for 1 epoch')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--batch_size', default=300, type=int, help='batch size')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')

parser.add_argument('--x_dim', type=int, default=100, help='dimensionality of input')
parser.add_argument('--n_dim', type=int, default=64, help='dimensionality of noise. kth: 32')
parser.add_argument('--h_dim', type=int, default=64, help='dimensionality of hidden state of RNN')

parser.add_argument('--max_seq_len', type=int, default=16, help='number of dynamic length of frames for training.')
parser.add_argument('--weight_align', type=float, default=0.5, help='weighting for alignment between latent space from encoder and frame predictor.')
parser.add_argument('--qual_iter', type=int, default=5, help='frequency to eval the quantitative results.')
parser.add_argument('--vid_root', default='/data/pzh/datasets/vox/images', help='root directory for videos')
parser.add_argument('--kps_root', default='/data/pzh/datasets/vox/tpsmm_kps', help='root directory for keypoints')
parser.add_argument('--n_critic', default=5, type=int)
parser.add_argument('--weight_recons', default=0., type=float)
parser.add_argument('--weight_lm', default=0., type=float)
parser.add_argument('--exp_name', type=str, default='')
parser.add_argument('--normalize_h0', action='store_true')
parser.add_argument('--g_len0', default=16, type=int)
parser.add_argument('--g_update_epoch', default=400, type=int)
parser.add_argument('--d_update_epoch', default=100, type=int)
parser.add_argument('--weight_gp', type=float, default=10.)
parser.add_argument('--glr', type=float, default=0.00005)
opt = parser.parse_args()

# gpu to use
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(opt.gpu)

# setup log_dir
if opt.ckpt == '':
    log_suffix = {
        'dataset': opt.dataset,
        'batch_size':opt.batch_size,
        'seq_len': opt.max_seq_len,
        'x_dim': opt.x_dim,
        'h_dim': opt.h_dim,
        'n_dim': opt.n_dim,
        'motion_loss_weight': opt.weight_lm
    }
    log_name = opt.exp_name
    for key, val in log_suffix.items():
        log_name += '-{}_{}'.format(key, val)
    opt.log_dir = os.path.join(opt.log_dir, log_name)

else:
    states = torch.load(opt.ckpt)

    log_suffix = {
        'dataset': opt.dataset,
        'batch_size':opt.batch_size,
        'seq_len': opt.max_seq_len,
        'x_dim': opt.x_dim,
        'h_dim': opt.h_dim,
        'n_dim': opt.n_dim,
        'motion_loss_weight': opt.weight_lm
    }
    log_name = opt.exp_name
    for key, val in log_suffix.items():
        log_name += '-{}_{}'.format(key, val)
    opt.log_dir = os.path.join(opt.log_dir, log_name)

os.makedirs('%s/gen_vis/' % opt.log_dir, exist_ok=True)

# tensorboard writer
tboard_dir = os.path.join(opt.log_dir, 'tboard')
try:
    writer = SummaryWriter(log_dir=tboard_dir)
except:
    writer = SummaryWriter(logdir=tboard_dir)

# setups starts here
# logger
logger = utils.get_logger(logpath=os.path.join(opt.log_dir, 'logs'), filepath=__file__)
logger.info(opt)

# store cmd
cmd = utils.store_cmd(opt=opt)

# set seed
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
logger.info("[*] Random Seed: %d" % opt.seed)
# setups ends here 


from data.nonoverlap_video_dataset import KpsNonOverlapDataset, KpsImageNonOverlapAugDataset
from modules.tpsmm import TPSMM

train_data = KpsNonOverlapDataset(vid_root=opt.vid_root, kps_root=opt.kps_root, nframes=opt.max_seq_len, dataset_type=opt.dataset)
train_loader = DataLoader(train_data,
                          num_workers=5,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)

def get_training_batch():
    while True:
        for sequence in train_loader:
            yield sequence 
training_batch_generator = get_training_batch()

val_data = KpsImageNonOverlapAugDataset(vid_root=opt.vid_root, kps_root=opt.kps_root, nframes=opt.max_seq_len, dataset_type=opt.dataset)
val_loader = DataLoader(val_data,
                        num_workers=5,
                        batch_size=5,
                        shuffle=True,
                        drop_last=True,
                        pin_memory=True)
def get_val_batch():
    while True:
        for sequence in val_loader:
            yield sequence 
val_batch_generator = get_val_batch()


tpsmm = TPSMM()

from modules.sivg_model import KPSeqGAN_MLP_PCAMD  as KPSeqGAN
model = KPSeqGAN(opt=opt)

model.cuda()
if opt.ckpt != '':
    states = torch.load(opt.ckpt)
    start_epoch = model.load(states=states)
    logger.info("[*] Load model from %s. Training continued at: %d" % (opt.ckpt, start_epoch))
else:
    start_epoch = 0
start_epoch = 0
    
# plot
def plot_seqx(x, gen_seq, sources, epoch, mode='prior'):
    nsample = len(gen_seq)
    nframes = gen_seq[0].shape[0]

    nrow = min(opt.batch_size, 2)
    to_plot = []
    gifs = [[] for t in range(nframes)]
    
    for i in range(nrow):
        # ground truth sequence
        gt_vid = tpsmm.animate(sources[i], x[:, i].unsqueeze(1))
        
        chunk = [gt_vid]
        for l in range(nsample):
            gen_vid = tpsmm.animate(sources[i], gen_seq[l][:, i].unsqueeze(1))
            chunk.append(gen_vid)
        to_plot += chunk
        for t in range(nframes):
            row = []
            for j in range(len(chunk)):
                row.append(chunk[j][t])
            gifs[t].append(row)
    if epoch % 50 == 0:
        fname = '%s/gen_vis/sample_%s_%d.png' % (opt.log_dir, mode, epoch) 
        utils.save_grid_imagesx(fname, to_plot)

    fname = '%s/gen_vis/sample_%s_%d.gif' % (opt.log_dir, mode, epoch) 
    utils.save_gifx(fname, gifs)

# training
logger.info('[*] Using gpu: %d' % opt.gpu)
logger.info('[*] log dir: %s' % opt.log_dir)

epoch_size = opt.epoch_size
cur_D_len = 1
cur_G_len = opt.g_len0

for epoch in range(start_epoch, opt.nepochs):
    model.train()

    epoch_loss_dict = {}
    cur_D_len = min(opt.max_seq_len - 1, max(cur_D_len, epoch // opt.d_update_epoch + 1))
    model.D_len = cur_D_len
    #simport pdb; pdb.set_trace()
    print("UPDATING D_LEN:", model.D_len)
    
    progress = utils.get_progress_bar('Training epoch: %d' % epoch, epoch_size)
    for i in range(epoch_size):
        progress.update(i+1)
        data = next(training_batch_generator)
        source_x = train_data.sample_aug_index()
        x = data['kps'].cuda()
        x0 = data['x0'].cuda()

        source_x = source_x.cuda()
        aug_x = tpsmm.augment_kp_seq(source_x, x)
        aug_x = aug_x.permute(1, 0, 2)
        #aug_x = x.permute(1, 0, 2)
        #x = data['kps'].permute(1, 0, 2).cuda()
        loss_dict = model(aug_x, x0, G_len=opt.max_seq_len-1)
        for k in loss_dict.keys():
            epoch_loss_dict[k] = epoch_loss_dict.get(k, 0.) + loss_dict[k]
        # log training info
        if i % 50 == 0 and i != 0:
            step = epoch * epoch_size + i
            for k, v in loss_dict.items():
                
                writer.add_scalar(f'Train/{k}', v, step)

            for name, param in model.named_parameters():
                if param.requires_grad:
                    name = name.replace('.', '/')
                    writer.add_histogram(name, param.data.cpu().numpy(), step)
                    try:
                        writer.add_histogram(name+'/grad', param.grad.data.cpu().numpy(), step)
                    except:
                        pass
    
    progress.finish()
    utils.clear_progressbar()
    log_error = '[%02d]' % epoch 
    for k, v in epoch_loss_dict.items():
        log_error += f'| {k}: %.5f' % (v / epoch_size) + '| '
    log_error += '(%d)' % (epoch * epoch_size * opt.batch_size)
    
    logger.info(log_error)
                                                                        
    ###### qualitative results ######
    model.eval()
    with torch.no_grad():
        if (epoch + 1) % opt.qual_iter == 0: # NOTE for fast training if set opt.quan_iter larger
            end = time.time()
            data = next(val_batch_generator)
            x, src_img = data['kps'].permute(1, 0, 2).cuda(), data['source'].cuda()
            length_to_gen = x.shape[0]
            gen_seqs = []   
            for i in range(3):
                gen_seq = model.p2p_generate(x, length_to_gen)
                gen_seq = torch.stack(gen_seq)
                gen_seqs.append(gen_seq)
            plot_seqx(x, gen_seqs, src_img, epoch, mode='prior')
            print("[*] Time for qualitative results: %.4f" % (time.time() - end))
            print(opt.log_dir)

    ###### qualitative results ######
    # import pdb; pdb.set_trace()
    fname = '%s/model_%d.pth' % (opt.log_dir, 0)
    model.save(fname, epoch)

    # save the model
    if epoch % 50 == 0:
        fname = '%s/model_%d.pth' % (opt.log_dir, epoch)
        model.save(fname, epoch)
        logger.info("[*] Model saved at: %s" % fname)
        os.system("cp %s/model_%d.pth %s/model.pth" % (opt.log_dir, epoch, opt.log_dir)) # latest ckpt
        logger.info('log dir: %s' % opt.log_dir)