import os
import sys
import time
import random
import logging
import argparse
from datetime import datetime

import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import itertools
import progressbar
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as vutils


from misc import utils
from misc import visualize
 

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int, help='gpu to use')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--log_dir', default='logs/fi', help='base directory to save logs')
parser.add_argument('--data_root', default='data_root', help='root directory for data')
parser.add_argument('--ckpt', type=str, default='', help='load ckpt for continued training') # load ckpt

parser.add_argument('--dataset', default='kps_vox', help='dataset to train with (mnist | weizmann | h36m | bair)')
parser.add_argument('--nepochs', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--epoch_size', type=int, default=300, help='how many batches for 1 epoch')
parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')

parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
parser.add_argument('--prior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
parser.add_argument('--x_dim', type=int, default=100, help='dimensionality of input')
parser.add_argument('--z_dim', type=int, default=30, help='dimensionality of z_t. kth: 32')
parser.add_argument('--g_dim', type=int, default=70, help='dimensionality of encoder output vector and decoder input vector')
parser.add_argument('--max_seq_len', type=int, default=16, help='number of dynamic length of frames for training.')
parser.add_argument('--qual_iter', type=int, default=10, help='frequency to eval the quantitative results.')
parser.add_argument('--vid_root', default='/data/pzh/datasets/vox/images/full_images', help='root directory for videos')
parser.add_argument('--kps_root', default='/data/pzh/datasets/vox/tpsmm_kps', help='root directory for keypoints')
parser.add_argument('--weight_recons', default=0., type=float)
parser.add_argument('--weight_lm', default=10., type=float)

opt = parser.parse_args()

# gpu to use
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(opt.gpu)

# setup log_dir
if opt.ckpt == '':
    log_suffix = {
        'dataset': opt.dataset,
        'batch_size':opt.batch_size,
        'x_dim': opt.x_dim,
        'g_dim': opt.g_dim,
        'z_dim': opt.z_dim,
        'rnn_size': opt.rnn_size,
        'weight_lm': opt.weight_lm,
        'weight_recons': opt.weight_recons
    }

    log_name = 'P2PModel'
    for key, val in log_suffix.items():
        log_name += '-{}_{}'.format(key, val)

    opt.log_dir = os.path.join(opt.log_dir, log_name)
else:
    states = torch.load(opt.ckpt)
    opt.log_dir = states['opt'].log_dir

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


# setup datasets
from data.KPSDataset_new import KpsDataset, KpsImageDataset
from modules.tpsmm import TPSMM

train_data = KpsDataset(kps_root=opt.kps_root, seq_len=opt.max_seq_len)
train_loader = DataLoader(train_data,
                          num_workers=20,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)
def get_training_batch():
    while True:
        for sequence in train_loader:
            yield sequence
training_batch_generator = get_training_batch()

val_data = KpsImageDataset(vid_root=opt.vid_root, kps_root=opt.kps_root, seq_len=opt.max_seq_len)
val_loader = DataLoader(val_data,
                          num_workers=10,
                          batch_size=10,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)
def get_val_batch():
    while True:
        for sequence in val_loader:
            yield sequence
val_batch_generator = get_val_batch()


tpsmm = TPSMM()

# from modules.p2p_modelx import P2PModelG
# model = P2PModelG(opt=opt)
from modules.cpc_model import P2PModelBDG_MLP_LM as P2PModel
model = P2PModel(n_dim=100, opt=opt)

# criterions
mse_criterion = nn.MSELoss()
model.cuda()


if opt.ckpt != '':
    states = torch.load(opt.ckpt)
    start_epoch = model.load(states=states)
    logger.info("[*] Load model from %s. Training continued at: %d" % (opt.ckpt, start_epoch))
else:
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
    if epoch % 20 == 0:
        fname = '%s/gen_vis/sample_%s_%d.png' % (opt.log_dir, mode, epoch) 
        utils.save_grid_imagesx(fname, to_plot)

    fname = '%s/gen_vis/sample_%s_%d.gif' % (opt.log_dir, mode, epoch) 
    utils.save_gifx(fname, gifs)
 


# training

# num of lengths to gen for qualitative results
#qual_lengths = [10, opt.max_seq_len]
qual_lengths = [10, 30]

logger.info('[*] Using gpu: %d' % opt.gpu)
logger.info('[*] log dir: %s' % opt.log_dir)

epoch_size = opt.epoch_size
cur_D_len = 1

for epoch in range(start_epoch, opt.nepochs):
    model.train()

    epoch_loss_dict = {}#{'loss_G': 0., 'loss_D': 0., 'loss_recon': 0.}
    progress = utils.get_progress_bar('Training epoch: %d' % epoch, epoch_size)
    for i in range(epoch_size):
        progress.update(i+1)

        data = next(training_batch_generator)
        x0, x1, real_seq = data['x0'].cuda(), data['x1'].cuda(), data['kps'].permute(1, 0, 2).cuda()
        # train p2p model
        start_ix = 0
        loss_dict = model(x0, x1, real_seq, i)
        for k in loss_dict.keys():
            epoch_loss_dict[k] = epoch_loss_dict.get(k, 0.) + loss_dict[k]
        # log training info
        if i % 50 == 0 and i != 0:
            step = epoch * epoch_size + i
            for k, v in epoch_loss_dict.items():
                writer.add_scalar(f'Train/{k}', v/i, step)

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
    ###### qualitative results ######
    # import pdb; pdb.set_trace()
    if epoch % 50 == 0:
        fname = '%s/model_%d.pth' % (opt.log_dir, epoch)
        model.save(fname, epoch)

    cur_D_len = min(opt.max_seq_len // 2, max(cur_D_len, epoch // 200 + 1))
    model.D_len = cur_D_len
    print("UPDATING D_LEN:", model.D_len)
