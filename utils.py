import math
import torch
import socket
import argparse
import os
import numpy as np
from sklearn.manifold import TSNE
import scipy.misc
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import functools

from scipy import signal
from scipy import ndimage
from PIL import Image, ImageDraw


from torchvision import datasets, transforms
from torch.autograd import Variable
import imageio


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def save_grid_images(fname, img_grid):
    # save the image grid as one img
    # basic element of img_gird should be a np arrary with shape of [H, W, 3]
    
    m, n = len(img_grid), len(img_grid[0])
    tmp = [np.concatenate(row, axis=1) for row in img_grid]
    tmp = np.concatenate(tmp)
    
    Image.fromarray(np.uint8(tmp * 255.)).save(fname)

def save_gif(filename, inputs, duration=0.25):
    images = []
 
    for frame in inputs:
        tmp = [np.concatenate(row, axis=1) for row in frame]
        tmp = np.concatenate(tmp)
        
        images.append(np.uint8(tmp * 255.)) 
    imageio.mimsave(filename, images, duration=duration)